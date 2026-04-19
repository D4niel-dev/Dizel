"""
training/sft.py — Supervised Fine-Tuning (SFT) for Dizel chat format.

SFT teaches the model to follow a simple conversation template:

    <|system|> You are Dizel... <|end|>
    <|user|>   Hello! <|end|>
    <|assistant|> Hi there! How can I help you? <|end|> </s>

Loss is only computed on assistant tokens (loss_mask = 0 on user/system).
This is equivalent to behaviour cloning on human–assistant demonstrations.

JSON output support
-------------------
Include examples in sft_data/chat.jsonl where the assistant turn contains
valid JSON.  The model will learn to produce structured output when the user
includes the [json] keyword.

Usage
-----
    python training/sft.py --base_checkpoint checkpoints/dizel-pretrain-best.pt
"""

import argparse
import os
import sys
import time
import math
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG, DizelConfig
from model.architecture import DizelLM
from training.dataset import Tokenizer, build_sft_loaders, build_mixed_sft_loaders
from training.data_mixing import DEFAULT_MIX, DatasetMixConfig, DatasetMixEntry
from training.pretrain import (
    cosine_lr_with_warmup, set_lr, evaluate, save_checkpoint, load_checkpoint,
    save_tokenizer,
)


# ---------------------------------------------------------------------------
# SFT evaluate (uses loss_mask)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sft_evaluate(model, val_loader, eval_iters, device, amp_ctx) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in val_loader:
        if count >= eval_iters:
            break
        x, y, mask = (t.to(device) for t in batch)
        with amp_ctx:
            _, loss = model(x, y, loss_mask=mask)
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def sft_train(cfg: DizelConfig, base_ckpt: str) -> None:
    sc = cfg.sft
    mc = cfg.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sft] Device : {device}")

    use_amp  = sc.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if sc.amp_dtype == "bfloat16" else torch.float16
    amp_ctx  = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer()
    mc.vocab_size = len(tokenizer)

    # ── Data ─────────────────────────────────────────────────────────────
    use_mix = getattr(cfg, '_use_mix', True)
    if use_mix:
        mix_config = getattr(cfg, '_mix_config', DEFAULT_MIX)
        print(f"[sft] Using mixed dataset ({len(mix_config.datasets)} sources)")
        try:
            train_loader, val_loader = build_mixed_sft_loaders(
                mix_config     = mix_config,
                tokenizer      = tokenizer,
                context_length = mc.context_length,
                batch_size     = sc.batch_size,
            )
        except Exception as e:
            print(f"[sft] ⚠ Mixed dataset failed ({e}), falling back to single file")
            use_mix = False

    if not use_mix:
        if not os.path.exists(sc.sft_data_path):
            print(f"[sft] ⚠ SFT data not found at '{sc.sft_data_path}'")
            print(f"[sft]   Run: python sft_data/generate_sft_data.py")
            sys.exit(1)
        train_loader, val_loader = build_sft_loaders(
            jsonl_path     = sc.sft_data_path,
            tokenizer      = tokenizer,
            context_length = mc.context_length,
            batch_size     = sc.batch_size,
        )

    # ── Model (load from pretrain checkpoint) ────────────────────────────
    model = DizelLM(mc).to(device)

    # Delay loading checkpoint until optimizer is created so we can load its state if resuming

    print(f"[sft] Model : {model}")

    # ── Optimiser (lower LR than pretrain) ───────────────────────────────
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
         "weight_decay": sc.weight_decay},
        {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=sc.lr, betas=(0.9, 0.95), eps=1e-8)

    # ── Training ─────────────────────────────────────────────────────────
    os.makedirs(sc.output_dir, exist_ok=True)
    best_val   = float("inf")
    step       = 0
    
    if base_ckpt and os.path.exists(base_ckpt):
        # We assume if the filename has 'sft', we are resuming an SFT run, not starting from pretrain
        is_resume = "sft" in base_ckpt.lower()
        loaded_step = load_checkpoint(base_ckpt, model, optimizer=optimizer if is_resume else None, device=str(device))
        if is_resume:
            step = loaded_step
    else:
        print(f"[sft] ⚠ No base checkpoint provided — SFT from scratch")
    data_iter  = iter(train_loader)
    micro_step = 0
    accum_loss = 0.0
    t0         = time.time()

    model.train()
    optimizer.zero_grad()

    print(f"\n[sft] Starting SFT  (steps={sc.max_steps})")
    print(f"      mask_prompt_loss : {sc.mask_prompt_loss}\n")

    while step < sc.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y, mask = (t.to(device) for t in batch)
        loss_mask  = mask if sc.mask_prompt_loss else None

        with amp_ctx:
            _, loss = model(x, y, loss_mask=loss_mask)
            loss    = loss / sc.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step % sc.grad_accum == 0:
            lr = cosine_lr_with_warmup(
                step, sc.warmup_steps, sc.max_steps, sc.lr, sc.min_lr
            )
            set_lr(optimizer, lr)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), sc.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            train_loss = accum_loss
            accum_loss = 0.0
            step      += 1

            if step % 25 == 0 or step == 1:
                elapsed = time.time() - t0
                t0      = time.time()
                print(
                    f"  step {step:4d}/{sc.max_steps} | "
                    f"loss {train_loss:.4f} | "
                    f"lr {lr:.2e} | {elapsed:.1f}s"
                )

            if step % sc.eval_interval == 0:
                val_loss = sft_evaluate(
                    model, val_loader, sc.eval_iters, device, amp_ctx
                )
                print(f"  *** step {step} | val_loss {val_loss:.4f} ***")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        model, optimizer, step, val_loss, CONFIG,
                        os.path.join(sc.output_dir, f"{sc.run_name}-best.pt"),
                        tokenizer=tokenizer,
                    )

            if step % sc.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, float("inf"), CONFIG,
                    os.path.join(sc.output_dir, f"{sc.run_name}-step{step}.pt"),
                    tokenizer=tokenizer,
                )

    save_checkpoint(
        model, optimizer, step, best_val, CONFIG,
        os.path.join(sc.output_dir, f"{sc.run_name}-final.pt"),
        tokenizer=tokenizer,
    )
    print(f"\n[sft] Done.  Best val loss: {best_val:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Dizel SFT")
    p.add_argument("--base_checkpoint", type=str, required=False, default="",
                   help="Path to pretrained checkpoint (.pt). Required unless resuming.")
    p.add_argument("--resume", type=str, default="",
                   help="Path to an SFT checkpoint (.pt) to resume from.")
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--max_steps", type=int,   default=None)
    p.add_argument("--no_amp",    action="store_true")
    p.add_argument("--no_mix",    action="store_true",
                   help="Disable mixed dataset loading, use single sft_data_path")
    p.add_argument("--mix_config", type=str, default="",
                   help="Path to dataset mix JSON config. Uses DEFAULT_MIX if empty.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CONFIG
    
    if not args.base_checkpoint and not args.resume:
        print("Error: Must provide either --base_checkpoint or --resume")
        sys.exit(1)
        
    if args.lr        is not None: cfg.sft.lr        = args.lr
    if args.max_steps is not None: cfg.sft.max_steps = args.max_steps
    if args.no_amp:                cfg.sft.use_amp   = False
    if args.base_checkpoint:       cfg.sft.base_checkpoint = args.base_checkpoint

    # Mixed dataset control
    cfg._use_mix = not args.no_mix
    if args.mix_config:
        import json as _json
        with open(args.mix_config) as f:
            data = _json.load(f)
        entries = [DatasetMixEntry(**e) for e in data.get("datasets", [])]
        cfg._mix_config = DatasetMixConfig(datasets=entries, seed=data.get("seed", 42))

    # Passing the resume path directly instead of base_checkpoint if resuming
    target_ckpt = args.resume if args.resume else args.base_checkpoint
    sft_train(cfg, target_ckpt)


if __name__ == "__main__":
    main()
