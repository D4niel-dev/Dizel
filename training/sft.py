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
from config import V15_CONFIG, ModelConfigV15, ModelConfig, DizelConfig
from model.architecture import DizelLM
from training.dataset import Tokenizer, build_sft_loaders, build_mixed_sft_loaders
from training.data_mixing import DEFAULT_MIX, DatasetMixConfig, DatasetMixEntry
from training.pretrain import (
    cosine_lr_with_warmup, set_lr, evaluate, save_checkpoint, load_checkpoint,
    save_tokenizer, make_schedule_tag, _enable_gradient_checkpointing,
)

from training.display_utils import (
    rule, section, info, ok, warning, error, dim,
    config_table, StepMeter, validation,
)

torch.serialization.add_safe_globals([ModelConfig, ModelConfigV15])

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

    rule("Supervised Fine-Tuning")

    section("Hardware")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device : {device}")

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
    section("Data")
    use_mix = getattr(cfg, '_use_mix', True)
    if use_mix:
        mix_config = getattr(cfg, '_mix_config', DEFAULT_MIX)
        ok(f"Mixed dataset ({len(mix_config.datasets)} sources)")
        try:
            train_loader, val_loader = build_mixed_sft_loaders(
                mix_config     = mix_config,
                tokenizer      = tokenizer,
                context_length = mc.context_length,
                batch_size     = sc.batch_size,
                cache_path     = getattr(cfg.sft, "cache_path", ".cache/sft_mixed_cache.pt"),
            )
        except Exception as e:
            warning(f"Mixed dataset failed ({e}), falling back to single file")
            use_mix = False

    if not use_mix:
        if not os.path.exists(sc.sft_data_path):
            error(f"SFT data not found at '{sc.sft_data_path}'")
            error("Run: python sft_data/generate_sft_data.py")
            sys.exit(1)
        train_loader, val_loader = build_sft_loaders(
            jsonl_path     = sc.sft_data_path,
            tokenizer      = tokenizer,
            context_length = mc.context_length,
            batch_size     = sc.batch_size,
        )

    # ── Model (load from pretrain checkpoint) ────────────────────────────
    section("Model")
    model = DizelLM(mc).to(device)

    # Gradient checkpointing (saves ~40% VRAM)
    if getattr(sc, "gradient_checkpointing", False):
        _enable_gradient_checkpointing(model)
        ok("Gradient checkpointing enabled")

    # Delay loading checkpoint until optimizer is created so we can load its state if resuming

    ok(f"Model: DizelLM({mc.d_model}, {mc.n_layers} layers)")

    config_table({
        "dimension":    mc.d_model,
        "layers":       mc.n_layers,
        "vocab size":   mc.vocab_size,
        "context":      mc.context_length,
    })
    target_steps = sc.max_steps

    config_table({
        "phase steps":   sc.max_steps,
        "batch size":    sc.batch_size,
        "grad accum":    sc.grad_accum,
        "learning rate": f"{sc.lr:.2e} → {sc.min_lr:.2e}",
        "warmup steps":  sc.warmup_steps,
        "mask prompts":  sc.mask_prompt_loss,
    })

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
    ckpt_phase0    = None
    ckpt_phase_tag = None

    if base_ckpt and os.path.exists(base_ckpt):
        # We assume if the filename has 'sft', we are resuming an SFT run, not starting from pretrain
        is_resume = "sft" in base_ckpt.lower()
        loaded_step, _meta = load_checkpoint(
            base_ckpt, model,
            optimizer=optimizer if is_resume else None,
            device=str(device),
        )
        ckpt_phase0    = _meta.get("phase0")
        ckpt_phase_tag = _meta.get("phase_tag")
        if is_resume:
            step = loaded_step
    else:
        warning("No base checkpoint provided — training from scratch")
    _resume_step = step

    # -- LR schedule anchor ----------------------------------------------
    # phase0 = global step where this phase first began; persisted in
    # checkpoints so a mid-phase resume CONTINUES decaying instead of
    # re-entering warmup. Legacy or cross-phase checkpoints start a fresh
    # schedule anchored at the current step.
    phase_tag = make_schedule_tag(sc.max_steps, sc.lr, sc.min_lr, sc.warmup_steps)
    if ckpt_phase0 is not None and ckpt_phase_tag == phase_tag:
        phase0 = ckpt_phase0
    else:
        phase0 = step
    # `--max_steps` is a PER-PHASE budget counted from the phase's FIRST step
    # (phase0), so the absolute stopping step stays fixed across mid-phase
    # resumes: resuming at step N simply gets closer to the same fixed end.
    target_steps = phase0 + sc.max_steps
    _resume_note = f" (resume {_resume_step:,} → end {target_steps:,})" if _resume_step else ""
    info(f"Steps: {sc.max_steps:,} this phase{_resume_note}")
    data_iter  = iter(train_loader)
    micro_step = 0
    accum_loss = 0.0

    model.train()
    optimizer.zero_grad()

    meter = StepMeter(
        target_steps,
        tokens_per_step=sc.batch_size * sc.grad_accum * mc.context_length,
        start_step=step,
    )

    while step < target_steps:
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
            # LR schedule — anchored to phase start so a mid-phase resume
            # continues decaying instead of re-entering warmup.
            local = step - phase0
            lr = cosine_lr_with_warmup(
                local, sc.warmup_steps, sc.max_steps, sc.lr, sc.min_lr
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

            if step % 25 == 0 or step == _resume_step + 1:
                meter.update(step=step, loss=train_loss, lr=lr)

            if step % sc.eval_interval == 0:
                val_loss = sft_evaluate(
                    model, val_loader, sc.eval_iters, device, amp_ctx
                )
                validation(step, val_loss)
                torch.cuda.empty_cache()
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        model, optimizer, step, val_loss, V15_CONFIG,
                        os.path.join(sc.output_dir, f"{sc.run_name}-best.pt"),
                        tokenizer=tokenizer,
                        phase0=phase0,
                        phase_tag=phase_tag,
                    )

            if not getattr(cfg, '_best_only', False) and step % sc.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, float("inf"), V15_CONFIG,
                    os.path.join(sc.output_dir, f"{sc.run_name}-step{step}.pt"),
                    tokenizer=tokenizer,
                    phase0=phase0,
                    phase_tag=phase_tag,
                )

    save_checkpoint(
        model, optimizer, step, best_val, V15_CONFIG,
        os.path.join(sc.output_dir, f"{sc.run_name}-final.pt"),
        tokenizer=tokenizer,
        phase0=phase0,
        phase_tag=phase_tag,
    )
    meter.finalise()
    config_table({
        "Best val loss": f"{best_val:.4f}",
        "Total steps":   f"{step:,}",
    })


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
    p.add_argument("--min_lr",    type=float, default=None)
    p.add_argument("--max_steps", type=int,   default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int,  default=None)
    p.add_argument("--grad_accum", type=int,  default=None)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing")
    p.add_argument("--no_amp",    action="store_true",
                   help="Disable AMP even on CUDA")
    p.add_argument("--use_amp",   action="store_true",
                   help="Enable AMP (overrides config)")
    p.add_argument("--amp_dtype", type=str, default=None, choices=["float16", "bfloat16"],
                   help="AMP dtype to use")
    p.add_argument("--sft_data",  type=str, default=None,
                   help="Path to SFT JSONL data file")
    p.add_argument("--no_mix",    action="store_true",
                   help="Disable mixed dataset loading, use single sft_data_path")
    p.add_argument("--mix_config", type=str, default="",
                   help="Path to dataset mix JSON config. Uses DEFAULT_MIX if empty.")
    p.add_argument("--best_only", action="store_true",
                   help="Save only the best checkpoint (skip periodic step checkpoints)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg  = V15_CONFIG
    
    if not args.base_checkpoint and not args.resume:
        print("Error: Must provide either --base_checkpoint or --resume")
        sys.exit(1)
        
    if args.lr        is not None: cfg.sft.lr        = args.lr
    if args.min_lr    is not None: cfg.sft.min_lr    = args.min_lr
    if args.max_steps is not None: cfg.sft.max_steps = args.max_steps
    if args.warmup_steps is not None: cfg.sft.warmup_steps = args.warmup_steps
    if args.batch_size is not None: cfg.sft.batch_size = args.batch_size
    if args.grad_accum is not None: cfg.sft.grad_accum = args.grad_accum
    if args.gradient_checkpointing: cfg.sft.gradient_checkpointing = True
    if args.no_amp:                cfg.sft.use_amp   = False
    if args.use_amp:               cfg.sft.use_amp   = True
    if args.amp_dtype is not None: cfg.sft.amp_dtype = args.amp_dtype
    if args.sft_data is not None:  cfg.sft.sft_data_path = args.sft_data
    if args.base_checkpoint:       cfg.sft.base_checkpoint = args.base_checkpoint
    cfg._best_only = args.best_only

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
