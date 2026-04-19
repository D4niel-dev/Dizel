"""
training/pretrain.py -- Next-token pre-training for Dizel.

Features
--------
* Mixed-precision training (bfloat16 / float16) via torch.amp
* Gradient accumulation (simulate large batch on small GPU)
* Cosine LR schedule with linear warm-up
* Gradient clipping
* Periodic evaluation on held-out val set
* Checkpoint saving (best val loss + periodic)
* Overfitting mitigation: window reshuffling, dropout, weight decay

Usage
-----
    python training/pretrain.py

    # Or override any config value:
    python training/pretrain.py --lr 5e-4 --max_steps 6000 --d_model 256
"""

import argparse
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from typing import Optional

import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG, DizelConfig, ModelConfig, PretrainConfig
from model.architecture import DizelLM
from training.dataset import Tokenizer, build_pretrain_loaders


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    """
    Linear warm-up then cosine decay from `lr` to `min_lr`.
    """
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + decay * (lr - min_lr)


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: DizelLM,
    val_loader,
    eval_iters: int,
    device: torch.device,
    amp_ctx,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in val_loader:
        if count >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(1, count)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_tokenizer(tokenizer: "Tokenizer", checkpoint_dir: str) -> str:
    """Copy the tokenizer .model file into the checkpoint directory.
    
    Returns the path where the tokenizer was saved.
    """
    src = tokenizer.sp.serialized_model_proto()
    dst_path = os.path.join(checkpoint_dir, "tokenizer.model")
    with open(dst_path, "wb") as f:
        f.write(src)
    print(f"  [ckpt] Tokenizer saved -> {dst_path}")
    return dst_path


def save_checkpoint(
    model: DizelLM,
    optimizer: optim.Optimizer,
    step: int,
    val_loss: float,
    cfg: DizelConfig,
    path: str,
    tokenizer: "Tokenizer" = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt_data = {
        "step":        step,
        "val_loss":    val_loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "model_cfg":   cfg.model,
    }
    # Embed tokenizer binary so checkpoint is fully self-contained
    if tokenizer is not None:
        ckpt_data["tokenizer_model"] = tokenizer.sp.serialized_model_proto()
        ckpt_data["vocab_size"] = len(tokenizer)
        # Also save standalone copy alongside checkpoint
        save_tokenizer(tokenizer, os.path.dirname(path))
    torch.save(ckpt_data, path)
    print(f"  [ckpt] Saved -> {path}  (val_loss={val_loss:.4f})")


def load_checkpoint(
    path: str,
    model: DizelLM,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """Load checkpoint and return the step it was saved at."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"]
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    # Match checkpoint keys to model keys (handle _orig_mod. in either direction)
    if model_keys != ckpt_keys:
        stripped = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        prefixed = {"_orig_mod." + k: v for k, v in state_dict.items()}
        if set(stripped.keys()) == model_keys:
            state_dict = stripped
        elif set(prefixed.keys()) == model_keys:
            state_dict = prefixed
    model.load_state_dict(state_dict)

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    step = ckpt.get("step", 0)
    print(f"[ckpt] Loaded '{path}' (step={step}, val_loss={ckpt.get('val_loss', '?')})")
    return step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: DizelConfig, resume_from: str = None) -> None:
    pc   = cfg.pretrain
    mc   = cfg.model

    # -- Device ----------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[train] Device : {device}")
    if device.type == "cuda":
        print(f"[train] GPU    : {torch.cuda.get_device_name()}")
        print(f"[train] VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -- Mixed precision --------------------------------------------------
    use_amp  = pc.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if pc.amp_dtype == "bfloat16" else torch.float16
    amp_ctx  = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )
    scaler   = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    print(f"[train] AMP    : {'ON (' + pc.amp_dtype + ')' if use_amp else 'OFF'}")

    # -- Tokenizer -------------------------------------------------------
    tokenizer = Tokenizer()

    # Sync vocab size from actual tokenizer
    mc.vocab_size = len(tokenizer)

    # -- Data ------------------------------------------------------------
    train_loader, val_loader, train_ds = build_pretrain_loaders(
        data_path      = pc.data_path,
        tokenizer      = tokenizer,
        context_length = mc.context_length,
        batch_size     = pc.batch_size,
        train_split    = pc.train_split,
        seed           = pc.seed,
    )

    # -- Model -----------------------------------------------------------
    model = DizelLM(mc).to(device)
    print(f"[train] Model  : {model}")
    print(f"[train] Params : {model.num_parameters() / 1e6:.2f} M")

    # Compile (PyTorch 2.x -- optional but ~30% faster on Ampere GPUs)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[train] torch.compile : enabled")
        except Exception as e:
            print(f"[train] torch.compile skipped: {e}")

    # -- Optimiser -------------------------------------------------------
    # Separate parameters into those with/without weight decay.
    # Embeddings and LayerNorm params should NOT be decayed.
    decay_params     = []
    no_decay_params  = []
    decay_names      = []
    no_decay_names   = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
            decay_names.append(name)
        else:
            no_decay_params.append(param)
            no_decay_names.append(name)

    param_groups = [
        {"params": decay_params,    "weight_decay": pc.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=pc.lr, betas=(0.9, 0.95), eps=1e-8)

    # -- Resume ----------------------------------------------------------
    start_step = 0
    best_val   = float("inf")
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(resume_from, model, optimizer, str(device))

    # -- Training loop ---------------------------------------------------
    os.makedirs(pc.checkpoint_dir, exist_ok=True)
    os.makedirs(pc.log_dir,        exist_ok=True)

    log_file = open(
        os.path.join(pc.log_dir, f"{pc.run_name}.csv"), "w"
    )
    log_file.write("step,train_loss,val_loss,lr\n")

    model.train()
    optimizer.zero_grad()

    step       = start_step
    data_iter  = iter(train_loader)
    t0         = time.time()
    accum_loss = 0.0
    micro_step = 0

    print(f"\n[train] Starting pre-training")
    print(f"        steps     : {pc.max_steps}")
    print(f"        batch     : {pc.batch_size} x {pc.grad_accum} = {pc.batch_size * pc.grad_accum} effective")
    print(f"        context   : {mc.context_length} tokens")
    print(f"        lr        : {pc.lr} -> {pc.min_lr}\n")

    while step < pc.max_steps:
        # -- Fetch micro-batch --------------------------------------------
        try:
            x, y = next(data_iter)
        except StopIteration:
            # Reshuffle every epoch to reduce overfitting
            train_ds.reshuffle(new_seed=pc.seed + step)
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # -- Forward / backward -------------------------------------------
        with amp_ctx:
            _, loss = model(x, y)
            loss    = loss / pc.grad_accum        # normalise for accumulation

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        # -- Optimiser step (every grad_accum micro-steps) -----------------
        if micro_step % pc.grad_accum == 0:
            # LR schedule
            lr = cosine_lr_with_warmup(
                step, pc.warmup_steps, pc.max_steps, pc.lr, pc.min_lr
            )
            set_lr(optimizer, lr)

            # Gradient clipping
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pc.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            train_loss = accum_loss
            accum_loss = 0.0
            step      += 1

            # -- Logging -------------------------------------------------
            if step % 50 == 0 or step == 1:
                elapsed = time.time() - t0
                t0      = time.time()
                print(
                    f"  step {step:5d}/{pc.max_steps} | "
                    f"loss {train_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{elapsed:.1f}s/50steps"
                )

            # -- Evaluation ----------------------------------------------
            if step % pc.eval_interval == 0:
                val_loss = evaluate(
                    model, val_loader, pc.eval_iters, device, amp_ctx
                )
                print(f"  *** step {step} | val_loss {val_loss:.4f} ***")
                log_file.write(f"{step},{train_loss:.4f},{val_loss:.4f},{lr:.6f}\n")
                log_file.flush()

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        model, optimizer, step, val_loss, cfg,
                        os.path.join(pc.checkpoint_dir, f"{pc.run_name}-best.pt"),
                        tokenizer=tokenizer,
                    )

            # -- Periodic checkpoint --------------------------------------
            if step % pc.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, float("inf"), cfg,
                    os.path.join(pc.checkpoint_dir, f"{pc.run_name}-step{step}.pt"),
                    tokenizer=tokenizer,
                )

            # -- Reshuffle windows periodically --------------------------
            if step % pc.reshuffle_every_n_steps == 0:
                train_ds.reshuffle(new_seed=pc.seed + step)
                data_iter = iter(train_loader)

    # -- Final save ------------------------------------------------------
    save_checkpoint(
        model, optimizer, step, best_val, cfg,
        os.path.join(pc.checkpoint_dir, f"{pc.run_name}-final.pt"),
        tokenizer=tokenizer,
    )
    log_file.close()
    print(f"\n[train] Pre-training complete. Best val loss: {best_val:.4f}")
    print(f"[train] Checkpoints saved to: {pc.checkpoint_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dizel pre-training")
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--min_lr",       type=float, default=None)
    p.add_argument("--max_steps",    type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--grad_accum",   type=int,   default=None)
    p.add_argument("--d_model",      type=int,   default=None)
    p.add_argument("--n_layers",     type=int,   default=None)
    p.add_argument("--n_heads",      type=int,   default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--data_path",    type=str,   default=None)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_amp",       action="store_true",
                   help="Disable mixed precision")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CONFIG   # global config

    # Apply CLI overrides
    if args.lr           is not None: cfg.pretrain.lr           = args.lr
    if args.min_lr       is not None: cfg.pretrain.min_lr       = args.min_lr
    if args.max_steps    is not None: cfg.pretrain.max_steps    = args.max_steps
    if args.batch_size   is not None: cfg.pretrain.batch_size   = args.batch_size
    if args.grad_accum   is not None: cfg.pretrain.grad_accum   = args.grad_accum
    if args.d_model      is not None: cfg.model.d_model         = args.d_model
    if args.n_layers     is not None: cfg.model.n_layers        = args.n_layers
    if args.n_heads      is not None: cfg.model.n_heads         = args.n_heads
    if args.dropout      is not None: cfg.model.dropout         = args.dropout
    if args.context_length is not None: cfg.model.context_length = args.context_length
    if args.data_path    is not None: cfg.pretrain.data_path    = args.data_path
    if args.no_amp:                    cfg.pretrain.use_amp      = False

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
