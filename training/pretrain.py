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
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import V15_CONFIG, ModelConfigV15, ModelConfig, DizelConfig
from model.architecture import DizelLM
from training.dataset import Tokenizer, build_pretrain_loaders

from training.display_utils import (
    rule, section, info, ok, error,
    config_table, StepMeter, validation,
)

# IMPORTANT!!
torch.serialization.add_safe_globals([ModelConfig, ModelConfigV15])

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


def make_schedule_tag(budget: int, lr: float, min_lr: float, warmup: int) -> str:
    """Stable identifier of a phase's LR schedule.

    Used to tell "resuming the SAME phase" (continue the schedule) apart from
    "launching a NEW phase from a previous phase's checkpoint" (start a fresh
    schedule). The checkpoint stores the tag of the run that wrote it; if it
    does not match the current run's schedule, the phase is treated as new.
    """
    return f"v1|b={int(budget)}|lr={float(lr)!r}|min={float(min_lr)!r}|w={int(warmup)}"


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
    ok(f"Tokenizer saved → {dst_path}")
    return dst_path


def save_checkpoint(
    model: DizelLM,
    optimizer: optim.Optimizer,
    step: int,
    val_loss: float,
    cfg: DizelConfig,
    path: str,
    tokenizer: "Tokenizer" = None,
    dataset_state: dict = None,
    phase0: int = None,
    phase_tag: str = None,
    extra_meta: dict = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt_data = {
        "step":            step,
        "val_loss":        val_loss,
        "model_state":     model.state_dict(),
        "optim_state":     optimizer.state_dict(),
        "model_cfg":       cfg.model,
        "_format_version": "1.5",
    }
    if dataset_state is not None:
        ckpt_data["dataset_state"] = dataset_state
    # Phase-anchor for the LR schedule: phase0 = the global step where this
    # phase first began. Persisted so a mid-phase resume CONTINUES the
    # schedule instead of re-entering warmup.
    if phase0 is not None:
        ckpt_data["phase0"] = phase0
    if phase_tag is not None:
        ckpt_data["phase_tag"] = phase_tag
    if extra_meta:
        ckpt_data.update(extra_meta)
    # Embed tokenizer binary so checkpoint is fully self-contained
    if tokenizer is not None:
        ckpt_data["tokenizer_model"] = tokenizer.sp.serialized_model_proto()
        ckpt_data["vocab_size"] = len(tokenizer)
        # Also save standalone copy alongside checkpoint
        save_tokenizer(tokenizer, os.path.dirname(path))
    torch.save(ckpt_data, path)
    ok(f"Checkpoint saved → {os.path.basename(path)}  (val_loss={val_loss:.4f})")


def load_checkpoint(
    path: str,
    model: DizelLM,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu",
    return_dataset_state: bool = False,
):
    """Load checkpoint.

    Returns
    -------
    (int, dict)              : (global step, meta) if return_dataset_state is False.
    (int, dict, dict)        : (step, dataset_state, meta) if return_dataset_state is True.
                               dataset_state is None if the checkpoint has no dataset state.
                               meta contains "phase0", "phase_tag", "base_checkpoint".
    """
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
    # Detect checkpoint format — swap blocks if v1.2 state dict vs v1.5 model
    fmt = ckpt.get("_format_version", "")
    if not fmt:
        if any("norm1.weight" in k for k in state_dict):
            fmt = "1.2.2"
        elif any("mlp.gate_proj.weight" in k for k in state_dict):
            fmt = "1.5"
    if fmt == "1.2.2":
        has_legacy = hasattr(model.transformer["blocks"][0], "norm1")
        if not has_legacy:
            # Model was built as v1.5 — swap in legacy decoder blocks
            from model.architecture import GQAAttention
            d_model = model.cfg.d_model
            for i in range(model.cfg.n_layers):
                legacy = nn.Module()
                legacy.norm1 = nn.LayerNorm(d_model, bias=False)
                legacy.attn  = GQAAttention(model.cfg)
                legacy.norm2 = nn.LayerNorm(d_model, bias=False)
                mlp = nn.Module()
                ffn_dim = int(d_model * getattr(model.cfg, 'ffn_mult', 4))
                mlp.fc1 = nn.Linear(d_model, ffn_dim, bias=model.cfg.bias)
                mlp.fc2 = nn.Linear(ffn_dim, d_model, bias=model.cfg.bias)
                legacy.mlp = mlp
                model.transformer["blocks"][i] = legacy.to(device)
            # Replace RMSNorm ln_f with LayerNorm
            model.transformer["ln_f"] = nn.LayerNorm(d_model, bias=False).to(device)
    model.load_state_dict(state_dict, strict=False)

    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    step = ckpt.get("step", 0)
    dataset_state = ckpt.get("dataset_state")
    meta = {
        "phase0":         ckpt.get("phase0"),
        "phase_tag":      ckpt.get("phase_tag"),
        "base_checkpoint": ckpt.get("base_checkpoint"),
    }
    info(f"Loaded checkpoint '{os.path.basename(path)}' (step={step}, val_loss={ckpt.get('val_loss', '?')})")
    if return_dataset_state:
        return step, dataset_state, meta
    return step, meta


# ---------------------------------------------------------------------------
# Gradient checkpointing helper
# ---------------------------------------------------------------------------
def _enable_gradient_checkpointing(model: "DizelLM") -> None:
    """Wrap each TransformerBlock forward with gradient checkpointing to save VRAM."""
    for block in model.transformer["blocks"]:
        orig_forward = block.forward
        block.forward = lambda x, orig=orig_forward: torch.utils.checkpoint.checkpoint(
            orig, x, use_reentrant=False
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: DizelConfig, resume_from: str = None) -> None:
    rule("Pre-training")

    pc   = cfg.pretrain
    mc   = cfg.model

    # -- Device ----------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    info(f"Device : {device}")
    if device.type == "cuda":
        info(f"GPU    : {torch.cuda.get_device_name()} / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM")

    section("Hardware")

    # -- Mixed precision --------------------------------------------------
    use_amp  = pc.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if pc.amp_dtype == "bfloat16" else torch.float16
    amp_ctx  = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )
    scaler   = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    info(f"AMP    : {'ON (' + pc.amp_dtype + ')' if use_amp else 'OFF'}")

    # -- Tokenizer -------------------------------------------------------
    tokenizer = Tokenizer()

    # Sync vocab size from actual tokenizer
    mc.vocab_size = len(tokenizer)

    section("Data")

    # -- Data ------------------------------------------------------------
    train_loader, val_loader, train_ds = build_pretrain_loaders(
        data_path      = pc.data_path,
        tokenizer      = tokenizer,
        context_length = mc.context_length,
        batch_size     = pc.batch_size,
        train_split    = pc.train_split,
        seed           = pc.seed,
        num_workers    = getattr(pc, "num_workers", 0),
        shard_size_mb  = getattr(pc, "shard_size_mb", 100),
        cache_dir      = getattr(pc, "cache_dir", ".cache/tokenized"),
        resume_enabled = getattr(pc, "resume_enabled", True),
    )

    section("Model")

    # -- Model -----------------------------------------------------------
    model = DizelLM(mc).to(device)
    info(f"Model  : DizelLM({mc.d_model}, {mc.n_layers} layers, {model.num_parameters() / 1e6:.1f}M params)")

    # Compile (PyTorch 2.x -- optional but ~30% faster on Ampere GPUs)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            ok("torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile skipped: {e}")

    # Gradient checkpointing (saves ~40% VRAM)
    if getattr(pc, "gradient_checkpointing", False):
        _enable_gradient_checkpointing(model)
        ok("Gradient checkpointing enabled")

    config_table({
        "Dimension":    str(mc.d_model),
        "Layers":       str(mc.n_layers),
        "Heads":        str(mc.n_heads),
        "KV Heads":     str(getattr(mc, 'kv_heads', mc.n_heads)),
        "Vocab":        f"{mc.vocab_size:,}",
        "Context":      str(mc.context_length),
        "FFN":          str(mc.intermediate_size if hasattr(mc, 'intermediate_size') else mc.ffn_dim),
    })

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
    dataset_state = None
    ckpt_phase0   = None
    ckpt_phase_tag = None
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_from}. Refusing to "
                "start a fresh run from a mistyped resume path."
            )
        start_step, dataset_state, _meta = load_checkpoint(
            resume_from, model, optimizer, str(device),
            return_dataset_state=True,
        )
        ckpt_phase0    = _meta.get("phase0")
        ckpt_phase_tag = _meta.get("phase_tag")

    # -- Training loop ---------------------------------------------------
    os.makedirs(pc.checkpoint_dir, exist_ok=True)
    os.makedirs(pc.log_dir,        exist_ok=True)

    log_file = open(
        os.path.join(pc.log_dir, f"{pc.run_name}.csv"), "w"
    )
    log_file.write("step,train_loss,val_loss,lr\n")

    model.train()
    optimizer.zero_grad()

    step         = start_step
    _resume_step = step

    # -- LR schedule anchor ----------------------------------------------
    # phase0 = global step where this phase first began. It is persisted in
    # checkpoints so a mid-phase resume CONTINUES the schedule (decaying)
    # instead of re-entering warmup. If the checkpoint belongs to a different
    # phase (different schedule tag) or is legacy (no phase0 key), anchor the
    # phase at the current step — this preserves fresh-launch behaviour:
    # warmup from 0 → decay over the phase budget.
    phase_tag = make_schedule_tag(pc.max_steps, pc.lr, pc.min_lr, pc.warmup_steps)
    if ckpt_phase0 is not None and ckpt_phase_tag == phase_tag:
        phase0 = ckpt_phase0
    else:
        phase0 = step
    # `--max_steps` is a PER-PHASE budget counted from the phase's FIRST step
    # (phase0), so the absolute stopping step stays fixed across mid-phase
    # resumes: resuming at step N simply gets closer to the same fixed end
    # (phase0 + budget) instead of pushing it out by N every resume.
    target_steps = phase0 + pc.max_steps
    batch_num    = 0
    accum_loss   = 0.0
    micro_step   = 0

    # -- Restore dataset position on resume ------------------------------
    if dataset_state is not None:
        # Restore before creating the iterator: DataLoader workers get a
        # private copy of the dataset, so load_state() after iter() would not
        # reach them and they would replay the initial window order.
        train_ds.load_state(dataset_state)
        pos = dataset_state.get("pos", 0)

        # The reshuffle happens at the END of this optimizer step in the
        # original run, but was NOT yet applied when the checkpoint was saved.
        # Start the next segment directly; replaying the finished segment
        # would only delay the resume.
        at_boundary = (
            pc.reshuffle_every_n_steps > 0
            and step > 0
            and step % pc.reshuffle_every_n_steps == 0
        )
        batch_num = 0
        if at_boundary:
            train_ds.reshuffle(new_seed=pc.seed + step)
            data_iter = iter(train_loader)
            info(
                f"Resume: step {step:,} is a reshuffle boundary; "
                f"starting segment with seed {train_ds.seed} "
                f"({pos:,} checkpointed windows not replayed)"
            )
        else:
            data_iter = iter(train_loader)
            for _ in range(pos // pc.batch_size):
                try:
                    next(data_iter)
                    batch_num += 1
                except StopIteration:
                    train_ds.reshuffle(new_seed=pc.seed + step)
                    data_iter = iter(train_loader)
                    batch_num = 0
                    break
            # pos is the window position within the current shuffled segment,
            # not an absolute dataset position.
            info(
                f"Resume: replayed {pos:,} windows of the checkpointed "
                f"segment (step={step:,}, seed={train_ds.seed})"
            )
    else:
        data_iter = iter(train_loader)

    _resume_note = f" (resume {_resume_step:,} → end {target_steps:,})" if _resume_step else ""
    config_table({
        "Steps":        f"{pc.max_steps:,} this phase{_resume_note}",
        "Batch":        f"{pc.batch_size} × {pc.grad_accum} = {pc.batch_size * pc.grad_accum}",
        "LR":           f"{pc.lr:.2e} → {pc.min_lr:.2e}",
        "Warmup":       str(pc.warmup_steps),
        "Weight Decay": str(pc.weight_decay),
        "Grad Clip":    str(pc.grad_clip),
    })

    meter = StepMeter(
        target_steps,
        tokens_per_step=pc.batch_size * pc.grad_accum * mc.context_length,
        start_step=step,
    )

    while step < target_steps:
        # -- Fetch micro-batch --------------------------------------------
        try:
            x, y = next(data_iter)
            batch_num += 1
        except StopIteration:
            # Reshuffle every epoch to reduce overfitting
            train_ds.reshuffle(new_seed=pc.seed + step)
            data_iter = iter(train_loader)
            batch_num = 1
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
            # LR schedule — anchored to the phase start step (phase0) so a
            # mid-phase resume continues decaying and never re-enters warmup.
            local = step - phase0
            lr = cosine_lr_with_warmup(
                local, pc.warmup_steps, pc.max_steps, pc.lr, pc.min_lr
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
            if step % 50 == 0 or step == _resume_step + 1:
                meter.update(
                    step=step,
                    loss=train_loss,
                    lr=lr,
                )

            # -- Build dataset state for checkpoint -----------------------
            if step % pc.eval_interval == 0 or step % pc.save_interval == 0:
                ds_state = train_ds.get_state()
                # Track window position in the main process. Using
                # train_ds.current_index is unreliable when num_workers>0
                # (__getitem__ runs in workers). batch_num × batch_size is
                # the exact number of windows already consumed.
                ds_state["pos"] = batch_num * pc.batch_size

            # -- Evaluation ----------------------------------------------
            if step % pc.eval_interval == 0:
                val_loss = evaluate(
                    model, val_loader, pc.eval_iters, device, amp_ctx
                )
                validation(step, val_loss)
                log_file.write(f"{step},{train_loss:.4f},{val_loss:.4f},{lr:.6f}\n")
                log_file.flush()
                torch.cuda.empty_cache()

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        model, optimizer, step, val_loss, cfg,
                        os.path.join(pc.checkpoint_dir, f"{pc.run_name}-best.pt"),
                        tokenizer=tokenizer,
                        dataset_state=ds_state,
                        phase0=phase0,
                        phase_tag=phase_tag,
                    )

            # -- Periodic checkpoint --------------------------------------
            if not getattr(cfg, '_best_only', False) and step % pc.save_interval == 0:
                save_checkpoint(
                    model, optimizer, step, float("inf"), cfg,
                    os.path.join(pc.checkpoint_dir, f"{pc.run_name}-step{step}.pt"),
                    tokenizer=tokenizer,
                    dataset_state=ds_state,
                    phase0=phase0,
                    phase_tag=phase_tag,
                )

            # -- Reshuffle windows periodically --------------------------
            if (
                pc.reshuffle_every_n_steps > 0
                and step % pc.reshuffle_every_n_steps == 0
            ):
                train_ds.reshuffle(new_seed=pc.seed + step)
                data_iter = iter(train_loader)
                batch_num = 0

    # -- Final save ------------------------------------------------------
    ds_state = train_ds.get_state()
    ds_state["pos"] = batch_num * pc.batch_size
    save_checkpoint(
        model, optimizer, step, best_val, cfg,
        os.path.join(pc.checkpoint_dir, f"{pc.run_name}-final.pt"),
        tokenizer=tokenizer,
        dataset_state=ds_state,
        phase0=phase0,
        phase_tag=phase_tag,
    )
    log_file.close()
    meter.finalise()
    rule("Complete")
    config_table({
        "Best val loss": f"{best_val:.4f}",
        "Total steps":   f"{step:,}",
    })


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
    p.add_argument("--warmup_steps",    type=int,   default=None,
                   help="Number of warmup steps")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing")
    p.add_argument("--best_only", action="store_true",
                   help="Save only the best checkpoint (skip periodic step checkpoints)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg  = V15_CONFIG  # v1.5 config (d_model=1024, layers=28 → 454.78M)

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
    if args.warmup_steps is not None: cfg.pretrain.warmup_steps = args.warmup_steps
    if args.gradient_checkpointing:   cfg.pretrain.gradient_checkpointing = True
    cfg._best_only = args.best_only

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
