"""
training/posttrain_checkpoint_v16.py — v1.6 checkpoint helpers for SFT/KD/DPO.

The pre-training saver is tied to ``PretrainConfigV16Flash`` because it writes
a pretraining scheduler block.  Post-training needs the same on-disk contract
(model config, tokenizer hash, optimizer/scaler, RNG, dataset position, phase
metadata) without pretending to be a pre-training run.  This module keeps the
contract in one place and delegates loading to ``load_checkpoint_v16``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Optional

import torch
import torch.optim as optim

from configs.v16_flash import (
    DPOConfigV16Flash,
    DistillConfigV16Flash,
    ModelConfigV16Flash,
    SFTConfigV16Flash,
)
from training.pretrain_v16 import (
    FORMAT_VERSION,
    _finite_or_none,
    _rng_state,
    _write_json_atomic,
    git_commit,
    load_checkpoint_v16,
    sync_to_drive,
)

__all__ = [
    "POSTTRAIN_FORMAT_VERSION",
    "make_schedule_tag",
    "posttrain_config_hash",
    "save_posttrain_checkpoint",
    "load_posttrain_checkpoint",
    "sync_checkpoint_to_drive",
]

POSTTRAIN_FORMAT_VERSION = FORMAT_VERSION


def make_schedule_tag(
    phase: str,
    max_steps: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
) -> str:
    """Stable identifier for a post-training phase's LR schedule."""
    return (
        f"v16-post|{phase}|b={int(max_steps)}|lr={float(lr)!r}|"
        f"min={float(min_lr)!r}|w={int(warmup_steps)}"
    )


def _as_dict(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return value
    return vars(value)


def posttrain_config_hash(model_cfg, tokenizer_cfg, phase: str, phase_cfg) -> str:
    payload = {
        "phase": phase,
        "model": _as_dict(model_cfg),
        "tokenizer": _as_dict(tokenizer_cfg),
        "phase_config": _as_dict(phase_cfg),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_posttrain_checkpoint(
    model,
    optimizer: Optional[optim.Optimizer],
    scaler,
    step: int,
    phase: str,
    phase_cfg,
    model_cfg,
    tokenizer_cfg,
    path: str,
    tokenizer=None,
    val_loss: Optional[float] = None,
    best_val: Optional[float] = None,
    dataset_state: Optional[dict] = None,
    phase0: Optional[int] = None,
    phase_tag: Optional[str] = None,
    base_checkpoint: Optional[str] = None,
    reference_checkpoint: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> str:
    """Atomically save a v1.6 post-training checkpoint and JSON sidecar."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tokenizer_name = getattr(tokenizer, "model_name", None)
    tokenizer_hash = tokenizer.fingerprint() if tokenizer is not None else None
    vocab_size = (
        len(tokenizer) if tokenizer is not None else int(model_cfg.vocab_size)
    )
    scaler_state = (
        scaler.state_dict()
        if scaler is not None and scaler.is_enabled()
        else None
    )
    scheduler = {
        "type": "cosine_warmup",
        "phase": phase,
        "max_steps": int(getattr(phase_cfg, "max_steps", 0)),
        "warmup_steps": int(getattr(phase_cfg, "warmup_steps", 0)),
        "lr": float(getattr(phase_cfg, "lr", 0.0)),
        "min_lr": float(getattr(phase_cfg, "min_lr", 0.0)),
    }
    ckpt = {
        "step": int(step),
        "tokens_seen": 0,
        "val_loss": val_loss,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler_state,
        "model_cfg": model_cfg,
        "_format_version": POSTTRAIN_FORMAT_VERSION,
        "arch_version": "dizel-1.6-preview-m1",
        "release_stage": "preview",
        "phase": phase,
        "config_hash": posttrain_config_hash(
            model_cfg, tokenizer_cfg, phase, phase_cfg
        ),
        "git_commit": git_commit(),
        "vocab_size": int(vocab_size),
        "tokenizer_name": tokenizer_name,
        "tokenizer_hash": tokenizer_hash,
        # The loader defaults v1.6 to ChatML + system turns anyway; writing
        # the flags makes the contract explicit for readers and future
        # non-v1.6 formats.
        "chat_format": "chatml",
        "supports_system_prompt": True,
        "num_experts": int(getattr(model_cfg, "num_experts", 1)),
        "top_k": int(getattr(model_cfg, "top_k", 1)),
        "router": {
            "aux_loss_coef": float(getattr(model_cfg, "router_aux_loss_coef", 0.0)),
            "z_loss_coef": float(getattr(model_cfg, "router_z_loss_coef", 0.0)),
            "dropless": True,
        },
        "scheduler": scheduler,
        "rng_state": _rng_state(),
    }
    if dataset_state is not None:
        ckpt["dataset_state"] = dataset_state
    if phase0 is not None:
        ckpt["phase0"] = int(phase0)
    if phase_tag is not None:
        ckpt["phase_tag"] = phase_tag
    if base_checkpoint:
        ckpt["base_checkpoint"] = base_checkpoint
    if reference_checkpoint:
        ckpt["reference_checkpoint"] = reference_checkpoint
    if extra_meta:
        ckpt.update(extra_meta)

    tmp = path + ".tmp"
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    sidecar = {
        "step": int(step),
        "val_loss": _finite_or_none(val_loss),
        "best_val": _finite_or_none(best_val),
        "phase": phase,
        "config_hash": ckpt["config_hash"],
        "tokenizer_hash": tokenizer_hash,
        "chat_format": ckpt["chat_format"],
        "supports_system_prompt": ckpt["supports_system_prompt"],
        "git_commit": ckpt["git_commit"],
        "num_experts": ckpt["num_experts"],
        "top_k": ckpt["top_k"],
        "arch_version": ckpt["arch_version"],
        "release_stage": ckpt["release_stage"],
    }
    if dataset_state is not None:
        sidecar["dataset_state"] = dataset_state
    _write_json_atomic(path + ".json", sidecar)
    print(f"  checkpoint saved -> {os.path.basename(path)} (step={step:,})")
    return path


def load_posttrain_checkpoint(
    path: str,
    model,
    optimizer=None,
    scaler=None,
    tokenizer=None,
    device: str = "cpu",
    restore_rng: bool = True,
):
    """Load a v1.6 checkpoint and expose post-training metadata.

    ``restore_rng=False`` is used for frozen reference models: loading a
    reference must never clobber the RNG state restored for the policy run.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = load_checkpoint_v16(
        path,
        model,
        optimizer=optimizer,
        scaler=scaler,
        tokenizer=tokenizer,
        device=device,
        restore_rng=restore_rng,
    )
    return state


def sync_checkpoint_to_drive(local_path: str, drive_checkpoint_dir: str):
    """Copy a checkpoint and sidecar to Drive, verified by size and SHA-256."""
    if not drive_checkpoint_dir:
        return None
    return sync_to_drive(local_path, drive_checkpoint_dir)
