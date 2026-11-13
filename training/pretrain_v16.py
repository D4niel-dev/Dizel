"""
training/pretrain_v16.py — Dizel 1.6 MoE pre-training entry point (Flash and Pro).

This is the shared v1.6 training path; ``--config flash|pro`` selects the tier.
It reuses the v1.5 data pipeline, LR schedule, evaluation and display helpers,
and adds the pieces the MoE plan requires:

* Qwen2.5 tokenizer with a vocabulary fingerprint (token-cache isolation)
* CE + router auxiliary loss + router z-loss, logged separately
* fail-loud checks that every configured loss term contributes and that
  every expert receives tokens
* BPB on the token-split validation set (proxy using the corpus-wide
  bytes/token ratio); the fixed byte-level holdout remains the separate
  release-gate evaluation
* v1.6 checkpoints with architecture/tokenizer/config hashes, RNG state,
  dataset position and a JSON sidecar for cheap "best" lookup
* atomic local saves plus verified Google Drive copies

``transformers`` is imported lazily inside :func:`train` only when no
tokenizer is injected, so the checkpoint helpers can be tested on CPU without
transformers or network access.
"""

import argparse
import copy
import csv
import dataclasses
import glob
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import ModelConfig, ModelConfigV15
from configs.v16_flash import (
    DIZEL_16_FLASH_CONFIG,
    DizelV16FlashConfig,
    ModelConfigV16Flash,
    PretrainConfigV16Flash,
)
from configs.v16_pro import (
    DIZEL_16_PRO_CONFIG,
    DizelV16ProConfig,
    ModelConfigV16Pro,
)
from model.architecture import DizelLM
from training.dataset import build_pretrain_loaders
from training.display_utils import (
    StepMeter,
    config_table,
    dim,
    error,
    info,
    ok,
    rule,
    section,
    validation,
    warning,
)
from training.pretrain import cosine_lr_with_warmup, evaluate, set_lr

# IMPORTANT: checkpoint configs are pickled dataclasses.
torch.serialization.add_safe_globals(
    [ModelConfig, ModelConfigV15, ModelConfigV16Flash, ModelConfigV16Pro]
)

FORMAT_VERSION = "1.6"
PHASE = "pretrain"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Architectural fields that must match exactly when resuming.
ARCH_FIELDS = (
    "vocab_size",
    "context_length",
    "d_model",
    "n_layers",
    "n_heads",
    "kv_heads",
    "head_dim",
    "intermediate_size",
    "norm_type",
    "norm_eps",
    "use_rope",
    "rope_theta",
    "bias",
    "weight_tying",
    "dropout",
    "num_experts",
    "top_k",
    "router_aux_loss_coef",
    "router_z_loss_coef",
)

_MISSING = object()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _schedule_tag_v16(budget: int, lr: float, min_lr: float, warmup: int) -> str:
    """Stable identifier of a phase's LR schedule (v1.6 namespace)."""
    return (
        f"v16|b={int(budget)}|lr={float(lr)!r}|min={float(min_lr)!r}|"
        f"w={int(warmup)}"
    )


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Current git commit, or ``"unknown"`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def config_hash(cfg) -> str:
    """SHA-256 of the model/tokenizer/pretrain configuration."""
    def _as_dict(value):
        if dataclasses.is_dataclass(value):
            return dataclasses.asdict(value)
        if isinstance(value, dict):
            return value
        return vars(value)

    payload = {
        "model": _as_dict(cfg.model),
        "tokenizer": _as_dict(cfg.tokenizer),
        "pretrain": _as_dict(cfg.pretrain),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rng_state() -> dict:
    """Capture Python, NumPy, torch CPU and (when present) CUDA RNG state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = None
    return state


def _restore_rng_state(state: Optional[dict], device: str = "cpu") -> None:
    if not state:
        return
    try:
        random.setstate(tuple(state["python"]))
    except Exception as exc:
        warning(f"Could not restore Python RNG state: {exc}")
    try:
        np_state = state["numpy"]
        np.random.set_state(
            (
                np_state[0],
                np.asarray(np_state[1], dtype=np.uint32),
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            )
        )
    except Exception as exc:
        warning(f"Could not restore NumPy RNG state: {exc}")
    try:
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            torch_state = torch.tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state.cpu())
    except Exception as exc:
        warning(f"Could not restore torch RNG state: {exc}")

    cuda_state = state.get("cuda")
    if cuda_state and torch.cuda.is_available() and device.startswith("cuda"):
        try:
            # torch.load(map_location="cuda") moves the saved RNG ByteTensors
            # to the GPU; set_rng_state_all requires CPU ByteTensors.
            restored = [
                item.detach().to(device="cpu", dtype=torch.uint8).contiguous()
                if isinstance(item, torch.Tensor)
                else torch.tensor(item, dtype=torch.uint8)
                for item in cuda_state
            ]
            torch.cuda.set_rng_state_all(restored)
        except Exception as exc:
            warning(f"Could not restore CUDA RNG state: {exc}")


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value):
    """JSON sidecars stay strict JSON: no Infinity/NaN literals."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_copy(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` through a temp file in the destination dir."""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    try:
        shutil.copy2(src, tmp)
        if os.path.getsize(tmp) != os.path.getsize(src):
            raise RuntimeError(
                f"Copy size mismatch: {tmp} != {src}"
            )
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
def save_checkpoint_v16(
    model: DizelLM,
    optimizer: Optional[optim.Optimizer],
    scaler,
    step: int,
    tokens_seen: int,
    val_loss: Optional[float],
    cfg: DizelV16FlashConfig,
    path: str,
    tokenizer=None,
    dataset_state: Optional[dict] = None,
    phase0: Optional[int] = None,
    phase_tag: Optional[str] = None,
    best_val: Optional[float] = None,
    extra_meta: Optional[dict] = None,
) -> str:
    """Atomically save a v1.6 checkpoint plus its JSON sidecar."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mc = cfg.model
    pc = cfg.pretrain

    tokenizer_name = getattr(tokenizer, "model_name", None)
    tokenizer_hash = tokenizer.fingerprint() if tokenizer is not None else None
    vocab_size = (
        len(tokenizer) if tokenizer is not None else int(mc.vocab_size)
    )
    scaler_state = (
        scaler.state_dict()
        if scaler is not None and scaler.is_enabled()
        else None
    )

    ckpt = {
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "val_loss": val_loss,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler_state,
        "model_cfg": mc,
        "_format_version": FORMAT_VERSION,
        "arch_version": cfg.preview_slug,
        "release_stage": cfg.release_stage,
        "phase": PHASE,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "vocab_size": int(vocab_size),
        "tokenizer_name": tokenizer_name,
        "tokenizer_hash": tokenizer_hash,
        "num_experts": int(mc.num_experts),
        "top_k": int(mc.top_k),
        "router": {
            "aux_loss_coef": float(mc.router_aux_loss_coef),
            "z_loss_coef": float(mc.router_z_loss_coef),
            "dropless": True,
        },
        "scheduler": {
            "type": "cosine_warmup",
            "max_steps": int(pc.max_steps),
            "warmup_steps": int(pc.warmup_steps),
            "lr": float(pc.lr),
            "min_lr": float(pc.min_lr),
        },
        "rng_state": _rng_state(),
    }
    if dataset_state is not None:
        ckpt["dataset_state"] = dataset_state
    if phase0 is not None:
        ckpt["phase0"] = int(phase0)
    if phase_tag is not None:
        ckpt["phase_tag"] = phase_tag
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
        "tokens_seen": int(tokens_seen),
        "val_loss": _finite_or_none(val_loss),
        "best_val": _finite_or_none(best_val),
        "config_hash": ckpt["config_hash"],
        "tokenizer_hash": tokenizer_hash,
        "git_commit": ckpt["git_commit"],
        "num_experts": int(mc.num_experts),
        "top_k": int(mc.top_k),
        "arch_version": cfg.preview_slug,
        "release_stage": cfg.release_stage,
    }
    _write_json_atomic(path + ".json", sidecar)

    ok(
        f"Checkpoint saved -> {os.path.basename(path)} "
        f"(step={step:,}, val_loss={val_loss if val_loss is None else f'{val_loss:.4f}'})"
    )
    return path


def _cfg_field(value, name: str):
    if isinstance(value, dict):
        return value.get(name, _MISSING)
    return getattr(value, name, _MISSING)


def load_checkpoint_v16(
    path: str,
    model: DizelLM,
    optimizer: Optional[optim.Optimizer] = None,
    scaler=None,
    tokenizer=None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Load a v1.6 checkpoint, validating architecture and tokenizer."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    fmt = ckpt.get("_format_version")
    if fmt != FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint '{path}' has format {fmt!r}, expected "
            f"{FORMAT_VERSION!r}. This is not a Dizel 1.6 checkpoint."
        )

    ckpt_cfg = ckpt.get("model_cfg")
    if ckpt_cfg is None:
        raise ValueError(f"Checkpoint '{path}' has no model_cfg")

    mismatches = []
    for field in ARCH_FIELDS:
        ckpt_value = _cfg_field(ckpt_cfg, field)
        model_value = getattr(model.cfg, field, _MISSING)
        if ckpt_value is _MISSING or model_value is _MISSING:
            mismatches.append(
                f"{field}: checkpoint={ckpt_value!r} model={model_value!r}"
            )
        elif ckpt_value != model_value:
            mismatches.append(
                f"{field}: checkpoint={ckpt_value!r} model={model_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Checkpoint architecture does not match the model:\n  "
            + "\n  ".join(mismatches)
        )

    if tokenizer is not None and ckpt.get("tokenizer_hash"):
        actual = tokenizer.fingerprint()
        if actual != ckpt["tokenizer_hash"]:
            raise ValueError(
                "Tokenizer hash mismatch: checkpoint has "
                f"{ckpt['tokenizer_hash'][:16]}..., supplied tokenizer has "
                f"{actual[:16]}... (checkpoint tokenizer: "
                f"{ckpt.get('tokenizer_name')!r})"
            )

    def _strip_prefix(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        return key

    state_dict = {
        _strip_prefix(key): value
        for key, value in ckpt["model_state"].items()
    }
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint state_dict does not match the model.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}"
        )
    model.load_state_dict(state_dict, strict=True)

    if optimizer is not None and ckpt.get("optim_state") is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if restore_rng:
        _restore_rng_state(ckpt.get("rng_state"), device=str(device))

    step = int(ckpt.get("step", 0))
    info(
        f"Loaded checkpoint '{os.path.basename(path)}' "
        f"(step={step:,}, tokens={int(ckpt.get('tokens_seen', 0)):,}, "
        f"val_loss={ckpt.get('val_loss')})"
    )

    return {
        "step": step,
        "tokens_seen": int(ckpt.get("tokens_seen", 0)),
        "val_loss": ckpt.get("val_loss"),
        "best_val": ckpt.get("best_val"),
        "dataset_state": ckpt.get("dataset_state"),
        "phase0": ckpt.get("phase0"),
        "phase_tag": ckpt.get("phase_tag"),
        "phase": ckpt.get("phase"),
        "base_checkpoint": ckpt.get("base_checkpoint"),
        "reference_checkpoint": ckpt.get("reference_checkpoint"),
        "scheduler": ckpt.get("scheduler"),
        "meta": {
            "config_hash": ckpt.get("config_hash"),
            "git_commit": ckpt.get("git_commit"),
            "tokenizer_name": ckpt.get("tokenizer_name"),
            "tokenizer_hash": ckpt.get("tokenizer_hash"),
            "arch_version": ckpt.get("arch_version"),
            "release_stage": ckpt.get("release_stage"),
            "scheduler": ckpt.get("scheduler"),
        },
    }


# ---------------------------------------------------------------------------
# Verified Drive sync
# ---------------------------------------------------------------------------
def sync_to_drive(local_path: str, drive_dir: str) -> Optional[str]:
    """Copy a checkpoint (and sidecar) to Drive, verifying size and hash."""
    if not drive_dir:
        return None

    local_path = os.path.abspath(local_path)
    drive_dir = os.path.abspath(drive_dir)
    destination = os.path.join(drive_dir, os.path.basename(local_path))
    if os.path.normcase(local_path) == os.path.normcase(destination):
        return destination

    os.makedirs(drive_dir, exist_ok=True)
    tmp = destination + ".tmp"
    local_size = os.path.getsize(local_path)
    local_hash = _sha256(local_path)

    try:
        shutil.copy2(local_path, tmp)
        copied_size = os.path.getsize(tmp)
        copied_hash = _sha256(tmp)
        if copied_size != local_size or copied_hash != local_hash:
            raise RuntimeError(
                f"Drive copy verification failed for {destination}: "
                f"size {copied_size}/{local_size}, hash "
                f"{copied_hash[:16]}/{local_hash[:16]}"
            )
        os.replace(tmp, destination)
    except Exception as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        error(f"Drive checkpoint sync failed: {exc}")
        raise

    sidecar = local_path + ".json"
    if os.path.exists(sidecar):
        sidecar_dst = destination + ".json"
        sidecar_tmp = sidecar_dst + ".tmp"
        try:
            shutil.copy2(sidecar, sidecar_tmp)
            if os.path.getsize(sidecar_tmp) != os.path.getsize(sidecar):
                raise RuntimeError(
                    f"Drive sidecar size mismatch for {sidecar_dst}"
                )
            os.replace(sidecar_tmp, sidecar_dst)
        except Exception as exc:
            if os.path.exists(sidecar_tmp):
                try:
                    os.remove(sidecar_tmp)
                except OSError:
                    pass
            error(f"Drive sidecar sync failed: {exc}")
            raise

    ok(f"Synced -> {destination} ({local_size / 1e9:.2f} GB)")
    return destination


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    cfg: Union[DizelV16FlashConfig, DizelV16ProConfig],
    resume_from: Optional[str] = None,
    tokenizer=None,
) -> None:
    rule(f"{cfg.display_name} - Pre-training")

    pc = cfg.pretrain
    mc = cfg.model

    # -- Validation -------------------------------------------------------
    for name in (
        "batch_size",
        "grad_accum",
        "max_steps",
        "eval_interval",
        "save_interval",
        "log_interval",
    ):
        value = getattr(pc, name)
        if int(value) <= 0:
            raise ValueError(
                f"PretrainConfigV16Flash.{name} must be > 0, got {value!r}"
            )
    if not str(pc.checkpoint_dir).strip():
        raise ValueError("PretrainConfigV16Flash.checkpoint_dir must not be empty")
    if not str(pc.log_dir).strip():
        raise ValueError("PretrainConfigV16Flash.log_dir must not be empty")
    using_pretokenized = bool(str(pc.tokenized_data_dir).strip())
    if using_pretokenized:
        manifest_path = os.path.join(pc.tokenized_data_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                "Pre-tokenized corpus manifest not found: "
                f"{manifest_path!r}. Point --tokenized_data_dir at the "
                "uploaded data/v16 directory."
            )
    elif not os.path.exists(pc.data_path):
        raise FileNotFoundError(
            f"Training corpus not found: {pc.data_path!r}"
        )

    # -- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info(
            f"GPU    : {torch.cuda.get_device_name()} / "
            f"{props.total_memory / 1e9:.1f} GB VRAM"
        )
        # fp32 parameters + gradients + two fp32 AdamW moments. Activation
        # memory is extra, so this is a floor, not a full estimate.
        state_gb = mc.param_estimate() * 16 / 1e9
        vram_gb = props.total_memory / 1e9
        if vram_gb < state_gb:
            raise RuntimeError(
                f"{cfg.display_name} needs ~{state_gb:.1f} GB for fp32 "
                "parameters, gradients, and AdamW states before activations, "
                f"but this GPU has {vram_gb:.1f} GB. Use a 40 GB+ GPU or add "
                "a sharded/8-bit optimizer path."
            )
        if vram_gb < state_gb + 3.0:
            warning(
                f"{cfg.display_name} training state needs ~{state_gb:.1f} GB "
                f"before activations; {vram_gb:.1f} GB VRAM leaves little "
                "headroom. Watch the first optimizer step."
            )

    # -- Mixed precision --------------------------------------------------
    use_amp = bool(pc.use_amp) and device.type == "cuda"
    if use_amp:
        # Recent PyTorch reports bf16 as "supported" on pre-Ampere GPUs because
        # it can emulate it. Emulation is far too slow for training, so require
        # native bf16 (Ampere or newer) and use fp16 on Turing/T4.
        native_bf16 = False
        if pc.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            try:
                native_bf16 = torch.cuda.is_bf16_supported(
                    including_emulation=False
                )
            except TypeError:
                native_bf16 = props.major >= 8
        if pc.amp_dtype == "bfloat16" and native_bf16:
            amp_dtype = torch.bfloat16
        else:
            if pc.amp_dtype == "bfloat16":
                warning(
                    "bf16 is only emulated on this GPU; falling back to fp16 "
                    "with GradScaler"
                )
            amp_dtype = torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_dtype = torch.float32
        amp_ctx = nullcontext()

    scaler = GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    amp_desc = (
        f"ON ({str(amp_dtype).replace('torch.', '')})" if use_amp else "OFF"
    )
    info(f"AMP    : {amp_desc}")
    info(f"Scaler : {'enabled' if scaler.is_enabled() else 'disabled'}")

    # -- Tokenizer --------------------------------------------------------
    if tokenizer is None:
        from training.tokenizer_qwen import QwenTokenizer

        tokenizer = QwenTokenizer(
            model_name=cfg.tokenizer.model_name,
            trust_remote_code=cfg.tokenizer.trust_remote_code,
            local_files_only=cfg.tokenizer.local_files_only,
            cache_dir=getattr(cfg.tokenizer, "cache_dir", None),
        )

    if len(tokenizer) > mc.vocab_size:
        raise ValueError(
            f"Tokenizer emits {len(tokenizer):,} ids, but the model only has "
            f"{mc.vocab_size:,} embedding rows. The tokenizer is larger than "
            "the model vocabulary."
        )
    if len(tokenizer) < mc.vocab_size:
        info(
            f"Tokenizer: {len(tokenizer):,} ids; model embedding rows: "
            f"{mc.vocab_size:,} ({mc.vocab_size - len(tokenizer):,} padded "
            "rows are unused). This is expected for Qwen2.5."
        )

    try:
        from training.tokenizer_qwen import verify_tokenizer
    except ImportError as exc:
        warning(f"Tokenizer verification helper unavailable: {exc}")
    else:
        verify_tokenizer(tokenizer)

    # -- Data -------------------------------------------------------------
    section("Data")
    if using_pretokenized:
        from training.pretokenized import build_pretokenized_loaders

        (
            train_loader,
            val_loader,
            train_ds,
            data_meta,
        ) = build_pretokenized_loaders(
            data_dir=pc.tokenized_data_dir,
            context_length=mc.context_length,
            batch_size=pc.batch_size,
            tokenizer=tokenizer,
            seed=pc.seed,
            num_workers=pc.num_workers,
        )
        total_bytes = int(data_meta["text_bytes"])
        total_tokens = int(data_meta["total_tokens"])
        bytes_per_token = total_bytes / max(total_tokens, 1)
        info(
            f"Data   : {data_meta['train_shards']} train shards / "
            f"{data_meta['val_shards']} val shards"
        )
        info(
            f"Tokens : {data_meta['train_tokens']:,} train / "
            f"{data_meta['val_tokens']:,} val"
        )
        dim("Pre-tokenized shards: no raw-text tokenization in this session.")
    else:
        cache_base = os.path.join(pc.cache_dir, tokenizer.cache_key())
        train_loader, val_loader, train_ds = build_pretrain_loaders(
            data_path=pc.data_path,
            tokenizer=tokenizer,
            context_length=mc.context_length,
            batch_size=pc.batch_size,
            train_split=pc.train_split,
            seed=pc.seed,
            num_workers=pc.num_workers,
            shard_size_mb=pc.shard_size_mb,
            cache_dir=cache_base,
            resume_enabled=pc.resume_enabled,
        )
        corpus_name = os.path.splitext(os.path.basename(pc.data_path))[0]
        tokens_bin = os.path.join(cache_base, corpus_name, "tokens.bin")
        total_bytes = os.path.getsize(pc.data_path)
        total_tokens = os.path.getsize(tokens_bin) // 4
        bytes_per_token = total_bytes / max(total_tokens, 1)

    # -- Bytes-per-token for BPB -----------------------------------------
    info(
        f"BPB basis: {total_bytes:,} bytes / {total_tokens:,} tokens "
        f"= {bytes_per_token:.4f} bytes/token"
    )
    dim(
        "BPB here is a token-split proxy (held-out tokens, corpus-wide "
        "bytes/token); the fixed byte-level holdout is a separate "
        "release-gate evaluation."
    )

    # -- Model ------------------------------------------------------------
    section("Model")
    model = DizelLM(mc).to(device)
    actual_params = model.num_parameters()
    info(f"Model  : {model}")
    info(
        f"Params : actual {actual_params / 1e6:.2f}M | "
        f"config total {mc.param_estimate() / 1e6:.1f}M | "
        f"config active {mc.active_param_estimate() / 1e6:.1f}M"
    )

    if pc.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        ok("Gradient checkpointing enabled")

    tokens_per_step = pc.batch_size * pc.grad_accum * mc.context_length
    if tokens_per_step != 65_536:
        warning(
            f"tokens/step is {tokens_per_step:,}, not the Phase A contract "
            "value of 65,536. The LR/schedule assume 65,536."
        )

    config_table(
        {
            "Dimension": str(mc.d_model),
            "Layers": str(mc.n_layers),
            "Heads": f"{mc.n_heads} (KV {mc.kv_heads})",
            "Vocab": f"{mc.vocab_size:,}",
            "Context": str(mc.context_length),
            "Experts": f"{mc.num_experts} (top-{mc.top_k}, dropless)",
            "Aux / z coef": f"{mc.router_aux_loss_coef} / {mc.router_z_loss_coef}",
            "Tokens/step": f"{tokens_per_step:,}",
        }
    )

    # -- Optimizer --------------------------------------------------------
    decay_params, no_decay_params = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": pc.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=pc.lr,
        betas=(pc.adam_beta1, pc.adam_beta2),
        eps=pc.adam_eps,
    )

    # -- Resume -----------------------------------------------------------
    start_step = 0
    tokens_seen = 0
    best_val = float("inf")
    best_bpb = None
    dataset_state = None
    ckpt_phase0 = None
    ckpt_phase_tag = None
    ckpt_val_loss = None

    best_path = os.path.join(pc.checkpoint_dir, f"{pc.run_name}-best.pt")
    last_path = os.path.join(pc.checkpoint_dir, f"{pc.run_name}-last.pt")
    final_path = os.path.join(pc.checkpoint_dir, f"{pc.run_name}-final.pt")

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_from}. Refusing to "
                "start a fresh run from an explicit --resume path."
            )
        loaded = load_checkpoint_v16(
            resume_from,
            model,
            optimizer=optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            device=str(device),
        )
        start_step = loaded["step"]
        tokens_seen = loaded["tokens_seen"]
        if loaded["best_val"] is not None:
            best_val = float(loaded["best_val"])
        if loaded["val_loss"] is not None:
            ckpt_val_loss = float(loaded["val_loss"])
            best_val = min(best_val, ckpt_val_loss)
        dataset_state = loaded["dataset_state"]
        ckpt_phase0 = loaded["phase0"]
        ckpt_phase_tag = loaded["phase_tag"]

    # A best checkpoint's sidecar is the cheapest source of truth after a
    # crash: never let a new run overwrite a better validation loss.
    best_sidecar = best_path + ".json"
    if os.path.exists(best_sidecar):
        try:
            with open(best_sidecar, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
            # A stale best from a different tokenizer or MoE shape must not
            # suppress a new best. Absent/None fields keep the old behaviour.
            current_tokenizer_hash = tokenizer.fingerprint()
            stale = []
            sidecar_tokenizer = sidecar.get("tokenizer_hash")
            if (
                sidecar_tokenizer is not None
                and sidecar_tokenizer != current_tokenizer_hash
            ):
                stale.append(
                    f"tokenizer_hash {str(sidecar_tokenizer)[:16]}... != "
                    f"{current_tokenizer_hash[:16]}..."
                )
            for field, actual in (
                ("num_experts", int(mc.num_experts)),
                ("top_k", int(mc.top_k)),
            ):
                sidecar_value = sidecar.get(field)
                if sidecar_value is not None and int(sidecar_value) != actual:
                    stale.append(
                        f"{field} {sidecar_value} != {actual}"
                    )
            sidecar_best = sidecar.get("val_loss")
            if stale:
                warning(
                    "Ignoring stale best-checkpoint sidecar "
                    f"'{os.path.basename(best_sidecar)}': "
                    + ", ".join(stale)
                )
            elif sidecar_best is not None:
                best_val = min(best_val, float(sidecar_best))
        except Exception as exc:
            warning(f"Could not read best-checkpoint sidecar: {exc}")

    # -- LR schedule anchor ----------------------------------------------
    phase_tag = _schedule_tag_v16(
        pc.max_steps, pc.lr, pc.min_lr, pc.warmup_steps
    )
    if ckpt_phase0 is not None and ckpt_phase_tag == phase_tag:
        phase0 = int(ckpt_phase0)
    else:
        phase0 = start_step
    target_steps = phase0 + pc.max_steps

    section("Training")
    _resume_note = (
        f" (resume {start_step:,} -> end {target_steps:,})"
        if start_step
        else ""
    )
    config_table(
        {
            "Steps": f"{pc.max_steps:,} this phase{_resume_note}",
            "Batch": f"{pc.batch_size} x {pc.grad_accum} = "
                     f"{pc.batch_size * pc.grad_accum}",
            "LR": f"{pc.lr:.2e} -> {pc.min_lr:.2e}",
            "Warmup": str(pc.warmup_steps),
            "Weight decay": str(pc.weight_decay),
            "Grad clip": str(pc.grad_clip),
            "Eval every": str(pc.eval_interval),
            "Save every": str(pc.save_interval),
        }
    )

    # -- Directories / CSV ------------------------------------------------
    os.makedirs(pc.checkpoint_dir, exist_ok=True)
    os.makedirs(pc.log_dir, exist_ok=True)
    log_path = os.path.join(pc.log_dir, f"{pc.run_name}.csv")
    new_log = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)

    header = [
        "step", "tokens_seen",
        "train_ce", "train_aux", "train_z", "train_total",
        "val_loss", "val_bpb",
        "lr", "grad_norm", "router_grad_norm",
        "expert_grad_min", "expert_grad_max",
        "tokens_per_sec", "vram_gb", "router_entropy",
    ]
    for expert_idx in range(mc.num_experts):
        header.extend(
            [
                f"expert_{expert_idx}_share",
                f"top1_{expert_idx}_share",
                f"top2_{expert_idx}_share",
            ]
        )
    if new_log:
        csv_writer.writerow(header)
        log_file.flush()
    else:
        info(f"Appending to existing CSV: {log_path}")

    def _num(value, spec: str) -> str:
        return "" if value is None else format(value, spec)

    def _write_csv_row(
        *,
        step: int,
        tokens_seen: int,
        metrics,
        lr,
        grad_norm,
        router_grad_norm,
        expert_grad_min,
        expert_grad_max,
        tokens_per_sec,
        vram_gb,
        router_entropy,
        share,
        top1_share,
        top2_share,
        val_loss,
        val_bpb,
    ) -> None:
        metrics = metrics or {}
        row = [
            str(step),
            str(tokens_seen),
            _num(metrics.get("ce"), ".6f"),
            _num(metrics.get("aux"), ".6f"),
            _num(metrics.get("z"), ".6f"),
            _num(metrics.get("total"), ".6f"),
            _num(val_loss, ".6f"),
            _num(val_bpb, ".6f"),
            _num(lr, ".8g"),
            _num(grad_norm, ".6g"),
            _num(router_grad_norm, ".6g"),
            _num(expert_grad_min, ".6g"),
            _num(expert_grad_max, ".6g"),
            _num(tokens_per_sec, ".1f"),
            _num(vram_gb, ".3f"),
            _num(router_entropy, ".6f"),
        ]
        for expert_idx in range(mc.num_experts):
            row.append(_num(share[expert_idx] if share else None, ".6f"))
            row.append(
                _num(top1_share[expert_idx] if top1_share else None, ".6f")
            )
            row.append(
                _num(top2_share[expert_idx] if top2_share else None, ".6f")
            )
        csv_writer.writerow(row)
        log_file.flush()

    # -- Dataset resume ---------------------------------------------------
    batch_num = 0

    if dataset_state is not None:
        from training.pretokenized import replay_loader_position

        (
            data_iter,
            batch_num,
            saved_pos,
            at_boundary,
        ) = replay_loader_position(
            train_ds,
            train_loader,
            dataset_state,
            start_step=start_step,
            batch_size=pc.batch_size,
            reshuffle_every_n_steps=pc.reshuffle_every_n_steps,
            seed=pc.seed,
        )
        if at_boundary:
            info(
                f"Resume: step {start_step:,} is a reshuffle boundary; "
                f"starting segment with seed {train_ds.seed} "
                f"({saved_pos:,} checkpointed windows not replayed)"
            )
        else:
            info(
                f"Resume: replayed {saved_pos:,} windows of the "
                f"checkpointed segment (step={start_step:,}, "
                f"seed={train_ds.seed})"
            )
    else:
        data_iter = iter(train_loader)

    def _dataset_state() -> dict:
        state = train_ds.get_state()
        # Main-process counter: correct with num_workers>0, unlike
        # train_ds.current_index which is set inside workers.
        state["pos"] = batch_num * pc.batch_size
        return state

    # -- Meter ------------------------------------------------------------
    meter = StepMeter(
        target_steps,
        tokens_per_step=tokens_per_step,
        start_step=start_step,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    resume_row_pending = start_step > 0
    last_written_step = None
    last_val_loss = ckpt_val_loss
    last_val_bpb = (
        last_val_loss / math.log(2) / max(bytes_per_token, 1e-9)
        if last_val_loss is not None
        else None
    )
    last_metrics = None
    last_saved_step = None

    expert_window = deque(maxlen=1000)
    last_collapse_warning_step = -10_000

    # -- Generation smoke -------------------------------------------------
    if getattr(pc, "generation_smoke", True):
        section("Generation smoke")
        model.eval()
        prompt_ids = tokenizer.encode("Hello")
        if not prompt_ids and tokenizer.eos_id is not None:
            prompt_ids = [tokenizer.eos_id]
        if prompt_ids:
            prompt = torch.tensor(
                [prompt_ids], dtype=torch.long, device=device
            )
            with torch.no_grad():
                generated = model.generate(
                    prompt, max_new_tokens=5, eos_ids=[]
                )
            excerpt = tokenizer.decode(generated[0].tolist())
            excerpt = "".join(
                ch if ch.isprintable() else " " for ch in excerpt
            )
            info(f"Generation: {excerpt[:80]!r}")
        else:
            warning("Generation smoke skipped: tokenizer produced no ids")
        model.train()

    def _grad_norm(select) -> float:
        """Total grad norm over selected params (no per-param .item())."""
        total = None
        for name, param in model.named_parameters():
            if param.grad is None or not select(name):
                continue
            value = param.grad.detach().float().pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return 0.0
        return float(total.sqrt().item())

    # -- Main loop --------------------------------------------------------
    while step < target_steps:
        step_started = time.time()

        # Fetch one micro-batch.
        try:
            x, y = next(data_iter)
            batch_num += 1
        except StopIteration:
            train_ds.reshuffle(new_seed=pc.seed + step)
            data_iter = iter(train_loader)
            batch_num = 0
            try:
                x, y = next(data_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    "Training loader produced no batches; the corpus is too "
                    "small for the configured context length."
                ) from exc
            batch_num = 1

        x, y = x.to(device), y.to(device)

        # -- Forward / backward over grad_accum micro-steps ---------------
        ce_acc = aux_acc = z_acc = total_acc = None
        expert_acc = top1_acc = top2_acc = None
        entropy_acc = None

        for micro in range(pc.grad_accum):
            with amp_ctx:
                _, ce_loss, aux = model(x, y, return_aux=True)
                total_loss = aux["total_loss"]
                loss = total_loss / pc.grad_accum

            for name, value in (
                ("ce", ce_loss),
                ("aux", aux["aux_loss"]),
                ("z", aux["z_loss"]),
                ("total", total_loss),
            ):
                if value is None or not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(
                        f"Non-finite {name} loss at step {step + 1} "
                        f"micro {micro + 1}: {value}"
                    )

            coef_aux = float(mc.router_aux_loss_coef)
            coef_z = float(mc.router_z_loss_coef)
            if step >= 3:
                if coef_aux > 0 and float(aux["aux_loss"].item()) == 0.0:
                    raise RuntimeError(
                        f"Router aux loss is exactly zero at step {step + 1} "
                        f"while router_aux_loss_coef={coef_aux}. A configured "
                        "loss term must never silently disappear."
                    )
                if coef_z > 0 and float(aux["z_loss"].item()) == 0.0:
                    raise RuntimeError(
                        f"Router z-loss is exactly zero at step {step + 1} "
                        f"while router_z_loss_coef={coef_z}. A configured "
                        "loss term must never silently disappear."
                    )
                expected_total = (
                    ce_loss + coef_aux * aux["aux_loss"] + coef_z * aux["z_loss"]
                )
                if not torch.allclose(
                    total_loss,
                    expected_total,
                    atol=1e-5,
                    rtol=1e-4,
                ):
                    raise RuntimeError(
                        f"total_loss does not equal ce + aux + z at step "
                        f"{step + 1}: {float(total_loss.item()):.6f} != "
                        f"{float(expected_total.item()):.6f}"
                    )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate detached metrics; convert to floats once per step.
            ce_d = ce_loss.detach().float()
            aux_d = aux["aux_loss"].detach().float()
            z_d = aux["z_loss"].detach().float()
            total_d = total_loss.detach().float()
            ce_acc = ce_d if ce_acc is None else ce_acc + ce_d
            aux_acc = aux_d if aux_acc is None else aux_acc + aux_d
            z_acc = z_d if z_acc is None else z_acc + z_d
            total_acc = total_d if total_acc is None else total_acc + total_d

            counts = aux["expert_counts"].detach().float().sum(dim=0)
            top1 = aux["top1_counts"].detach().float().sum(dim=0)
            top2 = aux["top2_counts"].detach().float().sum(dim=0)
            entropy = aux["router_entropy"].detach().float().mean()
            expert_acc = counts if expert_acc is None else expert_acc + counts
            top1_acc = top1 if top1_acc is None else top1_acc + top1
            top2_acc = top2 if top2_acc is None else top2_acc + top2
            entropy_acc = entropy if entropy_acc is None else entropy_acc + entropy

            if micro + 1 < pc.grad_accum:
                try:
                    x, y = next(data_iter)
                    batch_num += 1
                except StopIteration:
                    train_ds.reshuffle(new_seed=pc.seed + step)
                    data_iter = iter(train_loader)
                    batch_num = 0
                    try:
                        x, y = next(data_iter)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "Training loader produced no batches; the corpus "
                            "is too small for the configured context length."
                        ) from exc
                    batch_num = 1
                x, y = x.to(device), y.to(device)

        # -- Optimizer step -----------------------------------------------
        local_step = step - phase0
        lr = cosine_lr_with_warmup(
            local_step,
            pc.warmup_steps,
            pc.max_steps,
            pc.lr,
            pc.min_lr,
        )
        set_lr(optimizer, lr)

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = _grad_norm(lambda name: True)
        router_grad_norm = _grad_norm(lambda name: ".router." in name)

        expert_grad_sq = [None] * mc.num_experts
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            marker = ".experts."
            if marker not in name:
                continue
            tail = name.split(marker, 1)[1]
            expert_idx = int(tail.split(".", 1)[0])
            value = param.grad.detach().float().pow(2).sum()
            if expert_grad_sq[expert_idx] is None:
                expert_grad_sq[expert_idx] = value
            else:
                expert_grad_sq[expert_idx] = (
                    expert_grad_sq[expert_idx] + value
                )
        expert_grad_norms = [
            math.sqrt(float(value.item())) if value is not None else 0.0
            for value in expert_grad_sq
        ]

        torch.nn.utils.clip_grad_norm_(model.parameters(), pc.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_elapsed = max(time.time() - step_started, 1e-9)
        step += 1
        tokens_seen += tokens_per_step

        # -- Metrics ------------------------------------------------------
        ce_val = float(ce_acc.item()) / pc.grad_accum
        aux_val = float(aux_acc.item()) / pc.grad_accum
        z_val = float(z_acc.item()) / pc.grad_accum
        total_val = float(total_acc.item()) / pc.grad_accum
        entropy_val = float(entropy_acc.item()) / pc.grad_accum

        counts_total = float(expert_acc.sum().item())
        share = (
            (expert_acc / max(counts_total, 1.0)).tolist()
            if counts_total > 0
            else [0.0] * mc.num_experts
        )
        top1_total = float(top1_acc.sum().item())
        top1_share = (
            (top1_acc / max(top1_total, 1.0)).tolist()
            if top1_total > 0
            else [0.0] * mc.num_experts
        )
        top2_total = float(top2_acc.sum().item())
        top2_share = (
            (top2_acc / max(top2_total, 1.0)).tolist()
            if top2_total > 0
            else [0.0] * mc.num_experts
        )

        if bool((expert_acc <= 0).any()):
            raise RuntimeError(
                f"Expert utilization failure at step {step}: counts="
                f"{expert_acc.tolist()}. Every expert must receive tokens in "
                "a dropless MoE."
            )

        expert_window.append(torch.tensor(share, dtype=torch.float32))
        if len(expert_window) == expert_window.maxlen:
            mean_share = torch.stack(list(expert_window)).mean(dim=0).tolist()
            min_share = min(mean_share)
            if min_share < 0.02:
                raise RuntimeError(
                    f"Expert collapse stop: minimum mean share {min_share:.4f} "
                    f"over the last {expert_window.maxlen} steps "
                    f"(threshold 0.02, shares={mean_share})"
                )
            if min_share < 0.05 and step - last_collapse_warning_step >= 100:
                warning(
                    f"Expert collapse warning at step {step}: minimum mean "
                    f"share {min_share:.4f} over {expert_window.maxlen} steps "
                    f"(threshold 0.05)"
                )
                last_collapse_warning_step = step

        if step >= 3 and router_grad_norm == 0.0:
            raise RuntimeError(
                f"Router gradient norm is zero at step {step}; the router is "
                "not learning and the MoE is effectively disabled."
            )

        metrics = {
            "ce": ce_val,
            "aux": aux_val,
            "z": z_val,
            "total": total_val,
        }
        last_metrics = metrics
        grad_min = min(expert_grad_norms)
        grad_max = max(expert_grad_norms)
        step_tokens_per_sec = tokens_per_step / step_elapsed
        vram_gb = (
            torch.cuda.max_memory_allocated() / 1e9
            if device.type == "cuda"
            else 0.0
        )

        # -- Evaluation ---------------------------------------------------
        val_loss = None
        val_bpb = None
        is_eval = step % pc.eval_interval == 0
        if is_eval:
            val_loss = evaluate(
                model, val_loader, pc.eval_iters, device, amp_ctx
            )
            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at step {step}: {val_loss}"
                )
            val_bpb = val_loss / math.log(2) / max(bytes_per_token, 1e-9)
            last_val_loss = val_loss
            last_val_bpb = val_bpb
            validation(step, val_loss, extra=f"bpb {val_bpb:.4f}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # -- CSV row (log interval, eval, final, first resume step) -------
        should_log = (
            step % pc.log_interval == 0
            or step == start_step + 1
            or resume_row_pending
            or is_eval
            or step == target_steps
        )
        if should_log:
            _write_csv_row(
                step=step,
                tokens_seen=tokens_seen,
                metrics=metrics,
                lr=lr,
                grad_norm=grad_norm,
                router_grad_norm=router_grad_norm,
                expert_grad_min=grad_min,
                expert_grad_max=grad_max,
                tokens_per_sec=step_tokens_per_sec,
                vram_gb=vram_gb,
                router_entropy=entropy_val,
                share=share,
                top1_share=top1_share,
                top2_share=top2_share,
                val_loss=val_loss,
                val_bpb=val_bpb,
            )
            last_written_step = step
            resume_row_pending = False

            extra = (
                f"ce {ce_val:.4f} aux {aux_val:.4f} z {z_val:.4f} "
                f"gn {grad_norm:.2f} rgn {router_grad_norm:.2e} "
                f"exp {min(share):.1%}-{max(share):.1%}"
            )
            meter.update(
                step=step,
                loss=total_val,
                val_loss=val_loss if is_eval else None,
                lr=lr,
                extra=extra,
            )

        # -- Checkpoints --------------------------------------------------
        ds_state = _dataset_state()
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_bpb = val_bpb
            save_checkpoint_v16(
                model, optimizer, scaler, step, tokens_seen, val_loss, cfg,
                best_path,
                tokenizer=tokenizer,
                dataset_state=ds_state,
                phase0=phase0,
                phase_tag=phase_tag,
                best_val=best_val,
            )
            sync_to_drive(best_path, pc.drive_checkpoint_dir)

        if step != target_steps and step % pc.save_interval == 0:
            save_checkpoint_v16(
                model, optimizer, scaler, step, tokens_seen, last_val_loss,
                cfg, last_path,
                tokenizer=tokenizer,
                dataset_state=ds_state,
                phase0=phase0,
                phase_tag=phase_tag,
                best_val=best_val,
            )
            sync_to_drive(last_path, pc.drive_checkpoint_dir)
            last_saved_step = step

        if pc.milestone_interval > 0 and step % pc.milestone_interval == 0:
            milestone_path = os.path.join(
                pc.checkpoint_dir,
                f"{pc.run_name}-milestone-step{step:06d}.pt",
            )
            if last_saved_step == step and os.path.exists(last_path):
                _atomic_copy(last_path, milestone_path)
                if os.path.exists(last_path + ".json"):
                    _atomic_copy(last_path + ".json", milestone_path + ".json")
            else:
                save_checkpoint_v16(
                    model, optimizer, scaler, step, tokens_seen,
                    last_val_loss, cfg, milestone_path,
                    tokenizer=tokenizer,
                    dataset_state=ds_state,
                    phase0=phase0,
                    phase_tag=phase_tag,
                    best_val=best_val,
                )
            sync_to_drive(milestone_path, pc.drive_checkpoint_dir)
            _prune_milestones(milestone_path, pc, warning)

        # -- Periodic reshuffle -------------------------------------------
        # Skip the last step: checkpoints for it were just written, and
        # respawning workers right before exit would leave the final
        # checkpoint's dataset position out of sync with the last step.
        if (
            step < target_steps
            and pc.reshuffle_every_n_steps > 0
            and step % pc.reshuffle_every_n_steps == 0
        ):
            train_ds.reshuffle(new_seed=pc.seed + step)
            data_iter = iter(train_loader)
            batch_num = 0

    # -- Final row / checkpoint ------------------------------------------
    if last_written_step != step:
        _write_csv_row(
            step=step,
            tokens_seen=tokens_seen,
            metrics=last_metrics,
            lr=None,
            grad_norm=None,
            router_grad_norm=None,
            expert_grad_min=None,
            expert_grad_max=None,
            tokens_per_sec=None,
            vram_gb=None,
            router_entropy=None,
            share=None,
            top1_share=None,
            top2_share=None,
            val_loss=last_val_loss,
            val_bpb=last_val_bpb,
        )
        last_written_step = step

    ds_state = _dataset_state()
    final_val = last_val_loss if last_val_loss is not None else (
        best_val if math.isfinite(best_val) else None
    )
    save_checkpoint_v16(
        model, optimizer, scaler, step, tokens_seen, final_val, cfg,
        final_path,
        tokenizer=tokenizer,
        dataset_state=ds_state,
        phase0=phase0,
        phase_tag=phase_tag,
        best_val=best_val,
    )
    sync_to_drive(final_path, pc.drive_checkpoint_dir)

    log_file.close()
    meter.finalise()
    rule("Complete")

    if best_bpb is None and math.isfinite(best_val):
        best_bpb = best_val / math.log(2) / max(bytes_per_token, 1e-9)

    config_table(
        {
            "Best val loss": f"{best_val:.4f}" if math.isfinite(best_val) else "n/a",
            "Best BPB": f"{best_bpb:.4f}" if best_bpb is not None else "n/a",
            "Steps": f"{step:,}",
            "Tokens seen": f"{tokens_seen:,}",
            "Last checkpoint": last_path,
            "Best checkpoint": best_path,
            "Final checkpoint": final_path,
            "CSV": log_path,
        }
    )


def _prune_milestones(keep: str, pc, warn) -> None:
    """Keep exactly one local milestone and one Drive milestone."""
    keep_name = os.path.basename(keep)
    pattern = os.path.join(
        pc.checkpoint_dir, f"{pc.run_name}-milestone-step*.pt"
    )
    for stale in glob.glob(pattern):
        if os.path.basename(stale) == keep_name:
            continue
        os.remove(stale)
        stale_sidecar = stale + ".json"
        if os.path.exists(stale_sidecar):
            os.remove(stale_sidecar)
        dim(f"Removed old milestone {os.path.basename(stale)}")

    if not pc.drive_checkpoint_dir:
        return
    drive_pattern = os.path.join(
        pc.drive_checkpoint_dir, f"{pc.run_name}-milestone-step*.pt"
    )
    for stale in glob.glob(drive_pattern):
        if os.path.basename(stale) == keep_name:
            continue
        try:
            os.remove(stale)
            stale_sidecar = stale + ".json"
            if os.path.exists(stale_sidecar):
                os.remove(stale_sidecar)
        except OSError as exc:
            warn(f"Could not remove old Drive milestone {stale}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
PRETRAIN_CONFIGS = {
    "flash": DIZEL_16_FLASH_CONFIG,
    "pro": DIZEL_16_PRO_CONFIG,
}


def select_config(name: str):
    """Return a deep copy of the requested tier's pretraining config."""
    try:
        return copy.deepcopy(PRETRAIN_CONFIGS[name])
    except KeyError:
        raise ValueError(
            f"Unknown config {name!r}; expected one of "
            f"{', '.join(sorted(PRETRAIN_CONFIGS))}"
        ) from None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dizel 1.6 pre-training (MoE)"
    )
    parser.add_argument(
        "--config",
        choices=sorted(PRETRAIN_CONFIGS),
        default="flash",
        help="Model tier to train (default: flash).",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--tokenized_data_dir",
        type=str,
        default=None,
        help="Laptop-packed data/v16 directory (manifest.json + train/ + val/).",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--milestone_interval", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--drive_checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tokenizer_model", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--no_generation_smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cfg = select_config(args.config)
    pc = cfg.pretrain
    mc = cfg.model

    if args.data_path is not None:
        pc.data_path = args.data_path
    if args.tokenized_data_dir is not None:
        pc.tokenized_data_dir = args.tokenized_data_dir
    if args.max_steps is not None:
        pc.max_steps = args.max_steps
        pc.phase1_steps = args.max_steps
    if args.batch_size is not None:
        pc.batch_size = args.batch_size
    if args.grad_accum is not None:
        pc.grad_accum = args.grad_accum
    if args.lr is not None:
        pc.lr = args.lr
        pc.phase1_lr = args.lr
    if args.min_lr is not None:
        pc.min_lr = args.min_lr
        pc.phase1_min_lr = args.min_lr
    if args.warmup_steps is not None:
        pc.warmup_steps = args.warmup_steps
        pc.phase1_warmup = args.warmup_steps
    if args.context_length is not None:
        mc.context_length = args.context_length
    if args.eval_interval is not None:
        pc.eval_interval = args.eval_interval
    if args.save_interval is not None:
        pc.save_interval = args.save_interval
    if args.log_interval is not None:
        pc.log_interval = args.log_interval
    if args.milestone_interval is not None:
        pc.milestone_interval = args.milestone_interval
    if args.checkpoint_dir is not None:
        pc.checkpoint_dir = args.checkpoint_dir
    if args.drive_checkpoint_dir is not None:
        pc.drive_checkpoint_dir = args.drive_checkpoint_dir
    if args.log_dir is not None:
        pc.log_dir = args.log_dir
    if args.run_name is not None:
        pc.run_name = args.run_name
    if args.cache_dir is not None:
        pc.cache_dir = args.cache_dir
    if args.num_workers is not None:
        pc.num_workers = args.num_workers
    if args.seed is not None:
        pc.seed = args.seed
    if args.tokenizer_model is not None:
        cfg.tokenizer.model_name = args.tokenizer_model
    if args.local_files_only:
        cfg.tokenizer.local_files_only = True
    if args.no_amp:
        pc.use_amp = False
    if args.no_gradient_checkpointing:
        pc.gradient_checkpointing = False
    if args.no_generation_smoke:
        pc.generation_smoke = False

    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
