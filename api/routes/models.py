"""
api/routes/models.py
────────────────────
Discovery endpoints for the UI's model and mode pickers.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from model.naming import model_display_name

router = APIRouter(prefix="/models", tags=["models"])

_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _ROOT / "checkpoints"
_SUFFIXES = {".gguf", ".pt", ".safetensors"}

# Mirrors token_budget._PROFILES; used only if the private table can't be read.
_FALLBACK_PROFILES: Dict[str, List[str]] = {
    "Dizel": ["Fast", "Planning", "Coding", "Thinking"],
    "Mila": ["Fast", "Planning", "Coding", "Thinking"],
}


@router.get("/")
def list_models() -> List[Dict[str, Any]]:
    """Every loadable checkpoint under checkpoints/, newest first."""
    items: List[Dict[str, Any]] = []
    if _CHECKPOINT_DIR.exists():
        for path in _CHECKPOINT_DIR.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append(
                {
                    "name": path.name,
                    "stem": path.stem,
                    "display_name": model_display_name(path.name),
                    "path": str(path),
                    "size": stat.st_size,
                    "modified": int(stat.st_mtime),
                    "kind": path.suffix.lower().lstrip("."),
                }
            )
    items.sort(key=lambda item: item["modified"], reverse=True)
    return items


@router.get("/profiles")
def list_profiles() -> Dict[str, List[str]]:
    """Model variants and their available modes, for the composer pickers."""
    try:
        from dizel_core.token_budget import _PROFILES  # type: ignore[attr-defined]

        return {name: list(modes.keys()) for name, modes in _PROFILES.items()}
    except Exception:
        return _FALLBACK_PROFILES


@router.get("/info")
def model_info() -> Dict[str, Any]:
    """Static facts about the model family, for the Settings → Model tab."""
    try:
        from model import dizel_info as info
        from dizel_core.version import __version__

        return {
            "app_name": info.APP_NAME,
            "app_version": __version__,
            "model_name": info.MODEL_NAME,
            "version": info.VERSION,
            "model_size": info.MODEL_SIZE,
            "context_length": info.CONTEXT_LENGTH,
            "vocab_size": info.VOCAB_SIZE,
            "author": info.AUTHOR,
            "description": info.DESCRIPTION,
            "architecture": info.ARCHITECTURE,
            "num_layers": info.NUM_LAYERS,
            "num_heads": info.NUM_HEADS,
            "kv_heads": info.KV_HEADS,
            "hidden_dim": info.HIDDEN_DIM,
            "mlp_size": info.MLP_SIZE,
            "capabilities": sorted(info.CAPABILITIES),
            "training_data": info.TRAINING_DATA,
            "train_tokens": info.TRAIN_TOKENS,
            "train_steps": info.TRAIN_STEPS,
            "repository": info.REPOSITORY,
            "license": info.LICENSE,
            "build": info.BUILD,
        }
    except Exception as exc:
        return {"error": str(exc)}


@router.get("/defaults")
def defaults() -> Dict[str, Any]:
    """Where the app looks for checkpoints, plus the configured one."""
    from dizel_core.config_manager import ConfigManager

    config = ConfigManager.load()

    # Picking CUDA on a machine without it silently falls back to CPU, which
    # makes the header pill lie. Tell the UI whether the option is real.
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False

    return {
        "directory": str(_CHECKPOINT_DIR),
        "exists": _CHECKPOINT_DIR.exists(),
        "configured": config.get("checkpoint", ""),
        "device": config.get("device", "cpu"),
        "cuda_available": cuda,
        "sampling": config.get("sampling", {}),
        "system_prompt": config.get("system_prompt", ""),
        "appearance": config.get("appearance", {}),
        "cwd": os.getcwd(),
    }
