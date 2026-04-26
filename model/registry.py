"""
model/registry.py — Model registry for Dizel model family.

Provides a central place to define and look up model configurations.
Supports multiple model variants (Dizel v1.1, v1.2, Mila, etc.)
so the UI and training scripts can reference models by name.

Usage:
    from model.registry import MODEL_REGISTRY, get_model_config

    cfg = get_model_config("dizel-v1.2")
    model = DizelLM(cfg)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig


@dataclass
class ModelEntry:
    """A registered model variant."""
    name: str                       # Display name (e.g., "Dizel v1.2")
    key: str                        # Lookup key (e.g., "dizel-v1.2")
    config: ModelConfig             # Architecture config
    description: str = ""           # Human-readable summary
    author: str = "D4niel-dev"
    version: str = ""
    param_count: str = ""           # Approximate (e.g., "252M")
    is_default: bool = False        # Default model for the UI
    status: str = "available"       # "available", "coming_soon", "deprecated"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(entry: ModelEntry) -> None:
    """Add a model to the registry."""
    MODEL_REGISTRY[entry.key] = entry


def get_model_config(key: str) -> ModelConfig:
    """Look up a ModelConfig by registry key."""
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{key}' not found. Available: {available}")
    return MODEL_REGISTRY[key].config


def get_default_model() -> ModelEntry:
    """Return the default model entry."""
    for entry in MODEL_REGISTRY.values():
        if entry.is_default:
            return entry
    # Fallback: first registered
    return next(iter(MODEL_REGISTRY.values()))


def list_models() -> list:
    """Return all registered models as a list of ModelEntry."""
    return list(MODEL_REGISTRY.values())


# ---------------------------------------------------------------------------
# Built-in Models
# ---------------------------------------------------------------------------

# Dizel v1.1 (legacy, deprecated)
register_model(ModelEntry(
    name="Dizel v1.1",
    key="dizel-v1.1",
    config=ModelConfig(
        vocab_size=32000,
        context_length=2048,
        d_model=768,
        n_layers=12,
        n_heads=12,
        ffn_mult=4,
        dropout=0.05,
        weight_tying=True,
        bias=False,
        rope_base=10000.0,  # Not used — v1.1 used learned pos_emb
    ),
    description="The original 110.35M parameter model with learned positional embeddings.",
    version="v1.1.0",
    param_count="~110.35M",
    is_default=False,
    status="deprecated",
))

# Dizel v1.2.1 (current default)
register_model(ModelEntry(
    name="Dizel v1.2.1",
    key="dizel-v1.2",
    config=ModelConfig(
        vocab_size=32000,
        context_length=2048,
        d_model=896,
        n_layers=20,
        n_heads=16,
        ffn_mult=3.5,
        dropout=0.05,
        weight_tying=True,
        bias=False,
        rope_base=10000.0,
    ),
    description="~205.3M parameter model with RoPE, 2K context, and Code-First dataset mixing.",
    version="v1.2.1",
    param_count="~205.3M",
    is_default=True,
    status="available",
))

# Mila v1.0 (sister model — conversational, warm, friendly)
register_model(ModelEntry(
    name="Mila",
    key="mila-v1.0-beta",
    config=ModelConfig(
        vocab_size=32000,
        context_length=1024,
        d_model=768,
        n_layers=12,
        n_heads=12,
        ffn_mult=4,
        dropout=0.05,
        weight_tying=True,
        bias=False,
        rope_base=10000.0,
    ),
    description="The ~110M conversational sister model. Warm, talkative, encouraging personality.",
    author="D4niel-dev",
    version="v1.0.0-Beta",
    param_count="~110M",
    is_default=False,
    status="coming_soon",
))


# ---------------------------------------------------------------------------
# Quick info
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model Registry: {len(MODEL_REGISTRY)} models\n")
    for entry in list_models():
        default = " ★ DEFAULT" if entry.is_default else ""
        status = f" [{entry.status}]" if entry.status != "available" else ""
        print(f"  {entry.key:16s}  {entry.name:16s}  {entry.param_count:>8s}  ctx={entry.config.context_length}{default}{status}")
        print(f"                    {entry.description}")
        print()
