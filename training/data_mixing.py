"""
training/data_mixing.py — Weighted dataset mixing for SFT training.

Defines a DatasetMixConfig that describes which JSONL files to combine
and at what sampling weights. Used by MixedSFTDataset in dataset.py.

Weight semantics:
  - Higher weight = more frequently sampled during training
  - Weights are relative (3.0 vs 1.0 means 3× more samples from that source)
  - User's requested ratios: conv 3×, code 1.5×, factual 1.5×
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetMixEntry:
    """A single dataset source in the mix."""
    name: str               # Human-readable identifier
    path: str               # Path to JSONL file
    weight: float           # Sampling weight (higher = more frequent)
    max_samples: int = -1   # -1 = use all samples


@dataclass
class DatasetMixConfig:
    """Collection of datasets with their sampling weights."""
    datasets: List[DatasetMixEntry] = field(default_factory=list)
    seed: int = 42


# Default mix for Dizel v1.2 SFT
# Ratios: conversational 3×, code 1.5×, factual 1.5×
DEFAULT_MIX = DatasetMixConfig(datasets=[
    # ── Conversational (weight 3.0) — drives chat behavior ────────────
    DatasetMixEntry("ultrachat",    "data/processed/ultrachat.jsonl",    weight=3.0),
    DatasetMixEntry("oasst2",       "data/processed/oasst2.jsonl",       weight=3.0),
    DatasetMixEntry("sharegpt",     "data/processed/sharegpt.jsonl",     weight=3.0),

    # ── Instruction following (weight 2.0) — general task completion ──
    DatasetMixEntry("alpaca_gpt4",  "data/processed/alpaca_gpt4.jsonl",  weight=2.0),
    DatasetMixEntry("openorca",     "data/processed/openorca.jsonl",     weight=2.0, max_samples=100000),
    DatasetMixEntry("dolly",        "data/processed/dolly.jsonl",        weight=2.0),

    # ── Code (weight 1.5) ─────────────────────────────────────────────
    DatasetMixEntry("codealpaca",   "data/processed/codealpaca.jsonl",   weight=1.5),
    DatasetMixEntry("codefeedback", "data/processed/codefeedback.jsonl", weight=1.5),

    # ── Factual / editing (weight 1.5) ────────────────────────────────
    DatasetMixEntry("coedit",       "data/processed/coedit.jsonl",       weight=1.5),
    DatasetMixEntry("sciq",         "data/processed/sciq.jsonl",         weight=1.5),
])


if __name__ == "__main__":
    print(f"Default mix: {len(DEFAULT_MIX.datasets)} datasets")
    total_w = sum(d.weight for d in DEFAULT_MIX.datasets)
    for d in DEFAULT_MIX.datasets:
        pct = d.weight / total_w * 100
        cap = f" (max {d.max_samples})" if d.max_samples > 0 else ""
        print(f"  {d.name:16s}  weight={d.weight:.1f}  ({pct:4.1f}%){cap}")
