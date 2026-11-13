"""Fold unmerged LoRA adapters into base weights.

Distillation checkpoints are saved *unmerged*: a wrapped projection stores
``{prefix}.linear.weight`` plus ``{prefix}.lora_A`` / ``{prefix}.lora_B``.
Consumers that build a plain ``DizelLM`` must fold those adapters first.

Loading such a checkpoint with ``load_state_dict(..., strict=False)`` is not a
harmless no-op: every ``.linear.weight`` key is unexpected and every plain
``.weight`` key is missing, so the projection matrices stay randomly
initialized and generation collapses to noise.

This module is intentionally torch-agnostic; it only uses tensor methods.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def has_lora_adapters(state_dict: Dict[str, object]) -> bool:
    """True when ``state_dict`` contains at least one LoRA adapter pair."""
    return any(key.endswith(".lora_A") for key in state_dict)


def fold_lora_adapters(
    state_dict: Dict[str, object],
    *,
    lora_alpha: Optional[int] = None,
    lora_r: Optional[int] = None,
) -> Tuple[Dict[str, object], dict]:
    """Return ``(folded_state_dict, info)`` with ``W' = W + (B @ A) * alpha/r``.

    ``lora_alpha`` / ``lora_r`` normally come from checkpoint metadata. When
    absent, ``r`` is read from ``lora_A.shape[0]`` and alpha falls back to the
    repository convention ``alpha = 2 * r``.
    """
    out = dict(state_dict)
    adapters = 0
    alpha = lora_alpha
    r = lora_r

    for key in list(out.keys()):
        if not key.endswith(".linear.weight"):
            continue
        prefix = key[: -len(".linear.weight")]
        a_key, b_key = f"{prefix}.lora_A", f"{prefix}.lora_B"
        if a_key not in out or b_key not in out:
            continue  # plain nn.Linear, nothing to fold

        weight = out[key].float()
        a = out[a_key].float()
        b = out[b_key].float()
        r = lora_r if lora_r is not None else int(a.shape[0])
        alpha = lora_alpha if lora_alpha is not None else 2 * r
        out[f"{prefix}.weight"] = weight + (b @ a) * (alpha / r)
        del out[key], out[a_key], out[b_key]

        bias_key = f"{prefix}.linear.bias"
        if bias_key in out:
            out[f"{prefix}.bias"] = out.pop(bias_key)
        adapters += 1

    return out, {"adapters": adapters, "alpha": alpha, "r": r}
