"""model/naming.py — human-readable checkpoint names from file names.

The on-disk contract stays machine-readable
(``dizel-1.6-flash-preview-m1.pt``); the UI shows
``Dizel 1.6 Flash Preview``.  Milestones (``m1``, ``m2``), training phases
(``sft``, ``kd``, ``best``) and quantization suffixes never leak into the
label.  Non-Dizel checkpoints fall back to the Hugging Face convention of
joining the filename tokens with spaces.
"""

from __future__ import annotations

import re

__all__ = ["model_display_name"]

_CHECKPOINT_SUFFIXES = (
    ".safetensors",
    ".gguf",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
)

# Quantization names are the one place underscores are not separators:
# q4_k_m / q8_0 / iq4_xs must stay one token.
_TOKEN_RE = re.compile(
    r"(?:i?q\d+(?:_[a-z0-9]+)+)|[^-_\s]+",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"^v?(\d+(?:\.\d+)+)$", re.IGNORECASE)
_COMPACT_VERSION_RE = re.compile(r"^v(\d)(\d)$", re.IGNORECASE)
_PARAMS_RE = re.compile(r"^(\d+(?:\.\d+)?)([bmk])$", re.IGNORECASE)
_MILESTONE_RE = re.compile(r"^m\d+$", re.IGNORECASE)
_STEP_RE = re.compile(r"^(?:step|ckpt|checkpoint)[-_]?\d+$", re.IGNORECASE)
_QUANT_RE = re.compile(
    r"^(?:i?q\d+(?:_[a-z0-9]+)*|(?:f|fp|bf)\d+|int\d+|awq|gptq)$",
    re.IGNORECASE,
)
_PHASE_RE = re.compile(
    r"^(?:pretrain|sft|dpo|kd|best|last|final|latest|milestone|ckpt|checkpoint)$",
    re.IGNORECASE,
)

# Ordered the way a reader expects them: family before size before stage.
_FAMILIES = {
    "flash": "Flash",
    "pro": "Pro",
    "lite": "Lite",
    "mini": "Mini",
    "nano": "Nano",
    "air": "Air",
    "max": "Max",
    "ultra": "Ultra",
    "distill": "Distill",
}

_STAGES = {
    "preview": "Preview",
    "alpha": "Alpha",
    "beta": "Beta",
    "rc": "RC",
    "experimental": "Experimental",
}

_ACRONYMS = {
    "awq": "AWQ",
    "gptq": "GPTQ",
    "gguf": "GGUF",
    "fp16": "FP16",
    "bf16": "BF16",
    "f16": "F16",
    "int8": "INT8",
    "int4": "INT4",
    "it": "IT",
    "moe": "MoE",
    "dpo": "DPO",
    "sft": "SFT",
    "kd": "KD",
    "rc": "RC",
    "api": "API",
    "ui": "UI",
    "hf": "HF",
}


def _stem(filename: str) -> str:
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1].strip()
    lower = name.lower()
    for suffix in _CHECKPOINT_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _tokens(stem: str) -> list[str]:
    # Keep dots inside versions (1.6, Qwen2.5); treat other dots as separators.
    normalized = re.sub(r"(?<!\d)\.|\.(?!\d)", "-", stem)
    return [token for token in _TOKEN_RE.findall(normalized) if token]


def _version(token: str):
    match = _VERSION_RE.match(token)
    if match:
        return match.group(1)
    match = _COMPACT_VERSION_RE.match(token)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return None


def _params(token: str):
    match = _PARAMS_RE.match(token)
    if not match:
        return None
    return f"{match.group(1)}{match.group(2).upper()}"


def _title(token: str) -> str:
    lower = token.lower()
    if lower in _ACRONYMS:
        return _ACRONYMS[lower]
    if _version(token):
        return token
    params = _params(token)
    if params:
        return params
    if re.fullmatch(r"\d+(?:\.\d+)*", token):
        return token
    if token.isupper() and len(token) <= 5:
        return token
    return token[:1].upper() + token[1:]


def _dizel_display(tokens: list[str]) -> str:
    version = family = params = stage = None
    extras: list[str] = []

    for token in tokens:
        lower = token.lower()
        if (
            _MILESTONE_RE.match(lower)
            or _STEP_RE.match(lower)
            or _QUANT_RE.match(lower)
            or _PHASE_RE.match(lower)
        ):
            continue

        parsed_version = _version(lower)
        if parsed_version and version is None:
            version = parsed_version
            continue

        parsed_params = _params(lower)
        if parsed_params and params is None:
            params = parsed_params
            continue

        if lower in _FAMILIES and family is None:
            family = _FAMILIES[lower]
            continue

        if lower in _STAGES and stage is None:
            stage = _STAGES[lower]
            continue

        extras.append(_title(token))

    parts = ["Dizel"]
    if version:
        parts.append(version)
    if family:
        parts.append(family)
    if params:
        parts.append(params)
    if stage:
        parts.append(stage)
    parts.extend(extras)
    return " ".join(parts)


def _generic_display(tokens: list[str]) -> str:
    return " ".join(_title(token) for token in tokens)


def model_display_name(filename: str) -> str:
    """Return the label the UI should show for a checkpoint file.

    ``dizel-1.6-flash-preview-m1.pt`` -> ``Dizel 1.6 Flash Preview``
    ``dizel-v15-distill-best.pt``     -> ``Dizel 1.5 Distill``
    ``Qwen2.5-7B-Instruct.gguf``      -> ``Qwen2.5 7B Instruct``
    """
    stem = _stem(filename)
    tokens = _tokens(stem)
    if not tokens:
        return "Model"
    if tokens[0].lower() == "dizel":
        return _dizel_display(tokens[1:])
    return _generic_display(tokens)
