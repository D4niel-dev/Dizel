"""Prompt-compaction rules for the v1.5 checkpoints.

Kept free of torch so tests and prompt-shaping code can use it without loading
the inference stack. ``model_backend`` re-exports these names for compatibility.
"""

_V15_IDENTITY = (
    "You are Dizel, an AI assistant created by the Dizel team. "
    "Answer the user's request directly and accurately."
)
_V15_SYSTEM_PROMPT_MAX_CHARS = 240


def _v15_system_prompt(system_prompt: str) -> str:
    """Compact a system turn to what a v1.5 checkpoint can follow."""
    text = " ".join(str(system_prompt or "").split())
    if not text:
        return ""
    if len(text) > _V15_SYSTEM_PROMPT_MAX_CHARS:
        return _V15_IDENTITY
    if "dizel" in text.lower():
        return text
    return f"{_V15_IDENTITY} {text}"
