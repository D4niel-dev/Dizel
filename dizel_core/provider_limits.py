"""Per-provider context windows, used as each provider's usage limit.

Local checkpoints report their own window; API providers carry no metadata, so
they resolve to a per-provider default that the user can override in settings.
"""

from __future__ import annotations

from typing import Any

# Typical maximum context each provider serves. Conservative on purpose: a
# bigger number only hides truncation, it never prevents it.
DEFAULT_LIMITS = {
    "openai": 131_072,
    "groq": 131_072,
    "mistral": 131_072,
    "xai": 131_072,
    "cohere": 131_072,
    "azure": 131_072,
    "meta": 131_072,
    "anthropic": 200_000,
    "google": 1_048_576,
    "ai21": 262_144,
    "ollama": 32_768,
    "custom": 32_768,
}

FALLBACK = 32_768
LOCAL_FALLBACK = 2_048
MIN_LIMIT = 512
KEY = "provider_limits"


def default_for(slug: str) -> int:
    return DEFAULT_LIMITS.get((slug or "").lower(), FALLBACK)


def resolve(
    slug: str,
    model_ctx_len: int | None = None,
    overrides: Any = None,
) -> int:
    slug = (slug or "local").lower()
    if slug == "local":
        if model_ctx_len and int(model_ctx_len) > 0:
            return int(model_ctx_len)
        return LOCAL_FALLBACK

    if isinstance(overrides, dict):
        value = overrides.get(slug)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return max(value, MIN_LIMIT)

    return default_for(slug)


def get_override(api_router: Any, slug: str) -> int | None:
    if not isinstance(api_router, dict):
        return None
    overrides = api_router.get(KEY)
    if not isinstance(overrides, dict):
        return None
    value = overrides.get((slug or "").lower())
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def set_override(api_router: dict, slug: str, value: Any) -> None:
    """Store or clear one provider's override; 0/blank means 'use the default'."""
    slug = (slug or "").lower()
    overrides = api_router.get(KEY)
    if not isinstance(overrides, dict):
        overrides = {}
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    if number > 0:
        overrides[slug] = max(number, MIN_LIMIT)
    else:
        overrides.pop(slug, None)
    api_router[KEY] = overrides
