"""Conservative list-price estimates for hosted providers.

These are approximations from public list prices in USD per 1M tokens, not
billing. The usage window also counts policy tokens at a fixed rate, so every
figure derived from them is labelled as an estimate.
"""

from __future__ import annotations

from typing import Optional

# (input, output) USD per 1M tokens.
PRICES = {
    "openai": (2.50, 10.00),
    "anthropic": (3.00, 15.00),
    "google": (1.25, 5.00),
    "groq": (0.59, 0.79),
    "mistral": (0.40, 2.00),
    "xai": (2.00, 10.00),
    "cohere": (2.50, 10.00),
    "azure": (2.50, 10.00),
}

FREE = {"local", "ollama", "custom"}


def _slug(provider: str) -> str:
    return str(provider or "").strip().lower()


def estimate_cost(provider: str, tokens: float) -> Optional[float]:
    """Estimated blended cost, 0.0 for free/local, None when no price is known."""
    slug = _slug(provider)
    if slug in FREE:
        return 0.0
    pair = PRICES.get(slug)
    if pair is None:
        return None
    blended = (pair[0] + pair[1]) / 2
    return (max(0.0, float(tokens)) / 1_000_000.0) * blended


def rate_label(provider: str) -> str:
    slug = _slug(provider)
    if slug in FREE:
        return "free"
    pair = PRICES.get(slug)
    if pair is None:
        return "no list price"
    blended = (pair[0] + pair[1]) / 2
    return f"~${blended:.2f} / 1M blended (estimate)"
