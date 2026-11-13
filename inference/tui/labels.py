"""Display helpers shared by the composer and the welcome screen."""

from __future__ import annotations


def display_model(name: str, profile: str = "") -> str:
    """Drop a trailing profile word so "Dizel 1.6 Flash · Flash" does not repeat."""
    text = " ".join(str(name or "").split())
    profile = " ".join(str(profile or "").split())
    if profile and text.casefold().endswith(profile.casefold()):
        stem = text[: len(text) - len(profile)].rstrip(" ·-—")
        if stem:
            return stem
    return text
