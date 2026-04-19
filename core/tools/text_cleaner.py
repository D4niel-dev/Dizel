"""
text_cleaner.py — Normalize and clean extracted text before sending to Dizel.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Full cleaning pipeline."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = collapse_whitespace(text)
    text = strip_blank_lines(text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Normalize to NFC form (composed characters)."""
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters except newlines and tabs."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or not unicodedata.category(ch).startswith("C")
    )


def collapse_whitespace(text: str) -> str:
    """Replace runs of spaces/tabs with a single space (preserving newlines)."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        cleaned.append(re.sub(r"[ \t]+", " ", line).strip())
    return "\n".join(cleaned)


def strip_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines down to 2."""
    return re.sub(r"\n{3,}", "\n\n", text)
