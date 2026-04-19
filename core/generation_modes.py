"""
core/generation_modes.py — Generation parameter overrides for tool modes.

Deep Think modifies generation behavior without being an external tool:
  - Higher max_new_tokens for thorough responses
  - Lower temperature for more focused reasoning
  - Narrower top_k for reduced randomness
  - Chain-of-thought system prompt addendum
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationOverrides:
    """Parameter overrides applied before generation."""
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    system_addendum: str = ""


def get_deep_think_overrides() -> GenerationOverrides:
    """
    Deep Think mode: thorough, structured, step-by-step responses.

    Increases token budget to 512 (vs default 200), lowers temperature
    for more focused output, and adds a chain-of-thought instruction.
    """
    return GenerationOverrides(
        max_new_tokens=512,
        temperature=0.5,
        top_k=30,
        top_p=0.88,
        system_addendum=(
            "\n\nIMPORTANT: Think step by step. "
            "Break down complex problems into parts. "
            "Be thorough, precise, and structured in your answer."
        ),
    )
