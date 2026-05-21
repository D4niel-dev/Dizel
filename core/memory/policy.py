"""
core/memory/policy.py — Memory write/read policies, summarization, and pruning.

Controls token budgets, automatic summarization of old memories,
and garbage collection of expired or low-confidence items.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .schema import MemoryItem, MemoryType, can_read, can_write
from .store import MemoryQuery, MemoryStore


@dataclass
class TokenBudget:
    """Per-memory-type token budget for context injection."""
    working: int = 2000
    episodic: int = 1000
    semantic: int = 800
    system: int = 400
    agent: int = 300

    def for_type(self, mem_type: MemoryType) -> int:
        return {
            MemoryType.WORKING:  self.working,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.SYSTEM:   self.system,
            MemoryType.AGENT:    self.agent,
        }.get(mem_type, 200)

    @property
    def total(self) -> int:
        return self.working + self.episodic + self.semantic + self.system + self.agent


class MemoryPolicy:
    """Enforces read/write permissions, token budgets, and lifecycle rules."""

    def __init__(self, budget: Optional[TokenBudget] = None):
        self.budget = budget or TokenBudget()

    # ── Permission checks ─────────────────────────────────────────────────

    def check_write(self, agent: str, item: MemoryItem) -> bool:
        """Return True if this agent is allowed to write this item."""
        return can_write(agent, item.type)

    def check_read(self, agent: str, mem_type: MemoryType) -> bool:
        """Return True if this agent is allowed to read this memory type."""
        return can_read(agent, mem_type)

    # ── Context assembly ──────────────────────────────────────────────────

    def assemble_context(self, agent: str,
                         stores: Dict[MemoryType, MemoryStore],
                         query_text: Optional[str] = None) -> str:
        """
        Build a context string from all accessible memory stores,
        respecting token budgets per type.
        """
        blocks: List[str] = []

        # Priority order: working → agent → semantic → episodic → system
        priority = [
            MemoryType.WORKING,
            MemoryType.AGENT,
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
            MemoryType.SYSTEM,
        ]

        for mem_type in priority:
            if not self.check_read(agent, mem_type):
                continue
            store = stores.get(mem_type)
            if not store:
                continue

            budget = self.budget.for_type(mem_type)
            q = MemoryQuery(text=query_text, limit=50)
            items = store.query(q)

            block_lines = []
            char_count = 0
            approx_chars = budget * 4  # rough chars-per-token estimate

            for item in items:
                line = item.content
                if char_count + len(line) > approx_chars:
                    break
                block_lines.append(line)
                char_count += len(line)

            if block_lines:
                header = f"[{mem_type.value.upper()} MEMORY]"
                blocks.append(header + "\n" + "\n".join(block_lines))

        return "\n\n".join(blocks)

    # ── Pruning ───────────────────────────────────────────────────────────

    def prune(self, store: MemoryStore, mem_type: MemoryType,
              max_items: int = 1000) -> int:
        """
        Remove expired items and trim excess items by confidence.
        Returns number of items removed.
        """
        removed = store.prune_expired()

        # If still over limit, remove lowest-confidence items
        if store.count() > max_items:
            all_items = store.query(MemoryQuery(include_expired=True, limit=store.count()))
            all_items.sort(key=lambda m: m.confidence)
            excess = store.count() - max_items
            for item in all_items[:excess]:
                store.delete(item.id)
                removed += 1

        return removed

    # ── Summarization ─────────────────────────────────────────────────────

    def summarize_items(self, items: List[MemoryItem], max_length: int = 500) -> str:
        """
        Create a compressed summary from a list of memory items.
        Simple extractive approach: take the first sentence of each item
        until we hit max_length.
        """
        if not items:
            return ""

        sentences = []
        char_count = 0

        for item in items:
            # Take first sentence
            content = item.content.strip()
            first_sentence = content.split(".")[0] + "."
            if char_count + len(first_sentence) > max_length:
                break
            sentences.append(f"- {first_sentence} [{item.source}]")
            char_count += len(first_sentence)

        return "\n".join(sentences)

    def should_summarize(self, store: MemoryStore, threshold: int = 200) -> bool:
        """Check if a store has grown large enough to warrant summarization."""
        return store.count() > threshold
