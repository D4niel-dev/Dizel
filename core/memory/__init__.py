"""
core/memory — Dizel Memory Architecture.

Provides unified memory access across working, episodic, and semantic stores.

Usage:
    from core.memory import MemorySystem

    mem = MemorySystem()
    mem.working.add_message("user", "Hello!")
    mem.episodic.store_session_summary("s1", "User discussed X")
    mem.semantic.store_knowledge("Python uses indentation", ["python"])

    context = mem.get_context("dizel", query="python syntax")
"""

from .schema import MemoryItem, MemoryType, can_read, can_write, AGENT_PERMISSIONS
from .store import MemoryQuery, MemoryStore
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .policy import MemoryPolicy, TokenBudget

from typing import Dict, Optional


class MemorySystem:
    """
    Unified facade for all memory subsystems.

    Provides a single entry point for agents to read/write memory
    with automatic permission enforcement and context assembly.
    """

    def __init__(self, data_dir: Optional[str] = None):
        import os
        if data_dir:
            ep_path = os.path.join(data_dir, "episodic.db")
            sem_path = os.path.join(data_dir, "semantic.json")
        else:
            ep_path = None
            sem_path = None

        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path=ep_path)
        self.semantic = SemanticMemory(file_path=sem_path)
        self.policy = MemoryPolicy()

        self._stores: Dict[MemoryType, MemoryStore] = {
            MemoryType.WORKING:  self.working,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
        }

    def store(self, agent: str, item: MemoryItem) -> Optional[str]:
        """Write a memory item if the agent has permission."""
        if not self.policy.check_write(agent, item):
            return None
        target = self._stores.get(item.type)
        if not target:
            return None
        return target.write(item)

    def recall(self, agent: str, query: str, limit: int = 10) -> list:
        """Query all accessible stores and return merged results."""
        results = []
        for mem_type, store in self._stores.items():
            if not self.policy.check_read(agent, mem_type):
                continue
            items = store.query(MemoryQuery(text=query, limit=limit))
            results.extend(items)
        # Sort by relevance then recency
        results.sort(
            key=lambda m: (m.relevance_score, m.timestamp.timestamp()),
            reverse=True
        )
        return results[:limit]

    def get_context(self, agent: str, query: Optional[str] = None) -> str:
        """Assemble a context string for prompt injection."""
        return self.policy.assemble_context(agent, self._stores, query)

    def prune_all(self) -> int:
        """Run pruning across all stores."""
        total = 0
        for mem_type, store in self._stores.items():
            total += self.policy.prune(store, mem_type)
        return total

    def new_session(self):
        """Clear working memory for a fresh session."""
        self.working.clear()

    def end_session(self, session_id: str, summary: str):
        """Archive the current session into episodic memory."""
        self.episodic.store_session_summary(session_id, summary)
        self.working.clear()


__all__ = [
    "MemorySystem",
    "MemoryItem",
    "MemoryType",
    "MemoryQuery",
    "MemoryStore",
    "MemoryPolicy",
    "TokenBudget",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "can_read",
    "can_write",
    "AGENT_PERMISSIONS",
]
