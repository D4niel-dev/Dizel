"""
core/memory/schema.py — Memory type definitions and validation.

Defines MemoryItem, MemoryType, and the permission matrix
that controls which agents can read/write each memory type.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """Categories of memory in the Dizel ecosystem."""
    WORKING  = "working"    # Current session state, in-memory only
    EPISODIC = "episodic"   # Past interactions, disk-backed
    SEMANTIC = "semantic"   # Stable knowledge, indexed
    SYSTEM   = "system"     # Configuration state
    AGENT    = "agent"      # Inter-agent shared state


@dataclass
class MemoryItem:
    """Single unit of memory stored in the Dizel ecosystem."""

    type: MemoryType
    source: str                              # agent name or "user"
    content: str                             # the actual memory content
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0                  # 0.0 – 1.0
    relevance_score: float = 0.0             # computed at retrieval time
    expiry: Optional[datetime] = None        # None = permanent
    tags: List[str] = field(default_factory=list)

    # ── Validation ────────────────────────────────────────────────────────

    def __post_init__(self):
        if not self.content:
            raise ValueError("MemoryItem.content must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0–1.0, got {self.confidence}")
        if isinstance(self.type, str):
            self.type = MemoryType(self.type)

    # ── Helpers ───────────────────────────────────────────────────────────

    @property
    def is_expired(self) -> bool:
        if self.expiry is None:
            return False
        return datetime.utcnow() >= self.expiry

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for JSON / SQLite storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Deserialize from a plain dict."""
        data = dict(data)  # shallow copy
        data["type"] = MemoryType(data["type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("expiry"):
            data["expiry"] = datetime.fromisoformat(data["expiry"])
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"MemoryItem(id={self.id!r}, type={self.type.value}, "
            f"source={self.source!r}, len={len(self.content)}, "
            f"confidence={self.confidence:.2f})"
        )


# ── Agent Permission Matrix ──────────────────────────────────────────────────
# Defines which agents can read (R) and write (W) each memory type.
# "*" means all agents have access.

AGENT_PERMISSIONS: Dict[str, Dict[MemoryType, str]] = {
    "*": {
        MemoryType.WORKING:  "RW",
        MemoryType.SYSTEM:   "R",
    },
    "dizel": {
        MemoryType.WORKING:  "RW",
        MemoryType.EPISODIC: "RW",
        MemoryType.SEMANTIC: "RW",
        MemoryType.SYSTEM:   "RW",
        MemoryType.AGENT:    "RW",
    },
    "mila": {
        MemoryType.WORKING:  "RW",
        MemoryType.EPISODIC: "R",
        MemoryType.SEMANTIC: "R",
        MemoryType.SYSTEM:   "R",
        MemoryType.AGENT:    "RW",
    },
    "codelx": {
        MemoryType.WORKING:  "RW",
        MemoryType.EPISODIC: "R",
        MemoryType.SEMANTIC: "RW",
        MemoryType.SYSTEM:   "R",
        MemoryType.AGENT:    "RW",
    },
    "nova": {
        MemoryType.WORKING:  "RW",
        MemoryType.SYSTEM:   "R",
        MemoryType.AGENT:    "R",
    },
    "dict": {
        MemoryType.WORKING:  "RW",
        MemoryType.SYSTEM:   "R",
        MemoryType.AGENT:    "R",
    },
    "lily": {
        MemoryType.WORKING:  "RW",
        MemoryType.SEMANTIC: "W",
        MemoryType.SYSTEM:   "R",
        MemoryType.AGENT:    "R",
    },
}


def can_read(agent: str, mem_type: MemoryType) -> bool:
    """Check if an agent has read access to a memory type."""
    perms = AGENT_PERMISSIONS.get(agent, AGENT_PERMISSIONS.get("*", {}))
    return "R" in perms.get(mem_type, "")


def can_write(agent: str, mem_type: MemoryType) -> bool:
    """Check if an agent has write access to a memory type."""
    perms = AGENT_PERMISSIONS.get(agent, AGENT_PERMISSIONS.get("*", {}))
    return "W" in perms.get(mem_type, "")
