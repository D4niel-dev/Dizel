"""
core/orchestration/task.py — Task schema for agent orchestration.

Defines TaskPacket, TaskType, and TaskStatus used by the planner,
router, executor, and synthesizer.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    """Categories of work that can be routed to agents."""
    REASONING  = "reasoning"    # General Q&A, analysis, logic
    CODING     = "coding"       # Code generation, debugging
    FILE       = "file"         # File read/write/parse
    VISION     = "vision"       # Image analysis/generation
    VOICE      = "voice"        # Voice transcription
    SYNTHESIS  = "synthesis"    # Combining multi-agent results
    CONVERSATION = "conversation"  # Casual chat, social


class TaskStatus(str, Enum):
    """Lifecycle states of a task."""
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    HANDED_OFF = "handed_off"   # Delegated to another agent


@dataclass
class TaskPacket:
    """Single unit of work routed through the orchestration system."""

    type: TaskType
    input: Dict[str, Any]                          # structured input payload
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: Optional[str] = None                # for sub-tasks
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = "dizel"                   # default fallback
    output: Optional[Dict[str, Any]] = None         # structured result
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    step_count: int = 0
    max_steps: int = 10                             # safety limit
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    subtasks: List["TaskPacket"] = field(default_factory=list)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self.status = TaskStatus.RUNNING
        self.step_count += 1

    def complete(self, output: Dict[str, Any]):
        self.status = TaskStatus.COMPLETED
        self.output = output
        self.completed_at = datetime.utcnow()

    def fail(self, error: str):
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def handoff(self, new_agent: str):
        self.status = TaskStatus.HANDED_OFF
        self.assigned_agent = new_agent

    @property
    def is_done(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def exceeded_steps(self) -> bool:
        return self.step_count >= self.max_steps

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds()
        return None

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "type": self.type.value,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "input": self.input,
            "output": self.output,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "context": self.context,
            "error": self.error,
            "subtasks": [s.to_dict() for s in self.subtasks],
        }

    def __repr__(self) -> str:
        return (
            f"TaskPacket(id={self.id!r}, type={self.type.value}, "
            f"agent={self.assigned_agent!r}, status={self.status.value})"
        )
