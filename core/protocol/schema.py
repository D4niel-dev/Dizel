"""
core/protocol/schema.py — Tool message schema.

Defines ToolMessage and ToolStatus for structured
communication between agents and tools.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ToolStatus(str, Enum):
    """Lifecycle states of a tool invocation."""
    PENDING  = "pending"
    RUNNING  = "running"
    SUCCESS  = "success"
    ERROR    = "error"
    PARTIAL  = "partial"    # partial results available


@dataclass
class ToolMessage:
    """Structured message for tool invocation and response."""

    task_id: str                                   # links to TaskPacket
    source: str                                    # calling agent
    target: str                                    # tool name
    action: str                                    # "invoke" | "response" | "status" | "error"
    payload: Dict[str, Any]                        # input or output data
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ToolStatus = ToolStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_request(self) -> bool:
        return self.action == "invoke"

    @property
    def is_response(self) -> bool:
        return self.action == "response"

    @property
    def is_error(self) -> bool:
        return self.action == "error" or self.status == ToolStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "source": self.source,
            "target": self.target,
            "action": self.action,
            "payload": self.payload,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolMessage":
        data = dict(data)
        data["status"] = ToolStatus(data["status"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    @classmethod
    def invoke(cls, task_id: str, source: str, target: str,
               payload: Dict[str, Any], **kwargs) -> "ToolMessage":
        """Factory: create an invocation message."""
        return cls(
            task_id=task_id, source=source, target=target,
            action="invoke", payload=payload,
            status=ToolStatus.PENDING, **kwargs,
        )

    @classmethod
    def respond(cls, request: "ToolMessage", payload: Dict[str, Any],
                confidence: Optional[float] = None) -> "ToolMessage":
        """Factory: create a response to an invocation."""
        return cls(
            task_id=request.task_id, source=request.target,
            target=request.source, action="response",
            payload=payload, status=ToolStatus.SUCCESS,
            confidence=confidence,
            metadata={"request_id": request.id},
        )

    @classmethod
    def error_response(cls, request: "ToolMessage", error: str) -> "ToolMessage":
        """Factory: create an error response."""
        return cls(
            task_id=request.task_id, source=request.target,
            target=request.source, action="error",
            payload={}, status=ToolStatus.ERROR, error=error,
            metadata={"request_id": request.id},
        )

    def __repr__(self) -> str:
        return (
            f"ToolMessage({self.action} {self.source}→{self.target}, "
            f"status={self.status.value})"
        )
