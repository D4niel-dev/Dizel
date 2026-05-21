"""
evaluation/schema.py — Evaluation case, result, and scoring types.

Defines the data structures for benchmark evaluation,
scoring rubrics, and regression tracking.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalCategory(str, Enum):
    """Categories of evaluation benchmarks."""
    REASONING      = "reasoning"
    CODING         = "coding"
    CONVERSATION   = "conversation"
    MEMORY_RECALL  = "memory_recall"
    TOOL_SELECTION = "tool_selection"
    COLLABORATION  = "collaboration"
    FILE_UNDERSTANDING = "file_understanding"
    IMAGE_UNDERSTANDING = "image_understanding"
    VOICE          = "voice"
    FORMATTING     = "formatting"


class ScoringMethod(str, Enum):
    """How to score an evaluation case."""
    EXACT_MATCH  = "exact_match"
    CONTAINS     = "contains"
    SIMILARITY   = "similarity"     # fuzzy string similarity
    NUMERIC      = "numeric"        # within tolerance
    MANUAL       = "manual"         # human judgment
    CUSTOM       = "custom"         # custom scoring function


@dataclass
class EvalCase:
    """A single benchmark test case."""
    prompt: str
    expected: str                                  # expected output or pattern
    category: EvalCategory
    scoring: ScoringMethod = ScoringMethod.CONTAINS
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent: str = "dizel"                           # which agent to test
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0                            # importance weight
    tolerance: float = 0.8                         # similarity threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected": self.expected,
            "category": self.category.value,
            "scoring": self.scoring.value,
            "agent": self.agent,
            "tags": self.tags,
            "metadata": self.metadata,
            "weight": self.weight,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalCase":
        data = dict(data)
        data["category"] = EvalCategory(data["category"])
        data["scoring"] = ScoringMethod(data["scoring"])
        return cls(**data)


@dataclass
class EvalResult:
    """Result of running a single eval case."""
    case_id: str
    passed: bool
    score: float                                   # 0.0 – 1.0
    actual_output: str
    expected_output: str
    agent: str
    category: EvalCategory
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": self.score,
            "actual_output": self.actual_output[:500],  # truncate for storage
            "expected_output": self.expected_output[:500],
            "agent": self.agent,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class EvalRun:
    """A complete evaluation run across multiple cases."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    checkpoint: str = ""                           # model checkpoint name
    results: List[EvalResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total, 1)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def scores_by_category(self) -> Dict[str, float]:
        """Average score per category."""
        buckets: Dict[str, List[float]] = {}
        for r in self.results:
            buckets.setdefault(r.category.value, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "checkpoint": self.checkpoint,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "avg_score": round(self.avg_score, 4),
            "by_category": self.scores_by_category(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": [r.to_dict() for r in self.results],
        }
