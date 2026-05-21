"""
core/protocol/logger.py — Logging, replay, and audit trail.

Persists tool invocation history for debugging,
performance analysis, and replay capabilities.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schema import ToolMessage


_DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "logs", "protocol"
)


class ProtocolLogger:
    """Persistent audit logger for tool invocations."""

    def __init__(self, log_dir: Optional[str] = None, max_entries: int = 5000):
        self._log_dir = log_dir or os.path.normpath(_DEFAULT_LOG_DIR)
        os.makedirs(self._log_dir, exist_ok=True)
        self._max_entries = max_entries
        self._buffer: List[Dict[str, Any]] = []

    def log_invocation(self, request: ToolMessage, response: ToolMessage):
        """Record a request-response pair."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": request.target,
            "agent": request.source,
            "task_id": request.task_id,
            "action": request.action,
            "status": response.status.value,
            "duration_ms": self._calc_duration(request, response),
            "error": response.error,
            "request_id": request.id,
            "response_id": response.id,
        }
        self._buffer.append(entry)

        if len(self._buffer) >= 100:
            self.flush()

    def log_message(self, msg: ToolMessage):
        """Record a single message (for status updates, etc)."""
        self._buffer.append({
            "timestamp": datetime.utcnow().isoformat(),
            "tool": msg.target,
            "agent": msg.source,
            "task_id": msg.task_id,
            "action": msg.action,
            "status": msg.status.value,
            "message_id": msg.id,
        })

    def flush(self):
        """Write buffer to disk."""
        if not self._buffer:
            return

        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = os.path.join(self._log_dir, f"protocol_{date_str}.jsonl")

        with open(log_file, "a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._buffer.clear()

    def get_recent(self, limit: int = 50) -> List[Dict]:
        """Get recent log entries (from buffer + latest file)."""
        entries = list(self._buffer)

        # Also read from latest log file
        log_files = sorted(
            [f for f in os.listdir(self._log_dir) if f.endswith(".jsonl")],
            reverse=True,
        )
        if log_files:
            path = os.path.join(self._log_dir, log_files[0])
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                pass

        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:limit]

    def get_by_task(self, task_id: str) -> List[Dict]:
        """Get all log entries for a specific task."""
        all_entries = self.get_recent(limit=self._max_entries)
        return [e for e in all_entries if e.get("task_id") == task_id]

    def get_error_summary(self, limit: int = 20) -> List[Dict]:
        """Get recent errors for debugging."""
        all_entries = self.get_recent(limit=500)
        errors = [e for e in all_entries if e.get("error")]
        return errors[:limit]

    @staticmethod
    def _calc_duration(request: ToolMessage, response: ToolMessage) -> Optional[float]:
        delta = response.timestamp - request.timestamp
        return round(delta.total_seconds() * 1000, 2)

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass
