"""
core/orchestration/planner.py — Intent classification and task decomposition.

Analyzes user requests, classifies intent, and decomposes
complex requests into sub-tasks for routing.
"""

import re
from typing import List, Optional, Tuple

from .task import TaskPacket, TaskType


# Keyword-based intent signals with confidence weights
_INTENT_SIGNALS = {
    TaskType.CODING: {
        "keywords": [
            "code", "function", "class", "debug", "fix", "error", "bug",
            "implement", "refactor", "test", "compile", "syntax", "script",
            "api", "endpoint", "database", "query", "sql", "html", "css",
            "javascript", "python", "react", "import", "variable", "loop",
        ],
        "patterns": [
            r"```",                          # code blocks
            r"def |class |import |from ",    # python keywords
            r"function |const |let |var ",   # js keywords
            r"\.(py|js|ts|jsx|tsx|go|rs)\b",  # file extensions
        ],
        "base_confidence": 0.7,
    },
    TaskType.FILE: {
        "keywords": [
            "file", "read", "write", "upload", "download", "parse",
            "csv", "json", "pdf", "document", "spreadsheet", "extract",
            "open", "save", "folder", "directory", "path",
        ],
        "patterns": [
            r"\.(txt|csv|json|pdf|docx|xlsx)\b",
            r"[A-Z]:\\",                     # windows paths
            r"/home/|/usr/|~/",              # unix paths
        ],
        "base_confidence": 0.8,
    },
    TaskType.VISION: {
        "keywords": [
            "image", "picture", "photo", "screenshot", "visual",
            "diagram", "chart", "graph", "draw", "illustration",
            "look at", "what do you see", "describe this image",
        ],
        "patterns": [
            r"\.(png|jpg|jpeg|gif|webp|svg|bmp)\b",
        ],
        "base_confidence": 0.9,
    },
    TaskType.VOICE: {
        "keywords": [
            "voice", "speak", "listen", "transcribe", "audio",
            "recording", "microphone", "speech",
        ],
        "patterns": [
            r"\.(wav|mp3|ogg|m4a|flac)\b",
        ],
        "base_confidence": 0.9,
    },
    TaskType.CONVERSATION: {
        "keywords": [
            "hello", "hi", "hey", "thanks", "thank you", "bye",
            "how are you", "what's up", "tell me about yourself",
            "joke", "story", "chat", "talk",
        ],
        "patterns": [],
        "base_confidence": 0.5,
    },
}


class Planner:
    """Classifies intent and decomposes requests into TaskPackets."""

    def classify(self, user_input: str) -> Tuple[TaskType, float]:
        """
        Classify the primary intent of a user message.
        Returns (TaskType, confidence).
        """
        lower = user_input.lower()
        scores = {}

        for task_type, signals in _INTENT_SIGNALS.items():
            score = 0.0

            # Keyword matching
            keyword_hits = sum(1 for kw in signals["keywords"] if kw in lower)
            if keyword_hits > 0:
                score += min(keyword_hits * 0.15, 0.6)

            # Pattern matching
            for pattern in signals["patterns"]:
                if re.search(pattern, user_input, re.IGNORECASE):
                    score += 0.3
                    break

            if score > 0:
                score = min(score + signals["base_confidence"] * 0.3, 1.0)

            scores[task_type] = score

        if not scores or max(scores.values()) == 0:
            return TaskType.REASONING, 0.6

        best_type = max(scores, key=scores.get)
        return best_type, round(scores[best_type], 2)

    def decompose(self, user_input: str,
                  task_type: Optional[TaskType] = None) -> List[TaskPacket]:
        """
        Decompose a user request into one or more TaskPackets.
        Simple requests → single task. Complex → multiple sub-tasks.
        """
        if task_type is None:
            task_type, confidence = self.classify(user_input)
        else:
            confidence = 0.8

        # For now, single-task decomposition.
        # Multi-step decomposition will be enhanced once agents mature.
        primary = TaskPacket(
            type=task_type,
            input={"user_message": user_input},
            context={"confidence": confidence},
        )

        return [primary]

    def is_multi_step(self, user_input: str) -> bool:
        """Heuristic: does this request likely need multiple agents?"""
        lower = user_input.lower()
        multi_signals = [
            "and then", "after that", "also", "first", "next",
            "step 1", "step 2", "finally",
        ]
        return sum(1 for s in multi_signals if s in lower) >= 2
