"""
core/orchestration/synthesizer.py — Combine multi-agent results.

Merges outputs from multiple subtasks into a coherent
final response for the user.
"""

from typing import Any, Dict, List, Optional

from .task import TaskPacket, TaskStatus


class Synthesizer:
    """Combines results from multiple agents into a unified output."""

    def synthesize(self, task: TaskPacket) -> Dict[str, Any]:
        """
        Merge subtask outputs into a single coherent result.
        The parent task's context should contain subtask_outputs.
        """
        sub_outputs = task.context.get("subtask_outputs", [])

        if not sub_outputs:
            # Single-task result, just pass through
            return task.output or {}

        merged = {
            "source": "synthesizer",
            "parts": [],
            "errors": [],
            "summary": "",
        }

        for i, output in enumerate(sub_outputs):
            if not output:
                continue
            if "error" in output:
                merged["errors"].append(output["error"])
            else:
                merged["parts"].append(output)

        # Build a combined response text
        text_parts = []
        for part in merged["parts"]:
            if "response" in part:
                text_parts.append(part["response"])
            elif "text" in part:
                text_parts.append(part["text"])
            elif "result" in part:
                text_parts.append(str(part["result"]))

        merged["summary"] = "\n\n".join(text_parts)

        # Include errors as warnings
        if merged["errors"]:
            error_block = "\n".join(f"⚠ {e}" for e in merged["errors"])
            merged["summary"] += f"\n\n{error_block}"

        return merged

    def merge_contexts(self, tasks: List[TaskPacket]) -> Dict[str, Any]:
        """
        Merge context dicts from multiple completed tasks.
        Useful for building shared state across agents.
        """
        merged = {}
        for task in tasks:
            if task.status != TaskStatus.COMPLETED:
                continue
            if task.output:
                merged[task.assigned_agent] = task.output
        return merged

    def pick_best(self, tasks: List[TaskPacket]) -> Optional[TaskPacket]:
        """
        From multiple completed tasks, pick the one with the
        highest confidence or most complete output.
        """
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        if not completed:
            return None

        def score(t: TaskPacket) -> float:
            conf = t.context.get("confidence", 0.5)
            output_size = len(str(t.output)) if t.output else 0
            return conf + (output_size / 10000)

        return max(completed, key=score)
