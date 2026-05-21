"""
core/orchestration/router.py — Route tasks to the correct agent.

Maps TaskType → agent name using the routing rules defined
in the ecosystem plan, with confidence thresholds.
"""

from typing import Dict, List, Optional, Tuple

from .task import TaskPacket, TaskType, TaskStatus


# Agent routing table: TaskType → [(agent, min_confidence)]
_ROUTING_TABLE: Dict[TaskType, List[Tuple[str, float]]] = {
    TaskType.CODING:       [("codelx", 0.7), ("dizel", 0.3)],
    TaskType.FILE:         [("lily", 0.8),   ("dizel", 0.3)],
    TaskType.VISION:       [("dict", 0.9),   ("dizel", 0.3)],
    TaskType.VOICE:        [("nova", 0.9),   ("dizel", 0.3)],
    TaskType.CONVERSATION: [("mila", 0.5),   ("dizel", 0.3)],
    TaskType.REASONING:    [("dizel", 0.5)],
    TaskType.SYNTHESIS:    [("dizel", 0.5)],
}

# Agents that are currently available (updated at runtime)
_AVAILABLE_AGENTS = {"dizel", "mila", "codelx", "nova", "dict", "lily"}


class Router:
    """Routes TaskPackets to the appropriate agent based on type and confidence."""

    def __init__(self):
        self._available: set = set(_AVAILABLE_AGENTS)
        self._visited: Dict[str, set] = {}  # task_id → set of visited agents

    def set_available(self, agents: set):
        """Update which agents are currently online."""
        self._available = agents

    def route(self, task: TaskPacket) -> str:
        """
        Determine the best agent for a task.
        Falls back to 'dizel' if no specialist is available.
        """
        confidence = task.context.get("confidence", 0.5)
        candidates = _ROUTING_TABLE.get(task.type, [("dizel", 0.3)])

        # Track visited agents for loop prevention
        visited = self._visited.get(task.id, set())

        for agent, min_conf in candidates:
            if agent not in self._available:
                continue
            if agent in visited:
                continue
            if confidence >= min_conf:
                task.assigned_agent = agent
                visited.add(agent)
                self._visited[task.id] = visited
                return agent

        # Fallback: always route to dizel
        task.assigned_agent = "dizel"
        return "dizel"

    def can_handoff(self, task: TaskPacket, target_agent: str) -> bool:
        """Check if a handoff to target_agent is safe (no loops)."""
        if target_agent not in self._available:
            return False
        visited = self._visited.get(task.id, set())
        if target_agent in visited:
            return False
        if task.exceeded_steps:
            return False
        return True

    def record_handoff(self, task: TaskPacket, from_agent: str, to_agent: str):
        """Record a handoff for loop prevention tracking."""
        visited = self._visited.setdefault(task.id, set())
        visited.add(from_agent)
        visited.add(to_agent)
        task.assigned_agent = to_agent
        task.step_count += 1

    def clear_task(self, task_id: str):
        """Clean up tracking data for a completed task."""
        self._visited.pop(task_id, None)

    def get_agent_for_type(self, task_type: TaskType) -> str:
        """Quick lookup: primary agent for a given task type."""
        candidates = _ROUTING_TABLE.get(task_type, [("dizel", 0.3)])
        for agent, _ in candidates:
            if agent in self._available:
                return agent
        return "dizel"
