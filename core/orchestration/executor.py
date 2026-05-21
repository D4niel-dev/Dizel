"""
core/orchestration/executor.py — Execute tasks with step limits.

Runs TaskPackets through their assigned agents, enforcing
max-step safety limits and handling errors gracefully.
"""

import traceback
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .task import TaskPacket, TaskStatus, TaskType
from .router import Router


# Agent handler type: takes a TaskPacket, returns output dict
AgentHandler = Callable[[TaskPacket], Dict[str, Any]]


class Executor:
    """
    Executes TaskPackets by dispatching to registered agent handlers.
    Enforces step limits and provides error isolation.
    """

    def __init__(self, router: Optional[Router] = None):
        self._handlers: Dict[str, AgentHandler] = {}
        self._router = router or Router()
        self._history: list = []

    def register_agent(self, name: str, handler: AgentHandler):
        """Register a callable handler for an agent name."""
        self._handlers[name] = handler
        self._router.set_available(set(self._handlers.keys()) | {"dizel"})

    def unregister_agent(self, name: str):
        """Remove an agent handler."""
        self._handlers.pop(name, None)
        self._router.set_available(set(self._handlers.keys()) | {"dizel"})

    def execute(self, task: TaskPacket) -> TaskPacket:
        """
        Execute a single task. Routes to the assigned agent,
        runs the handler, and updates the task status.
        """
        # Safety: step limit
        if task.exceeded_steps:
            task.fail(f"Exceeded max steps ({task.max_steps})")
            self._record(task)
            return task

        # Route if not yet assigned
        if task.status == TaskStatus.PENDING:
            self._router.route(task)

        agent = task.assigned_agent
        handler = self._handlers.get(agent)

        if not handler:
            # No handler registered — try fallback to dizel
            if agent != "dizel" and "dizel" in self._handlers:
                task.assigned_agent = "dizel"
                handler = self._handlers["dizel"]
            else:
                task.fail(f"No handler registered for agent '{agent}'")
                self._record(task)
                return task

        # Execute
        task.start()
        try:
            output = handler(task)
            task.complete(output or {})
        except Exception as e:
            task.fail(f"{type(e).__name__}: {e}")

        self._record(task)
        self._router.clear_task(task.id)
        return task

    def execute_with_subtasks(self, task: TaskPacket) -> TaskPacket:
        """
        Execute a task and all its subtasks sequentially.
        Parent task aggregates subtask outputs.
        """
        # Execute subtasks first
        for sub in task.subtasks:
            self.execute(sub)

        # Collect subtask outputs into parent context
        sub_outputs = []
        for sub in task.subtasks:
            if sub.status == TaskStatus.COMPLETED:
                sub_outputs.append(sub.output)
            elif sub.status == TaskStatus.FAILED:
                sub_outputs.append({"error": sub.error})

        if sub_outputs:
            task.context["subtask_outputs"] = sub_outputs

        # Execute the parent task
        return self.execute(task)

    def handoff(self, task: TaskPacket, from_agent: str, to_agent: str) -> TaskPacket:
        """
        Hand off a running task from one agent to another.
        Respects loop prevention in the router.
        """
        if not self._router.can_handoff(task, to_agent):
            task.fail(f"Cannot hand off to '{to_agent}' (loop or unavailable)")
            return task

        self._router.record_handoff(task, from_agent, to_agent)
        task.status = TaskStatus.PENDING  # reset for re-execution
        return self.execute(task)

    @property
    def registered_agents(self) -> list:
        return list(self._handlers.keys())

    @property
    def history(self) -> list:
        return list(self._history)

    def _record(self, task: TaskPacket):
        self._history.append({
            "id": task.id,
            "type": task.type.value,
            "agent": task.assigned_agent,
            "status": task.status.value,
            "steps": task.step_count,
            "duration": task.duration_seconds,
            "error": task.error,
        })
        # Keep history bounded
        if len(self._history) > 500:
            self._history = self._history[-250:]
