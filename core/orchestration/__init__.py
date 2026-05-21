"""
core/orchestration — Dizel Agent Orchestration System.

Provides task planning, routing, execution, and synthesis
for coordinating work across multiple specialized agents.

Usage:
    from core.orchestration import Orchestrator

    orch = Orchestrator()
    orch.register_agent("dizel", my_dizel_handler)
    result = orch.process("Write a Python function to sort a list")
"""

from .task import TaskPacket, TaskType, TaskStatus
from .planner import Planner
from .router import Router
from .executor import Executor, AgentHandler
from .synthesizer import Synthesizer

from typing import Any, Callable, Dict, Optional


class Orchestrator:
    """
    Top-level facade for the orchestration system.

    Wires together the Planner → Router → Executor → Synthesizer pipeline.
    """

    def __init__(self):
        self.planner = Planner()
        self.router = Router()
        self.executor = Executor(router=self.router)
        self.synthesizer = Synthesizer()

    def register_agent(self, name: str, handler: AgentHandler):
        """Register an agent handler for task execution."""
        self.executor.register_agent(name, handler)

    def process(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Full pipeline: classify → decompose → route → execute → synthesize.
        Returns the final output dict.
        """
        # 1. Classify and decompose
        tasks = self.planner.decompose(user_input)

        # Inject external context
        if context:
            for task in tasks:
                task.context.update(context)

        # 2. Route each task
        for task in tasks:
            self.router.route(task)

        # 3. Execute
        if len(tasks) == 1:
            result_task = self.executor.execute(tasks[0])
            return result_task.output or {"error": result_task.error}

        # Multi-task: execute all, then synthesize
        for task in tasks:
            self.executor.execute(task)

        # 4. Synthesize
        parent = TaskPacket(
            type=TaskType.SYNTHESIS,
            input={"user_message": user_input},
            subtasks=tasks,
            context={"subtask_outputs": [t.output for t in tasks]},
        )
        return self.synthesizer.synthesize(parent)

    def classify(self, user_input: str):
        """Expose intent classification without execution."""
        return self.planner.classify(user_input)

    @property
    def agents(self) -> list:
        return self.executor.registered_agents

    @property
    def history(self) -> list:
        return self.executor.history


__all__ = [
    "Orchestrator",
    "TaskPacket",
    "TaskType",
    "TaskStatus",
    "Planner",
    "Router",
    "Executor",
    "Synthesizer",
    "AgentHandler",
]
