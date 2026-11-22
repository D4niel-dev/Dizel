from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command

PLAN_PROMPT = (
    "Draft a short numbered plan for this task. Call the plan tool with a "
    "short task line and the numbered steps; do not execute anything yet. "
    "If tools are unavailable, reply with the numbered plan only.\n\nTask: {task}"
)


class PlanCommand(Command):
    name = "plan"
    help_text = "Draft a numbered plan and ask before executing it."
    category = "Chat"
    usage = "/plan <task>"
    palette_hint = "/plan "
    examples = ["/plan add a doctor command"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        task = " ".join(invocation.args).strip()
        if not task:
            return "Usage: /plan <task>"
        if getattr(app, "generation_state", "IDLE") in ("STREAMING", "THINKING"):
            return "Still generating — press esc to stop first."
        shown = task if len(task) <= 60 else task[:59] + "…"
        app.set_mode("Planning", announce=False)
        return app.send_composed(f"Plan · {shown}", PLAN_PROMPT.format(task=task))
