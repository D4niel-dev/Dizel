from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command

RUN_HINT = (
    "Run the suite from the desktop (/scorecard run) or "
    "scripts/run_scorecard.py <checkpoint>."
)


class ScorecardCommand(Command):
    name = "scorecard"
    help_text = "Show the loaded checkpoint's scorecard; evaluations run from the desktop or CLI."
    category = "Runtime"
    usage = "/scorecard [run]"
    palette_hint = "/scorecard "
    examples = ["/scorecard", "/scorecard run"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        action = invocation.args[0].lower() if invocation.args else ""
        if action == "run":
            return RUN_HINT

        manager = getattr(getattr(app, "chat_bridge", None), "manager", None)
        path = str(getattr(manager, "checkpoint_path", "") or "") if manager else ""
        if not path:
            return "Scorecards evaluate local checkpoints — load one first."

        from dizel_core import scorecards

        card = scorecards.load(path)
        if card is None:
            return f"No scorecard for {path} yet. {RUN_HINT}"

        app.show_debug_output("Scorecard", scorecards.report(card))
        return ""
