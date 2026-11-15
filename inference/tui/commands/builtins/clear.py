from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class ClearCommand(Command):
    name = "clear"
    help_text = "Clear the workspace view (the model keeps its context)."
    category = "Workspace"
    usage = "/clear"
    palette_hint = "/clear"
    aliases = ["c"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        app.action_clear_workspace()
        # Feedback belongs above the composer, not in the transcript it just
        # cleared: a mounted result would dock the composer back to the bottom.
        app._log_system(
            "Workspace cleared · model context kept · /reset starts fresh.", "ok"
        )
        return ""
