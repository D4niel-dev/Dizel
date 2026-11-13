from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


def _reset(app: Any) -> str:
    app.chat_bridge.stop()
    app.session_bridge.create()
    try:
        app.chat_bridge.manager.new_session()
    except Exception:
        pass

    app.context_tokens = 0
    app.budget_tokens = 0
    app.query_one("WorkspacePanel").clear_workspace()
    app.refresh_sessions()
    return "New session. The model's context is empty."


class ResetCommand(Command):
    name = "reset"
    help_text = "Start a new session (clears the model's context)."
    category = "Session"
    usage = "/reset [force]"
    palette_hint = "/reset"
    examples = ["/reset", "/reset force"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if invocation.args and invocation.args[0].lower() in ("force", "yes", "-y"):
            return _reset(app)

        from inference.tui.panels.modals import ConfirmModal

        def _confirmed(confirmed: bool | None) -> None:
            if confirmed:
                app._log_system(_reset(app))

        app.push_screen(
            ConfirmModal(
                "Start a new session",
                "The transcript and the model's context will be cleared. "
                "Saved sessions stay on disk.",
                confirm_label="Start new",
                danger=False,
            ),
            _confirmed,
        )
        return ""
