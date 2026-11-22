from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class EditCommand(Command):
    name = "edit"
    help_text = "Edit the last prompt; sending replaces that turn and everything after it."
    category = "Chat"
    usage = "/edit"
    palette_hint = "/edit "
    examples = ["/edit"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if getattr(app, "generation_state", "IDLE") in ("STREAMING", "THINKING"):
            return "Still generating — press esc to stop first."
        bridge = getattr(app, "session_bridge", None)
        messages = list(getattr(bridge, "current_messages", []) or [])
        index = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if messages[i].get("role") == "user"
            ),
            None,
        )
        if index is None:
            return "Nothing to edit yet."
        message = messages[index]
        prefill = (
            (message.get("display") or {}).get("text")
            or message.get("content")
            or ""
        )

        from inference.tui.panels.multiline_input import MultilineInputModal

        def done(text: str | None) -> None:
            if text:
                app.edit_and_resend(index, text)

        app.push_screen(
            MultilineInputModal(
                initial_text=prefill,
                send_on_save=True,
                title="Edit message",
                subtitle="Change the prompt and send it again.",
                notice="Sending replaces this turn and every message after it.",
            ),
            done,
        )
        return ""
