from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation
from inference.tui.constants import MODES


class ModeCommand(Command):
    name = "mode"
    help_text = "Pick the response mode (Fast, Planning, Coding, Thinking, Deep Think)."
    category = "Runtime"
    usage = "/mode [fast|planning|coding|thinking|deep think]"
    palette_hint = "/mode "
    examples = ["/mode", "/mode coding", "/mode deep think"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            from inference.tui.panels.modals import ModePickerModal

            app.push_screen(
                ModePickerModal(),
                lambda mode: app.set_mode(mode) if mode else None,
            )
            return ""

        requested = " ".join(invocation.args).strip().lower().replace("-", " ")
        modes = {mode.lower(): mode for mode in MODES}
        modes.update({mode.lower().replace(" ", ""): mode for mode in MODES})
        new_mode = modes.get(requested.replace(" ", "")) or modes.get(requested)
        if not new_mode:
            return f"Unknown mode: {requested}\nAvailable: {', '.join(MODES)}"

        app.set_mode(new_mode)
        return ""
