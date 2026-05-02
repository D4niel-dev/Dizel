from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

VALID_MODES = ["Fast", "Planning", "Coding", "Review"]

class ModeCommand(Command):
    name = "mode"
    help_text = "Switch or list modes. Usage: /mode [name]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            lines = [f"Current mode: {app.active_mode}", "", "Available modes:"]
            for m in VALID_MODES:
                marker = " *" if m == app.active_mode else ""
                lines.append(f"  {m}{marker}")
            return "\n".join(lines)

        new_mode = invocation.args[0].capitalize()

        if new_mode not in VALID_MODES:
            return f"Unknown mode: {new_mode}\nAvailable: {', '.join(VALID_MODES)}"

        app.active_mode = new_mode

        # Apply profile with current model
        try:
            app.chat_bridge.manager.apply_profile(app.active_model, new_mode)
        except Exception:
            pass  # Profile may not exist for all combinations

        return f"Mode switched to: {new_mode}"
