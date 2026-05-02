from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

VALID_MODELS = ["Dizel Lite", "Dizel Pro", "Mila Lite", "Mila Pro"]

class ModelCommand(Command):
    name = "model"
    help_text = "Switch or list models. Usage: /model [name]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            lines = [f"Current model: {app.active_model}", "", "Available models:"]
            for m in VALID_MODELS:
                marker = " *" if m == app.active_model else ""
                lines.append(f"  {m}{marker}")
            return "\n".join(lines)

        new_model = " ".join(invocation.args)

        # Case-insensitive match
        matched = None
        for m in VALID_MODELS:
            if m.lower() == new_model.lower():
                matched = m
                break

        if not matched:
            return f"Unknown model: {new_model}\nAvailable: {', '.join(VALID_MODELS)}"

        app.active_model = matched

        # Apply profile with current mode
        try:
            app.chat_bridge.manager.apply_profile(matched, app.active_mode)
        except Exception:
            pass  # Profile may not exist for all combinations

        return f"Model switched to: {matched}"
