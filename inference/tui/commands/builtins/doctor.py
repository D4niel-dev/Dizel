from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class DoctorCommand(Command):
    name = "doctor"
    help_text = "Run offline runtime checks: packages, config, history, provider."
    category = "Runtime"
    usage = "/doctor"
    palette_hint = "/doctor "
    examples = ["/doctor"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import doctor

        app.show_debug_output("Runtime doctor", doctor.summary_text())
        return ""
