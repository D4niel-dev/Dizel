from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class CapabilitiesCommand(Command):
    name = "capabilities"
    help_text = "What each surface can do, and what is missing where."
    category = "Runtime"
    usage = "/capabilities [gaps]"
    palette_hint = "/capabilities "
    examples = ["/capabilities", "/capabilities gaps"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import capabilities

        if invocation.args and invocation.args[0].lower() == "gaps":
            return capabilities.report()
        return capabilities.summary()
