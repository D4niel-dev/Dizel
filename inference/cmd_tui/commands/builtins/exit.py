from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation

class ExitCommand(Command):
    name = "exit"
    help_text = "Exit the application."
    category = "Core"
    usage = "/exit"
    palette_hint = "/exit"
    aliases = ["quit", "q"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str | None:
        app.exit()
        return None
