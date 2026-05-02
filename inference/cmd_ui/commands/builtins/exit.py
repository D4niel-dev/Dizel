from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ExitCommand(Command):
    name = "exit"
    help_text = "Exit the application."
    aliases = ["quit", "q"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str | None:
        app.exit()
        return None
