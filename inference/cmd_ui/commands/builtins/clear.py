from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ClearCommand(Command):
    name = "clear"
    help_text = "Clear the current workspace."
    aliases = ["c"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        app.action_clear_workspace()
        return "Workspace cleared."
