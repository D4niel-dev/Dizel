from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ResetCommand(Command):
    name = "reset"
    help_text = "Clear messages in current session."
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        app.session_bridge.current_messages = []
        app.session_bridge.save()
        app.query_one("WorkspacePanel").clear_workspace()
        return "Session reset."
