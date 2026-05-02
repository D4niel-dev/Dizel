from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ContextCommand(Command):
    name = "context"
    help_text = "Show token usage and context status."
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        return f"Context Tokens: {app.context_tokens} / 4096\nBudget: {app.budget_tokens} tokens allocated."
