from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation

class ContextCommand(Command):
    name = "context"
    help_text = "Show token usage and context status."
    category = "Workspace"
    usage = "/context"
    palette_hint = "/context"
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        max_cap = int(app.usage_manager.max_capacity) if hasattr(app, "usage_manager") else 4096
        return f"Context Tokens: {app.context_tokens} / {max_cap}\nBudget: {app.budget_tokens} tokens allocated."
