from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class McpCommand(Command):
    name = "mcp"
    help_text = "Manage MCP servers and see the tools they expose."
    category = "Tools"
    usage = "/mcp"
    palette_hint = "/mcp"
    examples = ["/mcp"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from inference.tui.panels.modals import McpModal

        app.push_screen(McpModal())
        return ""
