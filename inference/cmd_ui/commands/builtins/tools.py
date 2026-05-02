from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

AVAILABLE_TOOLS = {
    "web_search": "Search the web for relevant information",
    "file_parse": "Parse and extract context from attached files",
    "deep_think": "Extended reasoning with higher token budget",
}

class ToolsCommand(Command):
    name = "tools"
    help_text = "List or toggle tools. Usage: /tools [name] [on|off]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        # Ensure the app has a tools state dict
        if not hasattr(app, "_tool_states"):
            app._tool_states = {name: True for name in AVAILABLE_TOOLS}

        if not invocation.args:
            lines = ["Available Tools:", ""]
            for name, desc in AVAILABLE_TOOLS.items():
                status = "ON" if app._tool_states.get(name, True) else "OFF"
                lines.append(f"  [{status}] {name} - {desc}")
            lines.append("")
            lines.append("Toggle: /tools <name> on|off")
            return "\n".join(lines)

        tool_name = invocation.args[0].lower()
        if tool_name not in AVAILABLE_TOOLS:
            return f"Unknown tool: {tool_name}\nAvailable: {', '.join(AVAILABLE_TOOLS.keys())}"

        if len(invocation.args) < 2:
            status = "ON" if app._tool_states.get(tool_name, True) else "OFF"
            return f"{tool_name}: {status} - {AVAILABLE_TOOLS[tool_name]}"

        action = invocation.args[1].lower()
        if action in ("on", "enable", "true", "1"):
            app._tool_states[tool_name] = True
            return f"{tool_name}: enabled"
        elif action in ("off", "disable", "false", "0"):
            app._tool_states[tool_name] = False
            return f"{tool_name}: disabled"
        else:
            return f"Invalid action: {action}. Use 'on' or 'off'."
