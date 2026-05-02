from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation
from inference.cmd_ui.commands.registry import registry

class HelpCommand(Command):
    name = "help"
    help_text = "Show available commands."
    aliases = ["h"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if invocation.args:
            cmd_name = invocation.args[0]
            cmd = registry.lookup(cmd_name)
            if cmd:
                return f"/{cmd.name} - {cmd.help_text}"
            return f"Unknown command: {cmd_name}"
            
        lines = ["Available Commands:"]
        for cmd in sorted(registry.list_all(), key=lambda c: c.name):
            lines.append(f"  /{cmd.name.ljust(10)} - {cmd.help_text}")
        return "\n".join(lines)
