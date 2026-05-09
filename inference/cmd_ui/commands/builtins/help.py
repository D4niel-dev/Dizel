from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation
from inference.cmd_ui.commands.registry import registry

class HelpCommand(Command):
    name = "help"
    help_text = "Show available commands."
    category = "Core"
    usage = "/help [command]"
    palette_hint = "/help "
    aliases = ["h"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if invocation.args:
            cmd_name = invocation.args[0].lstrip("/")
            cmd = registry.lookup(cmd_name)
            if cmd:
                lines = [
                    f"/{cmd.name}",
                    f"  {cmd.help_text}",
                    f"  Usage: {cmd.usage or '/' + cmd.name}",
                ]
                if cmd.aliases:
                    lines.append(f"  Aliases: {', '.join('/' + alias for alias in cmd.aliases)}")
                if cmd.examples:
                    lines.append("")
                    lines.append("Examples:")
                    lines.extend(f"  {example}" for example in cmd.examples)
                return "\n".join(lines)
            return f"Unknown command: {cmd_name}"
            
        lines = ["Available Commands:"]
        for cmd in sorted(registry.list_all(), key=lambda c: c.name):
            lines.append(f"  /{cmd.name.ljust(10)} {cmd.help_text}")
        lines.append("")
        lines.append("Use /help <command> for usage, aliases, and examples.")
        return "\n".join(lines)
