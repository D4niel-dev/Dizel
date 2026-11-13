from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import registry


class HelpCommand(Command):
    name = "help"
    help_text = "Browse commands and global keys."
    category = "Core"
    usage = "/help [command]"
    palette_hint = "/help "
    aliases = ["h"]
    examples = ["/help", "/help settings"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            from inference.tui.panels.modals import HelpBrowserModal

            app.push_screen(
                HelpBrowserModal(),
                lambda text: app.insert_prompt(text) if text else None,
            )
            return ""

        cmd_name = invocation.args[0].lstrip("/")
        cmd = registry.lookup(cmd_name)
        if cmd:
            lines = [
                f"### `/{cmd.name}`",
                f"{cmd.help_text}",
                f"**Usage:** `{cmd.usage or '/' + cmd.name}`",
            ]
            if cmd.aliases:
                lines.append(
                    f"**Aliases:** {', '.join('`/' + alias + '`' for alias in cmd.aliases)}"
                )
            if cmd.examples:
                lines.append("\n**Examples:**")
                lines.extend(f"- `{example}`" for example in cmd.examples)
            return "\n".join(lines)
        return f"Unknown command: `{cmd_name}`"
