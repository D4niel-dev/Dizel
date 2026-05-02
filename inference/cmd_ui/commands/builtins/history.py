from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class HistoryCommand(Command):
    name = "history"
    help_text = "List recent sessions. Usage: /history [search]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        query = " ".join(invocation.args) if invocation.args else None
        sessions = app.session_bridge.get_all(query)
        if not sessions:
            return "No sessions found."

        lines = ["Recent Sessions:", ""]
        for s in sessions[:10]:
            pin = " *" if s.get("pinned") else "  "
            active = ">" if s["id"] == app.session_id else " "
            title = s["title"][:30]
            lines.append(f"  {active}{pin} {s['id'][:8]} | {title}")
        return "\n".join(lines)
