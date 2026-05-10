from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation

class HistoryCommand(Command):
    name = "history"
    help_text = "List, search, or open recent sessions."
    category = "Session"
    usage = "/history [search|open <id>]"
    palette_hint = "/history "
    examples = ["/history", "/history open 20260502"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if invocation.args and invocation.args[0].lower() in ("open", "load"):
            if len(invocation.args) < 2:
                return "Usage: /history open <session-id-prefix>"
            return self._open_session(app, invocation.args[1])

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
        lines.append("")
        lines.append("Open: /history open <id-prefix>")
        return "\n".join(lines)

    def _open_session(self, app: Any, session_prefix: str) -> str:
        matches = [s for s in app.session_bridge.get_all() if s["id"].startswith(session_prefix)]
        if not matches:
            return f"No session found for id prefix: {session_prefix}"
        if len(matches) > 1:
            options = ", ".join(s["id"] for s in matches[:5])
            return f"Ambiguous session id prefix: {session_prefix}\nMatches: {options}"

        session = matches[0]
        app.load_session_to_workspace(session["id"])
        return f"Loaded session: {session['title']} ({session['id']})"
