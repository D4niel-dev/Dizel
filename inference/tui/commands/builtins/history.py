from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class HistoryCommand(Command):
    name = "history"
    help_text = "List, search, or open recent sessions."
    category = "Session"
    usage = "/history [search <text> | open <id>]"
    palette_hint = "/history "
    examples = ["/history", "/history search rust", "/history open 20260502"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        args = invocation.args

        if args and args[0].lower() in ("open", "load"):
            if len(args) < 2:
                return "Usage: /history open <session-id-prefix>"
            return self._open_session(app, args[1])

        if args and args[0].lower() == "search":
            args = args[1:]

        query = " ".join(args) if args else None
        sessions = app.session_bridge.get_all(query)
        if not sessions:
            return f"No sessions matching “{query}”." if query else "No sessions yet."

        header = f"Sessions matching “{query}”:" if query else "Recent sessions:"
        lines = [header, ""]
        for session in sessions[:12]:
            pin = "*" if session.get("pinned") else " "
            active = "▸" if session["id"] == app.session_id else " "
            title = session["title"][:32]
            tags = " ".join(f"#{t}" for t in (session.get("tags") or [])[:2])
            suffix = f"  {tags}" if tags else ""
            lines.append(f"  {active}{pin} {session['id'][:8]}  {title}{suffix}")
        lines.append("")
        lines.append("Open with: /history open <id-prefix>")
        return "\n".join(lines)

    def _open_session(self, app: Any, session_prefix: str) -> str:
        matches = [s for s in app.session_bridge.get_all() if s["id"].startswith(session_prefix)]
        if not matches:
            return f"No session found for id prefix: {session_prefix}"
        if len(matches) > 1:
            options = ", ".join(s["id"] for s in matches[:5])
            return f"Ambiguous id prefix: {session_prefix}\nMatches: {options}"

        session = matches[0]
        app.load_session_to_workspace(session["id"])
        app.refresh_sessions()
        return f"Loaded {session['title']} ({session['id']})."
