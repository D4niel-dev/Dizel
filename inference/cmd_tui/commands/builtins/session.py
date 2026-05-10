from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation

class SessionCommand(Command):
    name = "session"
    help_text = "Manage sessions. Usage: /session new|list|open|rename|pin|delete [args]"
    category = "Session"
    usage = "/session [new|list|open|rename|pin|delete] [args]"
    palette_hint = "/session "
    examples = ["/session new", "/session list", "/session open 20260502"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            return self._render_current(app)
            
        action = invocation.args[0].lower()
        
        if action == "new":
            app.session_bridge.create()
            app.query_one("WorkspacePanel").clear_workspace()
            return None

        elif action in ("list", "ls"):
            return self._render_list(app)

        elif action in ("open", "load"):
            if len(invocation.args) < 2:
                return "Usage: /session open <session-id-prefix>"
            return self._open_session(app, invocation.args[1])
            
        elif action == "rename":
            title = " ".join(invocation.args[1:])
            if not title:
                return "Please provide a title."
            app.session_bridge.save(title=title)
            return f"Session renamed to: {title}"
            
        elif action == "pin":
            if not app.session_id:
                return "No active session to pin."
            app.session_bridge.pin(app.session_id)
            return "Toggled pin status."
            
        elif action == "delete":
            target = invocation.args[1] if len(invocation.args) > 1 else app.session_id
            if target:
                app.session_bridge.delete(target)
                if target == app.session_id:
                    app.query_one("WorkspacePanel").clear_workspace()
                    return None
                return f"Deleted session {target}"
            return "No session specified."
            
        return f"Unknown action: {action}"

    def _render_current(self, app: Any) -> str:
        return (
            f"### Current Session\n"
            f"- **Session ID:** `{app.session_id or 'None'}`\n\n"
            f"**Actions:** `new` · `list` · `open <id>` · `rename <title>` · `pin` · `delete [id]`"
        )

    def _render_list(self, app: Any) -> str:
        sessions = app.session_bridge.get_all()
        if not sessions:
            return "**Sessions:** No saved sessions."

        lines = ["### Sessions"]
        for session in sessions[:15]:
            active = "**(current)**" if session["id"] == app.session_id else ""
            pinned = "📌" if session.get("pinned") else "📝"
            lines.append(f"- {pinned} `{session['id'][:8]}` | {session['title'][:36]} {active}")
        lines.append("")
        lines.append("**Open:** `/session open <id-prefix>`")
        return "\n".join(lines)

    def _open_session(self, app: Any, session_prefix: str) -> str:
        matches = [s for s in app.session_bridge.get_all() if s["id"].startswith(session_prefix)]
        if not matches:
            return f"**Error:** No session found for id prefix: `{session_prefix}`"
        if len(matches) > 1:
            options = ", ".join(f"`{s['id']}`" for s in matches[:5])
            return f"**Error:** Ambiguous session id prefix: `{session_prefix}`\nMatches: {options}"

        session = matches[0]
        app.load_session_to_workspace(session["id"])
        return f"**Loaded session:** {session['title']} (`{session['id']}`)"
