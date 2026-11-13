from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class SessionCommand(Command):
    name = "session"
    help_text = "Browse sessions, or run new/list/open/rename/pin/delete."
    category = "Session"
    usage = "/session [new|list|open|rename|pin|delete] [args]"
    palette_hint = "/session "
    examples = ["/session", "/session list", "/session open 20260502"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            from inference.tui.panels.modals import SessionManagerModal

            app.push_screen(
                SessionManagerModal(),
                lambda session_id: app.load_session_to_workspace(session_id)
                if session_id
                else None,
            )
            return ""

        action = invocation.args[0].lower()

        if action == "new":
            app.action_new_session()
            return ""

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
            if not app.session_id:
                return "No active session to rename."
            if app.session_bridge.rename(app.session_id, title):
                app.refresh_sessions()
                return f"Session renamed to: {title}"
            return "Could not rename this session."

        elif action == "pin":
            if not app.session_id:
                return "No active session to pin."
            pinned = app.session_bridge.pin(app.session_id)
            app.refresh_sessions()
            return "Session pinned." if pinned else "Session unpinned."

        elif action == "delete":
            target = invocation.args[1] if len(invocation.args) > 1 else app.session_id
            if not target:
                return "No session specified."

            # Deleting is permanent and a typo is one keystroke away, so it takes
            # an explicit second word. The GUI arms on a second click; the shell
            # has no dialog here, so the word is the gate.
            confirm = invocation.args[2].lower() if len(invocation.args) > 2 else ""
            if confirm not in ("confirm", "yes", "--yes"):
                name = target
                try:
                    found = next(
                        (s for s in app.session_bridge.get_all() if s.get("id") == target),
                        None,
                    )
                    if found:
                        name = found.get("title") or target
                except Exception:
                    pass
                return (
                    f"**Delete {name}?**\n\n"
                    "This cannot be undone. Run again with `confirm` to go ahead://n//n"
                    f"`/session delete {target} confirm`"
                )
            was_current = target == app.session_id
            app.session_bridge.delete(target)
            app.forget_tab(target)
            if was_current:
                if app.open_tabs:
                    app.load_session_to_workspace(app.open_tabs[-1])
                else:
                    app.action_new_session()
            app.refresh_sessions()
            return f"Deleted session {target}"

        return f"Unknown action: {action}"

    def _render_list(self, app: Any) -> str:
        sessions = app.session_bridge.get_all()
        if not sessions:
            return "**Sessions:** No saved sessions."

        lines = ["### Sessions"]
        for session in sessions[:15]:
            current = " *(current)*" if session["id"] == app.session_id else ""
            pinned = " `pinned`" if session.get("pinned") else ""
            lines.append(f"- `{session['id'][:8]}` · {session['title'][:36]}{pinned}{current}")
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
