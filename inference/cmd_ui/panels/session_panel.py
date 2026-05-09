from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class SessionPanel(Container):
    """Left sidebar showing chat session history with search and actions."""

    DEFAULT_CLASSES = "-hidden"

    def compose(self) -> ComposeResult:
        with Vertical(id="session-inner"):
            yield Static("SESSIONS", id="session-header")
            yield Input(placeholder="Search...", id="session-search")
            yield OptionList(id="session-list")
            yield Static(id="session-footer")

    def on_mount(self):
        self.display = False
        self._update_footer()
        self.update_sessions()
        self.set_interval(5.0, self.update_sessions)

    def toggle_panel(self) -> None:
        """Toggle the panel visibility."""
        if self.display:
            self.close_panel()
        else:
            self.open_panel()

    def open_panel(self) -> None:
        self.remove_class("-hidden")
        self.display = True
        self.update_sessions()
        self.app.call_after_refresh(self.query_one("#session-search", Input).focus)

    def close_panel(self) -> None:
        self.add_class("-hidden")
        self.display = False
        try:
            self.app.query_one("#prompt-input", Input).focus()
        except Exception:
            pass

    def update_sessions(self, query: str = "") -> None:
        """Refresh the session list from the history manager."""
        sessions = self.app.session_bridge.get_all(query=query if query else None)
        option_list = self.query_one("#session-list", OptionList)
        option_list.clear_options()

        if not sessions:
            option_list.add_option(Option("No sessions yet", id="__empty__", disabled=True))
            self._update_footer(count=0)
            return

        current_id = self.app.session_id
        pinned = [s for s in sessions if s.get("pinned")]
        unpinned = [s for s in sessions if not s.get("pinned")]

        if pinned:
            option_list.add_option(Option("PINNED", id="__label_pinned__", disabled=True))
            for session in pinned:
                option_list.add_option(Option(self._format_session(session, current_id), id=session["id"]))
            option_list.add_option(Option("RECENT", id="__label_recent__", disabled=True))

        for session in unpinned[:20]:
            option_list.add_option(Option(self._format_session(session, current_id), id=session["id"]))

        self._update_footer(count=len(sessions))

    def _format_session(self, session: dict, current_id: str) -> str:
        """Format a session entry as a stable one-line row."""
        title = self._clip_text(session.get("title") or "Untitled", 26)
        if session["id"] == current_id:
            prefix = ">*" if session.get("pinned") else ">"
        elif session.get("pinned"):
            prefix = "*"
        else:
            prefix = " "
        return f"{prefix} {escape(title)}"

    def _clip_text(self, value: str, max_length: int) -> str:
        text = " ".join(str(value).split())
        if len(text) <= max_length:
            return text
        return f"{text[:max_length - 3].rstrip()}..."

    def _update_footer(self, count: int = None) -> None:
        """Update the footer with session count and shortcuts."""
        footer = self.query_one("#session-footer", Static)
        parts = []
        if count is not None:
            label = "session" if count == 1 else "sessions"
            parts.append(f"[dim]{count} {label}[/]")
        parts.append("[dim]Ctrl+T/Ctrl+H[/]")
        footer.update("  ".join(parts))

    @on(Input.Changed, "#session-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        self.update_sessions(query=event.value.strip())

    @on(OptionList.OptionSelected, "#session-list")
    def on_session_selected(self, event: OptionList.OptionSelected) -> None:
        session_id = event.option.id
        if not session_id or session_id.startswith("__"):
            return
        if session_id != self.app.session_id:
            self.app.load_session_to_workspace(session_id)
            self.update_sessions()
