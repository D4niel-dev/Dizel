from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static, OptionList, Input
from textual.widgets.option_list import Option
from textual import on


class SessionPanel(Container):
    """Left sidebar showing chat session history with search and actions."""

    DEFAULT_CLASSES = "-hidden"

    def compose(self) -> ComposeResult:
        with Vertical(id="session-inner"):
            yield Static("  ◷  SESSIONS", id="session-header")
            yield Input(placeholder="Search...", id="session-search")
            yield OptionList(id="session-list")
            yield Static(id="session-footer")

    def on_mount(self):
        self._update_footer()
        self.update_sessions()
        self.set_interval(5.0, self.update_sessions)

    def toggle_panel(self) -> None:
        """Toggle the panel visibility."""
        if self.has_class("-hidden"):
            self.remove_class("-hidden")
            self.update_sessions()
        else:
            self.add_class("-hidden")

    def update_sessions(self, query: str = "") -> None:
        """Refresh the session list from the history manager."""
        sessions = self.app.session_bridge.get_all(query=query if query else None)
        option_list = self.query_one("#session-list", OptionList)
        option_list.clear_options()

        if not sessions:
            option_list.add_option(Option("  (no sessions)", id="__empty__", disabled=True))
            self._update_footer(count=0)
            return

        current_id = self.app.session_id
        pinned = [s for s in sessions if s.get("pinned")]
        unpinned = [s for s in sessions if not s.get("pinned")]

        if pinned:
            option_list.add_option(Option("  📌 Pinned", id="__label_pinned__", disabled=True))
            for s in pinned:
                label = self._format_session(s, current_id)
                option_list.add_option(Option(label, id=s["id"]))
            option_list.add_option(Option("  ─────────────", id="__sep__", disabled=True))

        for s in unpinned[:20]:
            label = self._format_session(s, current_id)
            option_list.add_option(Option(label, id=s["id"]))

        self._update_footer(count=len(sessions))

    def _format_session(self, session: dict, current_id: str) -> str:
        """Format a session entry for display."""
        is_active = session["id"] == current_id
        prefix = "▸ " if is_active else "  "
        title = session.get("title", "Untitled")[:22]
        preview = session.get("preview", "")[:18]

        if preview:
            return f"{prefix}{title}\n    [dim]{preview}[/]"
        return f"{prefix}{title}"

    def _update_footer(self, count: int = None) -> None:
        """Update the footer with session count and shortcuts."""
        footer = self.query_one("#session-footer", Static)
        parts = []
        if count is not None:
            parts.append(f"[dim]{count} session{'s' if count != 1 else ''}[/]")
        parts.append("[dim]Ctrl+H close[/]")
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
