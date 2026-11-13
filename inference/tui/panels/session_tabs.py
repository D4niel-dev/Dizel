from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Tab, Tabs


class SessionTabs(Horizontal):
    """Open chats as tabs. The fixed actions at the right create and close."""

    @staticmethod
    def _tab_id(session_id: str) -> str:
        # Textual IDs cannot begin with a digit, and session ids are timestamps.
        return f"session-{session_id}"

    @staticmethod
    def _session_id(tab_id: str | None) -> str:
        return (tab_id or "").removeprefix("session-")

    def compose(self) -> ComposeResult:
        yield Tabs(id="session-tabs")
        yield Static("×", id="tab-close", classes="tab-action")
        yield Static("+", id="tab-new", classes="tab-action")

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_tabs)

    def refresh_tabs(self) -> None:
        try:
            self.query_one("#session-tabs", Tabs)
        except Exception:
            return
        self.run_worker(self._sync_tabs, exclusive=True, group="session-tabs")

    async def _sync_tabs(self) -> None:
        try:
            tabs = self.query_one("#session-tabs", Tabs)
        except Exception:
            return

        wanted = list(getattr(self.app, "open_tabs", []) or [])
        current = str(getattr(self.app, "session_id", "") or "")
        existing = {self._session_id(tab.id): tab for tab in tabs.query(Tab)}

        for session_id, tab in list(existing.items()):
            if session_id not in wanted:
                await tabs.remove_tab(tab.id)

        for session_id in wanted:
            title = self.app.tab_title(session_id)
            if session_id in existing:
                existing[session_id].update(title)
            else:
                await tabs.add_tab(Tab(title, id=self._tab_id(session_id)))

        # A newer refresh may have changed the app state while this worker was
        # awaiting tab mounts. Never activate an id from the stale snapshot.
        latest = list(getattr(self.app, "open_tabs", []) or [])
        latest_current = str(getattr(self.app, "session_id", "") or "")
        if latest != wanted or latest_current != current:
            self.refresh_tabs()
            return

        live_ids = {self._session_id(tab.id) for tab in tabs.query(Tab)}
        active_id = self._tab_id(current)
        if wanted and current in live_ids and tabs.active != active_id:
            tabs.active = active_id

    async def _activate(self, event: Tabs.TabActivated) -> None:
        try:
            tabs = self.query_one("#session-tabs", Tabs)
        except Exception:
            return
        if event.tabs is not tabs:
            return
        tab_id = event.tab.id or ""
        session_id = self._session_id(tab_id)
        if not session_id or session_id == self.app.session_id:
            return
        # A tab removal can activate a neighbour before the app has chosen the
        # next session; ignore any activation the tab strip has already moved on from.
        if tab_id != (tabs.active or ""):
            return
        if not self.app.load_session_to_workspace(session_id):
            self.refresh_tabs()

    @on(Tabs.TabActivated)
    def _tab_activated(self, event: Tabs.TabActivated) -> None:
        self.run_worker(self._activate(event), group="session-tabs-activate")

    @on(events.Click, "#tab-new")
    def _new_tab(self) -> None:
        self.app.action_new_session()

    @on(events.Click, "#tab-close")
    def _close_tab(self) -> None:
        self.app.close_active_tab()
