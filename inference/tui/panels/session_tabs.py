from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Tab, Tabs

MAX_VISIBLE_TABS = 10


class TabClose(Button):
    """The close control attached to one session tab."""

    def __init__(self, session_id: str, active: bool) -> None:
        self.session_id = session_id
        classes = "tab-inline-close" + (" -active" if active else "")
        super().__init__(
            "×",
            id=f"tab-x-{session_id}",
            classes=classes,
            compact=True,
            tooltip="Close tab",
        )


class SessionTabsWidget(Tabs):
    """Tabs that keep the active tab and the trailing actions in view."""

    def _scroll_active_tab(self) -> None:
        if self.has_class("is-scrollable"):
            try:
                tabs = list(self.query("#tabs-list > Tab"))
                if tabs and self.active == (tabs[-1].id or ""):
                    self.query_one("#tabs-scroll").scroll_end(
                        animate=False,
                        force=True,
                        y_axis=False,
                    )
                    return
            except Exception:
                pass
        super()._scroll_active_tab()


class SessionTabs(Horizontal):
    """Open chats as tabs. Each tab closes itself; the newest carries create."""

    @staticmethod
    def _tab_id(session_id: str) -> str:
        # Textual IDs cannot begin with a digit, and session ids are timestamps.
        return f"session-{session_id}"

    @staticmethod
    def _session_id(tab_id: str | None) -> str:
        return (tab_id or "").removeprefix("session-")

    def compose(self) -> ComposeResult:
        yield SessionTabsWidget(id="session-tabs")

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

        scrollable = len(wanted) > MAX_VISIBLE_TABS
        self.set_class(scrollable, "is-scrollable")
        tabs.set_class(scrollable, "is-scrollable")
        new_tab = await self._sync_tab_actions(tabs, wanted, current)

        if (
            new_tab is not None
            and scrollable
            and wanted
            and current == wanted[-1]
        ):
            self.call_after_refresh(
                lambda: self.set_timer(0.05, lambda: self._scroll_to_widget(new_tab))
            )

    async def _sync_tab_actions(
        self,
        tabs: Tabs,
        wanted: list[str],
        current: str,
    ) -> Button | None:
        """Keep every close button beside its tab and create after the last."""
        actions = [
            *tabs.query(".tab-inline-close"),
            *tabs.query("#tab-new-inline"),
        ]
        for widget in actions:
            await widget.remove()

        try:
            tab_list = tabs.query_one("#tabs-list", Horizontal)
        except Exception:
            return None

        tab_by_session = {
            self._session_id(tab.id): tab for tab in tabs.query(Tab)
        }
        last: TabClose | None = None
        for session_id in wanted:
            tab = tab_by_session.get(session_id)
            if tab is None:
                continue
            close = TabClose(session_id, session_id == current)
            await tab_list.mount(close, after=tab)
            last = close

        new_tab = Button(
            "+",
            id="tab-new-inline",
            classes="tab-inline-new",
            compact=True,
            tooltip="New chat",
        )
        if last is not None:
            await tab_list.mount(new_tab, after=last)
        else:
            await tab_list.mount(new_tab)
        return new_tab

    def _scroll_to_widget(self, widget: Button) -> None:
        try:
            self.query_one("#tabs-scroll").scroll_to_widget(
                widget,
                animate=False,
                force=True,
            )
        except Exception:
            pass

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

    @on(Button.Pressed, "#tab-new-inline")
    def _new_tab(self) -> None:
        self.app.action_new_session()

    @on(Button.Pressed, ".tab-inline-close")
    def _close_tab(self, event: Button.Pressed) -> None:
        event.stop()
        session_id = getattr(event.button, "session_id", "")
        if not session_id:
            return
        self.app.close_tab(session_id)
        try:
            self.app.query_one("#prompt-input").focus()
        except Exception:
            pass
