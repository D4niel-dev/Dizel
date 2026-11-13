"""Session browser with open, pin, rename and delete actions."""

from __future__ import annotations

from datetime import datetime

from textual.binding import Binding

from inference.tui.panels.modals.common import (
    ConfirmModal,
    PickerItem,
    PickerModal,
    TextPromptModal,
)


def _ago(created: str) -> str:
    try:
        moment = datetime.fromisoformat(created)
    except (TypeError, ValueError):
        return ""
    seconds = max(0, int((datetime.now() - moment).total_seconds()))
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


class SessionManagerModal(PickerModal):
    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("f3", "pin_session", "Pin"),
        Binding("f2", "rename_session", "Rename"),
        Binding("f4", "delete_session", "Delete"),
    ]

    def __init__(self) -> None:
        super().__init__(
            "Sessions",
            [],
            subtitle="Saved conversations",
            placeholder="Search titles, previews or tags…",
            footer="enter open · ctrl+n new · f3 pin · f2 rename · f4 delete",
            empty_text="No saved sessions yet · ctrl+n starts one",
        )

    def _after_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        sessions = self.app.session_bridge.get_all()
        current = self.app.session_id
        current_title = ""
        items: list[PickerItem] = []

        for session in sessions:
            badges: list[str] = []
            if session.get("pinned"):
                badges.append("pinned")
            if session["id"] == current:
                badges.append("current")
                current_title = session.get("title", "")
            for tag in session.get("tags", [])[:2]:
                badges.append(f"#{tag}")
            age = _ago(session.get("created", ""))
            if age:
                badges.append(age)

            preview = session.get("preview", "") or "No assistant reply yet"
            items.append(
                PickerItem(
                    session["id"],
                    session.get("title", "Untitled"),
                    preview,
                    trailing=" · ".join(badges),
                    active=session["id"] == current,
                )
            )

        if current_title:
            subtitle = f"{len(items)} saved · current: {current_title}"
        elif current:
            subtitle = f"{len(items)} saved · current session is unsaved"
        else:
            subtitle = f"{len(items)} saved"
        self.set_items(items, subtitle=subtitle)

    def _selected(self) -> tuple[str, str] | None:
        options = self._list
        if options.highlighted is None:
            return None
        option = options.get_option_at_index(options.highlighted)
        if not option.id or option.id == "__empty__":
            return None
        item = self._by_id.get(option.id)
        if item is None:
            return None
        return item.id, item.label

    def action_new_session(self) -> None:
        self.app.chat_bridge.stop()
        self.app.generation_state = "IDLE"
        self.app.create_empty_session()
        self.dismiss(None)

    def action_pin_session(self) -> None:
        selected = self._selected()
        if not selected:
            return
        session_id, _ = selected
        self.app.session_bridge.pin(session_id)
        self.app.refresh_sessions()
        self._refresh()

    def action_rename_session(self) -> None:
        selected = self._selected()
        if not selected:
            return
        session_id, title = selected

        def _renamed(new_title: str | None) -> None:
            if not new_title:
                return
            self.app.session_bridge.rename(session_id, new_title)
            self.app.refresh_sessions()
            self._refresh()

        self.app.push_screen(
            TextPromptModal(
                "Rename session",
                "Title",
                value=title,
                placeholder="A short name for this conversation",
                submit_label="Rename",
            ),
            _renamed,
        )

    def action_delete_session(self) -> None:
        selected = self._selected()
        if not selected:
            return
        session_id, title = selected
        is_current = session_id == self.app.session_id

        def _confirmed(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self.app.session_bridge.delete(session_id)
            self.app.forget_tab(session_id)
            if is_current:
                if self.app.open_tabs:
                    self.app.load_session_to_workspace(self.app.open_tabs[-1])
                else:
                    self.app.create_empty_session()
            self.app.refresh_sessions()
            self._refresh()

        self.app.push_screen(
            ConfirmModal(
                "Delete session",
                f"“{title}” will be removed from disk. This cannot be undone.",
                confirm_label="Delete",
            ),
            _confirmed,
        )
