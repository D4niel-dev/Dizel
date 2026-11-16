"""The bundled release notes, rendered as markdown in a scrollable modal."""

from __future__ import annotations

from rich.markdown import Markdown
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from inference.tui.panels.modals.common import plain_text
from inference.tui.theme import CODE_THEME


class ReleaseNotesModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "cancel", "Close")]

    def __init__(self, text: str, version: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text
        self._version = version

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card release-notes-card"):
            with Horizontal(classes="modal-head"):
                yield Static("Release notes", classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            subtitle = f"Dizel {self._version}" if self._version else "Bundled changelog"
            yield Static(plain_text(subtitle), classes="modal-sub")
            with VerticalScroll(classes="release-notes-scroll"):
                yield Static(
                    Markdown(self._text, code_theme=CODE_THEME),
                    id="release-notes-text",
                )
            yield Static("esc close", classes="modal-footer")

    def action_cancel(self) -> None:
        self.dismiss(None)