"""Scrollable debug-command output, outside the saved transcript."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class DebugPanelModal(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, title: str, body: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card debug-card"):
            with Horizontal(classes="modal-head"):
                yield Static(self._title, classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            with VerticalScroll(classes="debug-scroll"):
                yield Static(Text(self._body), classes="debug-body")
            yield Static(
                "Debug output only — this is not saved to the chat.",
                classes="debug-footer",
            )

    def action_close(self) -> None:
        self.dismiss(None)
