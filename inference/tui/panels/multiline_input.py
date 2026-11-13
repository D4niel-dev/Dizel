from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import TextArea, Static
from textual.containers import Horizontal, Vertical
from textual import events


class MultilineInputModal(ModalScreen[str]):
    """A larger editor for multi-line prompts (ctrl+y)."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card multiline-card"):
            with Horizontal(classes="modal-head"):
                yield Static("Compose", classes="modal-title")
                yield Static("ctrl+s", classes="modal-kbd")
            yield Static("Draft a longer prompt before sending it.", classes="modal-sub")
            self.text_area = TextArea(self.initial_text, language="markdown")
            yield self.text_area
            verb = "send" if self.send_on_save else "save and return"
            yield Static(
                f"ctrl+s {verb} · esc cancel",
                classes="modal-footer",
            )

    def __init__(self, initial_text: str = "", send_on_save: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.initial_text = initial_text
        self.send_on_save = send_on_save

    def on_mount(self) -> None:
        self.text_area.focus()
        lines = self.initial_text.splitlines() or [""]
        self.text_area.cursor_location = (len(lines) - 1, len(lines[-1]))

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
        elif event.key == "ctrl+s":
            event.stop()
            self.dismiss(self.text_area.text)
