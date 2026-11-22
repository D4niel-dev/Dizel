from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import TextArea, Static
from textual.containers import Horizontal, Vertical
from textual import events


class MultilineInputModal(ModalScreen[str]):
    """A larger editor for multi-line prompts (ctrl+y, /edit)."""

    def __init__(
        self,
        initial_text: str = "",
        send_on_save: bool = False,
        title: str = "Compose",
        subtitle: str = "Draft a longer prompt before sending it.",
        notice: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_text = initial_text
        self.send_on_save = send_on_save
        self.title_text = title
        self.subtitle = subtitle
        self.notice_text = notice
        self._armed = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card multiline-card"):
            with Horizontal(classes="modal-head"):
                yield Static(self.title_text, classes="modal-title")
                yield Static("ctrl+s", classes="modal-kbd")
            yield Static(self.subtitle, classes="modal-sub")
            self.notice_widget = Static(self.notice_text, classes="modal-notice")
            self.notice_widget.display = bool(self.notice_text)
            yield self.notice_widget
            self.text_area = TextArea(self.initial_text, language="markdown")
            yield self.text_area
            self.footer = Static(
                f"ctrl+s {self._verb()} · esc cancel", classes="modal-footer"
            )
            yield self.footer

    def _verb(self) -> str:
        if self.notice_text:
            return "replace" if self._armed else "send"
        return "send" if self.send_on_save else "save and return"

    def on_mount(self) -> None:
        self.text_area.focus()
        lines = self.initial_text.splitlines() or [""]
        self.text_area.cursor_location = (len(lines) - 1, len(lines[-1]))

    def _disarm(self) -> None:
        self._armed = False
        self.notice_widget.update(self.notice_text)
        self.notice_widget.remove_class("-danger")
        self.footer.update(f"ctrl+s {self._verb()} · esc cancel")

    def _arm(self) -> None:
        self._armed = True
        self.notice_widget.update(
            "Press ctrl+s again to replace this turn and every message after it — "
            "esc goes back."
        )
        self.notice_widget.add_class("-danger")
        self.footer.update(f"ctrl+s {self._verb()} · esc cancel")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            if self._armed:
                self._disarm()
                return
            self.dismiss(None)
        elif event.key == "ctrl+s":
            event.stop()
            if self.notice_text and not self._armed:
                self._arm()
                return
            self.dismiss(self.text_area.text)
