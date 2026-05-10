from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import TextArea, Static
from textual.containers import Vertical
from textual import events

class MultilineInputModal(ModalScreen[str]):
    """A modal overlay containing a multi-line TextArea."""

    CSS = """
    MultilineInputModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }

    #multiline-container {
        width: 80%;
        height: 70%;
        background: #18181B;
        border: solid #3B82F6;
        padding: 1 2;
    }

    #multiline-header {
        height: 1;
        text-align: center;
        color: #60A5FA;
        text-style: bold;
        margin-bottom: 1;
    }

    TextArea {
        height: 1fr;
        background: #09090B;
        border: none;
    }

    #multiline-footer {
        height: 1;
        text-align: right;
        color: #71717A;
        margin-top: 1;
    }
    """

    def __init__(self, initial_text: str = "", **kwargs):
        super().__init__(**kwargs)
        self.initial_text = initial_text

    def compose(self) -> ComposeResult:
        with Vertical(id="multiline-container"):
            yield Static("Multi-line Editor (Ctrl+S to save, ESC to cancel)", id="multiline-header")
            self.text_area = TextArea(self.initial_text, language="markdown")
            yield self.text_area
            yield Static("[ctrl+s] save and return  [esc] cancel", id="multiline-footer")

    def on_mount(self) -> None:
        self.text_area.focus()
        # Put cursor at the end
        self.text_area.cursor_location = (len(self.initial_text.splitlines()) - 1, len(self.initial_text.splitlines()[-1]) if self.initial_text else 0)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
        elif event.key == "ctrl+s":
            self.dismiss(self.text_area.text)
