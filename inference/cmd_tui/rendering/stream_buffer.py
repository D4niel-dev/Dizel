from rich.text import Text
from rich.console import RenderableType
from rich.markdown import Markdown

class StreamBuffer:
    def __init__(self):
        self.raw_text = ""
        self.finished = False

    def append(self, token: str):
        self.raw_text += token

    def finish(self):
        self.finished = True

    def __rich__(self) -> RenderableType:
        if not self.raw_text:
            return Text("...", style="dim")
        try:
            return Markdown(self.raw_text)
        except Exception:
            return Text(self.raw_text)
