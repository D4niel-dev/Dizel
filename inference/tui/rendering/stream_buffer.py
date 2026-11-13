import re

from rich.text import Text
from rich.console import RenderableType
from rich.markdown import Markdown

def _strip_think_tags(text: str) -> str:
    return re.sub(r'</?think(?:ing)?>', '', text)

class StreamBuffer:
    def __init__(self):
        self.raw_text = ""
        self.finished = False

    def append(self, token: str):
        self.raw_text += token

    def finish(self):
        self.finished = True

    def __rich__(self) -> RenderableType:
        display = _strip_think_tags(self.raw_text)
        if not display:
            return Text("...", style="dim")
        try:
            return Markdown(display)
        except Exception:
            return Text(display)
