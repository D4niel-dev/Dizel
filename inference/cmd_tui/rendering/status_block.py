from rich.text import Text
from rich.console import RenderableType

class StatusBlock:
    def __init__(self, message: str):
        self.message = message

    def __rich__(self) -> RenderableType:
        return Text(f"[SYSTEM] {self.message}", style="dim italic")
