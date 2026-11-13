from rich.console import RenderableType
from rich.text import Text

from inference.tui.theme import ACCENT, DANGER, RAIL, TEXT_3


class StatusBlock:
    """Dim system notice with an optional state colour."""

    def __init__(self, message: str, level: str = "info"):
        self.message = message
        self.level = level

    def __rich__(self) -> RenderableType:
        color = ACCENT if self.level == "ok" else DANGER if self.level == "error" else TEXT_3
        return Text.assemble((f"{RAIL} ", color), (self.message, TEXT_3))
