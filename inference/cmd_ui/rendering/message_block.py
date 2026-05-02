from rich.console import RenderableType, Group
from rich.text import Text
from rich.padding import Padding
from inference.cmd_ui.theme import STYLE_USER, STYLE_ASSISTANT, STYLE_SYSTEM

class MessageBlock:
    def __init__(self, role: str, content: str | RenderableType):
        self.role = role
        self.content = content

    def __rich__(self) -> RenderableType:
        if self.role == "USER":
            style = STYLE_USER
            title = "> USER"
        elif self.role == "ASSISTANT":
            style = STYLE_ASSISTANT
            title = "* ASSISTANT"
        else:
            style = STYLE_SYSTEM
            title = f"# {self.role}"

        header = Text(title, style=f"bold {style}")
        return Padding(Group(header, self.content), (0, 0, 1, 0))
