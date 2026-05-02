from rich.console import RenderableType, Group
from rich.text import Text
from rich.padding import Padding
from inference.cmd_ui.theme import COLOR_TOOL, COLOR_STREAMING, COLOR_ERROR, COLOR_THINKING

class ToolBlock:
    def __init__(self, tool_name: str, status: str = "RUNNING", detail: str = ""):
        self.tool_name = tool_name
        self.status = status
        self.detail = detail

    def __rich__(self) -> RenderableType:
        if self.status == "RUNNING":
            color = COLOR_THINKING
            icon = "..."
        elif self.status == "DONE":
            color = COLOR_STREAMING
            icon = "+"
        else:
            color = COLOR_ERROR
            icon = "x"

        header = Text(f"  {icon} {self.tool_name}", style=f"bold {COLOR_TOOL}")
        detail_text = Text(f"    {self.status} ", style=f"bold {color}")
        if self.detail:
            detail_text.append(f"- {self.detail}", style="dim")

        return Padding(Group(header, detail_text), (0, 0, 0, 0))
