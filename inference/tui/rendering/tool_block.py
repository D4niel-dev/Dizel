from rich.console import RenderableType
from rich.text import Text

from inference.tui.theme import ACCENT, DANGER, RAIL, TEXT, TEXT_2, WARN

_ICONS = {"RUNNING": "✻", "DONE": "✓", "FAILED": "✗"}
_VERBS = {"RUNNING": "running", "DONE": "done", "FAILED": "failed"}


def _preview(detail: str, limit: int = 90) -> str:
    """One line, because a tool row is a status line and not its log."""
    first = " ".join((detail or "").split())
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


class ToolBlock:
    """One line per tool event: rail, icon gutter, name, state, detail.

    Running is amber, done blue, failed red — the only place in the transcript
    where colour changes for something other than a role. The two-cell gutter
    mirrors OpenCode's inline tool rows.
    """

    def __init__(self, tool_name: str, status: str = "RUNNING", detail: str = ""):
        self.tool_name = tool_name
        self.status = status
        self.detail = detail

    def __rich__(self) -> RenderableType:
        status = self.status.upper()
        if status == "DONE":
            color = ACCENT
        elif status == "FAILED":
            color = DANGER
        else:
            color = WARN
            status = "RUNNING"

        name = self.tool_name.replace("_", " ")
        line = Text.assemble(
            (f"{RAIL} ", color),
            (f"{_ICONS[status]} ", color),
            (name, f"bold {TEXT}"),
            (f"  {_VERBS[status]}", color),
        )
        detail = _preview(self.detail)
        if detail:
            line.append(f"  {detail}", style=TEXT_2)
        return line
