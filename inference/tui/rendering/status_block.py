import textwrap

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text

from inference.tui.theme import ACCENT, BORDER_HI, DANGER, TEXT, TEXT_2, TEXT_3


class StatusBlock:
    """A quiet railed system note: `~` info, `✓` success, `×` failure.

    Tool rows are bare agent-log lines. System notes carry a rail so a command
    result or error reads as a separate voice, without growing into a card that
    competes with the user's prompt.
    """

    def __init__(self, message: str, level: str = "info"):
        self.message = message
        self.level = level

    def _style(self) -> tuple[str, str, str, str]:
        if self.level == "ok":
            return "✓", ACCENT, ACCENT, TEXT_2
        if self.level == "error":
            return "×", DANGER, DANGER, TEXT
        return "~", TEXT_3, BORDER_HI, TEXT_2

    @staticmethod
    def _wrap(message: str, width: int) -> list[str]:
        lines: list[str] = []
        for paragraph in str(message).splitlines() or [""]:
            if not paragraph.strip():
                lines.append("")
                continue
            lines.extend(
                textwrap.wrap(
                    paragraph,
                    width=max(1, width),
                    break_on_hyphens=False,
                )
                or [""]
            )
        return lines

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        glyph, glyph_color, rail_color, body_color = self._style()
        width = max(1, options.max_width)
        for index, line in enumerate(self._wrap(self.message, width - 4)):
            text = Text()
            text.append("▌ ", style=rail_color)
            text.append(f"{glyph} " if index == 0 else "  ", style=glyph_color)
            text.append(line, style=body_color)
            yield text
