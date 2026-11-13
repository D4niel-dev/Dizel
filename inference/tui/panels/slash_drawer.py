"""The slash-command drawer.

Typing ``/`` opens a list of commands under the composer. Each keystroke narrows
it, including fuzzy matches, and the letters already typed stay highlighted.
Up/down/page-up/page-down move the selection, enter runs it, ctrl+f opens the
full list, esc puts it away.
"""

from __future__ import annotations

from rich.markup import escape
from textual.widgets import Static

from inference.tui.theme import ACCENT, SELECT_BG, TEXT, TEXT_3, VIOLET

NAME_WIDTH = 14
MAX_ROWS = 7
# The composer's measure. Painting happens before layout, so self.size.width is
# still 0 at that point — measuring there rendered the drawer at 40 columns.
DRAWER_WIDTH = 96


class SlashDrawer(Static):
    """A floating list of commands, shown only while a slash command is typed."""

    DEFAULT_CSS = """
    SlashDrawer {
        display: none;
        height: auto;
        width: 96;
        max-width: 100%;
        background: $surface;
        padding: 0 1;
    }

    SlashDrawer.-open {
        display: block;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._matches: list = []
        self._prefix = ""
        self._index = 0
        self._window_start = 0

    # ── state ────────────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self.has_class("-open") and bool(self._matches)

    def selected(self):
        if not self._matches:
            return None
        return self._matches[max(0, min(self._index, len(self._matches) - 1))]

    def show(self, matches: list, prefix: str) -> None:
        self._matches = list(matches)
        self._prefix = prefix
        self._index = 0
        self._window_start = 0
        self.add_class("-open")
        self._paint()

    def hide(self) -> None:
        self._matches = []
        self._index = 0
        self._window_start = 0
        self.remove_class("-open")

    def move(self, delta: int) -> None:
        if not self._matches:
            return
        self._index = (self._index + delta) % len(self._matches)
        self._ensure_window()
        self._paint()

    def page(self, delta: int) -> None:
        if not self._matches:
            return
        self._index = max(0, min(len(self._matches) - 1, self._index + delta * MAX_ROWS))
        self._ensure_window()
        self._paint()

    def _ensure_window(self) -> None:
        if self._index < self._window_start:
            self._window_start = self._index
        elif self._index >= self._window_start + MAX_ROWS:
            self._window_start = self._index - MAX_ROWS + 1

    # ── mouse ────────────────────────────────────────────────────────────
    def on_click(self, event) -> None:
        """Header opens the full list; a row runs that command."""
        row = event.offset.y
        if row == 0:
            self.hide()
            self.app.action_toggle_palette()
            event.stop()
            return

        index = self._window_start + row - 1
        if 0 <= index < len(self._matches):
            name = self._matches[index].name
            self.hide()
            from inference.tui.panels.input_bar import InputBar

            self.app.query_one(InputBar).run_command(name)
            event.stop()

    # ── rendering ────────────────────────────────────────────────────────
    def _paint(self) -> None:
        width = DRAWER_WIDTH
        visible = self._matches[self._window_start:self._window_start + MAX_ROWS]
        rows = [self._header(width)]
        for position, command in enumerate(visible):
            absolute = self._window_start + position
            rows.append(self._row(command, absolute == self._index, width))
        self.update("\n".join(rows))
        self.styles.height = len(rows)

    def _header(self, width: int) -> str:
        label = "COMMANDS"
        count = f"{len(self._matches)} match" + ("" if len(self._matches) == 1 else "es")
        hint = "⌃F full"
        gap = max(1, width - len(label) - len(count) - len(hint) - 4)
        return (
            f"[b {VIOLET}]{label}[/]{' ' * gap}"
            f"[{TEXT_3}]{count}  {hint}[/]"
        )

    def _row(self, command, selected: bool, width: int) -> str:
        name = f"/{command.name}"
        description = command.help_text or ""
        source = getattr(getattr(command, "skill", None), "source", "builtin")
        if source == "project":
            description = f"local · {description}"
        elif source == "user":
            description = f"user · {description}"
        # One line per command: anything longer is clipped rather than allowed to
        # run past the drawer and off the screen.
        room = width - NAME_WIDTH - 3
        if len(description) > room:
            description = description[: max(0, room - 1)].rstrip() + "…"

        if selected:
            text = f" {name:<{NAME_WIDTH}}{description}"
            return f"[bold {TEXT} on {SELECT_BG}]{escape(text[: width - 2]):<{width - 2}}[/]"

        typed = len(self._prefix)
        head = name[: typed + 1]
        tail = name[typed + 1 :]
        coloured = f"[b {ACCENT}]{escape(head)}[/]{escape(tail)}"
        pad = " " * max(1, NAME_WIDTH - len(name))
        body = f" {coloured}{pad}[{TEXT_3}]{escape(description)}[/]"
        return body
