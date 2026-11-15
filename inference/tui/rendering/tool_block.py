import re

from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.widgets import Static

from inference.tui.settings_apply import chat_pref
from inference.tui.theme import (
    ACCENT,
    CODE_BG,
    CODE_THEME,
    DANGER,
    DIFF_ADD,
    DIFF_ADD_BG,
    DIFF_DEL,
    DIFF_DEL_BG,
    DIFF_HUNK,
    SPINNER_FRAMES,
    TEXT,
    TEXT_2,
    TEXT_3,
    WARN,
)

_SEARCH = ("grep", "search", "find")
_READ = ("read", "open", "fetch", "file", "glob", "list")
_URL = re.compile(r"(https?://[^\s<>\"']+)")
_READ_HEADER = re.compile(r"^.+\s+—\s+lines (\d+)-(\d+) of (\d+)\n")
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DETAIL_WIDTH = 240
_LEXERS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".xml": "xml",
    ".css": "css",
    ".scss": "scss",
    ".sh": "bash",
    ".bash": "bash",
    ".ps1": "powershell",
    ".cmd": "batch",
    ".bat": "batch",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".sql": "sql",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".txt": "text",
}


def _preview(detail: str, limit: int = 90) -> str:
    """One line, because a tool row is a status line and not its log."""
    first = " ".join((detail or "").split())
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def _glyph(name: str, status: str, frame: str | None = None) -> str:
    if status == "FAILED":
        return "×"
    if status == "RUNNING":
        return frame or SPINNER_FRAMES[0]
    key = name.lower()
    if any(word in key for word in _SEARCH):
        return "*"
    if any(word in key for word in _READ):
        return "→"
    return "✓"


def _line(value: str, style: str = TEXT_2) -> Text:
    """One detail line with a quiet rail and clickable URLs."""
    value = value if len(value) <= _DETAIL_WIDTH else value[: _DETAIL_WIDTH - 1] + "…"
    row = Text("  │ ", style=TEXT_3)
    cursor = 0
    for match in _URL.finditer(value):
        row.append(value[cursor : match.start()], style=style)
        url = match.group(0).rstrip(".,);]")
        tail = match.group(0)[len(url) :]
        row.append(url, style=f"link {url} {ACCENT}")
        row.append(tail, style=style)
        cursor = match.start() + len(match.group(0))
    row.append(value[cursor:], style=style)
    return row


def _diff_rows(diff: str) -> list[Text]:
    rows: list[Text] = []
    old_line = 0
    new_line = 0
    in_hunk = False

    for line in diff.splitlines():
        if line.startswith(("--- ", "+++ ", "\\")):
            rows.append(_line(line, TEXT_3))
            continue
        if line.startswith("@@"):
            match = _HUNK.match(line)
            if match:
                old_line = int(match.group(1))
                new_line = int(match.group(2))
                in_hunk = True
            rows.append(_line(line, f"bold {DIFF_HUNK}"))
            continue

        sign = line[0] if line else " "
        if sign == "+":
            number = new_line if in_hunk else None
            new_line += 1
            style = f"{DIFF_ADD} on {DIFF_ADD_BG}"
        elif sign == "-":
            number = old_line if in_hunk else None
            old_line += 1
            style = f"{DIFF_DEL} on {DIFF_DEL_BG}"
        else:
            number = new_line if in_hunk else None
            if in_hunk:
                old_line += 1
                new_line += 1
            style = TEXT_2

        row = Text("  │ ", style=TEXT_3)
        row.append(f"{number:>4} " if number is not None else "     ", style=TEXT_3)
        row.append(sign if sign in "+-" else " ", style=style)
        row.append(line[1:] if sign in "+- " else line, style=style)
        rows.append(row)
    return rows


def _lexer_for(path: str) -> str:
    name = str(path or "")
    if "." not in name:
        return "text"
    return _LEXERS.get("." + name.rsplit(".", 1)[-1].lower(), "text")


class ToolBlock:
    """OpenCode-style inline tool activity: glyph, tool, one-line detail.

    No rail and no "running/done" verb; the glyph and colour carry state. Search
    tools use `*`, read/list tools use `→`, failures use `×`. A running row gets
    the one-cell activity spinner instead of a final icon.
    """

    def __init__(
        self,
        tool_name: str,
        status: str = "RUNNING",
        detail: str = "",
        frame: str | None = None,
        expandable: bool = False,
        expanded: bool = False,
    ):
        self.tool_name = tool_name
        self.status = status
        self.detail = detail
        self.frame = frame
        self.expandable = expandable
        self.expanded = expanded

    def __rich__(self) -> RenderableType:
        status = self.status.upper()
        if status not in ("RUNNING", "DONE", "FAILED"):
            status = "RUNNING"
        color = WARN if status == "RUNNING" else DANGER if status == "FAILED" else TEXT_3

        name = self.tool_name.replace("_", " ")
        line = Text.assemble(
            (f"{_glyph(name, status, self.frame)} ", color),
            (name, f"bold {TEXT}"),
        )
        detail = _preview(self.detail)
        if detail:
            line.append(f"  {detail}", style=TEXT_2)
        if self.expandable:
            line.append("  ▾" if self.expanded else "  ▸", style=TEXT_3)
        return line


class ToolLine(Static):
    """A tool row that keeps its glyph moving while the call is running.

    Finished calls carry a chevron; clicking the row (or pressing enter) expands
    an inline drawer with the diff, written content, or result. It never opens a
    separate screen.
    """

    can_focus = True
    BINDINGS = [Binding("enter", "toggle", "Expand", show=False)]

    def __init__(
        self,
        tool_name: str,
        status: str = "RUNNING",
        detail: str = "",
        details: dict | None = None,
    ):
        super().__init__()
        self.add_class("tool-line")
        self.tool_name = tool_name
        self.status = status.upper()
        self.detail = detail
        self.details = details or {}
        self.expanded = False
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        self._render_row()
        if self.status == "RUNNING" and chat_pref(self.app, "animations", True):
            self._timer = self.set_interval(0.12, self._tick)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _has_details(self) -> bool:
        return any(
            bool(self.details.get(key))
            for key in ("diff", "content", "result", "old", "new")
        )

    def _render_row(self) -> None:
        frame = SPINNER_FRAMES[self._frame] if self.status == "RUNNING" else None
        expandable = self._has_details()
        lines = [
            ToolBlock(
                self.tool_name,
                self.status,
                self.detail,
                frame,
                expandable=expandable,
                expanded=self.expanded,
            )
        ]
        if self.expanded:
            lines.extend(self._meta_lines())
            lines.extend(self._detail_lines())
        self.update(Group(*lines))

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
        self._render_row()

    def finish(self, status: str, detail: str | None = None, details: dict | None = None) -> None:
        self.status = status.upper()
        if detail is not None:
            self.detail = detail
        if details is not None:
            self.details = details
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._render_row()

    def _meta_lines(self) -> list[Text]:
        rows: list[Text] = []

        def add(label: str, key: str) -> None:
            value = self.details.get(key)
            if value:
                rows.append(_line(f"{label:<8} {value}", TEXT_3))

        add("path", "path")
        add("query", "query")
        add("glob", "glob")
        add("action", "action")
        add("url", "url")
        add("search", "search_url")
        return rows

    def output_text(self) -> str:
        details = self.details
        for key in ("diff", "content", "result", "old", "new"):
            value = details.get(key)
            if value:
                return str(value)
        return ""

    def _code_block(
        self, code: str, path: str, start_line: int = 1
    ) -> RenderableType:
        code = code.rstrip("\n")
        if not code:
            return _line("(empty file)", TEXT_3)
        if not chat_pref(self.app, "syntax_highlighting", True):
            return Group(*[_line(line, TEXT_2) for line in code.splitlines()])
        return Padding(
            Syntax(
                code,
                _lexer_for(path),
                theme=CODE_THEME,
                line_numbers=True,
                start_line=start_line,
                word_wrap=True,
                background_color=CODE_BG,
            ),
            (0, 0, 0, 1),
        )

    def _detail_lines(self) -> list[RenderableType]:
        if self.status == "RUNNING" or not self.details:
            return []

        details = self.details
        name = self.tool_name.lower()
        diff = str(details.get("diff") or "")
        content = str(details.get("content") or "")
        result = str(details.get("result") or "")
        path = str(details.get("path") or "")

        if name == "write_file" and content:
            return self._finish_drawer(
                [self._code_block(content, path)], details, "content_truncated"
            )

        if diff:
            return self._finish_drawer(
                _diff_rows(diff), details, "diff_truncated"
            )

        if name == "edit_file" and (details.get("old") or details.get("new")):
            old = str(details.get("old") or "").splitlines()
            new = str(details.get("new") or "").splitlines()
            rows = [
                _line("- " + line, f"{DIFF_DEL} on {DIFF_DEL_BG}") for line in old
            ]
            rows += [
                _line("+ " + line, f"{DIFF_ADD} on {DIFF_ADD_BG}") for line in new
            ]
            return self._finish_drawer(rows, details, "replacement_truncated")

        if name == "read_file" and result:
            match = _READ_HEADER.match(result)
            if match:
                start = int(match.group(1))
                header = _line(
                    f"lines {match.group(1)}–{match.group(2)} of {match.group(3)}",
                    TEXT_3,
                )
                return self._finish_drawer(
                    [header, self._code_block(result[match.end():], path, start)],
                    details,
                    "result_truncated",
                )

        if content:
            return self._finish_drawer(
                [self._code_block(content, path)], details, "content_truncated"
            )

        if not result:
            return []
        rows = [
            _line(line, DANGER if self.status == "FAILED" else TEXT_2)
            for line in result.splitlines()
        ]
        return self._finish_drawer(rows, details, "result_truncated")

    def _finish_drawer(
        self, rows: list[RenderableType], details: dict, truncation_key: str
    ) -> list[RenderableType]:
        if details.get(truncation_key):
            rows.append(_line("… clipped at the UI limit", TEXT_3))
        if len(rows) > 14:
            rows.append(_line("click the row to collapse", TEXT_3))
        return rows

    def action_toggle(self) -> None:
        if not self._has_details():
            return
        self.expanded = not self.expanded
        self._render_row()
        try:
            self.app.query_one("WorkspacePanel")._follow_if_pinned()
        except Exception:
            pass

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.action_toggle()
