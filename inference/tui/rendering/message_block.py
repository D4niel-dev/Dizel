import re
import time

from rich.console import RenderableType, Group
from rich.text import Text
from rich.padding import Padding
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.panel import Panel
from rich import box

from inference.tui.settings_apply import chat_pref
from inference.tui.theme import (
    ACCENT,
    BORDER,
    BORDER_HI,
    CODE_BG,
    CODE_THEME,
    DIFF_ADD,
    DIFF_ADD_BG,
    DIFF_DEL,
    DIFF_DEL_BG,
    DIFF_HUNK,
    RAIL,
    STYLE_ASSISTANT,
    STYLE_SYSTEM,
    STYLE_USER,
    SURFACE,
    TEXT_2,
    TEXT_3,
    WARN,
)


def _parse_think_blocks(text: str):
    """Split text into (main, thought) around <think>…</think> segments."""
    parts = re.split(r"<(?:think|thinking)>(.*?)</(?:think|thinking)>", text, flags=re.DOTALL)
    main_parts = []
    thought_parts = []

    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            unclosed_match = re.search(r"<(?:think|thinking)>(.*)", chunk, flags=re.DOTALL)
            if unclosed_match:
                main_parts.append(chunk[:unclosed_match.start()])
                thought_parts.append(unclosed_match.group(1))
            else:
                main_parts.append(chunk)
        else:
            thought_parts.append(chunk)

    main_text = "".join(main_parts).strip()
    thought_text = "\n\n".join(t.strip() for t in thought_parts if t.strip())
    return main_text, thought_text


_FENCE = re.compile(r"```([\w+-]*)[ \t]*\n(.*?)```", re.S)

# Left rail only, the way OpenCode marks a reasoning block. Rich boxes are
# eight four-character lines: top, head, head_row, mid, row, foot_row, foot,
# bottom. Blank lines suppress the horizontal rules.
THOUGHT_BOX = box.Box(
    "┃   \n"
    "┃   \n"
    "    \n"
    "┃   \n"
    "    \n"
    "    \n"
    "┃   \n"
    "    \n"
)

_DIFF_LANGS = {"diff", "patch", "udiff"}


def _markdown(text: str) -> Markdown:
    return Markdown(text, code_theme=CODE_THEME)


def format_clock(ts: float, clock_24h: bool = True) -> str:
    local = time.localtime(ts)
    if clock_24h:
        return time.strftime("%H:%M", local)
    return time.strftime("%I:%M %p", local).lstrip("0")


def message_style(app, timestamp: float | None = None) -> dict:
    """The shared appearance prefs, in MessageBlock's vocabulary.

    A timestamp is only shown when the message actually has one: sessions saved
    before timestamps existed cannot invent a time.
    """
    stamp = None
    if timestamp is not None and chat_pref(app, "show_timestamps"):
        stamp = format_clock(timestamp, bool(chat_pref(app, "clock_24h")))
    return {
        "timestamp": stamp,
        "separators": bool(chat_pref(app, "separators")),
        "spacing": chat_pref(app, "message_spacing"),
        "syntax": bool(chat_pref(app, "syntax_highlighting")),
    }


def _diff_panel(code: str, language: str) -> Panel:
    """A patch reads as a patch: green additions, red deletions, blue hunks."""
    text = Text()
    for line in code.splitlines():
        if line.startswith(("diff ", "index ", "--- ", "+++ ")):
            text.append(line + "\n", style=TEXT_3)
        elif line.startswith("@@"):
            text.append(line + "\n", style=f"bold {DIFF_HUNK}")
        elif line.startswith("+"):
            text.append(line + "\n", style=f"{DIFF_ADD} on {DIFF_ADD_BG}")
        elif line.startswith("-"):
            text.append(line + "\n", style=f"{DIFF_DEL} on {DIFF_DEL_BG}")
        else:
            text.append(line + "\n", style=TEXT_2)
    return Panel(
        text,
        title=Text(language, style=f"bold {ACCENT}"),
        title_align="left",
        border_style=BORDER_HI,
        box=box.ROUNDED,
        padding=(0, 1),
        style=f"on {CODE_BG}",
        expand=True,
    )


def _render_body(text: str, syntax: bool = True) -> RenderableType:
    """Prose as Markdown, fenced code as a card.

    Rich's Markdown renders a fence as bare coloured text. A code block is a
    block: it gets a border, a background and a language label, the way the
    desktop wraps one.
    """
    parts: list[RenderableType] = []
    cursor = 0
    for match in _FENCE.finditer(text):
        prose = text[cursor : match.start()].strip()
        if prose:
            parts.append(_markdown(prose))

        language = (match.group(1) or "text").lower()
        code = match.group(2).rstrip()
        if code:
            if language in _DIFF_LANGS:
                parts.append(_diff_panel(code, language))
            else:
                body = (
                    Syntax(code, language, theme=CODE_THEME, word_wrap=True)
                    if syntax
                    else Text(code, style=TEXT_2)
                )
                parts.append(
                    Panel(
                        body,
                        title=Text(language, style=f"bold {ACCENT}"),
                        title_align="left",
                        border_style=BORDER_HI,
                        box=box.ROUNDED,
                        padding=(0, 1),
                        style=f"on {CODE_BG}",
                        expand=True,
                    )
                )
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        parts.append(_markdown(tail))

    if not parts:
        return _markdown(text)
    return parts[0] if len(parts) == 1 else Group(*parts)


class MessageBlock:
    """A role label on a one-cell rail, with the body indented beneath it.

    The rail carries the role colour; the prose stays in the normal text colour
    so a long transcript reads as writing, not as a colour chart.
    """

    def __init__(
        self,
        role: str,
        content: str | RenderableType,
        *,
        timestamp: str | None = None,
        separators: bool = False,
        spacing: str = "comfortable",
        syntax: bool = True,
    ):
        self.role = role
        self.timestamp = timestamp
        self.separators = separators
        self.spacing = spacing
        if isinstance(content, str):
            main_text, thought_text = _parse_think_blocks(content)
            renderables = []
            if thought_text:
                renderables.append(
                    Panel(
                        _markdown(thought_text),
                        title=f"✻ Thought · {len(thought_text):,} chars",
                        title_align="left",
                        border_style=WARN,
                        box=THOUGHT_BOX,
                        padding=(0, 1),
                    )
                )
            if main_text:
                renderables.append(_render_body(main_text, syntax))
            if not renderables:
                renderables.append(_render_body(main_text or thought_text, syntax))
            self.content = Group(*renderables) if len(renderables) > 1 else renderables[0]
        else:
            self.content = content

    def _role_style(self):
        role = self.role.upper()
        if role == "USER":
            return "you", STYLE_USER, "▌"
        if role == "ASSISTANT":
            return "dizel", STYLE_ASSISTANT, "◆"
        return role.lower(), STYLE_SYSTEM, RAIL

    def __rich__(self) -> RenderableType:
        label, color, glyph = self._role_style()
        header = Text.assemble((f"{glyph} ", color), (label, f"bold {color}"))
        if self.timestamp:
            header.append(f"  {self.timestamp}", style=TEXT_3)

        pad = (1, 0, 1, 0) if self.spacing != "compact" else (0, 0, 0, 0)

        # What you said reads as a block on a raised surface; what the model said
        # reads as writing. Same distinction the desktop makes between the user
        # bubble and the assistant turn.
        if self.role.upper() == "USER":
            bubble = Panel(
                self.content,
                box=box.SIMPLE,
                padding=(0, 1),
                style=f"on {SURFACE}",
            )
            body = Padding(Group(header, bubble), pad)
        else:
            body = Padding(Group(header, Padding(self.content, (0, 0, 0, 2))), pad)
        if self.separators:
            return Group(Rule(characters="─", style=BORDER), body)
        return body


class SystemLine:
    """A single dim line for system notices — quieter than a message."""

    def __init__(self, message: str):
        self.message = message

    def __rich__(self) -> RenderableType:
        return Padding(
            Text.assemble(("· ", TEXT_3), (self.message, TEXT_3)),
            (0, 0, 1, 0),
        )
