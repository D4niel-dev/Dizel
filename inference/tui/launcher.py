"""Pre-start chooser shown by ``dizel.cmd``.

The batch file can only print lines and wait for one keypress. This is the same
decision with room to breathe: the brand, what each surface is for, and keyboard
navigation — using the same Textual stack and palette as the terminal client
itself, so the first thing you see is made of the thing you are choosing.

Exit codes are the contract with ``dizel.cmd``:

    1  terminal UI
    2  desktop app
    3  cancelled (Esc, Ctrl+C, or closed)
    4  terminal UI, resuming the most recent saved chat
"""

from __future__ import annotations

from rich.markup import escape

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Middle, Vertical
from textual.widgets import ListItem, ListView, Static

from .rendering.empty_state import Wordmark
from .theme import (
    ACCENT,
    ACCENT_INK,
    BG_DEEP,
    BORDER,
    SUCCESS,
    SURFACE,
    SURFACE_HI,
    TEXT,
    TEXT_2,
    TEXT_3,
)

TERMINAL = 1
DESKTOP = 2
CANCELLED = 3
RESUME = 4

BASE_CHOICES: list[tuple[int, str, str, str, str]] = [
    (TERMINAL, "1", "Terminal UI", "Keyboard-first, stays in the shell.", ">_"),
    (DESKTOP, "2", "Desktop app", "Full window, with motion and visuals.", "▤"),
]


def _clip(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _version() -> str:
    try:
        from dizel_core.version import __version__

        return __version__
    except Exception:
        return ""


class ChoiceItem(ListItem):
    """One console row: keycap, glyph, name, and what it gives you."""

    def __init__(self, code: int, key: str, title: str, hint: str, glyph: str) -> None:
        super().__init__()
        self.code = code
        self.key = key
        self.title_text = title
        self.hint = hint
        self.glyph = glyph

    def compose(self) -> ComposeResult:
        with Horizontal(classes="choice-row"):
            yield Static("▌", classes="choice-rail")
            yield Static(self.glyph, classes="choice-glyph")
            with Vertical(classes="choice-text"):
                yield Static(
                    f"[b {TEXT}]{escape(self.title_text)}[/]", classes="choice-title"
                )
                yield Static(f"[{TEXT_3}]{escape(self.hint)}[/]", classes="choice-hint")
            yield Static(self.key, classes="choice-key")
            yield Static("→", classes="choice-arrow")


class Launcher(App):
    """A small, deliberate screen: pick a surface, or leave."""

    TITLE = "Dizel"
    CSS = f"""
    Screen {{
        background: {BG_DEEP};
    }}

    #launcher-root {{
        width: 100%;
        height: 100%;
        layout: vertical;
    }}

    /* ── Chrome ─────────────────────────────────────────────────────── */
    #topbar {{
        width: 100%;
        height: 2;
        background: {BG_DEEP};
        border-bottom: solid {BORDER};
        layout: horizontal;
        padding: 0 2;
        align: left middle;
    }}

    #topbar-left {{
        width: 1fr;
        height: 1;
        color: {TEXT};
    }}

    #topbar-right {{
        width: auto;
        height: 1;
        content-align: right middle;
        color: {TEXT_3};
    }}

    #launcher-middle {{
        width: 100%;
        height: 1fr;
        align: center middle;
        padding: 0 2;
    }}

    #shell {{
        width: 72;
        max-width: 100%;
        height: auto;
    }}

    #shell.-wide {{
        width: 96;
    }}

    #deck {{
        width: 100%;
        height: auto;
        layout: horizontal;
    }}

    #deck.-stacked {{
        layout: vertical;
    }}

    /* ── Brand and session ledger ───────────────────────────────────── */
    #brand-panel {{
        width: 38;
        height: auto;
        padding-right: 2;
        border-right: solid {BORDER};
    }}

    #deck.-stacked #brand-panel {{
        width: 100%;
        padding-right: 0;
        padding-bottom: 1;
        border-right: none;
    }}

    .launcher-wordmark {{
        width: 100%;
        height: auto;
        content-align: center middle;
        margin-bottom: 0;
    }}

    #brand-compact {{
        width: 100%;
        height: 1;
        content-align: center middle;
        color: {TEXT};
    }}

    #kicker {{
        width: 100%;
        height: 1;
        content-align: center middle;
        color: {TEXT_3};
        text-style: bold;
        margin-top: 1;
    }}

    #tagline {{
        width: 100%;
        height: auto;
        content-align: center middle;
        color: {TEXT_2};
        margin-top: 1;
    }}

    #meta {{
        width: 100%;
        height: 1;
        content-align: center middle;
        color: {TEXT_3};
    }}

    #brand-facts {{
        display: none;
        width: 100%;
        height: auto;
        margin-top: 2;
        padding-top: 1;
        border-top: solid {BORDER};
    }}

    .fact {{
        width: 100%;
        height: 1;
    }}

    .fact-label {{
        width: 10;
        height: 1;
        color: {TEXT_3};
        text-style: bold;
    }}

    .fact-value {{
        width: 1fr;
        height: 1;
        color: {TEXT_2};
    }}

    /* ── Choices ────────────────────────────────────────────────────── */
    #choice-panel {{
        width: 1fr;
        height: auto;
        padding-left: 2;
    }}

    #deck.-stacked #choice-panel {{
        width: 100%;
        padding-left: 0;
    }}

    #choice-head {{
        width: 100%;
        height: auto;
        layout: horizontal;
        align: left middle;
        padding-bottom: 1;
        border-bottom: solid {BORDER};
    }}

    #choice-label {{
        width: 1fr;
        height: 1;
        color: {TEXT_3};
        text-style: bold;
    }}

    #choice-meta {{
        width: auto;
        height: 1;
        content-align: right middle;
        color: {TEXT_3};
    }}

    ListView {{
        height: auto;
        background: transparent;
        border: none;
        padding: 0;
        margin-top: 2;
    }}

    ListView:focus {{
        border: none;
    }}

    ListItem {{
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        background: transparent;
        border: none;
    }}

    ListItem.-highlight {{
        background: {SURFACE};
    }}

    .choice-row {{
        width: 100%;
        height: auto;
        align: left middle;
    }}

    .choice-rail {{
        width: 2;
        height: auto;
        color: {BORDER};
        content-align: left middle;
    }}

    .choice-glyph {{
        width: 3;
        height: 1;
        content-align: left middle;
        color: {TEXT_3};
    }}

    .choice-text {{
        width: 1fr;
        height: auto;
        padding-left: 1;
    }}

    .choice-title,
    .choice-hint {{
        width: 100%;
        height: auto;
    }}

    .choice-key {{
        width: 3;
        height: 1;
        margin-left: 1;
        content-align: center middle;
        background: {SURFACE_HI};
        color: {TEXT_2};
        text-style: bold;
    }}

    .choice-arrow {{
        width: 2;
        height: auto;
        color: {TEXT_3};
        content-align: right middle;
    }}

    ListItem.-highlight .choice-rail,
    ListItem.-highlight .choice-glyph,
    ListItem.-highlight .choice-arrow {{
        color: {ACCENT};
    }}

    ListItem.-highlight .choice-key {{
        background: {ACCENT};
        color: {ACCENT_INK};
    }}

    ListItem.-highlight .choice-hint {{
        color: {TEXT_2};
    }}

    #choice-foot {{
        display: none;
        width: 100%;
        height: auto;
        layout: horizontal;
        align: left middle;
        margin-top: 1;
        padding-top: 1;
        border-top: solid {BORDER};
    }}

    .foot-label {{
        width: 10;
        height: 1;
        color: {TEXT_3};
        text-style: bold;
    }}

    .foot-value {{
        width: 1fr;
        height: 1;
        color: {TEXT_3};
    }}

    /* ── Status line ────────────────────────────────────────────────── */
    #launcher-status {{
        width: 100%;
        height: 2;
        background: {BG_DEEP};
        border-top: solid {BORDER};
        layout: horizontal;
        padding: 0 2;
        align: left middle;
    }}

    #launcher-status-left {{
        width: 1fr;
        height: 1;
        color: {TEXT_3};
    }}

    #launcher-status-right {{
        width: 1fr;
        height: 1;
        content-align: right middle;
        color: {TEXT_3};
    }}

    /* ── Compact terminal ───────────────────────────────────────────── */
    #shell.-compact #brand-panel {{
        padding-bottom: 0;
    }}

    #shell.-compact #kicker {{
        margin-top: 0;
    }}

    #shell.-compact #tagline {{
        margin-top: 0;
    }}

    #shell.-compact #meta {{
        display: none;
    }}

    #shell.-compact #choice-head {{
        padding-bottom: 0;
        border-bottom: none;
    }}

    #shell.-compact ListView {{
        margin-top: 1;
    }}

    #shell.-tiny #meta,
    #shell.-tiny .choice-hint {{
        display: none;
    }}

    #shell.-tiny ListView {{
        margin-top: 0;
    }}
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("q", "cancel", "Cancel", show=False),
        Binding("1", "pick(0)", "Terminal", show=False),
        Binding("2", "pick(1)", "Desktop", show=False),
        Binding("3", "pick(2)", "Resume", show=False),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sessions = self._sessions()
        self.choices = list(BASE_CHOICES)
        self.last_title = ""
        last = self.sessions[0] if self.sessions else None
        if last:
            raw_title = " ".join(str(last.get("title") or "").split())
            title = _clip(raw_title, 32)
            self.last_title = _clip(raw_title, 24)
            hint = f"Continue: {title}" if title else "Continue where you left off."
            self.choices.append((RESUME, "3", "Resume last chat", hint, "↺"))

    @staticmethod
    def _sessions():
        try:
            from dizel_core.history_manager import list_sessions

            return list_sessions()
        except Exception:
            return []

    def _saved_text(self) -> str:
        count = len(self.sessions)
        if count == 0:
            return "No saved chats yet"
        return f"{count} saved chat" + ("" if count == 1 else "s")

    def _capture(self, label: str) -> str:
        return f"[{TEXT} on {SURFACE_HI}] {label} [/]"

    def _hints(self) -> str:
        return (
            f"{self._capture('↑↓')} [{TEXT_3}]move[/]  "
            f"{self._capture('↵')} [{TEXT_3}]start[/]  "
            f"{self._capture('esc')} [{TEXT_3}]quit[/]"
        )

    def _apply_layout(self, width: int, height: int) -> None:
        wide = width >= 100
        compact = width < 100 or height < 28
        tiny = height < 22
        try:
            shell = self.query_one("#shell")
            shell.set_class(wide, "-wide")
            shell.set_class(compact, "-compact")
            shell.set_class(tiny, "-tiny")
            self.query_one("#deck").set_class(not wide, "-stacked")
            self.query_one("#brand-facts").display = wide
            self.query_one("#choice-foot").display = wide
            self.query_one("#wordmark").display = not tiny
            self.query_one("#brand-compact").display = tiny
            self.query_one("#launcher-status-right", Static).update(self._hints())
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        saved = self._saved_text()
        version = _version()
        last = self.last_title or "No saved chats yet."
        facts = (
            ("SESSIONS", saved),
            ("LAST CHAT", last),
            ("TABS", "Open chats as tabs."),
        )
        with Vertical(id="launcher-root"):
            with Horizontal(id="topbar"):
                yield Static(
                    f"[{ACCENT}]◆[/]  [b {TEXT}]DIZEL[/]  "
                    f"[{TEXT_3}]·[/]  [{TEXT_3}]LAUNCHER[/]",
                    id="topbar-left",
                )
                right = f"[{TEXT_3}]LOCAL WORKSPACE[/]"
                if version:
                    right += f"  [{TEXT_3}]·[/]  [{TEXT_3}]v{escape(version)}[/]"
                yield Static(right, id="topbar-right")

            with Middle(id="launcher-middle"):
                with Center():
                    with Vertical(id="shell"):
                        with Horizontal(id="deck"):
                            with Vertical(id="brand-panel"):
                                yield Wordmark(id="wordmark", classes="launcher-wordmark")
                                yield Static(
                                    f"[b {ACCENT}]◆[/]  [b {TEXT}]D I Z E L[/]  "
                                    f"[b {ACCENT}]◆[/]",
                                    id="brand-compact",
                                )
                                yield Static(
                                    f"[b {TEXT_3}]LOCAL-FIRST INTELLIGENCE[/]",
                                    id="kicker",
                                )
                                yield Static(
                                    f"[{TEXT_2}]Choose where to work.[/]",
                                    id="tagline",
                                )
                                yield Static(
                                    f"[{TEXT_3}]Same model. Two surfaces.[/]",
                                    id="meta",
                                )
                                with Vertical(id="brand-facts"):
                                    for label, value in facts:
                                        with Horizontal(classes="fact"):
                                            yield Static(label, classes="fact-label")
                                            yield Static(
                                                escape(value), classes="fact-value"
                                            )

                            with Vertical(id="choice-panel"):
                                with Horizontal(id="choice-head"):
                                    yield Static(
                                        f"[b {TEXT_3}]SURFACES[/]", id="choice-label"
                                    )
                                    yield Static(
                                        f"[{TEXT_3}]{len(self.choices)} OPTIONS[/]",
                                        id="choice-meta",
                                    )
                                view = ListView(
                                    *[
                                        ChoiceItem(code, key, title, hint, glyph)
                                        for code, key, title, hint, glyph in self.choices
                                    ]
                                )
                                yield view
                                with Horizontal(id="choice-foot"):
                                    yield Static("LOCAL", classes="foot-label")
                                    yield Static(
                                        "Chats and settings stay on disk.",
                                        classes="foot-value",
                                    )

            with Horizontal(id="launcher-status"):
                yield Static(
                    f"[{SUCCESS}]●[/] [{TEXT_2}]ready to start[/]  "
                    f"[{TEXT_3}]·[/]  [{TEXT_3}]{saved}[/]",
                    id="launcher-status-left",
                )
                yield Static(self._hints(), id="launcher-status-right")

    def on_mount(self) -> None:
        self._apply_layout(self.size.width, self.size.height)
        self.query_one(ListView).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_layout(event.size.width, event.size.height)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ChoiceItem):
            self.exit(return_code=item.code)

    def action_pick(self, index: int) -> None:
        if 0 <= index < len(self.choices):
            self.exit(return_code=self.choices[index][0])

    def action_cancel(self) -> None:
        self.exit(return_code=CANCELLED)


def main() -> int:
    """Run the chooser and return the exit code ``dizel.cmd`` switches on.

    ``App.run()`` returns ``return_value`` — the *result* passed to ``exit()`` — not
    the process code. Reading ``return_code`` off the instance is the only reliable
    way to hand a number back to the batch file.
    """
    app = Launcher()
    try:
        app.run()
    except Exception:
        return CANCELLED
    return app.return_code or CANCELLED


if __name__ == "__main__":
    raise SystemExit(main())
