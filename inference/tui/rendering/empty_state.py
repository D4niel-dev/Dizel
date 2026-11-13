import random

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Static
from rich.markup import escape

from inference.tui.labels import display_model
from inference.tui.rendering.welcome_copy import (
    day_part,
    fill_welcome,
    greeting,
    sample_welcome,
)
from inference.tui.settings_apply import chat_pref
from inference.tui.theme import ACCENT, SURFACE_HI, TEXT, TEXT_2, TEXT_3


# ── Wordmark ─────────────────────────────────────────────────────────────
# OpenCode opens with a big wordmark; without one an empty chat is a void with
# small text in it. pyfiglet renders the shape, then we add the light: a vertical
# gradient and a highlight that drifts across it.

_FONT = "ansi_shadow"

# Hand-drawn fallback for when pyfiglet is not installed.
_GLYPHS = {
    "D": ("███ ", "█  █", "█  █", "█  █", "███ "),
    "I": ("███ ", " █  ", " █  ", " █  ", "███ "),
    "Z": ("████", "   █", "  █ ", " █  ", "████"),
    "E": ("████", "█   ", "███ ", "█   ", "████"),
    "L": ("█   ", "█   ", "█   ", "█   ", "████"),
}


def _fallback(word: str) -> list[str]:
    rows = [""] * 5
    for letter in word:
        glyph = _GLYPHS.get(letter)
        if not glyph:
            continue
        for i in range(5):
            rows[i] += glyph[i] + " "
    return rows


def _shape(word: str = "DIZEL") -> list[str]:
    try:
        import pyfiglet

        rendered = pyfiglet.figlet_format(word, font=_FONT)
        rows = [line.rstrip() for line in rendered.splitlines()]
        while rows and not rows[-1].strip():
            rows.pop()
        return rows or _fallback(word)
    except Exception:
        return _fallback(word)


_SHAPE = _shape()

# Light from above: bright at the top row, deeper at the base.
_TOP = (0x6F, 0xB4, 0xFF)
_BOTTOM = (0x0A, 0x5F, 0xC8)
_HIGHLIGHT = (0xD8, 0xEC, 0xFF)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % rgb


def _wordmark(phase: float = -1.0) -> str:
    """The wordmark as markup, with a gradient and an optional moving highlight."""
    height = max(1, len(_SHAPE))
    width = max(len(row) for row in _SHAPE)

    # The sheen crosses the whole wordmark inside the active window (2.2 s of a
    # 6 s cycle). Deriving it from raw elapsed seconds meant it was still off the
    # left edge when the window closed, so nothing ever lit up.
    if phase < 0:
        beam = -999.0
    else:
        span = 2.2
        progress = (phase % 6.0) / span
        beam = -6.0 + progress * (width + 12)

    out = []
    for y, row in enumerate(_SHAPE):
        base = _mix(_TOP, _BOTTOM, y / max(1, height - 1))
        # Group consecutive cells that share a colour into one span. Tagging
        # every character individually meant ~200 tags per frame, at 11 fps.
        line: list[str] = []
        run_colour = None
        run: list[str] = []

        def flush() -> None:
            if run:
                line.append(f"[{_hex(run_colour)}]{''.join(run)}[/]")
                run.clear()

        for x, char in enumerate(row):
            colour = base
            distance = abs(x - beam)
            if char != " " and distance < 6:
                colour = _mix(base, _HIGHLIGHT, 0.75 * (1 - distance / 6))
            if char == " ":
                flush()
                line.append(" ")
                run_colour = None
                continue
            if colour != run_colour:
                flush()
                run_colour = colour
            run.append(char)
        flush()
        out.append("".join(line))
    return "\n".join(out)


class Wordmark(Static):
    """The logo, with a slow sheen drifting across it."""

    def on_mount(self) -> None:
        self._phase = 0.0
        self._sheen_on = True
        self._timer = self.set_interval(0.09, self._tick)
        self.update(_wordmark(-1))

    def _tick(self) -> None:
        # The shared animations switch stills the sheen; the gradient stays.
        on = bool(chat_pref(self.app, "animations"))
        if on != self._sheen_on:
            self._sheen_on = on
            self.update(_wordmark(self._phase if on else -1))
        if not on:
            return
        self._phase += 0.09
        # One pass every ~6s, then idle for the rest of the cycle.
        if self._phase % 6.0 < 2.2:
            self.update(_wordmark(self._phase))
        elif self._phase % 6.0 < 2.35:
            self.update(_wordmark(-1))

    def on_unmount(self) -> None:
        try:
            self._timer.stop()
        except Exception:
            pass


SUGGESTION_COUNT = 3

# A starter must stand on its own: clicking fills the composer, so the prompt
# cannot depend on something the user has not pasted yet.
SUGGESTION_POOL = (
    ("Reverse a string in Python", "reverse a string in python"),
    ("Explain how attention works", "explain how attention works, briefly"),
    ("Plan a small desktop app", "make a plan for a small desktop app"),
    ("Lists vs tuples in Python", "explain the difference between lists and tuples in python"),
    ("Write a binary search", "write a binary search function in python"),
    ("Debug a slow Python script", "help me debug why a python script is slow"),
    ("Summarize the water cycle", "summarize the water cycle in five bullets"),
    ("SQL vs NoSQL for a small app", "compare sql and nosql for a small app"),
    ("Draft a project README", "draft a concise readme for a small python project"),
    ("Git rebase vs merge", "explain git rebase vs merge with a short example"),
    ("Write an email regex", "write a regex for email addresses and explain it"),
    ("Plan a week of Python", "plan a week of learning python for a beginner"),
    ("Outline a local-AI post", "outline a blog post about local ai"),
    ("Write unit tests", "write unit tests for a python function that parses dates"),
    ("Explain Big O simply", "explain big o notation simply with examples"),
    ("Design a small REST API", "design a simple rest api for a todo app"),
    ("Sketch a todo-app model", "sketch a data model for a todo app"),
    ("Write a polite decline email", "write a polite email declining a meeting"),
    ("Build a launch checklist", "make a checklist for a small product launch"),
    ("Fix a Python traceback", "explain common python traceback errors and how to fix them"),
    ("Study plan for two weeks", "make a study plan for two weeks of python"),
    ("Write a haiku about the moon", "write a haiku about the moon"),
    ("Explain quantum entanglement", "explain quantum entanglement in simple terms"),
    ("Plan a morning routine", "give me a simple morning routine"),
)


class SuggestionRow(Horizontal):
    """A full-width starter: glyph, prompt, and the enter affordance."""

    class Chosen(Message):
        def __init__(self, prompt: str) -> None:
            super().__init__()
            self.prompt = prompt

    def __init__(self, label: str, prompt: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.text_label = label
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        yield Static("›", classes="suggestion-glyph")
        yield Static(self.text_label, classes="suggestion-label")
        yield Static("↵", classes="suggestion-arrow")

    def on_click(self, event: events.Click) -> None:
        self.post_message(self.Chosen(self.prompt))
        event.stop()


def _configured_username() -> str:
    try:
        from dizel_core.config_manager import ConfigManager

        profile = ConfigManager.load().get("user_profile", {})
        name = str(profile.get("username") or "").strip()
        return name or "User"
    except Exception:
        return "User"


class EmptyStateBlock(Container):
    """The welcome: greeting, prompt and three starters.

    No logo. The header already carries the wordmark, and the desktop welcome
    opens with a greeting too — two wordmarks in one window read as a template.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._part = day_part()
        self._headline, self._subline = sample_welcome(self._part)
        self._username = _configured_username()
        self._suggestions = random.sample(SUGGESTION_POOL, k=SUGGESTION_COUNT)

    def on_mount(self) -> None:
        # The model is usually resolved *after* this widget is composed, so the
        # first paint says "no model". Watch it and rewrite the line.
        try:
            for prop in ("active_model", "active_tier", "active_mode"):
                self.watch(self.app, prop, self._refresh_context_line)
        except Exception:
            pass
        self.set_interval(60, self._refresh_copy)
        self.apply_suggestions()
        self.call_after_refresh(self._apply_density)

    def apply_suggestions(self) -> None:
        """The shared starter-suggestions switch, live even while the welcome is up."""
        show = bool(chat_pref(self.app, "show_suggestions"))
        for row in self.query(".empty-suggestion"):
            row.display = show

    def on_resize(self, event: events.Resize) -> None:
        self._apply_density()

    def _apply_density(self) -> None:
        """The welcome is a composition; at short heights only its essentials fit."""
        height = self.app.size.height
        # The composer centres with the welcome now, so the density steps arrive
        # one composer-height (6 rows) sooner to keep the whole column floating.
        self.set_class(height < 40, "-compact")
        self.set_class(height < 32, "-tiny")
        self.set_class((self.size.width or 96) < 64, "-narrow")

    def _refresh_copy(self) -> None:
        part = day_part()
        if part == self._part:
            return
        self._part = part
        self._headline, self._subline = sample_welcome(part)
        try:
            self.query_one(".empty-greeting", Static).update(
                f"[b {TEXT}]{escape(self._greeting_line())}[/]"
            )
            self.query_one(".empty-prompt", Static).update(
                f"[{TEXT_3}]{escape(self._subline)}[/]"
            )
        except Exception:
            pass

    def _greeting_line(self) -> str:
        return fill_welcome(
            self._headline,
            greeting_text=greeting(),
            name=self._username,
        )

    def _refresh_context_line(self, *_: object) -> None:
        try:
            self.query_one(".empty-meta", Static).update(
                f"[{TEXT_3}]{self._context_line()}[/]"
            )
        except Exception:
            pass

    def _context_line(self) -> str:
        """What is actually about to answer — the desktop puts this on the chip."""
        try:
            app = self.app
            tier = getattr(app, "response_label", app.active_tier) or "Fast"
            model = display_model(app.active_model or "no model", tier)
            return f"{model} · {tier}"
        except Exception:
            return ""

    def compose(self) -> ComposeResult:
        keycap = f"[{TEXT_2} on {SURFACE_HI}]"
        with Vertical(id="empty-content"):
            with Vertical(id="empty-column"):
                yield Static(
                    f"[bold {ACCENT}]◆[/] [{TEXT_3}]LOCAL WORKSPACE[/]",
                    classes="empty-kicker",
                )
                yield Wordmark(classes="empty-logo")
                yield Static(
                    f"[b {TEXT}]{escape(self._greeting_line())}[/]",
                    classes="empty-greeting",
                )
                yield Static(f"[{TEXT_3}]{self._context_line()}[/]", classes="empty-meta")
                yield Static(
                    f"[{TEXT_3}]{escape(self._subline)}[/]",
                    classes="empty-prompt",
                )
                for label, prompt in self._suggestions:
                    yield SuggestionRow(label, prompt, classes="empty-suggestion")
                yield Static(
                    f"{keycap} enter [/] [{TEXT_3}]send[/]   "
                    f"{keycap} ctrl+k [/] [{TEXT_3}]palette[/]   "
                    f"{keycap} ctrl+t [/] [{TEXT_3}]sessions[/]",
                    classes="empty-hint",
                )

    @on(SuggestionRow.Chosen)
    def pick_suggestion(self, event: SuggestionRow.Chosen) -> None:
        prompt = event.prompt
        if not prompt:
            return
        try:
            inp = self.app.query_one("#prompt-input", Input)
        except Exception:
            return
        inp.value = prompt
        inp.cursor_position = len(prompt)
        inp.focus()
