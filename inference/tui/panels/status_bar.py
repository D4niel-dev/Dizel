from textual import events
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from inference.tui.theme import ACCENT, SPINNER_FRAMES, SUCCESS, TEXT_2, TEXT_3, WARN

_DEFAULT_HINT = "ctrl+k commands   ctrl+t sessions   /help"
_COMPACT_HINT = "^K commands   ^T sessions   /help"


def _version() -> str:
    try:
        from dizel_core.version import __version__

        return __version__
    except Exception:
        return ""


class StatusBar(Container):
    """One quiet line: what the app is doing, and what you can press."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hint = ""
        self._frame = 0
        self._glyph = "●"

    def compose(self) -> ComposeResult:
        yield Static(id="status-left")
        yield Static(id="status-right")

    def on_mount(self) -> None:
        self.update_status()
        self.set_interval(0.12, self._tick)
        for prop in (
            "active_mode",
            "active_tier",
            "active_provider",
            "generation_state",
            "context_tokens",
            "budget_tokens",
        ):
            self.watch(self.app, prop, self.update_status)

    def _tick(self) -> None:
        """One cell of motion while the app is working — the terminal's spinner."""
        from inference.tui.settings_apply import chat_pref

        if self.app.generation_state not in ("STREAMING", "THINKING", "LOADING"):
            if self._glyph != "●":
                self._glyph = "●"
                self.update_status()
            return
        if not chat_pref(self.app, "animations", True):
            if self._glyph != "●":
                self._glyph = "●"
                self.update_status()
            return
        self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
        self._glyph = SPINNER_FRAMES[self._frame]
        self.update_status()

    def set_hint(self, text: str) -> None:
        """Replace the key hints while the composer has something to say."""
        if text == self._hint:
            return
        self._hint = text
        self.update_status()

    def reset_hint(self) -> None:
        self.set_hint("")

    def update_status(self, *_: object) -> None:
        state = self.app.generation_state
        if state == "STREAMING":
            color, label, hint = ACCENT, "generating", "esc stop"
        elif state == "THINKING":
            color, label, hint = WARN, "thinking", "esc stop"
        elif state == "LOADING":
            color, label, hint = WARN, "loading model", "wait"
        else:
            color, label, hint = SUCCESS, "ready", ""

        used = int(self.app.context_tokens)
        context = f"{used:,} tokens"
        budget = int(self.app.budget_tokens)
        if budget:
            context += f" · budget {budget:,}"

        from inference.tui.settings_apply import pref

        glyph = self._glyph if state in ("STREAMING", "THINKING", "LOADING") else "●"
        left = f"[{color}]{glyph}[/] [{TEXT_2}]{label}[/]"
        if hint:
            left += f"   [{WARN}]{hint}[/]"
        if pref(self.app, "show_tokens", True):
            left += f"   [{TEXT_3}]·[/]   [{TEXT_3}]{context}[/]"
        self.query_one("#status-left").update(left)

        width = self.size.width or 80
        if self._hint:
            hints = self._hint
        elif pref(self.app, "show_hints", True):
            hints = _DEFAULT_HINT if width >= 110 else _COMPACT_HINT
        else:
            hints = ""
        if width >= 110:
            version = _version()
            if version:
                if hints:
                    hints += f"   [{TEXT_3}]·[/]   [{TEXT_3}]v{version}[/]"
                else:
                    hints = f"[{TEXT_3}]v{version}[/]"
        self.query_one("#status-right").update(f"[{TEXT_3}]{hints}[/]")

    def on_resize(self, event: events.Resize) -> None:
        self.update_status()
