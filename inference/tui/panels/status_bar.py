from textual import events
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from inference.tui.theme import (
    ACCENT,
    ACCENT_INK,
    SPINNER_FRAMES,
    SUCCESS,
    TEXT_2,
    TEXT_3,
    VIOLET,
    WARN,
)

_DEFAULT_HINT = "ctrl+k commands   ctrl+t sessions   /help"
_COMPACT_HINT = "^K commands   ^T sessions   /help"
_TIGHT_HINT = "^K  ^T  /help"


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
        self._trust_tick = 0
        self._trust_persistent = False
        self._trust_revision_seen = -1

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
            "last_reply",
            "debug_mode",
            "update_ready",
            "trust_revision",
            "session_id",
        ):
            self.watch(self.app, prop, self.update_status)

    def _tick(self) -> None:
        """One cell of motion while the app is working — the terminal's spinner."""
        from inference.tui.settings_apply import chat_pref

        self._trust_tick += 1
        trust_due = self._trust_tick % 16 == 0
        if trust_due:
            self._refresh_trust()
        if self.app.generation_state not in ("STREAMING", "THINKING", "LOADING"):
            if self._glyph != "●" or trust_due:
                self._glyph = "●"
                self.update_status()
            return
        if not chat_pref(self.app, "animations", True):
            if self._glyph != "●" or trust_due:
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

    def _refresh_trust(self) -> None:
        from dizel_core import tool_trust

        self._trust_persistent = tool_trust.persistent_enabled()

    def _trust_marker(self) -> str:
        active = self._trust_persistent or (
            bool(self.app.session_id)
            and self.app.session_id in self.app._trusted_sessions
        )
        return f"[bold {WARN}]writes:auto[/]   " if active else ""

    def _reply_metrics(self, width: int) -> str:
        reply = getattr(self.app, "last_reply", None)
        if not reply:
            return ""
        seconds, chars, per_second = reply
        if seconds <= 0 or chars <= 0:
            return ""

        from inference.tui.settings_apply import pref

        speed = per_second > 0 and pref(self.app, "show_speed", True)
        if width >= 116:
            parts = [f"{seconds:.1f}s", f"{chars:,} chars"]
            if speed:
                parts.append(f"~{per_second} tok/s")
        else:
            parts = [f"{seconds:.1f}s", f"{chars:,}ch"]
            if speed:
                parts.append(f"~{per_second}t/s")
        return f"[{TEXT_3}]◷[/] [{TEXT_2}]{' · '.join(parts)}[/]"

    def update_status(self, *_: object) -> None:
        # Disk reads stay off the repaint hot path: refresh only on a trust
        # revision bump or the ~2s poll, never on every spinner tick.
        revision = int(self.app.trust_revision)
        if revision != self._trust_revision_seen:
            self._trust_revision_seen = revision
            self._refresh_trust()
        state = self.app.generation_state
        if state == "STREAMING":
            color, label, hint = ACCENT, "generating", "esc stop"
        elif state == "THINKING":
            color, label, hint = WARN, "thinking", "esc stop"
        elif state == "LOADING":
            color, label, hint = WARN, "loading model", "wait"
        else:
            color, label, hint = SUCCESS, "ready", ""

        width = self.size.width or 80
        metrics = self._reply_metrics(width) if state == "IDLE" else ""

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
        if metrics:
            left += f"   [{TEXT_3}]·[/]   {metrics}"
        show_context = pref(self.app, "show_tokens", True) and not (
            metrics and width < 70
        )
        if show_context:
            left += f"   [{TEXT_3}]·[/]   [{TEXT_3}]{context}[/]"
        self.query_one("#status-left").update(left)

        tight = bool(metrics) and width < 90
        if self._hint:
            hints = self._hint
        elif pref(self.app, "show_hints", True):
            if width >= 110:
                hints = _DEFAULT_HINT
            elif tight:
                hints = _TIGHT_HINT
            else:
                hints = _COMPACT_HINT
        else:
            hints = ""
        if width >= 110:
            version = _version()
            if version:
                if hints:
                    hints += f"   [{TEXT_3}]·[/]   [{TEXT_3}]v{version}[/]"
                else:
                    hints = f"[{TEXT_3}]v{version}[/]"
        update = str(getattr(self.app, "update_ready", "") or "")
        badge = ""
        if update:
            label = (
                f"UPDATE v{update}"
                if width >= 104
                else f"↑ v{update}" if width >= 68 else "↑"
            )
            badge = f"[bold {ACCENT_INK} on {ACCENT}] {label} [/]   "
        debug = (
            f"[bold {VIOLET}]DEBUG[/]   "
            if getattr(self.app, "debug_mode", False)
            else ""
        )
        self.query_one("#status-right").update(
            f"{badge}{debug}{self._trust_marker()}[{TEXT_3}]{hints}[/]"
        )

    def on_resize(self, event: events.Resize) -> None:
        self.update_status()
