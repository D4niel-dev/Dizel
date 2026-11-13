import time

from rich.markdown import Markdown

from textual.widgets import Static, Collapsible
from textual.containers import Vertical

from inference.tui.rendering.stream_buffer import StreamBuffer
from inference.tui.rendering.message_block import (
    _parse_think_blocks,
    MessageBlock,
    message_style,
)
from inference.tui.settings_apply import chat_pref
from inference.tui.theme import CODE_THEME, TEXT_3


def _format_thought_clock(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s"


# Seconds between paints per typing mode. "off" rides the ticker itself, so
# text lands as fast as the terminal can draw it.
_CADENCE = {"off": 0.0, "fast": 0.04, "smooth": 0.08, "cinematic": 0.12}
_TICK = 0.02
# Cinematic reveals at roughly reading pace instead of token-burst pace.
_REVEAL_CHARS = 11
_THINK_TAGS = ("<thinking>", "</thinking>", "<think>", "</think>")


def _safe_reveal(text: str, end: int) -> int:
    """Hold back a cut that lands inside a think tag, so it never flashes raw."""
    for tag in _THINK_TAGS:
        for n in range(len(tag) - 1, 0, -1):
            start = end - n
            if start >= 0 and text.startswith(tag, start):
                return start
    return end


class AssistantMessage(Vertical):
    """Streaming assistant turn.

    Tokens arrive many times a second but Markdown is expensive to re-render,
    so updates are coalesced to ~12 fps instead of one parse per token.
    """

    def __init__(self):
        super().__init__(classes="assistant-msg")
        self.stream = StreamBuffer()
        self._finalized = False
        self._dirty = False
        self._pending: str | None = None
        self._finished_at: float | None = None
        self._started_at = time.monotonic()
        self._started_wall = time.time()
        self._thought_text = ""
        self._thought_started_at: float | None = None
        self._thought_ms: float | None = None
        self._thought_timer = None
        self._flush_timer = None
        self._last_paint = 0.0
        self._revealed = 0

    def compose(self):
        self._think_body = Static("", id="asm-think-body")
        self._think_coll = Collapsible(
            self._think_body, title="✻ Thinking", collapsed=False, classes="asm-collapsible"
        )
        yield self._think_coll
        self._stream = Static("", id="asm-stream")
        yield self._stream

    def on_mount(self) -> None:
        if self._pending is not None:
            text, self._pending = self._pending, None
            self._paint(text)
        self._finalize_thought()
        self._flush_timer = self.set_interval(_TICK, self._flush)

    def on_unmount(self) -> None:
        if self._thought_timer is not None:
            self._thought_timer.stop()
            self._thought_timer = None
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

    def _tick_thought(self) -> None:
        if (
            self._thought_ms is not None
            or self._thought_started_at is None
            or not hasattr(self, "_think_coll")
        ):
            return
        elapsed = time.monotonic() - self._thought_started_at
        self._think_coll.title = f"✻ Thinking for {_format_thought_clock(elapsed)}"

    def _end_thought(self) -> None:
        if self._thought_started_at is None or self._thought_ms is not None:
            return
        self._thought_ms = max(0.0, time.monotonic() - self._thought_started_at)
        if hasattr(self, "_think_coll"):
            self._think_coll.title = (
                f"✻ Thought for {_format_thought_clock(self._thought_ms)}"
            )
        if self._thought_timer is not None:
            self._thought_timer.stop()
            self._thought_timer = None

    def update_stream(self) -> None:
        if not self._finalized:
            self._dirty = True

    def _flush(self) -> None:
        if self._finalized:
            return
        raw = self.stream.raw_text
        mode = chat_pref(self.app, "typing_animation", "smooth")
        now = time.monotonic()
        period = _CADENCE.get(mode, 0.08)
        if now - self._last_paint < period:
            return

        if mode == "cinematic":
            # Reveal at reading pace; the buffer can be far ahead of the paint.
            if self._revealed >= len(raw):
                return
            end = _safe_reveal(raw, min(len(raw), self._revealed + _REVEAL_CHARS))
            self._revealed = end
            self._last_paint = now
            self._paint(raw[:end])
            return

        if not self._dirty:
            return
        self._dirty = False
        self._last_paint = now
        self._revealed = len(raw)
        self._paint(raw)

    def _paint(self, text: str) -> None:
        # finalize() can land in the same tick as the mount, before compose()
        # has created the children. Hold the text until on_mount instead of
        # raising AttributeError.
        if not hasattr(self, "_stream"):
            self._pending = text
            return

        main_text, thought_text = _parse_think_blocks(text)
        self._thought_text = thought_text

        if thought_text:
            self._think_body.update(Markdown(thought_text, code_theme=CODE_THEME))
            self._think_coll.display = True
            # OpenCode reports reasoning as a compact line, not a full panel.
            # While streaming the body stays open; finalize() folds it.
            if self._thought_started_at is None:
                self._thought_started_at = time.monotonic()
                self._thought_timer = self.set_interval(1.0, self._tick_thought)
            if self._thought_ms is None:
                if main_text:
                    self._end_thought()
                else:
                    elapsed = time.monotonic() - self._thought_started_at
                    self._think_coll.title = (
                        f"✻ Thinking for {_format_thought_clock(elapsed)}"
                    )
        else:
            self._think_coll.display = False

        self._stream.update(
            MessageBlock(
                "ASSISTANT",
                main_text or "…",
                **message_style(self.app, timestamp=self._started_wall),
            )
        )

    def _finalize_thought(self) -> None:
        if self._finished_at is None or not hasattr(self, "_think_coll"):
            return
        if not self._thought_text:
            return
        self._end_thought()
        if self._thought_ms is not None:
            self._think_coll.title = (
                f"✻ Thought for {_format_thought_clock(self._thought_ms)}"
            )
        self._think_coll.collapsed = True

    def finalize(self, full_text: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._finished_at = time.monotonic()
        self._dirty = False
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._paint(full_text)
        self._finalize_thought()

        # The desktop's meta line: which model answered, how long it took, how
        # big the reply is, and the throughput that says how the model is doing.
        if full_text.strip():
            from rich.text import Text as _Text

            from inference.tui.labels import display_model

            elapsed = time.monotonic() - self._started_at
            parts = []
            if chat_pref(self.app, "show_model_tag"):
                model = display_model(
                    self.app.active_model or "",
                    getattr(self.app, "response_label", self.app.active_tier) or "",
                )
                if model and model != "no model":
                    parts.append(model)
            if elapsed > 0.2:
                parts.append(f"{elapsed:.1f}s")
            parts.append(f"{len(full_text):,} chars")
            # Estimated at 4 chars/token — the same ratio the context meter uses.
            if chat_pref(self.app, "show_speed") and elapsed > 0.5:
                per_second = round(len(full_text) / 4 / elapsed)
                if per_second > 0:
                    parts.append(f"~{per_second} tok/s")
            meta = _Text.assemble(
                ("◷ ", TEXT_3), (f"{' · '.join(parts)}", TEXT_3)
            )
            self.mount(Static(meta, classes="asm-meta"))
