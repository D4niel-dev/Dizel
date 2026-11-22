import math
import random
import re
import time

from rich.markdown import Markdown

from textual.color import Color
from textual.content import Content
from textual.widgets import Static, Collapsible
from textual.containers import Vertical

from inference.tui.rendering.stream_buffer import StreamBuffer
from inference.tui.rendering.message_block import (
    _parse_think_blocks,
    MessageBlock,
    message_style,
    reasoning_summary,
)
from inference.tui.settings_apply import chat_pref
from inference.tui.theme import ACCENT, CODE_THEME, TEXT_2, TEXT_3, WARN


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
_THINK_CLOSED = re.compile(r"</(?:think|thinking)>")
_THINK_TICK = 0.08

# The GUI's reasoning copy. Rotating the line is what keeps a long pass from
# looking frozen when the model has no streamed reasoning of its own.
THOUGHTS = (
    "Reading the question…",
    "Gathering context…",
    "Sketching an outline…",
    "Weighing approaches…",
    "Drafting a first pass…",
    "Tightening the wording…",
    "Still working through it…",
    "Checking assumptions…",
    "Comparing trade-offs…",
    "Testing the edge cases…",
    "Building the answer…",
    "Rereading the details…",
    "Working the hard part…",
    "Taking the time it needs…",
    "Finding the clearest path…",
    "Sorting signal from noise…",
    "Double-checking the logic…",
    "Choosing the right depth…",
    "Turning it into a plan…",
    "Keeping it grounded…",
)
_DOT_FRAMES = ("·  ", "·· ", "···", " ··", "  ·")
_WAIT_TICK = 0.18
_WAIT_PHRASE_TICKS = 13  # 13 × 0.18s ≈ the GUI's 2.4s phrase clock.
_THOUGHT_PHRASE_SECONDS = 3.2
_PULSE_SECONDS = 2.8
_PULSE_DIM = "#0057b8"


def _mix_hex(start: str, end: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    left = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    right = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(
        round(a + (b - a) * amount) for a, b in zip(left, right)
    )


class ThinkingCollapsible(Collapsible):
    """The GUI reasoning block, with a terminal-native marker and label.

    Textual exposes the collapsed/expanded symbols on the title widget only;
    keeping that touch here means the rest of the stream never reaches into it.
    """

    def set_header(
        self,
        label: Content | str,
        expanded: str,
        collapsed: str,
        *,
        color: str | None = None,
    ) -> None:
        plain = label.plain if isinstance(label, Content) else label
        if self.title != plain:
            self.title = plain
        title_widget = self._title
        title_widget.label = label
        title_widget.expanded_symbol = expanded
        title_widget.collapsed_symbol = collapsed
        if color is not None:
            title_widget.styles.color = Color.parse(color)
        title_widget._update_label()

    @property
    def marker(self) -> str:
        title_widget = self._title
        return (
            title_widget.collapsed_symbol
            if self.collapsed
            else title_widget.expanded_symbol
        )


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
        self._thought_title: str | None = None
        self._thought_started_at: float | None = None
        self._thought_ms: float | None = None
        self._thought_timer = None
        self._think_phrase_idx = 0
        self._think_phrase_at = 0.0
        self._think_clock = ""
        self._wait_timer = None
        self._wait_frame = 0
        self._wait_ticks = 0
        self._wait_phrase_idx = 0
        self._wait_started = False
        self._last_thought_body: str | None = None
        self._flush_timer = None
        self._last_paint = 0.0
        self._revealed = 0

    def compose(self):
        self._wait = Static("", id="asm-wait", classes="asm-wait")
        yield self._wait
        self._think_body = Static("", id="asm-think-body")
        self._think_coll = ThinkingCollapsible(
            self._think_body,
            title="Thinking",
            collapsed=False,
            expanded_symbol="✻",
            collapsed_symbol="✻",
            classes="asm-collapsible",
        )
        yield self._think_coll
        self._stream = Static("", id="asm-stream")
        yield self._stream

    def on_mount(self) -> None:
        # The GUI's generating row carries the first-token wait; the reasoning
        # block takes over once the model actually emits a thought.
        self._think_coll.display = False
        self._stream.display = False
        self._wait.display = True
        self._start_wait()
        if self._pending is not None:
            text, self._pending = self._pending, None
            self._paint(text)
        self._finalize_thought()
        self._flush_timer = self.set_interval(_TICK, self._flush)

    def on_unmount(self) -> None:
        for timer in (self._thought_timer, self._wait_timer, self._flush_timer):
            if timer is not None:
                timer.stop()
        self._thought_timer = None
        self._wait_timer = None
        self._flush_timer = None

    def _thought_end_label(self) -> str:
        duration = _format_thought_clock(self._thought_ms or 0.0)
        if self._thought_title:
            return f"Thought: {self._thought_title} · {duration}"
        return f"Thought for {duration}"

    def _tick_thought(self) -> None:
        if (
            self._thought_ms is not None
            or self._thought_started_at is None
            or not hasattr(self, "_think_coll")
        ):
            return
        now = time.monotonic()
        elapsed = now - self._thought_started_at
        animations = chat_pref(self.app, "animations", True)
        phrase_changed = False
        if now >= self._think_phrase_at:
            self._think_phrase_idx = (self._think_phrase_idx + 1) % len(THOUGHTS)
            self._think_phrase_at = now + _THOUGHT_PHRASE_SECONDS
            phrase_changed = True
        clock = _format_thought_clock(elapsed)
        if animations:
            wave = 0.5 + 0.5 * math.sin(
                (elapsed / _PULSE_SECONDS) * math.tau
            )
            pulse = _mix_hex(_PULSE_DIM, ACCENT, 0.62 + 0.38 * wave)
        else:
            pulse = ACCENT
        if animations or phrase_changed or clock != self._think_clock:
            phrase = THOUGHTS[self._think_phrase_idx]
            label = Content.assemble(
                (phrase, pulse),
                ("  ", TEXT_3),
                (clock, TEXT_3),
            )
            self._think_coll.set_header(label, "✻", "✻", color=pulse)
        self._think_clock = clock

    def _start_thought(self) -> None:
        if self._thought_started_at is not None:
            return
        now = time.monotonic()
        self._thought_started_at = now
        self._think_phrase_idx = random.randrange(len(THOUGHTS))
        self._think_phrase_at = now + _THOUGHT_PHRASE_SECONDS
        self._think_clock = ""
        self._think_coll.display = True
        self._think_coll.collapsed = False
        self._think_coll.add_class("-thinking")
        self._tick_thought()
        self._thought_timer = self.set_interval(_THINK_TICK, self._tick_thought)

    def _end_thought(self) -> None:
        if self._thought_started_at is None or self._thought_ms is not None:
            return
        self._thought_ms = max(0.0, time.monotonic() - self._thought_started_at)
        self._think_coll.remove_class("-thinking")
        self._think_coll.set_header(self._thought_end_label(), "-", "+", color=WARN)
        if self._thought_timer is not None:
            self._thought_timer.stop()
            self._thought_timer = None

    def _paint_wait(self) -> None:
        animations = chat_pref(self.app, "animations", True)
        dots = _DOT_FRAMES[self._wait_frame] if animations else "···"
        phrase = THOUGHTS[self._wait_phrase_idx]
        self._wait.update(
            Content.assemble(
                (dots, ACCENT if animations else TEXT_3),
                ("  ", TEXT_3),
                (phrase, TEXT_2),
            )
        )

    def _tick_wait(self) -> None:
        if self._finalized or not self._wait.display:
            self._stop_wait()
            return
        if chat_pref(self.app, "animations", True):
            self._wait_frame = (self._wait_frame + 1) % len(_DOT_FRAMES)
        self._wait_ticks += 1
        if self._wait_ticks >= _WAIT_PHRASE_TICKS:
            self._wait_ticks = 0
            self._wait_phrase_idx = (self._wait_phrase_idx + 1) % len(THOUGHTS)
        self._paint_wait()

    def _start_wait(self) -> None:
        if self._wait_timer is not None:
            return
        if not self._wait_started:
            self._wait_started = True
            self._wait_phrase_idx = random.randrange(len(THOUGHTS))
            self._wait_frame = 0
            self._wait_ticks = 0
            self._paint_wait()
        self._wait_timer = self.set_interval(_WAIT_TICK, self._tick_wait)

    def _stop_wait(self) -> None:
        try:
            if self._wait.is_mounted:
                self._wait.display = False
        except Exception:
            pass
        if self._wait_timer is not None:
            self._wait_timer.stop()
            self._wait_timer = None

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
        # Markdown is parsed whole on every paint, so a long answer costs more
        # per frame than a short one. Stretch the interval with length instead of
        # holding 12 fps into a 20k-character reply — the reader cannot tell the
        # difference, the CPU can.
        if mode != "cinematic":
            period += min(0.20, len(raw) / 4000 * 0.03)
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
        self._thought_title, thought_body = reasoning_summary(thought_text)

        if thought_text:
            if thought_body != self._last_thought_body:
                self._think_body.update(
                    Markdown(thought_body, code_theme=CODE_THEME, style=TEXT_3)
                )
                self._last_thought_body = thought_body
            self._think_coll.display = True
            # The GUI keeps the reasoning body open while the pass is live;
            # finalize() folds it to the one-line Thought header.
            if self._thought_started_at is None:
                self._start_thought()
            if self._thought_ms is None and (
                main_text or _THINK_CLOSED.search(text)
            ):
                self._end_thought()
        else:
            self._think_coll.display = False

        # A model without a reasoning channel still gets the GUI's generating
        # row until its first answer token. Once the answer starts, the row
        # leaves so the transcript stays free of a placeholder line.
        waiting = not self._finalized and not main_text and not thought_text
        self._wait.display = waiting
        if waiting:
            self._start_wait()
        else:
            self._stop_wait()

        self._stream.display = bool(main_text)
        if main_text:
            self._stream.update(
                MessageBlock(
                    "ASSISTANT",
                    main_text,
                    **message_style(self.app, timestamp=self._started_wall),
                )
            )
        else:
            self._stream.update("")

    def _finalize_thought(self) -> None:
        if self._finished_at is None or not hasattr(self, "_think_coll"):
            return
        if not self._thought_text:
            return
        self._end_thought()
        self._think_coll.collapsed = True

    def stop(self) -> None:
        """Stop a failed/interrupted turn without leaving animation running."""
        if self._finalized:
            return
        self._finalized = True
        self._end_thought()
        self._stop_wait()
        for timer in (self._thought_timer, self._flush_timer):
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        self._thought_timer = None
        self._flush_timer = None

    def finalize(self, full_text: str, record_metrics: bool = True) -> None:
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

        # Reply statistics live in the status line, not under every message:
        # elapsed time, visible answer size, and the estimated throughput.
        if record_metrics:
            answer, _thought = _parse_think_blocks(full_text)
            answer = answer.strip()
            if answer:
                elapsed = max(0.1, time.monotonic() - self._started_at)
                per_second = round(len(answer) / 4 / elapsed) if elapsed > 0.5 else 0
                self.app.record_reply_metrics(elapsed, len(answer), per_second)
