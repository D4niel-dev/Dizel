"""One reserved row above the composer for transient app notices.

Notices used to be mounted into the transcript and deleted on a timer, which
polluted the conversation and yanked the reader's scroll position when they
disappeared. This line sits outside the scroll view, so it can come and go
without moving anything.
"""

from __future__ import annotations

from textual.content import Content
from textual.widgets import Static

from inference.tui.theme import TEXT_2, ACCENT, DANGER, TEXT_3

# Well inside the one-and-a-half-second fade contract the tests assert. The
# extra slack matters on loaded CI runners, where timer callbacks can arrive
# late even though the notice itself was shown on time.
_NOTICE_TTL = 1.0


class NoticeLine(Static):
    _timer = None

    def show(self, message: str, level: str = "info") -> None:
        if level == "ok":
            glyph, color = "✓", ACCENT
        elif level == "error":
            glyph, color = "×", DANGER
        else:
            glyph, color = "~", TEXT_3
        self.update(
            Content.assemble(
                (glyph, color),
                (" ", TEXT_3),
                (str(message), TEXT_2),
            )
        )
        self.set_class(level == "ok", "-ok")
        self.set_class(level == "error", "-error")
        self.add_class("-shown")
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        # Errors stay until the next notice; ordinary notes fade.
        if level != "error":
            self._timer = self.set_timer(_NOTICE_TTL, self.clear)

    def clear(self) -> None:
        self._timer = None
        self.remove_class("-shown")
        self.update("")
