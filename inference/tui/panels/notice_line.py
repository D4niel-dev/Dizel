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

_NOTICE_TTL = 20.0


class NoticeLine(Static):
    _timer = None

    def show(self, message: str, level: str = "info") -> None:
        color = ACCENT if level == "ok" else DANGER if level == "error" else TEXT_3
        self.update(Content.assemble((str(message), TEXT_2)))
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
