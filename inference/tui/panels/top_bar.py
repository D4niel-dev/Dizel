"""Title block: the chat's own name, with its context cost on the right.

The terminal twin of OpenCode's header — a raised block carrying ``# title`` and
the token spend, instead of a row of chips. The model and tier moved into the
composer footer, so repeating them here would be the same sentence twice.
"""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from inference.tui.theme import TEXT, TEXT_3


class TopBar(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static(id="top-title")
        yield Static(id="top-stats")

    def on_mount(self) -> None:
        self.refresh_bar()
        for prop in ("context_tokens", "budget_tokens", "active_model", "session_id"):
            self.watch(self.app, prop, self.refresh_bar)

    def _title(self) -> str:
        for message in getattr(self.app.session_bridge, "current_messages", []):
            if message.get("role") == "user" and message.get("content"):
                text = " ".join(str(message["content"]).split())
                return text[:58] + ("…" if len(text) > 58 else "")
        return "New chat"

    def refresh_bar(self, *_: object) -> None:
        app = self.app
        try:
            ctx = int(app.chat_bridge.manager.effective_ctx_len or 2048)
        except Exception:
            ctx = 2048
        used = int(getattr(app, "context_tokens", 0) or 0)
        percent = round(used * 100 / ctx) if ctx else 0

        try:
            self.query_one("#top-title", Static).update(
                f"[b {TEXT}]# {escape(self._title())}[/]"
            )
            self.query_one("#top-stats", Static).update(
                f"[{TEXT_3}]{used:,}  {percent}% of {ctx:,}[/]"
            )
        except Exception:
            pass
