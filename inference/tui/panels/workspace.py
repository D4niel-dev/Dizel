from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual import on

import time

from inference.tui.rendering.assistant_message import AssistantMessage
from inference.tui.rendering.message_block import MessageBlock, message_style
from inference.tui.rendering.tool_block import ToolBlock
from inference.tui.rendering.status_block import StatusBlock
from inference.tui.bridge.chat_bridge import ChatBridge
from inference.tui.bridge.tool_bridge import ToolBridge
from inference.tui.rendering.empty_state import EmptyStateBlock
from inference.tui.settings_apply import chat_pref


class WorkspacePanel(VerticalScroll):
    """The transcript.

    The view follows new output only while the reader is already at the
    bottom. Scrolling up mid-generation should let you read, not fight a
    ``scroll_end`` on every token.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_message: AssistantMessage | None = None

    def on_mount(self) -> None:
        self._set_empty_mode(bool(self.query(EmptyStateBlock)))

    def _set_empty_mode(self, empty: bool) -> None:
        # The desktop composer sits mid-screen with the welcome on an empty
        # chat, then docks to the bottom once a conversation starts. This class
        # drives the same placement in the terminal.
        try:
            self.app.query_one("#workspace-container").set_class(empty, "-empty")
        except Exception:
            pass

    # ── Scroll ───────────────────────────────────────────────────────
    def _follow(self) -> None:
        if not chat_pref(self.app, "auto_scroll"):
            return
        self.call_after_refresh(self.scroll_end, animate=False)

    def _follow_if_pinned(self) -> None:
        if self.is_vertical_scroll_end:
            self._follow()

    # ── Output ───────────────────────────────────────────────────────
    def add_system_message(self, text: str, level: str = "info") -> None:
        """A command result or error that belongs in the conversation.

        Deliberately does *not* dismiss the welcome screen: a command result is
        not a conversation. It does dock the composer, though — a long result
        must scroll in the transcript instead of growing the centered column
        until the composer falls off screen.
        """
        self._set_empty_mode(False)
        self.mount(Static(StatusBlock(text, level), classes="system-line"))
        self._follow()

    def mount_system_line(self, renderable, ttl: float = 0.0) -> Static:
        """Mount a line in the transcript. Persistent unless a ttl is given."""
        widget = Static(renderable, classes="system-line")
        self.mount(widget)
        self._follow()
        if ttl and ttl > 0:
            self.set_timer(ttl, lambda: self._remove_system_line(widget))
        return widget

    def _remove_system_line(self, widget: Static) -> None:
        # Removing a line never moves the reader: the transcript only shrinks.
        try:
            if widget.is_mounted:
                widget.remove()
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield EmptyStateBlock()

    def _remove_empty_state(self) -> None:
        had_empty = bool(self.query(EmptyStateBlock))
        for child in self.query(EmptyStateBlock):
            child.remove()
        if had_empty:
            # Back to a normal transcript: the composer returns to the bottom.
            self._set_empty_mode(False)

    def add_user_message(self, text: str) -> None:
        self._remove_empty_state()
        self.mount(
            Static(MessageBlock("USER", text, **message_style(self.app, timestamp=time.time())))
        )
        self._follow()

    def start_assistant_message(self) -> None:
        self._remove_empty_state()
        self.current_message = AssistantMessage()
        self.mount(self.current_message)
        self._follow()

    @on(ChatBridge.TokenReceived)
    def on_token(self, event: ChatBridge.TokenReceived) -> None:
        if not self.current_message:
            return
        stick = self.is_vertical_scroll_end
        self.current_message.stream.append(event.token)
        self.current_message.update_stream()
        if stick:
            self._follow()
        if self.app.generation_state != "STREAMING":
            self.app.generation_state = "STREAMING"

    @on(ChatBridge.GenerationDone)
    def on_done(self, event: ChatBridge.GenerationDone) -> None:
        stick = self.is_vertical_scroll_end
        if self.current_message:
            self.current_message.stream.finish()
            self.current_message.finalize(event.full_text)
            self.current_message = None
            if stick:
                self._follow()
        self.app.generation_state = "IDLE"
        self.app.session_bridge.add_message("assistant", event.full_text)
        self.app.refresh_tabs()

        # A long reply finishes while you are looking elsewhere; the bell is the
        # only way a terminal can tell you.
        try:
            from inference.tui.settings_apply import bell_enabled

            if bell_enabled():
                self.app.bell()
        except Exception:
            pass

        try:
            from inference.tui.panels.artifact_panel import ArtifactPanel

            self.app.query_one(ArtifactPanel).add_artifacts(event.full_text)
        except Exception:
            pass

    @on(ChatBridge.GenerationError)
    def on_error(self, event: ChatBridge.GenerationError) -> None:
        self.current_message = None
        self.add_system_message(f"Error: {event.error_msg}", "error")
        self.app.generation_state = "IDLE"

    @on(ChatBridge.SystemLog)
    def on_system_log(self, event: ChatBridge.SystemLog) -> None:
        self.app._log_system(event.message)

    @on(ToolBridge.ToolStarted)
    def on_tool_start(self, event: ToolBridge.ToolStarted) -> None:
        self.mount(Static(ToolBlock(event.tool_name, "RUNNING", event.detail)))
        self._follow_if_pinned()

    @on(ToolBridge.ToolCompleted)
    def on_tool_completed(self, event: ToolBridge.ToolCompleted) -> None:
        self.mount(Static(ToolBlock(event.tool_name, "DONE", event.detail)))
        self._follow_if_pinned()

    @on(ToolBridge.ToolFailed)
    def on_tool_failed(self, event: ToolBridge.ToolFailed) -> None:
        self.mount(Static(ToolBlock(event.tool_name, "FAILED", event.error_msg)))
        self._follow_if_pinned()

    def clear_workspace(self, show_empty: bool = True) -> None:
        for child in list(self.children):
            child.remove()
        self.current_message = None
        if show_empty:
            self.mount(EmptyStateBlock())
        self._set_empty_mode(show_empty)
