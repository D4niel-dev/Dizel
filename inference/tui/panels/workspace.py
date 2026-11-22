from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual import on

import time

from rich.markup import escape

from inference.tui.rendering.assistant_message import AssistantMessage
from inference.tui.rendering.message_block import MessageBlock, message_style
from inference.tui.rendering.tool_block import ToolLine
from inference.tui.rendering.status_block import StatusBlock
from inference.tui.bridge.chat_bridge import ChatBridge
from inference.tui.bridge.tool_bridge import ToolBridge
from inference.tui.panels.context_panel import ContextPanel
from inference.tui.rendering.empty_state import EmptyStateBlock, StarterShelf
from inference.tui.settings_apply import chat_pref
from inference.tui.theme import ACCENT


class WorkspacePanel(VerticalScroll):
    """The transcript.

    The view follows new output only while the reader is already at the
    bottom. Scrolling up mid-generation should let you read, not fight a
    ``scroll_end`` on every token.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_message: AssistantMessage | None = None
        self._context_rail_visible = False
        self._tool_rows: dict[int, ToolLine] = {}
        self._tool_results: dict[int, tuple[str, str, dict]] = {}
        # Every turn that landed, in order, with the widget that owns it. The
        # transcript is the only place a long reply can hide, so /find needs a
        # map from text back to a row it can scroll to.
        self._turns: list[tuple[str, Static]] = []
        self._find_query = ""
        self._find_hits: list[tuple[int, Static]] = []
        self._find_cursor = -1
        self._find_flash = None

    def on_mount(self) -> None:
        self._set_empty_mode(bool(self.query(EmptyStateBlock)))
        self._set_context_rail(False)

    def _set_empty_mode(self, empty: bool) -> None:
        # The desktop composer sits mid-screen with the welcome on an empty
        # chat, then docks to the bottom once a conversation starts. This class
        # drives the same placement in the terminal.
        try:
            self.app.query_one("#workspace-container").set_class(empty, "-empty")
        except Exception:
            pass
        self.call_after_refresh(self._sync_starter_shelf)

    def _set_context_rail(self, visible: bool) -> None:
        self._context_rail_visible = visible
        self.call_after_refresh(self._sync_context_rail)

    def _sync_context_rail(self) -> None:
        """The rail belongs to a transcript: hidden on the welcome only."""
        try:
            panel = self.app.query_one(ContextPanel)
            panel.set_class(self._context_rail_visible, "-visible")
            if not self._context_rail_visible and panel.has_focus:
                self.app.query_one("#prompt-input").focus()
        except Exception:
            pass

    def _sync_starter_shelf(self) -> None:
        """The starter prompts sit under the composer, only while empty."""
        try:
            container = self.app.query_one("#workspace-container")
            row = self.app.query_one("#starter-row")
        except Exception:
            return
        empty = container.has_class("-empty")
        shelves = list(row.query(StarterShelf))
        if empty and not shelves:
            row.mount(StarterShelf())
        elif not empty:
            for shelf in shelves:
                shelf.remove()

    # ── Scroll ───────────────────────────────────────────────────────
    def _follow(self) -> None:
        if not chat_pref(self.app, "auto_scroll"):
            return
        self.call_after_refresh(self.scroll_end, animate=False)

    def _follow_if_pinned(self) -> None:
        if self.is_vertical_scroll_end:
            self._follow()

    # ── Search ───────────────────────────────────────────────────────
    def find(self, query: str = "") -> dict:
        """Jump to the next turn containing `query`. Empty repeats the last one.

        The unit of a hit is a turn, not a character: this view is a list of
        mounted widgets, so the useful promise is "the row that has it is now
        on screen", not an in-line highlight offset.
        """
        needle = query.strip() or self._find_query
        if not needle:
            return {"query": "", "index": 0, "turns": 0, "matches": 0}

        lowered = needle.lower()
        if lowered != self._find_query.lower():
            self._find_query = needle
            self._find_hits = []
            for text, widget in self._turns:
                count = text.lower().count(lowered)
                if count and getattr(widget, "is_mounted", False):
                    self._find_hits.append((count, widget))
            self._find_cursor = -1

        if not self._find_hits:
            return {"query": needle, "index": 0, "turns": 0, "matches": 0}

        self._find_cursor = (self._find_cursor + 1) % len(self._find_hits)
        total = sum(count for count, _ in self._find_hits)
        _, widget = self._find_hits[self._find_cursor]
        self._flash(widget)
        try:
            self.scroll_to_widget(widget, animate=False, top=True)
        except Exception:
            pass
        return {
            "query": needle,
            "index": self._find_cursor + 1,
            "turns": len(self._find_hits),
            "matches": total,
        }

    def _flash(self, widget: Static) -> None:
        """Tint the row long enough to find it, then hand the colour back."""
        try:
            widget.add_class("-find-hit")
        except Exception:
            return
        if self._find_flash is not None:
            self._find_flash.stop()
        self._find_flash = self.set_timer(1.6, lambda: widget.remove_class("-find-hit"))

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
            # Back to a normal transcript: the composer returns to the bottom
            # and the context rail docks for the rest of the session.
            self._set_empty_mode(False)
            self._set_context_rail(True)

    def add_user_message(
        self, text: str, display: dict | None = None, ts: float | None = None
    ) -> None:
        self._remove_empty_state()
        if display and display.get("skills"):
            self.mount(Static(self._skill_chips(display), classes="skill-chips"))
        if text or not display:
            block = Static(
                MessageBlock(
                    "USER",
                    text,
                    **message_style(self.app, timestamp=ts or time.time()),
                )
            )
            self.mount(block)
            self._remember(text, block)
        self._follow()

    def _remember(self, text: str, widget: Static) -> None:
        if text and text.strip():
            self._turns.append((text, widget))

    @staticmethod
    def _skill_chips(display: dict) -> str:
        return "   ".join(
            f"[{ACCENT}]▣ {escape(str(skill.get('title') or skill.get('name') or 'skill'))}[/]"
            for skill in display.get("skills", [])
        )

    def add_debug_turn(self, text: str) -> None:
        """Mount a completed assistant turn without charging model metrics."""
        self._remove_empty_state()
        message = AssistantMessage()
        self.mount(message)
        message.stream.append(text)
        message.update_stream()
        message.stream.finish()
        message.finalize(text, record_metrics=False)
        self._follow()

    def add_debug_tools(self) -> None:
        """Mount finished tool rows with representative details for inspection."""
        self._remove_empty_state()
        details = [
            (
                9001,
                "grep",
                "TODO · src",
                {
                    "path": "src",
                    "query": "TODO",
                    "glob": "*.tsx",
                    "result": (
                        "src/App.tsx:12: // TODO: remove the fixture marker\n"
                        "src/lib/tools.ts:4: // TODO: shared renderer\n\n"
                        "2 matches in 2 files."
                    ),
                    "result_truncated": False,
                },
            ),
            (
                9002,
                "read_file",
                "src/App.tsx:1-12",
                {
                    "path": "src/App.tsx",
                    "result": (
                        "src/App.tsx — lines 1-12 of 48\n"
                        "import { useState } from \"react\";\n\n"
                        "export function App() {\n"
                        "  const [ready, setReady] = useState(false);\n\n"
                        "  return (\n"
                        "    <main>\n"
                        "      <h1>Dizel</h1>\n"
                        "      <p>{ready ? \"ready\" : \"loading\"}</p>\n"
                        "    </main>\n"
                        "  );\n"
                        "}\n"
                    )
                },
            ),
            (
                9003,
                "edit_file",
                "src/App.tsx",
                {
                    "diff": (
                        "--- a/src/App.tsx\n"
                        "+++ b/src/App.tsx\n"
                        "@@ -1,3 +1,4 @@\n"
                        "-export function App() {\n"
                        "+export function App() {\n"
                        "+  // reviewed by the tool fixture\n"
                        "   return null;\n"
                        " }"
                    ),
                    "path": "src/App.tsx",
                    "additions": 2,
                    "deletions": 1,
                    "action": "edit",
                },
            ),
            (
                9004,
                "write_file",
                "src/debug-fixture.ts",
                {
                    "content": (
                        "const fixture = {\n"
                        "  surface: 'terminal',\n"
                        "  detail: 'visible after the call',\n"
                        "};\n\n"
                        "export default fixture;\n"
                    ),
                    "path": "src/debug-fixture.ts",
                    "action": "create",
                    "result": "Created src/debug-fixture.ts (+6/-0)",
                },
            ),
            (
                9005,
                "web_search",
                "dizel local model",
                {
                    "query": "dizel local model",
                    "search_url": "https://duckduckgo.com/?q=dizel+local+model",
                    "result": (
                        "[1] Dizel 1.5 — local-first model\n"
                        "A small model and desktop app that run locally.\n"
                        "Source: https://github.com/D4niel-dev/Dizel\n\n"
                        "[2] Dizel documentation\n"
                        "Install, launch, and model notes.\n"
                        "Source: https://d4niel-dev.github.io/Dizel/"
                    )
                },
            ),
        ]
        for key, name, detail, payload in details:
            row = ToolLine(name, "DONE", detail, details=payload)
            self._tool_rows[key] = row
            self.mount(row)
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
            finished = self.current_message
            finished.stream.finish()
            finished.finalize(event.full_text)
            self._remember(event.full_text, finished)
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
        if self.current_message:
            self.current_message.stop()
        self.current_message = None
        self.add_system_message(f"Error: {event.error_msg}", "error")
        self.app.generation_state = "IDLE"

    @on(ChatBridge.SystemLog)
    def on_system_log(self, event: ChatBridge.SystemLog) -> None:
        self.app._log_system(event.message)

    @on(ToolBridge.ToolStarted)
    def on_tool_start(self, event: ToolBridge.ToolStarted) -> None:
        row = self._tool_rows.get(event.key)
        if row is None or not row.is_mounted:
            row = ToolLine(event.tool_name, "RUNNING", event.detail)
            self._tool_rows[event.key] = row
            self.mount(row)
        result = self._tool_results.pop(event.key, None)
        if result is not None:
            row.finish(*result)
        self._follow_if_pinned()

    @on(ToolBridge.ToolCompleted)
    def on_tool_completed(self, event: ToolBridge.ToolCompleted) -> None:
        row = self._tool_rows.get(event.key)
        if row is None or not row.is_mounted:
            self._tool_results[event.key] = ("DONE", event.detail, event.details)
        else:
            row.finish("DONE", event.detail, event.details)
        self._follow_if_pinned()

    @on(ToolBridge.ToolFailed)
    def on_tool_failed(self, event: ToolBridge.ToolFailed) -> None:
        row = self._tool_rows.get(event.key)
        if row is None or not row.is_mounted:
            self._tool_results[event.key] = ("FAILED", event.error_msg, event.details)
        else:
            row.finish("FAILED", event.error_msg, event.details)
        self._follow_if_pinned()

    def clear_workspace(self, show_empty: bool = True) -> None:
        for child in list(self.children):
            child.remove()
        self.current_message = None
        self._tool_rows.clear()
        self._tool_results.clear()
        self._turns.clear()
        self._find_query = ""
        self._find_hits = []
        self._find_cursor = -1
        try:
            self.app.clear_reply_metrics()
        except Exception:
            pass
        if show_empty:
            self.mount(EmptyStateBlock())
        self._set_empty_mode(show_empty)
        self._set_context_rail(not show_empty)
