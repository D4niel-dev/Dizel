from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Input, Static
from textual import events, on

from inference.tui.commands.parser import parse_command
from inference.tui.commands.registry import registry
from inference.tui.labels import display_model
from inference.tui.panels.status_bar import StatusBar
from inference.tui.settings_apply import chat_pref
from inference.tui.theme import (
    ACCENT,
    ACCENT_INK,
    BORDER_HI,
    DANGER,
    TEXT,
    TEXT_3,
    WARN,
)

PLACEHOLDERS = (
    'Ask anything...  "Fix a TODO in the codebase"',
    "Explain how attention works",
    "Make a plan for a small app",
    "Write a function that reverses a string",
)


class PromptInput(Input):
    BINDINGS = [
        Binding("tab", "tab_pressed", "Tab", show=False),
        Binding("up", "up_pressed", "Up", show=False),
        Binding("down", "down_pressed", "Down", show=False),
        Binding("pageup", "page_up_pressed", "Page up", show=False),
        Binding("pagedown", "page_down_pressed", "Page down", show=False),
    ]

    class TabPressed(Message):
        pass

    class UpPressed(Message):
        pass

    class DownPressed(Message):
        pass

    class PageUpPressed(Message):
        pass

    class PageDownPressed(Message):
        pass

    class SubmittedFromEditor(Input.Submitted):
        """Sent by the multi-line editor: already composed, so Enter-mode
        preferences must not bounce it back into the editor."""

    def action_tab_pressed(self) -> None:
        self.post_message(self.TabPressed())

    def action_up_pressed(self) -> None:
        self.post_message(self.UpPressed())

    def action_down_pressed(self) -> None:
        self.post_message(self.DownPressed())

    def action_page_up_pressed(self) -> None:
        self.post_message(self.PageUpPressed())

    def action_page_down_pressed(self) -> None:
        self.post_message(self.PageDownPressed())


class InputBar(Container):
    """The composer. State and model live in the header and status line, so
    this is only the input itself."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: list[str] = []
        self.history_idx = -1
        self._placeholder_idx = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="input-row"):
            yield Static("❯", id="prompt-icon")
            yield PromptInput(
                placeholder=PLACEHOLDERS[0],
                id="prompt-input",
            )
        # OpenCode puts agent + model *inside* the input box rather than in a
        # separate header chip. One box, one place to look.
        with Horizontal(id="input-foot"):
            yield Static(id="input-agent")
            yield Static(id="input-model")
            yield Static(id="input-stats")
            yield Static(id="input-hint")

    def on_mount(self) -> None:
        self.refresh_footer()
        self.set_interval(4.5, self._rotate_placeholder)
        for prop in (
            "active_model",
            "active_provider",
            "active_tier",
            "active_mode",
            "context_tokens",
            "budget_tokens",
            "generation_state",
        ):
            self.watch(self.app, prop, self.refresh_footer)

    def _rotate_placeholder(self) -> None:
        """Home-screen prompt rotation, only while the composer is untouched."""
        try:
            inp = self.query_one(PromptInput)
        except Exception:
            return
        if inp.value.strip() or self.app.generation_state != "IDLE":
            return
        if not chat_pref(self.app, "animations", True):
            return
        self._placeholder_idx = (self._placeholder_idx + 1) % len(PLACEHOLDERS)
        inp.placeholder = PLACEHOLDERS[self._placeholder_idx]

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    @staticmethod
    def _gauge(percent: int, cells: int) -> tuple[str, str]:
        filled = max(0, min(cells, int(percent * cells / 100 + 0.5)))
        color = ACCENT
        if percent >= 90:
            color = DANGER
        elif percent >= 75:
            color = WARN
        bar = f"[{color}]{'▰' * filled}[/][{BORDER_HI}]{'▱' * (cells - filled)}[/]"
        return color, bar

    def refresh_footer(self, *_: object) -> None:
        app = self.app
        width = self.size.width or 96
        compact = width < 88
        narrow = width < 64
        hide_stats = width < 54
        agent = getattr(app, "response_label", app.active_tier) or "Fast"
        model = self._clip(
            display_model(app.active_model or "no model", agent),
            16 if narrow else (18 if compact else 34),
        )
        provider = (
            "Local"
            if getattr(app, "active_provider", "local") == "local"
            else app.active_provider
        )

        used = int(getattr(app, "context_tokens", 0) or 0)
        try:
            total = int(app.chat_bridge.manager.effective_ctx_len or 0)
        except Exception:
            total = 0
        percent = round(used * 100 / total) if total else 0
        color, bar = self._gauge(percent, 3 if compact else 5)
        if total:
            stats = (
                f"[{TEXT_3}]ctx[/]  {bar}  "
                f"[{color}]{percent}%[/]"
            )
        else:
            stats = f"[{TEXT_3}]ctx  —[/]"
        budget = int(getattr(app, "budget_tokens", 0) or 0)
        if budget and not compact:
            stats += f"  [{TEXT_3}]· {budget:,}[/]"

        try:
            has_text = bool(self.query_one(PromptInput).value.strip())
        except Exception:
            has_text = False

        state = app.generation_state
        if state in ("STREAMING", "THINKING"):
            hint = f"[{WARN}]esc stop[/]" if not compact else f"[{WARN}]esc[/]"
        elif state == "LOADING":
            hint = f"[{WARN}]loading[/]"
        elif has_text:
            verb = "send" if chat_pref(app, "enter_to_send", True) else "compose"
            hint = f"[b {ACCENT}]↵[/] [{TEXT_3}]{verb}[/]"
        else:
            hint = f"[{TEXT_3}]shift+tab mode[/]"

        try:
            self.query_one("#input-agent", Static).update(
                f"[bold {ACCENT_INK} on {ACCENT}]  {escape(agent)}  [/]"
            )
            model_markup = f"[{TEXT}]{escape(model)}[/]"
            if not compact:
                model_markup += f"   [{TEXT_3}]·  {escape(provider)}[/]"
            self.query_one("#input-model", Static).update(model_markup)
            self.query_one("#input-stats", Static).update(
                "" if hide_stats else stats
            )
            show_hint = not narrow or state != "IDLE"
            self.query_one("#input-hint", Static).update(hint if show_hint else "")
        except Exception:
            pass

    def on_resize(self, event: events.Resize) -> None:
        self.refresh_footer()

    def _drawer(self):
        from inference.tui.panels.slash_drawer import SlashDrawer

        try:
            return self.app.query_one(SlashDrawer)
        except Exception:
            return None

    def _refresh_drawer(self, text: str) -> None:
        drawer = self._drawer()
        if drawer is None:
            return
        # Only while the command name is being typed: once there is a space the
        # user is writing arguments and a menu would just be in the way.
        if not text.startswith("/") or " " in text:
            drawer.hide()
            return
        prefix = text[1:].lower()
        matches = registry.search(prefix)
        if matches:
            drawer.show(matches, prefix)
        else:
            drawer.hide()

    def on_key(self, event) -> None:
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            if event.key == "escape":
                drawer.hide()
                event.stop()
                return
            if event.key == "ctrl+f":
                drawer.hide()
                self.app.action_toggle_palette()
                event.stop()
                return

        # Esc wipes the composer only when it would not steal the generation
        # interrupt — a streaming reply owns Escape first.
        if event.key == "escape" and chat_pref(self.app, "escape_clears_draft", False):
            inp = self.query_one(PromptInput)
            if inp.value and self.app.generation_state == "IDLE":
                inp.value = ""
                event.stop()
                return

        if event.key == "shift+tab":
            event.stop()
            event.prevent_default()
            self.app.action_switch_agent_back()

    def run_command(self, name: str) -> None:
        """Run a command by name — used by clicking a row in the drawer."""
        inp = self.query_one(PromptInput)
        inp.value = f"/{name}"
        inp.post_message(Input.Submitted(inp, inp.value))

    def _status(self) -> StatusBar | None:
        try:
            return self.app.query_one(StatusBar)
        except Exception:
            return None

    # ── Command hints ────────────────────────────────────────────────
    @on(Input.Changed, "#prompt-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_drawer(event.value)
        self.refresh_footer()
        status = self._status()
        if status is None:
            return

        text = event.value
        if not text.startswith("/"):
            status.reset_hint()
            return

        prefix = text[1:]
        if not prefix:
            status.set_hint("type a command, tab to complete")
            return

        matches = registry.search(prefix)
        if matches:
            count = len(matches)
            label = f"{count} match" + ("" if count == 1 else "es")
            status.set_hint(
                f"tab  complete  [bold {ACCENT}]/{matches[0].name}[/]  ·  {label}"
            )
        else:
            status.set_hint("no matching command")

    # ── Submit ───────────────────────────────────────────────────────
    @on(Input.Submitted, "#prompt-input")
    async def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        from_editor = isinstance(event, PromptInput.SubmittedFromEditor)
        compose_mode = not chat_pref(self.app, "enter_to_send", True)
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            picked = drawer.selected()
            drawer.hide()
            if picked is not None:
                # Enter runs what is highlighted, not what was half-typed.
                text = f"/{picked.name}"
        if not text:
            if compose_mode and not from_editor:
                self.app.action_multiline_input(send=True)
            return

        # Enter composes rather than sends: the multi-line editor is this
        # terminal's newline surface, and ctrl+s there sends for real.
        # Commands still run on Enter — they are actions, not prose.
        if compose_mode and not from_editor and not text.startswith("/"):
            self.app.action_multiline_input(send=True)
            return

        if not self.history or self.history[-1] != text:
            self.history.append(text)
        self.history_idx = len(self.history)

        workspace = self.app.query_one("WorkspacePanel")

        if text.startswith("/"):
            event.input.value = ""
            invocation = parse_command(text)
            if invocation.flags.get("error"):
                workspace.add_system_message(f"Command parse error: {invocation.flags['error']}", "error")
                return

            cmd = registry.lookup(invocation.name)
            if not cmd:
                workspace.add_system_message(f"Unknown command: /{invocation.name}", "error")
                return
            try:
                result = await cmd.execute(self.app, invocation)
                if result:
                    workspace.add_system_message(result)
            except Exception as error:
                workspace.add_system_message(f"Error: {error}", "error")
            return

        if self.app.generation_state in ("STREAMING", "THINKING"):
            workspace.add_system_message("Still generating — press esc to stop first.", "error")
            return

        event.input.value = ""
        self.app.generation_state = "THINKING"
        workspace.add_user_message(text)
        self.app.session_bridge.add_message("user", text)
        self.app.refresh_tabs()
        self.app.tool_bridge.run_tools_and_send(text)

    # ── History navigation ───────────────────────────────────────────
    @on(PromptInput.UpPressed)
    def handle_up(self, event: PromptInput.UpPressed) -> None:
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            drawer.move(-1)
            return
        if not chat_pref(self.app, "arrow_up_edits_last", True):
            return
        inp = self.query_one(PromptInput)
        # A half-typed draft is not history yet — never overwrite it.
        if inp.value.strip() and self.history_idx >= len(self.history):
            return
        if self.history and self.history_idx > 0:
            self.history_idx -= 1
            inp.value = self.history[self.history_idx]
            inp.action_end()

    @on(PromptInput.DownPressed)
    def handle_down(self, event: PromptInput.DownPressed) -> None:
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            drawer.move(1)
            return
        if not chat_pref(self.app, "arrow_up_edits_last", True):
            return
        if self.history and self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            inp = self.query_one(PromptInput)
            inp.value = self.history[self.history_idx]
            inp.action_end()
        elif self.history_idx == len(self.history) - 1:
            self.history_idx = len(self.history)
            self.query_one(PromptInput).value = ""

    @on(PromptInput.PageUpPressed)
    def handle_page_up(self, event: PromptInput.PageUpPressed) -> None:
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            drawer.page(-1)

    @on(PromptInput.PageDownPressed)
    def handle_page_down(self, event: PromptInput.PageDownPressed) -> None:
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            drawer.page(1)

    @on(PromptInput.TabPressed)
    def handle_tab(self, event: PromptInput.TabPressed) -> None:
        inp = self.query_one(PromptInput)
        text = inp.value
        drawer = self._drawer()
        if drawer is not None and drawer.is_open:
            picked = drawer.selected()
            if picked is not None:
                inp.value = f"/{picked.name} "
                inp.action_end()
                return
        if text.startswith("/"):
            matches = registry.search(text[1:])
            if matches:
                inp.value = f"/{matches[0].name} "
                inp.action_end()
        else:
            self.app.action_switch_agent()
