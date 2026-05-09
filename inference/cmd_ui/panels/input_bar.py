from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Static
from textual import events
from textual import on

from inference.cmd_ui.commands.parser import parse_command
from inference.cmd_ui.commands.registry import registry

from textual.message import Message

from textual.binding import Binding
from textual.message import Message

class PromptInput(Input):
    BINDINGS = [
        Binding("tab", "tab_pressed", "Tab", show=False),
        Binding("up", "up_pressed", "Up", show=False),
        Binding("down", "down_pressed", "Down", show=False),
    ]

    class TabPressed(Message):
        pass
    class UpPressed(Message):
        pass
    class DownPressed(Message):
        pass

    def action_tab_pressed(self) -> None:
        self.post_message(self.TabPressed())

    def action_up_pressed(self) -> None:
        self.post_message(self.UpPressed())

    def action_down_pressed(self) -> None:
        self.post_message(self.DownPressed())


class InputBar(Container):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
        self.history_idx = -1

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal
        from textual.widgets import Static
        with Horizontal(id="input-row"):
            yield Static(">", id="prompt-icon")
            yield PromptInput(placeholder="Ask anything... or type /command", id="prompt-input")
        yield Static(id="input-mode-indicator")
        
    def on_mount(self):
        self.update_indicator()
        self.watch(self.app, "active_mode", self.update_indicator)
        self.watch(self.app, "active_model", self.update_indicator)
        self.watch(self.app, "active_provider", self.update_indicator)
        
    def update_indicator(self, *args):
        app = self.app
        mode = app.active_mode.lower()
        mode_colors = {
            "fast": "#10B981",       # Emerald
            "planning": "#8B5CF6",   # Violet
            "coding": "#F59E0B",     # Amber
            "reasoning": "#06B6D4",  # Cyan
            "debug": "#EF4444",      # Red
            "writer": "#EC4899",     # Pink
        }
        color = mode_colors.get(mode, "#3B82F6") # Default Blue
        text = f"[b {color}]{app.active_mode.capitalize()}[/]  [dim]{app.active_model} | {app.active_provider.capitalize()}[/]"
        self.query_one("#input-mode-indicator").update(text)

    @on(Input.Changed, "#prompt-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        text = event.value
        if text.startswith("/"):
            prefix = text[1:]
            if prefix:
                completions = registry.complete(prefix)
                if completions:
                    self.query_one("#input-mode-indicator").update(f"[dim]Tab to complete: /{completions[0]}[/dim]")
                    return
            else:
                self.query_one("#input-mode-indicator").update("[dim]Type a command...[/dim]")
                return
        self.update_indicator()

    @on(Input.Submitted, "#prompt-input")
    async def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
            
        # Add to history
        if not self.history or self.history[-1] != text:
            self.history.append(text)
        self.history_idx = len(self.history)
            
        event.input.value = ""
        
        if text.startswith("/"):
            invocation = parse_command(text)
            if invocation.flags.get("error"):
                from inference.cmd_ui.rendering.message_block import MessageBlock
                workspace = self.app.query_one("WorkspacePanel")
                workspace.mount(Static(MessageBlock("SYSTEM", f"Command parse error: {invocation.flags['error']}")))
                workspace.scroll_end(animate=False)
                return

            cmd = registry.lookup(invocation.name)
            workspace = self.app.query_one("WorkspacePanel")
            if cmd:
                try:
                    result = await cmd.execute(self.app, invocation)
                    if result:
                        from inference.cmd_ui.rendering.message_block import MessageBlock
                        workspace.mount(Static(MessageBlock("SYSTEM", result)))
                        workspace.scroll_end(animate=False)
                except Exception as e:
                    from inference.cmd_ui.rendering.message_block import MessageBlock
                    workspace.mount(Static(MessageBlock("SYSTEM", f"Error: {e}")))
                    workspace.scroll_end(animate=False)
            else:
                from inference.cmd_ui.rendering.message_block import MessageBlock
                workspace.mount(Static(MessageBlock("SYSTEM", f"Unknown command: /{invocation.name}")))
                workspace.scroll_end(animate=False)
        else:
            self.app.generation_state = "THINKING"
            workspace = self.app.query_one("WorkspacePanel")
            workspace.add_user_message(text)
            self.app.session_bridge.add_message("user", text)
            self.app.tool_bridge.run_tools_and_send(text)

    @on(PromptInput.UpPressed)
    def handle_up(self, event: PromptInput.UpPressed) -> None:
        if self.history and self.history_idx > 0:
            self.history_idx -= 1
            inp = self.query_one(PromptInput)
            inp.value = self.history[self.history_idx]
            inp.action_end()

    @on(PromptInput.DownPressed)
    def handle_down(self, event: PromptInput.DownPressed) -> None:
        if self.history and self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            inp = self.query_one(PromptInput)
            inp.value = self.history[self.history_idx]
            inp.action_end()
        elif self.history_idx == len(self.history) - 1:
            self.history_idx = len(self.history)
            self.query_one(PromptInput).value = ""

    @on(PromptInput.TabPressed)
    def handle_tab(self, event: PromptInput.TabPressed) -> None:
        inp = self.query_one(PromptInput)
        text = inp.value
        if text.startswith("/"):
            prefix = text[1:]
            completions = registry.complete(prefix)
            if len(completions) > 0:
                inp.value = f"/{completions[0]} "
                inp.action_end()
        else:
            self.app.action_switch_agent()

    def on_key(self, event: events.Key) -> None:
        if event.key == "shift+tab":
            event.stop()
            event.prevent_default()
            self.app.action_switch_agent_back()
