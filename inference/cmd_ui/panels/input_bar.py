from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, Static
from textual import events
from textual import on

from inference.cmd_ui.commands.parser import parse_command
from inference.cmd_ui.commands.registry import registry

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
            yield Input(placeholder="Ask anything... or type /command", id="prompt-input")
        yield Static(id="input-mode-indicator")
        
    def on_mount(self):
        self.update_indicator()
        self.watch(self.app, "active_mode", self.update_indicator)
        self.watch(self.app, "active_model", self.update_indicator)
        self.watch(self.app, "active_provider", self.update_indicator)
        
    def update_indicator(self, *args):
        app = self.app
        from inference.cmd_ui.theme import STYLE_TOOL
        # Example format: "Build Claude Opus 4.5 Anthropic" using rich markup
        text = f"[{STYLE_TOOL}]{app.active_mode.capitalize()}[/]  [dim]{app.active_model} {app.active_provider.capitalize()}[/]"
        self.query_one("#input-mode-indicator").update(text)

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

    def on_key(self, event: events.Key) -> None:
        if event.key == "up":
            if self.history and self.history_idx > 0:
                self.history_idx -= 1
                inp = self.query_one(Input)
                inp.value = self.history[self.history_idx]
                inp.action_end()
        elif event.key == "down":
            if self.history and self.history_idx < len(self.history) - 1:
                self.history_idx += 1
                inp = self.query_one(Input)
                inp.value = self.history[self.history_idx]
                inp.action_end()
            elif self.history_idx == len(self.history) - 1:
                self.history_idx = len(self.history)
                self.query_one(Input).value = ""
        elif event.key == "tab":
            event.stop()
            event.prevent_default()
            inp = self.query_one(Input)
            text = inp.value
            if text.startswith("/"):
                prefix = text[1:]
                completions = registry.complete(prefix)
                if len(completions) == 1:
                    inp.value = f"/{completions[0]} "
                    inp.action_end()
            else:
                # Switch agent mode forward
                self.app.action_switch_agent()
        elif event.key == "shift+tab":
            event.stop()
            event.prevent_default()
            self.app.action_switch_agent_back()
