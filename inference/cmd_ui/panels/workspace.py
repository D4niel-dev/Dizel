from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual import on

from inference.cmd_ui.rendering.message_block import MessageBlock
from inference.cmd_ui.rendering.stream_buffer import StreamBuffer
from inference.cmd_ui.rendering.tool_block import ToolBlock
from inference.cmd_ui.rendering.status_block import StatusBlock
from inference.cmd_ui.bridge.chat_bridge import ChatBridge
from inference.cmd_ui.bridge.tool_bridge import ToolBridge
from inference.cmd_ui.rendering.empty_state import EmptyStateBlock

class WorkspacePanel(VerticalScroll):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_stream: StreamBuffer | None = None
        self.current_message_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield EmptyStateBlock()

    def _remove_empty_state(self) -> None:
        for child in self.query(EmptyStateBlock):
            child.remove()

    def add_user_message(self, text: str) -> None:
        self._remove_empty_state()
        self.mount(Static(MessageBlock("USER", text)))
        self.scroll_end(animate=False)

    def start_assistant_message(self) -> None:
        self._remove_empty_state()
        self.current_stream = StreamBuffer()
        self.current_message_widget = Static(MessageBlock("ASSISTANT", self.current_stream))
        self.mount(self.current_message_widget)
        self.scroll_end(animate=False)

    @on(ChatBridge.TokenReceived)
    def on_token(self, event: ChatBridge.TokenReceived) -> None:
        if self.current_stream and self.current_message_widget:
            self.current_stream.append(event.token)
            self.current_message_widget.update(MessageBlock("ASSISTANT", self.current_stream))
            self.scroll_end(animate=False)
            if self.app.generation_state != "STREAMING":
                self.app.generation_state = "STREAMING"

    @on(ChatBridge.GenerationDone)
    def on_done(self, event: ChatBridge.GenerationDone) -> None:
        if self.current_stream and self.current_message_widget:
            self.current_stream.finish()
            self.current_message_widget.update(MessageBlock("ASSISTANT", event.full_text))
            self.current_stream = None
            self.current_message_widget = None
            self.scroll_end(animate=False)
        self.app.generation_state = "IDLE"
        self.app.session_bridge.add_message("assistant", event.full_text)

    @on(ChatBridge.GenerationError)
    def on_error(self, event: ChatBridge.GenerationError) -> None:
        self.mount(Static(MessageBlock("SYSTEM", f"Error: {event.error_msg}")))
        self.current_stream = None
        self.current_message_widget = None
        self.scroll_end(animate=False)
        self.app.generation_state = "IDLE"
        
    @on(ChatBridge.SystemLog)
    def on_system_log(self, event: ChatBridge.SystemLog) -> None:
        self.mount(Static(StatusBlock(event.message)))
        self.scroll_end(animate=False)
        
    @on(ToolBridge.ToolStarted)
    def on_tool_start(self, event: ToolBridge.ToolStarted) -> None:
        self.mount(Static(ToolBlock(event.tool_name, "RUNNING", event.detail)))
        self.scroll_end(animate=False)

    @on(ToolBridge.ToolCompleted)
    def on_tool_completed(self, event: ToolBridge.ToolCompleted) -> None:
        self.mount(Static(ToolBlock(event.tool_name, "DONE", event.detail)))
        self.scroll_end(animate=False)

    @on(ToolBridge.ToolFailed)
    def on_tool_failed(self, event: ToolBridge.ToolFailed) -> None:
        self.mount(Static(ToolBlock(event.tool_name, "FAILED", event.error_msg)))
        self.scroll_end(animate=False)
        
    def clear_workspace(self) -> None:
        for child in self.children:
            child.remove()
        self.mount(EmptyStateBlock())
