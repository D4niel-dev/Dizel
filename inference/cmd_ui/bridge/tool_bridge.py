from textual.message import Message
from textual.app import App
from textual import work
import time

class ToolBridge:
    class ToolStarted(Message):
        def __init__(self, tool_name: str, detail: str):
            self.tool_name = tool_name
            self.detail = detail
            super().__init__()

    class ToolCompleted(Message):
        def __init__(self, tool_name: str, detail: str):
            self.tool_name = tool_name
            self.detail = detail
            super().__init__()

    class ToolFailed(Message):
        def __init__(self, tool_name: str, error_msg: str):
            self.tool_name = tool_name
            self.error_msg = error_msg
            super().__init__()

    def __init__(self, app: App):
        self.app = app

    def run_tools_and_send(self, text: str, files: list = None) -> None:
        self.app.run_worker(lambda: self._run(text, files), exclusive=True, thread=True)
        
    def _run(self, text: str, files: list = None) -> None:
        workspace = self.app.query_one("WorkspacePanel")
        
        # Simulated tools based on active mode
        if files:
            for f in files:
                self.app.call_from_thread(workspace.post_message, self.ToolStarted("LilyAgent", f"Processing {f}"))
                time.sleep(0.5)
                self.app.call_from_thread(workspace.post_message, self.ToolCompleted("LilyAgent", "tokens extracted"))

        if self.app.active_mode == "Planning":
            self.app.call_from_thread(workspace.post_message, self.ToolStarted("web_search", f'Searching "{text[:20]}..."'))
            time.sleep(1.0)
            self.app.call_from_thread(workspace.post_message, self.ToolCompleted("web_search", "results found"))

        # Now start assistant generation
        self.app.call_from_thread(workspace.start_assistant_message)
        self.app.chat_bridge.send(text)
