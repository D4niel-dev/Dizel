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
        
        # Parse files if any
        file_context = ""
        if files:
            for f in files:
                self.app.call_from_thread(workspace.post_message, self.ToolStarted("LilyAgent", f"Processing {f}"))
                # TODO: Wire real file extractors if needed, currently simulated
                time.sleep(0.5)
                self.app.call_from_thread(workspace.post_message, self.ToolCompleted("LilyAgent", "tokens extracted"))
                file_context += f"[FILE: {f}]\nContent processed.\n"

        # Real Web Search Integration
        web_results = ""
        # Check if tools state exists and if web search is enabled (default True for Planning)
        is_web_enabled = getattr(self.app, "_tool_states", {}).get("web_search", self.app.active_mode == "Planning")
        
        if is_web_enabled:
            from core.tools.web_search import search_web
            self.app.call_from_thread(workspace.post_message, self.ToolStarted("web_search", f'Searching "{text[:20]}..."'))
            results = search_web(text)
            if results:
                self.app.call_from_thread(workspace.post_message, self.ToolCompleted("web_search", "results found"))
                web_results = results
            else:
                self.app.call_from_thread(workspace.post_message, self.ToolFailed("web_search", "no results"))

        # Check if thinking is enabled
        is_thinking_enabled = getattr(self.app, "_tool_states", {}).get("thinking", False)

        # Inject context directly into text for the LLM
        final_text = text
        if web_results or file_context or is_thinking_enabled:
            context_block = ""
            if file_context:
                context_block += f"[FILE CONTEXT]\n{file_context}\n\n"
            if web_results:
                context_block += f"[WEB SEARCH RESULTS]\n{web_results}\n\n"
            if is_thinking_enabled:
                context_block += f"[SYSTEM]\nPlease think step-by-step and output your reasoning inside <think>...</think> tags before providing your final answer.\n\n"
            final_text = f"{context_block}[USER REQUEST]\n{text}"

        # Now start assistant generation
        self.app.call_from_thread(workspace.start_assistant_message)
        self.app.chat_bridge.send(final_text)
