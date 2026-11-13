import threading

from textual.app import App
from textual.message import Message

from inference.tui.commands.builtins.tools import AVAILABLE_TOOLS


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
        self._busy = False

    # ── Thread marshalling ───────────────────────────────────────────
    def _ui(self, fn, *args) -> None:
        if threading.current_thread() is threading.main_thread():
            fn(*args)
            return
        try:
            self.app.call_from_thread(fn, *args)
        except Exception:
            pass

    def _post(self, message: Message) -> None:
        def deliver():
            try:
                self.app.query_one("WorkspacePanel").post_message(message)
            except Exception:
                pass

        self._ui(deliver)

    # ── Tool run ─────────────────────────────────────────────────────
    def run_tools_and_send(self, text: str) -> None:
        if self._busy:
            self._post(self.app.chat_bridge.SystemLog("A tool run is already in progress."))
            return
        self._busy = True
        self.app.run_worker(
            lambda: self._run(text),
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _run(self, text: str) -> None:
        try:
            states = getattr(self.app, "_tool_states", {})
            web_results = ""
            # `web_search` is a callable tool now, not a pre-search toggle. The
            # key is kept so an old session file cannot crash the bridge.
            if states.get("web_search", False):
                web_results = self._web_search(text)

            deep_think = states.get("deep_think", False) and self.app.active_mode != "Deep Think"

            context_block = ""
            if web_results:
                context_block += f"[WEB SEARCH RESULTS]\n{web_results}\n\n"
            if deep_think:
                context_block += (
                    "[SYSTEM]\nThink step by step and put your reasoning inside "
                    "<think>...</think> before the final answer.\n\n"
                )

            final_text = f"{context_block}[USER REQUEST]\n{text}" if context_block else text

            self._post(self.app.chat_bridge.SystemLog(
                f"Routing to {self.app.active_provider} · {self.app.active_model}"
            ))
            self._ui(self._start_assistant)
            self.app.chat_bridge.send(final_text)
        except Exception as error:
            self._post(self.ToolFailed("tool run", str(error)))
            self._post(self.app.chat_bridge.GenerationError(str(error)))
        finally:
            self._busy = False

    def _start_assistant(self) -> None:
        try:
            self.app.query_one("WorkspacePanel").start_assistant_message()
        except Exception:
            pass

    def _web_search(self, text: str) -> str:
        self._post(self.ToolStarted("web_search", f'Searching "{text[:24]}…"'))
        try:
            from core.tools.web_search import search_web

            results = search_web(text)
        except Exception as error:
            self._post(self.ToolFailed("web_search", str(error)))
            return ""

        if results:
            self._post(self.ToolCompleted("web_search", "results found"))
            return results
        self._post(self.ToolFailed("web_search", "no results"))
        return ""
