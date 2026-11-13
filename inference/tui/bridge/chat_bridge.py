import threading

from textual.app import App
from textual.message import Message

from dizel_core.chat_manager import ChatManager
from dizel_core.token_budget import allocate_token_budget, classify_task


class ChatBridge:
    class TokenReceived(Message):
        def __init__(self, token: str):
            self.token = token
            super().__init__()

    class GenerationDone(Message):
        def __init__(self, full_text: str):
            self.full_text = full_text
            super().__init__()

    class GenerationError(Message):
        def __init__(self, error_msg: str):
            self.error_msg = error_msg
            super().__init__()

    class SystemLog(Message):
        def __init__(self, message: str):
            self.message = message
            super().__init__()

    def __init__(self, app: App, manager=None):
        self.app = app
        # A manager that already has its model loaded can be handed in: loading
        # 1 GB of weights can block the process on some machines, and it must not
        # happen while the TUI event loop is running.
        self.manager = manager or ChatManager()
        self._session_id = ""
        self._cost = 0.0
        self._tokens_uncounted = 0
        self._prev_max_tokens = None

    # ── Thread marshalling ───────────────────────────────────────────
    def _ui(self, fn, *args) -> None:
        """Run fn on the UI thread, whether we are already on it or not."""
        if threading.current_thread() is threading.main_thread():
            fn(*args)
            return
        try:
            self.app.call_from_thread(fn, *args)
        except Exception:
            pass  # app is shutting down

    def _post(self, message: Message) -> None:
        def deliver():
            try:
                self.app.query_one("WorkspacePanel").post_message(message)
            except Exception:
                pass

        self._ui(deliver)

    def _bump_context(self, count: int) -> None:
        self.app.context_tokens = int(self.app.context_tokens) + count

    def _settle(self) -> None:
        """Restore per-request sampling and persist usage exactly once."""
        if self._prev_max_tokens is not None:
            self.manager.max_new_tokens = self._prev_max_tokens
            self._prev_max_tokens = None
        if self._tokens_uncounted:
            self.app.context_tokens = int(self.app.context_tokens) + self._tokens_uncounted
            self._tokens_uncounted = 0
        # Usage is charged by the shared fixed rate at the point each side of the
        # turn is known, not accumulated per token here.
        self._cost = 0.0

    # ── Generation ───────────────────────────────────────────────────
    def send(self, text: str) -> None:
        self._sync_provider_state()
        self._session_id = self.app.session_id
        self._cost = 0.0
        self._tokens_uncounted = 0

        has_tools = any(getattr(self.app, "_tool_states", {}).values())
        task_type = classify_task(text, has_tools)
        input_tokens = len(text.split()) * 2
        max_cap = int(self.app.usage_manager.max_capacity) if hasattr(self.app, "usage_manager") else 4096

        budget = allocate_token_budget(
            task_type=task_type,
            input_token_count=input_tokens,
            context_tokens=self.app.context_tokens,
            model_ctx_length=max_cap,
        )
        self._ui(setattr, self.app, "budget_tokens", budget)

        try:
            if hasattr(self.app, "usage_manager"):
                self.app.usage_manager.charge_text(text)
        except Exception:
            pass

        # Per-request ceiling only. The configured value is restored in _settle()
        # so /settings sampling max_new_tokens keeps meaning something.
        self._prev_max_tokens = self.manager.max_new_tokens
        self.manager.max_new_tokens = max(64, min(int(budget), 2048))


        def on_token(token: str) -> None:
            self._tokens_uncounted += 1
            if self._tokens_uncounted >= 8:
                pending, self._tokens_uncounted = self._tokens_uncounted, 0
                self._ui(self._bump_context, pending)
            self._post(self.TokenReceived(token))

        def on_done(full_text: str) -> None:
            try:
                if hasattr(self.app, "usage_manager"):
                    self.app.usage_manager.charge_text(full_text)
            except Exception:
                pass
            self._ui(self._settle)
            if self.app.session_id == self._session_id:
                self._post(self.GenerationDone(full_text))
            else:
                self._post(self.SystemLog("Reply discarded — the session changed while it generated."))

        def on_tool(call: dict, outcome: str) -> None:
            try:
                if hasattr(self.app, "usage_manager"):
                    self.app.usage_manager.charge_tool_calls(1)
            except Exception:
                pass
            name = call.get("name", "?")
            args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
            first = (outcome or "").strip().splitlines()
            preview = first[0][:70] if first else "(no output)"
            self._post(self.SystemLog(f"tool {name}({args}) → {preview}"))

        def on_error(error_msg: str) -> None:
            self._ui(self._settle)
            self._post(self.GenerationError(error_msg))

        self.manager.send_with_tools(
            user_text=text,
            attachments=[],
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
            on_tool=on_tool,
        )

    def _sync_provider_state(self) -> None:
        mgr = self.manager
        app_provider = self.app.active_provider
        app_model = self.app.active_model

        needs_reload = mgr._provider_slug != app_provider
        if app_provider != "local" and mgr._api_model != app_model:
            mgr._api_model = app_model

        if needs_reload:
            mgr.reload_provider()

    def stop(self) -> None:
        self.manager.stop_generation()
        self._ui(self._settle)
