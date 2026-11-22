import itertools
import threading

from textual.app import App
from textual.message import Message

from dizel_core import tools as tool_registry
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
        self._tool_keys = itertools.count(1)

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

    def _approve_tool(self, change) -> bool:
        """Block the tool thread until the user approves or rejects the diff."""
        from dizel_core import tool_trust
        from dizel_core.tools import PlanChange, _project_root

        if isinstance(change, PlanChange):
            return self._approve_plan(change)

        persistent = tool_trust.persistent_enabled()
        if tool_trust.should_auto_approve(
            change,
            self._session_id,
            session_grants=self.app._trusted_sessions,
            persistent=persistent,
            root=_project_root(),
        ):
            tier = "persistent" if persistent else "session"
            self._post(
                self.SystemLog(f"Auto-approved {change.tool} {change.path} ({tier})")
            )
            return True

        event = threading.Event()
        holder: dict = {"modal": None, "approved": False}

        def done(decision: str | None) -> None:
            decision = decision if decision in tool_trust.DECISIONS else tool_trust.DENY
            if decision == tool_trust.ALWAYS_SESSION:
                tool_trust.record_session_grant(
                    self.app._trusted_sessions, self._session_id
                )
                self.app.trust_revision += 1
            elif decision == tool_trust.ALWAYS_PERSISTENT:
                tool_trust.set_persistent(True)
                self.app.trust_revision += 1
            holder["approved"] = tool_trust.approved_for(decision)
            event.set()

        def show() -> None:
            from inference.tui.panels.modals.tool_approval import ToolApprovalModal

            try:
                modal = ToolApprovalModal(change)
                holder["modal"] = modal
                self.app.push_screen(modal, done)
            except Exception:
                event.set()

        try:
            self.app.call_from_thread(show)
        except Exception:
            return False
        if not event.wait(180):
            def _dismiss_if_open() -> None:
                modal = holder.get("modal")
                try:
                    if modal is not None and self.app.screen is modal:
                        modal.dismiss(tool_trust.DENY)
                except Exception:
                    pass

            try:
                self.app.call_from_thread(_dismiss_if_open)
            except Exception:
                pass
            return False
        return bool(holder["approved"])

    def _approve_plan(self, change) -> bool:
        """Block the tool thread until the user executes or discards the plan."""
        event = threading.Event()
        holder: dict = {"modal": None, "approved": False}

        def done(approved) -> None:
            holder["approved"] = bool(approved)
            event.set()

        def show() -> None:
            from inference.tui.panels.modals.plan_approval import PlanApprovalModal

            try:
                modal = PlanApprovalModal(change)
                holder["modal"] = modal
                self.app.push_screen(modal, done)
            except Exception:
                event.set()

        try:
            self.app.call_from_thread(show)
        except Exception:
            return False
        if not event.wait(180):
            def _dismiss_if_open() -> None:
                modal = holder.get("modal")
                try:
                    if modal is not None and self.app.screen is modal:
                        modal.dismiss(False)
                except Exception:
                    pass

            try:
                self.app.call_from_thread(_dismiss_if_open)
            except Exception:
                pass
            return False
        return bool(holder["approved"])

    def enforce_usage(self) -> str | None:
        """Apply the active provider's usage policy before a turn.

        Returns a message when the turn must not run, None when it may proceed.
        Compaction happens here so the trimmed history and the notice are in
        place before the model is called.
        """
        usage = getattr(self.app, "usage_manager", None)
        if usage is None:
            return None
        usage.activate(self.app.active_provider, int(self.manager.effective_ctx_len))
        if usage.is_exhausted:
            return (
                f"Usage window spent for {usage.active_provider} — resets in "
                f"{usage.time_until_reset()}. Start a new chat or wait for the reset."
            )
        if usage.needs_compaction:
            if not usage.can_compact():
                return (
                    "Context limit reached — this chat already used all "
                    f"{usage.MAX_COMPACTIONS} compactions in this window. "
                    "Start a new chat or wait for the reset."
                )
            try:
                self.manager.compact_context()
                usage.run_compaction()
                self._post(
                    self.SystemLog(
                        f"Context compacted at {usage.COMPACTION_THRESHOLD}% — earlier "
                        f"turns were summarised. {usage.compaction_count}/"
                        f"{usage.MAX_COMPACTIONS} compactions used."
                    )
                )
            except Exception:
                pass
        return None

    # ── Generation ───────────────────────────────────────────────────
    def send(self, text: str, display: dict | None = None) -> None:
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

        def on_tool_start(call: dict) -> None:
            from inference.tui.bridge.tool_bridge import ToolBridge

            key = next(self._tool_keys)
            call["_tui_key"] = key
            name = str(call.get("name") or "?")
            self._post(
                ToolBridge.ToolStarted(
                    name, tool_registry.display_detail(call), key=key
                )
            )

        def on_tool(call: dict, outcome: str) -> None:
            try:
                if hasattr(self.app, "usage_manager"):
                    self.app.usage_manager.charge_tool_calls(1)
            except Exception:
                pass
            from inference.tui.bridge.tool_bridge import ToolBridge

            name = str(call.get("name") or "?")
            key = int(call.get("_tui_key") or 0)
            details = tool_registry.ui_payload(call, outcome)
            if tool_registry.outcome_failed(name, outcome):
                first = (outcome or "").strip().splitlines()
                preview = first[0][:90] if first else "failed"
                prefix = f"[{name}] "
                if preview.startswith(prefix):
                    preview = preview[len(prefix) :]
                self._post(ToolBridge.ToolFailed(name, preview, key=key, details=details))
            else:
                self._post(
                    ToolBridge.ToolCompleted(
                        name, tool_registry.display_detail(call), key=key, details=details
                    )
                )

        def on_error(error_msg: str) -> None:
            self._ui(self._settle)
            self._post(self.GenerationError(error_msg))

        self.manager.send_with_tools(
            user_text=text,
            attachments=[],
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
            verbosity=getattr(self.app, "active_verbosity", "normal"),
            on_tool=on_tool,
            on_tool_start=on_tool_start,
            approve_tool=self._approve_tool,
            display=display,
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
