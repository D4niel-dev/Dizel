"""
api/services/chat_service.py
────────────────────────────
Service wrapper around the desktop `ChatManager`.

The manager is the same object the PySide6 app uses — it owns conversation
history, provider routing, token budgeting and generation. This module only
adapts its callback-based API to HTTP: SSE for streaming, polling for model
load progress.
"""

import asyncio
import json
import os
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import HTTPException

from dizel_core import history_manager
from dizel_core.chat_manager import ChatManager
from model.naming import model_display_name

# Slash commands that touch the host machine. Disabled unless the operator
# explicitly opts in — the desktop app is trusted, an HTTP surface is not.
_SHELL_PREFIXES = ("/sh ", "/py ")
_ALLOW_SHELL = os.environ.get("DIZEL_ALLOW_SHELL", "").lower() in ("1", "true", "yes")


def _sse(payload: Dict[str, Any]) -> Dict[str, str]:
    """Shape one event for sse-starlette. Data must be a pre-serialised string."""
    return {"data": json.dumps(payload, ensure_ascii=False)}


class ChatService:
    """Thin HTTP adapter over a single shared ChatManager."""

    _manager: Optional[ChatManager] = None
    _manager_lock = threading.Lock()

    # Model loading is slow enough (a 1.1 GB GGUF on CPU is 10–30s) that it
    # must not block the request. A polled status object keeps the UI honest.
    _load_lock = threading.Lock()
    _load_state: Dict[str, Any] = {
        "status": "idle",       # idle | loading | ready | error
        "message": "",
        "error": "",
        "checkpoint": "",
    }

    # ── manager ──────────────────────────────────────────────────────────

    @classmethod
    def get_manager(cls) -> ChatManager:
        """Lazily build the manager so importing the app stays cheap."""
        if cls._manager is None:
            with cls._manager_lock:
                if cls._manager is None:
                    cls._manager = ChatManager()
        return cls._manager

    # ── model lifecycle ──────────────────────────────────────────────────

    @classmethod
    def context_usage(cls, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        What the context window actually contains.

        Counted on the server with the model's own tokenizer: the old client-side
        `chars / 4` estimate ignored the system prompt and attachments, and
        disagreed with the side that actually does the trimming.
        """
        manager = cls.get_manager()
        total = int(manager.effective_ctx_len)

        messages: List[Dict[str, Any]] = []
        if session_id:
            data = history_manager.load_session(session_id) or {}
            messages = data.get("messages", []) or []

        system = manager.count_tokens(manager.system_prompt or "")
        conversation = sum(manager.count_tokens(m.get("content", "")) for m in messages)
        used = system + conversation
        reserved = int(getattr(manager, "last_budget", 0) or 0)

        return {
            "system": system,
            "conversation": conversation,
            "reserved": reserved,
            "trimmed": int(getattr(manager, "last_trimmed", 0) or 0),
            "used": used,
            "total": total,
            "percent": min(100, round(used * 100 / total)) if total else 0,
            "exact": bool(manager.exact_counts),
            "task": getattr(manager, "last_task", ""),
        }

    @classmethod
    def start_load(cls, checkpoint_path: str, device: Optional[str] = None) -> Dict[str, Any]:
        """Kick off a background model load and return immediately."""
        with cls._load_lock:
            if cls._load_state["status"] == "loading":
                return dict(cls._load_state)
            cls._load_state = {
                "status": "loading",
                "message": "Starting…",
                "error": "",
                "checkpoint": checkpoint_path,
            }

        manager = cls.get_manager()

        def _progress(msg: str) -> None:
            with cls._load_lock:
                cls._load_state["message"] = msg

        def _run() -> None:
            try:
                manager.load_model(checkpoint_path, device=device, on_progress=_progress)
                with cls._load_lock:
                    cls._load_state.update(
                        status="ready",
                        message="Model ready",
                        error="",
                    )
            except Exception as exc:  # surfaced verbatim to the UI
                with cls._load_lock:
                    cls._load_state.update(status="error", error=str(exc))

        threading.Thread(target=_run, name="dizel-model-load", daemon=True).start()
        return dict(cls._load_state)

    @classmethod
    def load_status(cls) -> Dict[str, Any]:
        with cls._load_lock:
            state = dict(cls._load_state)
        state["is_ready"] = cls.get_manager().is_ready
        checkpoint = state.get("checkpoint", "")
        state["display_name"] = (
            model_display_name(checkpoint) if checkpoint else ""
        )
        return state

    @classmethod
    def switch_provider(cls) -> Dict[str, Any]:
        manager = cls.get_manager()
        manager.reload_provider()
        return {"status": "ok", "active_provider": manager.active_provider_slug}

    @classmethod
    def apply_profile(cls, model_name: str, mode_name: str) -> Dict[str, Any]:
        cls.get_manager().apply_profile(model_name, mode_name)
        return {"status": "ok", "profile": f"{model_name} / {mode_name}"}

    @classmethod
    def reload_settings(cls) -> Dict[str, Any]:
        """Re-read settings.json into the live manager after a Settings save."""
        manager = cls.get_manager()
        manager.reload_sampling()
        # Provider instances bake in their base URL at construction, so an
        # edited Ollama/custom URL only takes effect after a rebuild.
        manager.reload_provider()
        return {
            "status": "ok",
            "sampling": {
                "temperature": manager.temperature,
                "top_k": manager.top_k,
                "top_p": manager.top_p,
                "repetition_penalty": manager.repetition_penalty,
                "max_new_tokens": manager.max_new_tokens,
            },
        }

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        manager = cls.get_manager()
        return {
            "is_ready": manager.is_ready,
            "is_generating": manager.is_generating,
            "provider": manager.active_provider_slug,
            "api_model": manager.active_api_model,
            "model_info": {
                **(manager.model_info or {}),
                "ctx_len": manager.effective_ctx_len,
                # True when no model reported its window (providers don't).
                "ctx_assumed": not (manager.model_info or {}).get("ctx_len"),
            },
            "load": cls.load_status(),
        }

    # ── generation ───────────────────────────────────────────────────────

    @classmethod
    def stop(cls) -> Dict[str, Any]:
        manager = cls.get_manager()
        manager.stop_generation()
        return {"status": "stopping", "was_generating": manager.is_generating}

    @classmethod
    async def stream_chat(
        cls,
        user_text: str,
        session_id: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        continuation: bool = False,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Stream one assistant turn as SSE events.

        Events: start | token | done | error
        """
        manager = cls.get_manager()

        if not manager.is_ready:
            yield _sse({"type": "error", "message": "Model not loaded. Choose a checkpoint or provider in Settings."})
            return

        if manager.is_generating:
            if manager.is_stopping:
                # A stop was requested: give the provider socket a moment to
                # unwind so an immediate retry is not refused.
                for _ in range(40):
                    await asyncio.sleep(0.025)
                    if not manager.is_generating:
                        break
            if manager.is_generating:
                yield _sse({"type": "error", "message": "A response is already being generated."})
                return

        if not _ALLOW_SHELL and user_text.strip().startswith(_SHELL_PREFIXES):
            yield _sse({
                "type": "error",
                "message": "Shell and Python execution are disabled over the API. "
                           "Start the sidecar with DIZEL_ALLOW_SHELL=1 to enable.",
            })
            return

        # ── restore or start the session ─────────────────────────────────
        if session_id:
            from api.services.session_service import SessionService
            session = SessionService.get_session(session_id)
            if session and session.get("messages"):
                manager.load_history(session["messages"])
            manager.session_id = session_id
        else:
            from dizel_core.history_manager import new_session_id
            manager.new_session()
            session_id = new_session_id()
            manager.session_id = session_id

        yield _sse({"type": "start", "session_id": session_id})

        # ── bridge thread callbacks into asyncio ─────────────────────────
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

        def _put(event: Dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def on_token(token: str) -> None:
            _put({"type": "token", "text": token})

        def on_thinking(piece: str) -> None:
            # Reasoning models stream their working-out before the answer. It
            # goes on its own channel so the client can show it apart from the
            # reply instead of rendering it as one long preamble.
            _put({"type": "thinking", "text": piece})

        def on_done(full_text: str) -> None:
            try:
                from api.services.session_service import SessionService
                SessionService.create_or_update_session(manager.get_history(), session_id=session_id)
            except Exception as exc:
                print(f"[dizel-api] session save failed: {exc}", flush=True)
            # Record what this turn cost. Without this the 12-hour usage window
            # never moves — the TUI did it, the desktop path did not.
            try:
                # Fixed, generous rate rather than a real tokenizer count — the
                # window is a policy limit, not a prediction. See UsageManager.
                usage = get_usage_manager()
                usage.charge_text(user_text)
                usage.charge_text(full_text or "")
            except Exception as exc:
                print(f"[dizel-api] usage record failed: {exc}", flush=True)

            _put({"type": "done", "text": full_text, "session_id": session_id})

        def on_error(message: str) -> None:
            _put({"type": "error", "message": message})

        # ── Usage window gate ────────────────────────────────────────────
        # Ported from the v1.0.2 GUI: compact at the threshold, allow five per
        # window, then refuse. Done here rather than in the renderer so it holds
        # for every client and cannot be skipped.
        usage = get_usage_manager()
        try:
            usage.set_capacity(int(manager.effective_ctx_len))
        except Exception:
            pass

        if usage.is_exhausted:
            note = (
                "**Usage window spent**\n\n"
                f"{int(usage.usage)} of {int(usage.max_capacity)} tokens used.\n\n"
                f"Resets in **{usage.time_until_reset()}**. Start a new chat, or wait "
                "for the window to reset."
            )
            yield _sse({"type": "done", "text": note, "session_id": session_id or ""})
            return

        if usage.needs_compaction:
            if not usage.can_compact():
                note = (
                    "**Context limit reached**\n\n"
                    f"This chat has already been compacted **{usage.MAX_COMPACTIONS}** "
                    "times in this window.\n\nStart a **new chat**, or wait for the "
                    "usage window to reset."
                )
                yield _sse({"type": "done", "text": note, "session_id": session_id or ""})
                return
            try:
                manager.compact_context()
                usage.run_compaction()
                yield _sse(
                    {
                        "type": "notice",
                        "text": (
                            f"Context compacted at {usage.COMPACTION_THRESHOLD}% — "
                            f"earlier turns were summarised. "
                            f"{usage.compaction_count}/{usage.MAX_COMPACTIONS} compactions used."
                        ),
                    }
                )
            except Exception as exc:
                print(f"[dizel-api] compaction failed: {exc}", flush=True)

        def on_tool(call, outcome) -> None:
            # Surfaced as a system line so the user can see what ran, and what it
            # was asked to do, without it looking like part of the answer.
            name = call.get("name", "?")
            args = ", ".join(f"{k}={v!r}" for k, v in (call.get("args") or {}).items())
            preview = (outcome or "").strip().splitlines()
            first = preview[0][:90] if preview else "(no output)"
            try:
                get_usage_manager().charge_tool_calls(1)
            except Exception:
                pass
            _put({"type": "notice", "text": f"tool {name}({args}) → {first}"})

        terminal = False
        try:
            manager.send_with_tools(
                user_text=user_text,
                attachments=attachments or [],
                on_token=on_token,
                on_done=on_done,
                on_error=on_error,
                on_tool=on_tool,
                on_thinking=on_thinking,
                continuation=continuation,
            )

            # ── drain ────────────────────────────────────────────────────
            # A hard ceiling stops a wedged backend from holding the socket open
            # forever. The UI can always cancel client-side.
            deadline = asyncio.get_running_loop().time() + 900
            while True:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    yield _sse({"type": "error", "message": "Generation timed out."})
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=min(timeout, 15))
                except asyncio.TimeoutError:
                    # Comment frame keeps proxies/browsers from dropping the stream.
                    yield {"comment": "keepalive"}
                    continue

                yield _sse(event)
                if event["type"] in ("done", "error"):
                    terminal = True
                    return
        finally:
            # A client that hangs up mid-reply must not leave the generation
            # slot claimed; otherwise the next prompt (or new chat) is refused
            # with "already being generated" until the provider finishes.
            if not terminal and manager.is_generating:
                manager.stop_generation()


__all__ = ["ChatService"]


# ── Usage window ─────────────────────────────────────────────────────────
# The manager survived the rewrite; nothing exposed it. The old GUI showed a
# reset countdown and blocked sending once the 12-hour window was spent.
_usage_manager = None


def get_usage_manager():
    global _usage_manager
    if _usage_manager is None:
        from dizel_core.usage_manager import UsageManager

        _usage_manager = UsageManager()
    return _usage_manager


def usage_snapshot() -> Dict[str, Any]:
    manager = get_usage_manager()
    try:
        manager.set_capacity(int(get_manager().effective_ctx_len))
    except Exception:
        pass
    return {
        "used": int(manager.usage),
        "capacity": int(manager.max_capacity),
        "percent": manager.percentage,
        "exhausted": manager.is_exhausted,
        "resets_in": manager.time_until_reset(),
        "compactions_used": manager.compaction_count,
        "compactions_remaining": manager.compactions_remaining,
        "can_compact": manager.can_compact(),
    }
