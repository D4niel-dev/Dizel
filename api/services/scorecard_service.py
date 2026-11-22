"""Background evaluation of the loaded local checkpoint.

Runs the fixed core suite against the loaded model, one case at a time, with
the user's chat history set aside, and stores the result as a scorecard. Only
one run at a time; generation is refused by the sidecar while it runs.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict

from dizel_core import scorecards
from evaluation import suite


class ScorecardService:
    _lock = threading.Lock()
    _state: Dict[str, Any] = {
        "running": False,
        "done": 0,
        "total": 0,
        "error": None,
        "scorecard": None,
        "path": "",
    }

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._state = {
                "running": False,
                "done": 0,
                "total": 0,
                "error": None,
                "scorecard": None,
                "path": "",
            }

    @classmethod
    def is_running(cls) -> bool:
        with cls._lock:
            return bool(cls._state["running"])

    @classmethod
    def state(cls) -> Dict[str, Any]:
        with cls._lock:
            return dict(cls._state)

    @classmethod
    def start(cls) -> Dict[str, Any]:
        from api.services.chat_service import ChatService

        manager = ChatService.get_manager()
        path = str(getattr(manager, "checkpoint_path", "") or "")
        provider = str(getattr(manager, "active_provider_slug", "local") or "local")

        with cls._lock:
            if cls._state["running"]:
                return {
                    "started": False,
                    "error": "A scorecard run is already in progress.",
                }
            if provider != "local" or not path:
                return {
                    "started": False,
                    "error": "Scorecards evaluate local checkpoints — load one first.",
                }
            if not getattr(manager, "is_ready", False):
                return {"started": False, "error": "The model is not ready yet."}
            if getattr(manager, "is_generating", False):
                return {
                    "started": False,
                    "error": "The model is generating — wait for the reply first.",
                }
            total = len(suite.EvalRunner().load_cases())
            cls._state = {
                "running": True,
                "done": 0,
                "total": total,
                "error": None,
                "scorecard": None,
                "path": path,
            }

        threading.Thread(
            target=cls._run, args=(manager, path), name="dizel-scorecard", daemon=True
        ).start()
        return {"started": True, "total": total}

    @classmethod
    def _generate(cls, manager, prompt: str, timeout: float = 180.0) -> str:
        done = threading.Event()
        box: Dict[str, Any] = {"text": "", "error": None}

        def on_token(token: str) -> None:
            pass

        def on_done(text: str) -> None:
            box["text"] = text
            done.set()

        def on_error(message: str) -> None:
            box["error"] = message
            done.set()

        manager.send_message(prompt, [], on_token, on_done, on_error)
        if not done.wait(timeout):
            raise TimeoutError("The model took too long to answer an evaluation case.")
        if box["error"]:
            raise RuntimeError(box["error"])
        return str(box["text"])

    @classmethod
    def _run(cls, manager, path: str) -> None:
        history = manager.get_history()
        system_prompt = getattr(manager, "system_prompt", "")
        try:
            manager.system_prompt = ""
            manager.load_history([])

            def handler(case_prompt: str) -> str:
                text = cls._generate(manager, case_prompt)
                with cls._lock:
                    cls._state["done"] += 1
                return text

            card = suite.run_core_suite(handler, checkpoint=os.path.basename(path))
            record = scorecards.save(path, card)
            with cls._lock:
                cls._state.update(running=False, scorecard=scorecards.summary(record))
        except Exception as error:  # a failed run must leave the app usable
            with cls._lock:
                cls._state.update(running=False, error=str(error))
        finally:
            manager.system_prompt = system_prompt
            manager.load_history(history)
