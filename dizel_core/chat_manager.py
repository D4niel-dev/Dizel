"""
dizel_ui/logic/chat_manager.py
────────────────────────────────
Adapter between the desktop UI and the Dizel inference engine.

Design principles
-----------------
* The UI never calls torch directly — everything goes through ChatManager.
* Generation runs in a daemon background thread so the UI never freezes.
* ChatManager raises clear exceptions when the model is not ready yet.
* The caller provides two callbacks:
    on_token(str)   — called for each decoded piece (streaming)
    on_done(str)    — called once with the full response when generation ends
    on_error(str)   — called if something goes wrong

Usage (from the UI layer)
--------------------------
    mgr = ChatManager()
    mgr.load_model("checkpoints/dizel-sft-best.pt", device="cpu")

    mgr.send_message(
        user_text  = "Hello!",
        on_token   = lambda tok: print(tok, end="", flush=True),
        on_done    = lambda full: update_bubble(full),
        on_error   = lambda msg: show_error(msg),
    )
"""

import os
import sys
import threading
from typing import Callable, List, Dict, Optional

# Allow importing from project root. This module lives in dizel_core/, so the
# repo root is exactly one level up; resolving further up breaks checkpoints
# whose tokenizer path is stored relative to the repo.
_UI_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_UI_DIR)   # project root (Dizel/)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)


# ---------------------------------------------------------------------------
# Lazy imports — torch / model are only loaded when load_model() is called
# ---------------------------------------------------------------------------
_torch           = None
_F               = None
_DizelLM         = None
_Tokenizer       = None
_ROLE_TOKENS     = None
_END_TOKEN       = None
_CONFIG          = None


def _lazy_import() -> None:
    """Import heavy dependencies on first use."""
    global _torch, _F, _DizelLM, _Tokenizer, _ROLE_TOKENS, _END_TOKEN, _CONFIG
    if _torch is not None:
        return
    import torch
    import torch.nn.functional as F
    from model.architecture import DizelLM
    from training.dataset   import Tokenizer, ROLE_TOKENS, END_TOKEN
    from config             import CONFIG
    _torch       = torch
    _F           = F
    _DizelLM     = DizelLM
    _Tokenizer   = Tokenizer
    _ROLE_TOKENS = ROLE_TOKENS
    _END_TOKEN   = END_TOKEN
    _CONFIG      = CONFIG


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Dizel. You are not any other AI model or assistant — you are only Dizel. "
    "You are a highly capable, intelligent, and helpful AI assistant. "
    "You answer thoughtfully, concisely, and accurately. "
    "You use Markdown formatting to organize your thoughts and provide clear, structured text. "
    "Adapt your tone to the context — professional when needed, warm when appropriate."
)


from .config_manager import ConfigManager

# ---------------------------------------------------------------------------
# ChatManager
# ---------------------------------------------------------------------------
class ChatManager:
    """
    Manages model loading, conversation history, and async generation.

    Thread safety
    -------------
    load_model() and send_message() are safe to call from the UI thread.
    All heavy work happens in daemon threads and results are delivered
    via callbacks — the UI must marshal those callbacks to its own thread
    if needed (CustomTkinter's .after() is appropriate).
    """

    def __init__(self) -> None:
        self._backend        = None
        self._is_generating  = False
        self._stop_requested = False
        self._lock           = threading.Lock()
        # Guards the backend reference itself. Loading builds a backend off to
        # the side and swaps it in only after it is ready, so a failed or slow
        # load can never leave ChatManager pointing at a half-built model.
        self._backend_lock   = threading.RLock()
        self._load_in_progress = False

        # Load persistent settings
        self._cfg = ConfigManager.load()

        # Conversation state
        self.history:         List[Dict] = []
        self.system_prompt:   str        = self._cfg.get("system_prompt", SYSTEM_PROMPT)
        self.session_id:      Optional[str] = None

        # Active model variant & mode (set via apply_profile)
        self._active_model: str = "Dizel"
        self._active_mode:  str = "Fast"
        self._active_profile = None  # ModelProfile, set by apply_profile()
        self.last_budget: int = 0    # tokens granted to the most recent reply
        self.last_task: str = ""     # intent classified for that reply

        # Sampling defaults (from disk)
        samp = self._cfg.get("sampling", {})
        self.temperature         = samp.get("temperature", 0.8)
        self.top_k               = samp.get("top_k", 50)
        self.top_p               = samp.get("top_p", 0.92)
        self.repetition_penalty  = samp.get("repetition_penalty", 1.15)
        self.max_new_tokens      = samp.get("max_new_tokens", 2048)

        # Detect whether the user has customized sampling in Settings.
        # When customized, the active profile's / task-type sampling must
        # NOT silently replace the user's manual slider values during
        # local generation (see _generate_worker Step 6). Defaults come
        # from ConfigManager._DEFAULTS so an untouched settings.json
        # compares equal and the flag stays False.
        _default_samp = ConfigManager._DEFAULTS.get("sampling", {})
        self._sampling_customized = any(
            float(samp.get(k, _default_samp.get(k))) != float(_default_samp.get(k))
            for k in ("temperature", "top_k", "top_p", "repetition_penalty")
        )

        # ── API Provider state ──────────────────────────────────────
        self._provider_slug  = "local"
        self._api_provider   = None   # BaseProvider instance
        self._api_key        = ""
        self._api_model      = ""
        self._load_provider_state()

    def _load_provider_state(self) -> None:
        """Load API provider config from disk."""
        from .config_manager import decrypt_key
        api_cfg = self._cfg.get("api_router", {})
        self._provider_slug = api_cfg.get("provider", "local")
        self._api_model = api_cfg.get("model", "")
        self._api_key = decrypt_key(api_cfg.get("api_key", ""))

        if self._provider_slug != "local":
            try:
                from .providers import get_provider
                kwargs = {}
                if self._provider_slug == "ollama":
                    kwargs["ollama_url"] = api_cfg.get("ollama_url", "http://localhost:11434")
                elif self._provider_slug == "custom":
                    kwargs["custom_name"] = api_cfg.get("custom_name", "")
                    kwargs["custom_url"] = api_cfg.get("custom_url", "")
                self._api_provider = get_provider(self._provider_slug, **kwargs)
                print(f"[ChatManager] API provider loaded: {self._provider_slug}", flush=True)
            except Exception as e:
                print(f"[ChatManager] Failed to load provider '{self._provider_slug}': {e}", flush=True)
                self._api_provider = None
                self._provider_slug = "local"

        # Auto-select first available model when none is saved
        if self._provider_slug != "local" and not self._api_model:
            avail = api_cfg.get("available_models", [])
            if avail:
                first = avail[0]
                m_id = first.get("id") if isinstance(first, dict) else getattr(first, "id", "")
                if m_id:
                    self._api_model = m_id
                    api_cfg["model"] = m_id
                    self._cfg["api_router"] = api_cfg
                    from .config_manager import ConfigManager
                    ConfigManager.save(self._cfg)
                    print(f"[ChatManager] Auto-selected model: {m_id}", flush=True)

    def reload_provider(self) -> None:
        """Re-read provider config (called after settings change)."""
        self._cfg = ConfigManager.load()
        self._load_provider_state()

    def reload_sampling(self) -> None:
        """Re-read sampling settings after the user edits them.

        The values captured in __init__ go stale the moment Settings writes a
        new sampling block, so anything that saves settings must call this.
        """
        self._cfg = ConfigManager.load()
        samp = self._cfg.get("sampling", {})

        self.temperature = samp.get("temperature", self.temperature)
        self.top_k = samp.get("top_k", self.top_k)
        self.top_p = samp.get("top_p", self.top_p)
        self.repetition_penalty = samp.get("repetition_penalty", self.repetition_penalty)
        self.max_new_tokens = samp.get("max_new_tokens", self.max_new_tokens)

        defaults = ConfigManager._DEFAULTS.get("sampling", {})
        self._sampling_customized = any(
            float(samp.get(k, defaults.get(k))) != float(defaults.get(k))
            for k in ("temperature", "top_k", "top_p", "repetition_penalty")
        )

        print(
            f"[ChatManager] Sampling reloaded: temp={self.temperature} "
            f"top_k={self.top_k} top_p={self.top_p} max={self.max_new_tokens} "
            f"custom={self._sampling_customized}",
            flush=True,
        )

    def apply_profile(self, model_name: str, mode_name: str) -> None:
        """
        Apply a model variant + mode profile.

        Changes the system prompt and stores the profile so that
        _generate_worker can use its sampling overrides and budget multiplier.
        """
        from .token_budget import get_model_profile
        profile = get_model_profile(model_name, mode_name)

        self._active_model   = model_name
        self._active_mode    = mode_name
        self._active_profile = profile
        self.system_prompt   = profile.system_prompt

        print(f"[ChatManager] Profile applied: {profile.label}", flush=True)

    # ── Model loading ──────────────────────────────────────────────────

    def load_model(
        self,
        checkpoint_path: str,
        device:          str = None,
        on_progress:     Callable[[str], None] = None,
    ) -> None:
        """
        Load the model from a checkpoint file.

        Runs synchronously (blocking). Call from a background thread in the
        UI to avoid freezing while the model loads.

        The new backend is built and loaded into a local variable, then swapped
        in under ``_backend_lock``. Until that swap the previously loaded model
        stays fully usable; if the load fails it is left untouched. Loading and
        generation are mutually exclusive so a backend can never be freed while
        a generation thread is still inside it.

        Parameters
        ----------
        checkpoint_path : path to the .pt checkpoint
        device          : "cuda" or "cpu"; auto-detected if None
        on_progress     : optional callback(msg: str) for status updates
        """
        def _report(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        with self._backend_lock:
            if self._is_generating:
                raise RuntimeError(
                    "A response is still being generated. Stop it before "
                    "loading another model."
                )
            if self._load_in_progress:
                raise RuntimeError("A model load is already in progress.")
            self._load_in_progress = True

        backend = None
        try:
            _report("Checking checkpoint…")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: '{checkpoint_path}'\n"
                    "Train the model first or update the path in Settings."
                )

            _report("Resolving device…")
            # Auto-detect GPU if no device specified
            if device is None:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            _report("Importing inference backend…")
            from .model_backend import NativeBackend, GGUFBackend, SafeTensorsBackend

            _report("Preparing backend…")
            ext = os.path.splitext(checkpoint_path)[1].lower()
            if ext == ".gguf":
                backend = GGUFBackend()
                kwargs = {}
                if hasattr(self, "_cfg"):
                    kwargs["chat_format"] = self._cfg.get("advanced", {}).get("gguf_chat_format", "auto")
            elif ext == ".safetensors":
                backend = SafeTensorsBackend()
                kwargs = {}
            else:
                backend = NativeBackend()
                kwargs = {}

            _report(f"Loading {os.path.basename(checkpoint_path)}…")
            # load() runs synchronously and may take a while
            backend.load(checkpoint_path, device, _report, **kwargs)
        except BaseException:
            # A half-loaded backend can own gigabytes of native memory and a
            # llama.cpp context. Release it before the error reaches the UI.
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            raise
        else:
            with self._backend_lock:
                previous = self._backend
                self._backend = backend
                self._device = device
                self._checkpoint_path = checkpoint_path
            if previous is not None and previous is not backend:
                # Release after the swap: a failure here cannot lose the new
                # model, and the old one is no longer reachable by generation.
                _report("Releasing previous model…")
                try:
                    previous.close()
                except Exception:
                    pass
            _report("Model ready.")
        finally:
            with self._backend_lock:
                self._load_in_progress = False

    def unload_model(self) -> None:
        """Release the local model and its native memory.

        Refused while a response is generating: the generation thread holds a
        reference to the backend and would otherwise use freed weights.
        """
        with self._backend_lock:
            if self._is_generating:
                raise RuntimeError(
                    "A response is still being generated. Stop it before "
                    "unloading the model."
                )
            previous = self._backend
            self._backend = None
            self._checkpoint_path = ""
        if previous is not None:
            try:
                previous.close()
            except Exception:
                pass

    def count_tokens(self, text: str) -> int:
        """
        Real token count when a model is loaded, 4-chars/token otherwise.

        The estimate is only a fallback: it ignores the system prompt, markdown
        overhead and attachments, which is why the context meter used to read low.
        """
        if not text:
            return 0
        backend = self._current_backend()
        if backend is not None and backend.is_ready:
            try:
                return int(backend.count_tokens(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _current_backend(self):
        """The backend reference generation should use, read atomically."""
        with self._backend_lock:
            return self._backend

    @property
    def exact_counts(self) -> bool:
        backend = self._current_backend()
        return backend is not None and backend.is_ready

    @property
    def is_ready(self) -> bool:
        # Ready if local model loaded OR an API provider is configured
        if self._provider_slug != "local" and self._api_provider is not None:
            return True
        backend = self._current_backend()
        return backend is not None and backend.is_ready

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    @property
    def is_stopping(self) -> bool:
        """True between a stop request and the worker's cleanup."""
        return self._stop_requested

    @property
    def active_provider_slug(self) -> str:
        return self._provider_slug

    @property
    def active_api_model(self) -> str:
        return self._api_model

    @property
    def effective_ctx_len(self) -> int:
        """
        Context window for whatever is actually answering.

        A local checkpoint reports its own `ctx_len`. Provider models carry no
        metadata at all, so the window comes from `api_router.context_length` in
        settings — without it the UI assumed 2048 for every provider, which is
        wrong by orders of magnitude for something like gpt-4o.
        """
        info = self.model_info or {}
        if info.get("ctx_len"):
            return int(info["ctx_len"])

        configured = (self._cfg.get("api_router") or {}).get("context_length") or 0
        try:
            configured = int(configured)
        except (TypeError, ValueError):
            configured = 0
        return configured if configured > 0 else 2048

    @property
    def checkpoint_path(self) -> str:
        """Path of the checkpoint currently loaded, or "" if none."""
        return getattr(self, "_checkpoint_path", "") or ""

    @property
    def model_info(self) -> dict:
        if self._provider_slug != "local" and self._api_provider is not None:
            from .providers import model_supports_vision

            return {
                "provider": getattr(self._api_provider, "label", self._provider_slug),
                "vision": model_supports_vision(
                    self._provider_slug, self._api_model
                ),
            }
        backend = self._current_backend()
        if backend is None or not backend.is_ready:
            return {}
        info = dict(backend.model_info)
        # Local Dizel backends are text-only; a multimodal backend must opt in
        # explicitly once it can actually accept image input.
        info.setdefault("vision", False)
        return info

    # ── Conversation management ────────────────────────────────────────

    def new_session(self) -> None:
        """Clear history and start fresh."""
        self.history.clear()
        self.session_id = None

    def clear_history(self) -> None:
        self.history.clear()

    def compact_context(self) -> str:
        """Compact conversation history into a short extractive summary.

        Keeps the last 4 messages; older messages are replaced by a system
        note containing a one-line summary. Returns the summary string.
        """
        from .context_trimmer import summarize_old_context
        if not self.history:
            return ""
        summary = summarize_old_context(self.history, max_messages=4)
        kept = self.history[-4:]
        note_content = f"[Compaction: {summary}]" if summary else "[Earlier conversation was compacted.]"
        self.history = [{"role": "system", "content": note_content}] + kept
        return summary or ""

    def get_history(self) -> List[Dict]:
        return list(self.history)

    def load_history(self, messages: List[Dict]) -> None:
        self.history = list(messages)

    # ── Message sending ────────────────────────────────────────────────

    def regenerate_last(
        self,
        on_token:    Callable[[str], None],
        on_done:     Callable[[str], None],
        on_error:    Callable[[str], None],
        verbosity:   Optional[str] = None,
    ) -> None:
        """Regenerate the last prompt by popping the assistant message."""
        if not self.history:
            on_error("History is empty.")
            return

        # Pop any trailing assistant messages
        while self.history and self.history[-1].get("role") == "assistant":
            self.history.pop()

        if not self.history or self.history[-1].get("role") != "user":
            on_error("Last message is not from the user.")
            return

        last_user_msg = self.history.pop()
        user_text = last_user_msg.get("content", "")
        attachments = last_user_msg.get("attachments", [])

        self.send_message(user_text, attachments, on_token, on_done, on_error, verbosity)

    def send_message(
        self,
        user_text:   str,
        attachments: List[str],
        on_token:    Callable[[str], None],
        on_done:     Callable[[str], None],
        on_error:    Callable[[str], None],
        verbosity:   Optional[str] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        continuation: bool = False,
    ) -> None:
        """
        Send a user message and generate a response asynchronously.

        Routes to API provider or local torch inference based on active provider.
        Callbacks are called from the background thread.
        """
        # --- Native Tools Interception ---
        if user_text.startswith("/sh "):
            cmd = user_text[4:].strip()
            import subprocess
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                output = result.stdout
                if result.stderr:
                    output += "\n[stderr]\n" + result.stderr
                if not output.strip():
                    output = "[Command executed successfully with no output]"
            except Exception as e:
                output = str(e)
            user_text = f"User ran shell command: `{cmd}`\nOutput:\n```\n{output}\n```\nAnalyze the output or acknowledge it."
            
        elif user_text.startswith("/read "):
            filepath = user_text[6:].strip()
            import os
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(8000)
                    user_text = f"User requested to read file: `{filepath}`\nContents:\n```\n{content}\n```\nAnalyze the file contents."
                except Exception as e:
                    user_text = f"User requested to read file: `{filepath}`, but an error occurred: {e}"
            else:
                user_text = f"User requested to read file: `{filepath}`, but it was not found."
                
        elif user_text.startswith("/py "):
            code = user_text[4:].strip()
            import subprocess
            try:
                result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=10)
                output = result.stdout
                if result.stderr:
                    output += "\n[stderr]\n" + result.stderr
                if not output.strip():
                    output = "[Python script executed successfully with no output]"
            except Exception as e:
                output = str(e)
            user_text = f"User ran python code:\n```python\n{code}\n```\nOutput:\n```\n{output}\n```\nAnalyze the result."
            
        elif user_text.startswith("/fetch "):
            url = user_text[7:].strip()
            import urllib.request
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                # Simple strip of HTML tags for text context
                import re
                text_content = re.sub('<[^<]+>', ' ', html)
                text_content = " ".join(text_content.split())
                user_text = f"User requested to fetch URL: `{url}`\nExtracted Text (truncated):\n```\n{text_content[:8000]}\n```\nAnalyze the content."
            except Exception as e:
                user_text = f"User requested to fetch URL: `{url}`, but an error occurred: {e}"
        # ---------------------------------

        if not self.is_ready:
            on_error("Model not loaded. Configure a provider or checkpoint in Settings.")
            return

        # Claim the generation slot before the thread starts. load_model()
        # checks the same flags under the same lock, so a backend can never be
        # freed while a generation thread is still inside it.
        with self._backend_lock:
            if self._load_in_progress:
                on_error("A model is loading. Try again when it finishes.")
                return
            if self._is_generating:
                on_error("Already generating. Please wait.")
                return

            backend = None
            # Route: API provider or local inference
            if self._provider_slug != "local" and self._api_provider is not None:
                thread = threading.Thread(
                    target=self._generate_via_api,
                    args=(
                        user_text,
                        attachments,
                        on_token,
                        on_done,
                        on_error,
                        on_thinking,
                        continuation,
                    ),
                    daemon=True,
                )
            elif self._backend is not None and self._backend.is_ready:
                backend = self._backend
                thread = threading.Thread(
                    target=self._generate_worker,
                    args=(
                        user_text,
                        attachments,
                        on_token,
                        on_done,
                        on_error,
                        verbosity,
                        backend,
                        continuation,
                    ),
                    daemon=True,
                )
            else:
                on_error("No local model loaded. Load a checkpoint in Settings first.")
                return
            # Claiming a fresh slot clears any stop left over from the previous
            # turn. Workers must not reset this themselves: a stop pressed right
            # after send would be erased when the thread finally starts.
            self._is_generating = True
            self._stop_requested = False

        try:
            thread.start()
        except Exception:
            with self._backend_lock:
                self._is_generating = False
            raise

    def stop_generation(self) -> None:
        """Request early stop of the current generation."""
        self._stop_requested = True
        backend = self._current_backend()
        if backend:
            backend.stop()

        # API providers stream over HTTP: closing the response is what actually
        # unblocks iter_lines(). Without this the worker keeps waiting for the
        # provider's next chunk, the generation slot stays claimed, and the next
        # prompt is refused as "already being generated".
        provider = self._api_provider
        if self._provider_slug != "local" and provider is not None:
            try:
                provider.cancel()
            except Exception:
                pass
            # Let the worker run its cleanup before the stop call returns, so a
            # prompt sent immediately afterwards is accepted.
            import time

            deadline = time.monotonic() + 1.0
            while self._is_generating and time.monotonic() < deadline:
                time.sleep(0.02)

    # ── API provider generation ────────────────────────────────────────

    def _commit_reply(self, text: str, continuation: bool) -> None:
        """Record an assistant reply.

        A continuation is not a new turn: the model kept writing the same answer,
        so the text is appended to the message that was already there. That keeps
        the saved transcript matching what the client shows, and stops a reload
        from revealing the synthetic prompt.
        """
        if continuation and self.history and self.history[-1].get("role") == "assistant":
            previous = self.history[-1].get("content", "")
            self.history[-1]["content"] = f"{previous.rstrip()}\n\n{text.strip()}"
        else:
            self.history.append({"role": "assistant", "content": text})

    def _generate_via_api(
        self,
        user_text: str,
        attachments: List[str],
        on_token:  Callable[[str], None],
        on_done:   Callable[[str], None],
        on_error:  Callable[[str], None],
        on_thinking: Optional[Callable[[str], None]] = None,
        continuation: bool = False,
    ) -> None:
        """Background thread: stream from an external API provider."""
        full_response: List[str] = []
        stopped = False

        def finish_stopped() -> None:
            # A stop is user intent, not a failure: keep whatever was already
            # on screen and end the turn instead of reporting an API error.
            partial = "".join(full_response)
            if partial:
                self._commit_reply(partial, continuation)
            on_done(partial)

        try:
            if self._stop_requested:
                finish_stopped()
                return
            images = [
                value
                for value in attachments
                if isinstance(value, str) and value.startswith("data:image/")
            ]
            # Record user message. Images ride with the turn so follow-ups in
            # the same conversation still give a vision model its context.
            if not continuation:
                entry = {"role": "user", "content": user_text}
                if images:
                    entry["images"] = images
                self.history.append(entry)

            # Build messages for the API (OpenAI format)
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            for msg in self.history:
                item = {
                    "role": msg["role"],
                    "content": msg.get("content", ""),
                }
                if msg.get("images"):
                    item["images"] = msg["images"]
                messages.append(item)

            # Get provider-specific kwargs
            kwargs = {}
            api_cfg = self._cfg.get("api_router", {})
            if self._provider_slug == "ollama":
                kwargs["ollama_url"] = api_cfg.get("ollama_url", "http://localhost:11434")
            elif self._provider_slug == "azure":
                kwargs["azure_resource"] = api_cfg.get("azure_resource", "")
                kwargs["azure_deployment"] = api_cfg.get("azure_deployment", "")

            token_count = 0
            for kind, piece in self._api_provider.stream_events(
                messages=messages,
                model=self._api_model,
                key=self._api_key,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                **kwargs,
            ):
                if self._stop_requested:
                    stopped = True
                    break
                if kind == "thinking":
                    # Reasoning stays out of the reply text and out of history;
                    # it is shown live and then discarded.
                    if on_thinking is not None:
                        on_thinking(piece)
                    continue
                full_response.append(piece)
                on_token(piece)
                token_count += 1

            if stopped:
                finish_stopped()
                return
            response_text = "".join(full_response)
            print(f"[ChatManager] API stream done: {token_count} tokens, total_len={len(response_text)}", flush=True)
            if token_count > 0 and len(response_text) < 500:
                print(f"[ChatManager]  full response={response_text!r}", flush=True)
            self._commit_reply(response_text, continuation)
            on_done(response_text)

        except (ConnectionError, RuntimeError, ImportError) as e:
            if self._stop_requested:
                finish_stopped()
            else:
                on_error(str(e))
        except Exception as e:
            if self._stop_requested:
                finish_stopped()
            else:
                on_error(f"API error: {e}")
        finally:
            with self._backend_lock:
                self._is_generating  = False
                self._stop_requested = False

    # ── Internal generation ────────────────────────────────────────────

    def _generate_worker(
        self,
        user_text:   str,
        attachments: List[str],
        on_token:    Callable[[str], None],
        on_done:     Callable[[str], None],
        on_error:    Callable[[str], None],
        verbosity:   Optional[str] = None,
        backend=None,
        continuation: bool = False,
    ) -> None:
        """Background thread: runs generation and fires callbacks."""
        import time

        # The caller claimed the generation slot and handed over the exact
        # backend to use. Holding that reference keeps the weights alive even
        # if a future lifecycle change swaps the manager's backend.
        backend = backend or self._current_backend()
        if backend is None:
            on_error("No local model loaded. Load a checkpoint in Settings first.")
            with self._backend_lock:
                self._is_generating = False
            return

        self._is_generating = True

        try:
            from .token_budget import (
                classify_task, allocate_token_budget, get_task_sampling,
                log_budget_decision, TaskType, SamplingOverrides,
            )
            from .context_trimmer import trim_context_if_needed, estimate_history_tokens

            ctx = backend.context_length

            # ── Step 1: Classify the task ────────────────────────────────
            has_tools = bool(attachments)
            task_type = classify_task(user_text, has_tools_active=has_tools)

            # ── Step 2: Add user turn to history ─────────────────────────
            msg_dict = {"role": "user", "content": user_text}
            if attachments:
                msg_dict["attachments"] = attachments
            if not continuation:
                self.history.append(msg_dict)
            print(f"[ChatManager] Generating response for: {user_text[:60]}...", flush=True)

            # ── Step 3: Smart context trimming ───────────────────────────
            budget_cfg = self._cfg.get("token_budget", {})
            # GUI context limit for local Dizel/Mila: 10k, but never
            # exceed the backend's real window (ctx from GGUF metadata).
            max_ctx_tokens = budget_cfg.get("max_context_tokens", 10000)
            if ctx and ctx > 0:
                max_ctx_tokens = min(int(max_ctx_tokens), max(int(ctx) - 100, 1))
            trimmed_history, n_trimmed = trim_context_if_needed(
                self.history, max_ctx_tokens,
            )
            # The drop is silent by nature — nothing in the reply says the model
            # never saw those turns. Publish the count so the UI can say it.
            self.last_trimmed = int(n_trimmed)

            # ── Step 4: Estimate token counts ────────────────────────────
            ctx_text = self.system_prompt + "".join([m["content"] for m in trimmed_history])
            context_tokens = backend.count_tokens(ctx_text)
            input_token_count = backend.count_tokens(user_text)
            
            print(f"[ChatManager] Context approx tokens: {context_tokens} (trimmed {n_trimmed} msgs)", flush=True)

            # ── Step 5: Allocate dynamic token budget ────────────────────
            # Try to get verbosity from the argument, fallback to config
            if verbosity is None:
                verbosity = budget_cfg.get("verbosity", "normal")
            
            hard_limit = budget_cfg.get("hard_output_limit", 0)
            custom_budgets = {
                "chat":       budget_cfg.get("chat_budget", 150),
                "coding":     budget_cfg.get("coding_budget", 350),
                "complex":    budget_cfg.get("complex_budget", 500),
                "factual":    budget_cfg.get("factual_budget", 100),
                "tool_based": budget_cfg.get("tool_budget", 300),
            }

            effective_max = allocate_token_budget(
                task_type=task_type,
                input_token_count=input_token_count,
                context_tokens=context_tokens,
                model_ctx_length=ctx,
                verbosity=verbosity,
                custom_budgets=custom_budgets,
                hard_output_limit=hard_limit,
            )

            # Apply model profile budget multiplier (Pro/Planning = more tokens)
            if self._active_profile:
                effective_max = int(effective_max * self._active_profile.budget_multiplier)
                # Leave 10% of remaining context as breathing room
                effective_max = max(30, min(effective_max, int((ctx - context_tokens) * 0.9)))

            self.last_budget = int(effective_max)
            self.last_task = getattr(task_type, "value", str(task_type))

            # ── Step 5b: Honor the user's explicit max-new-tokens setting ──
            # settings.json / Settings dialog stores sampling.max_new_tokens
            # into ChatManager.max_new_tokens. The dynamic task budget above
            # silently capped output at ~70-105 tokens (e.g. FACTUAL 100 ×
            # Fast profile 0.7) regardless of what the user configured, which
            # made the GUI appear to ignore Settings. The user's explicit
            # setting is authoritative for output length: it is scaled by the
            # active profile multiplier and bounded only by the remaining
            # context window (10% breathing room).
            user_max = int(getattr(self, "max_new_tokens", 0) or 0)
            ctx_safe = int((ctx - context_tokens) * 0.9)
            if user_max > 0:
                mult = self._active_profile.budget_multiplier if self._active_profile else 1.0
                effective_max = min(int(user_max * mult), ctx_safe)
            else:
                effective_max = min(effective_max, ctx_safe)
            effective_max = max(30, effective_max)
            print(f"[budget] user_max={user_max} -> effective_max={effective_max}", flush=True)

            # ── Step 6: Apply sampling overrides ─────────────────────────
            # The user's manual Settings values are authoritative once they
            # have customized them. Otherwise (flag False), profile sampling
            # takes priority and task sampling is the fallback.
            if self._sampling_customized:
                sampling = SamplingOverrides(
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                )
            else:
                if self._active_profile:
                    sampling = self._active_profile.sampling
                else:
                    sampling = get_task_sampling(task_type)

            saved_temp   = self.temperature
            saved_top_k  = self.top_k
            saved_top_p  = self.top_p
            saved_rep    = self.repetition_penalty

            self.temperature        = sampling.temperature
            self.top_k              = sampling.top_k
            self.top_p              = sampling.top_p
            self.repetition_penalty = sampling.repetition_penalty

            print(f"[sampling] user_customized={self._sampling_customized} temp={self.temperature} top_k={self.top_k} top_p={self.top_p} rep={self.repetition_penalty}", flush=True)

            # ── Step 7: Log the decision ─────────────────────────────────
            profile_label = self._active_profile.label if self._active_profile else "default"
            log_budget_decision(
                task_type=task_type,
                budget=effective_max,
                context_tokens=context_tokens,
                model_ctx_length=ctx,
                verbosity=verbosity,
                trimmed_msgs=n_trimmed,
                sampling=sampling,
            )
            print(f"[budget] profile={profile_label}", flush=True)

            # ── Step 8: Delegate to ModelBackend ─────────────────────────
            
            sampling_params = {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
            }
            
            full_response = backend.generate(
                messages=trimmed_history,
                system_prompt=self.system_prompt,
                sampling_params=sampling_params,
                max_tokens=effective_max,
                on_token=on_token
            )

            if full_response:
                print(f"[ChatManager] Response: {full_response[:100]}...", flush=True)
            else:
                print("[ChatManager] Response was EMPTY", flush=True)

            # Handle empty responses (model hit end token immediately)
            if not full_response:
                full_response = "(The model produced an empty response. Try rephrasing your question or adjusting sampling settings.)"

            # Append assistant turn to history
            self._commit_reply(full_response, continuation)

            on_done(full_response)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            on_error(f"Generation error: {exc}")
        finally:
            # Restore original sampling params (undo task-specific overrides)
            try:
                self.temperature        = saved_temp
                self.top_k              = saved_top_k
                self.top_p              = saved_top_p
                self.repetition_penalty = saved_rep
            except NameError:
                pass  # saved_* vars not yet assigned if error happened early
            with self._backend_lock:
                self._is_generating  = False
                self._stop_requested = False
            print("[ChatManager] Generation finished, _is_generating = False", flush=True)

    # ── Tool-augmented generation ──────────────────────────────────────

    def send_with_tools(
        self,
        user_text:   str,
        attachments: List[str],
        on_token:    Callable[[str], None],
        on_done:     Callable[[str], None],
        on_error:    Callable[[str], None],
        verbosity:   Optional[str] = None,
        max_rounds:  int = 3,
        on_tool:     Optional[Callable[[dict, str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        continuation: bool = False,
    ) -> None:
        """Answer a prompt, running any tools the model asks for on the way.

        `send_message` is threaded, so this chains on the callback instead of
        looping inline: done -> parse -> run -> send again -> done.

        Bounded by `max_rounds` so a model that keeps asking for tools cannot spin
        forever, and the system prompt is restored on every exit path.
        """
        from . import tools as tool_registry

        # Opt-in: the prompt block costs tokens on every request, which is real
        # money at a 2,048-token window.
        if not tool_registry.is_enabled():
            self.send_message(
                user_text,
                attachments,
                on_token,
                on_done,
                on_error,
                verbosity,
                on_thinking,
                continuation,
            )
            return

        execute = tool_registry.execute
        parse_calls = tool_registry.parse_calls

        original = self.system_prompt
        self.system_prompt = (original or "") + "\n\n" + tool_registry.prompt_block()
        state = {"round": 0, "done": False}

        def restore() -> None:
            self.system_prompt = original

        def fail(message: str) -> None:
            if state["done"]:
                return
            state["done"] = True
            restore()
            on_error(message)

        def round_done(text: str) -> None:
            if state["done"]:
                return
            state["round"] += 1
            calls = parse_calls(text)

            if not calls:
                state["done"] = True
                restore()
                on_done(text)
                return

            if state["round"] >= max_rounds:
                state["done"] = True
                restore()
                on_done(
                    text
                    + f"\n\n_[stopped after {max_rounds} tool rounds]_"
                )
                return

            results = []
            for call in calls:
                outcome = execute(call)
                if on_tool is not None:
                    try:
                        on_tool(call, outcome)
                    except Exception:
                        pass
                results.append(
                    f'<tool_result name="{call.get("name", "")}">\n{outcome}\n</tool_result>'
                )

            try:
                self.send_message(
                    "Tool output://n//n"
                    + "\n\n".join(results)
                    + "\n\nContinue and answer the original question.",
                    [],
                    on_token,
                    round_done,
                    fail,
                    verbosity,
                )
            except Exception as error:
                fail(f"Tool round failed: {error}")

        try:
            self.send_message(user_text, attachments, on_token, round_done, fail, verbosity)
        except Exception as error:
            fail(str(error))

    def send_message_with_tools(
        self,
        augmented_text: str,
        deep_think:     bool,
        attachments:    List[str],
        on_token,
        on_done,
        on_error,
        verbosity:      Optional[str] = None,
    ) -> None:
        """
        Send a message with tool-augmented context.

        If Deep Think is active, temporarily overrides generation params
        (higher max_new_tokens, lower temperature). Params are restored
        after generation completes.
        """
        if not deep_think:
            self.send_message(augmented_text, attachments, on_token, on_done, on_error, verbosity)
            return

        # Save originals
        from core.generation_modes import get_deep_think_overrides
        overrides = get_deep_think_overrides()

        orig_max   = self.max_new_tokens
        orig_temp  = self.temperature
        orig_topk  = self.top_k
        orig_topp  = self.top_p
        orig_sys   = self.system_prompt

        # Apply overrides
        if overrides.max_new_tokens is not None:
            self.max_new_tokens = overrides.max_new_tokens
        if overrides.temperature is not None:
            self.temperature = overrides.temperature
        if overrides.top_k is not None:
            self.top_k = overrides.top_k
        if overrides.top_p is not None:
            self.top_p = overrides.top_p
        if overrides.system_addendum:
            self.system_prompt += overrides.system_addendum

        print(
            f"[ChatManager] Deep Think ON: max_tokens={self.max_new_tokens}, "
            f"temp={self.temperature}, top_k={self.top_k}",
            flush=True,
        )

        def _restore():
            self.max_new_tokens = orig_max
            self.temperature    = orig_temp
            self.top_k          = orig_topk
            self.top_p          = orig_topp
            self.system_prompt  = orig_sys
            print("[ChatManager] Deep Think params restored", flush=True)

        def on_done_wrapper(full_text):
            _restore()
            on_done(full_text)

        def on_error_wrapper(msg):
            _restore()
            on_error(msg)

        self.send_message(augmented_text, attachments, on_token, on_done_wrapper, on_error_wrapper, verbosity)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
__all__ = ["ChatManager", "SYSTEM_PROMPT"]

