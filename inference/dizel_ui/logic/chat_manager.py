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

# Allow importing from project root
_UI_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_UI_DIR)))   # project root (Dizel/)
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
    "You are Dizel, a highly capable, intelligent, and helpful AI assistant. "
    "You answer thoughtfully, concisely, and accurately. "
    "You use formatting like markdown to organize your thoughts and provide clear, structured text."
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
        self._model          = None
        self._tokenizer      = None
        self._device         = "cpu"
        self._is_generating  = False
        self._stop_requested = False
        self._lock           = threading.Lock()

        # Load persistent settings
        self._cfg = ConfigManager.load()

        # Conversation state
        self.history:         List[Dict] = []
        self.system_prompt:   str        = self._cfg.get("system_prompt", SYSTEM_PROMPT)
        self.session_id:      Optional[str] = None

        # Active model variant & mode (set via apply_profile)
        self._active_model: str = "Dizel Lite"
        self._active_mode:  str = "Fast"
        self._active_profile = None  # ModelProfile, set by apply_profile()

        # Sampling defaults (from disk)
        samp = self._cfg.get("sampling", {})
        self.temperature         = samp.get("temperature", 0.8)
        self.top_k               = samp.get("top_k", 50)
        self.top_p               = samp.get("top_p", 0.92)
        self.repetition_penalty  = samp.get("repetition_penalty", 1.15)
        self.max_new_tokens      = samp.get("max_new_tokens", 200)

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

        Parameters
        ----------
        checkpoint_path : path to the .pt checkpoint
        device          : "cuda" or "cpu"; auto-detected if None
        on_progress     : optional callback(msg: str) for status updates
        """
        def _report(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        _report("Importing dependencies…")
        _lazy_import()

        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint_path}'\n"
                "Train the model first or update the path in Settings."
            )

        _report(f"Loading checkpoint from {checkpoint_path}…")
        ckpt      = _torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_cfg = ckpt.get("model_cfg", _CONFIG.model)

        _report("Building model…")
        state_dict = ckpt["model_state"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Detect checkpoint version: v1.1 has pos_emb, v1.2 uses RoPE
        is_legacy = "transformer.pos_emb.weight" in cleaned

        if is_legacy:
            _report("Detected v1.1 checkpoint (learned pos_emb) — loading in legacy mode…")
            # Add pos_emb to the model so the state_dict can load
            pos_emb_weight = cleaned["transformer.pos_emb.weight"]
            ctx_len, d_model = pos_emb_weight.shape

            model = _DizelLM(model_cfg).to(device)
            # Register the missing parameter so load_state_dict succeeds
            model.transformer["pos_emb"] = _torch.nn.Embedding(ctx_len, d_model).to(device)

            model.load_state_dict(cleaned)

            # Monkey-patch forward to ADD pos_emb to token embeddings (v1.1 behavior)
            _original_forward = model.forward
            def _legacy_forward(idx, targets=None, loss_mask=None):
                B, T = idx.shape
                tok = model.transformer["tok_emb"](idx)
                pos = model.transformer["pos_emb"](_torch.arange(T, device=idx.device))
                x = model.transformer["emb_drop"](tok + pos)
                for block in model.transformer["blocks"]:
                    x = block(x)
                x = model.transformer["ln_f"](x)
                logits = model.lm_head(x)
                if targets is None:
                    return logits, None
                loss = _torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
                return logits, loss
            model.forward = _legacy_forward
        else:
            _report("Detected v1.2 checkpoint (RoPE) — loading normally…")
            model = _DizelLM(model_cfg).to(device)
            model.load_state_dict(cleaned)

        model.eval()

        _report("Loading tokenizer…")
        # Prefer embedded tokenizer from checkpoint (guarantees vocab match)
        if "tokenizer_model" in ckpt:
            import tempfile
            tok_bytes = ckpt["tokenizer_model"]
            tok_tmp = os.path.join(os.path.dirname(checkpoint_path), "_embedded_tokenizer.model")
            with open(tok_tmp, "wb") as f:
                f.write(tok_bytes)
            tokenizer = _Tokenizer(model_path=tok_tmp)
            _report(f"Using embedded tokenizer (vocab={len(tokenizer)})")
        else:
            tokenizer_path = os.path.join(_ROOT_DIR, _CONFIG.tokenizer.model_path)
            tokenizer = _Tokenizer(model_path=tokenizer_path)

        with self._lock:
            self._model     = model
            self._tokenizer = tokenizer

        step = ckpt.get("step", "?")
        vl   = ckpt.get("val_loss", float("inf"))
        _report(
            f"Ready  — step={step}  val_loss={vl:.4f}  "
            f"params={model.num_parameters()/1e6:.2f}M  device={device}"
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    @property
    def model_info(self) -> dict:
        if not self.is_ready:
            return {}
        return {
            "params":   f"{self._model.num_parameters()/1e6:.2f} M",
            "device":   self._device,
            "d_model":  self._model.cfg.d_model,
            "n_layers": self._model.cfg.n_layers,
            "n_heads":  self._model.cfg.n_heads,
            "ctx_len":  self._model.cfg.context_length,
            "vocab":    self._model.cfg.vocab_size,
        }

    # ── Conversation management ────────────────────────────────────────

    def new_session(self) -> None:
        """Clear history and start fresh."""
        self.history.clear()
        self.session_id = None

    def clear_history(self) -> None:
        self.history.clear()

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

        self.send_message(user_text, attachments, on_token, on_done, on_error)

    def send_message(
        self,
        user_text:   str,
        attachments: List[str],
        on_token:    Callable[[str], None],
        on_done:     Callable[[str], None],
        on_error:    Callable[[str], None],
    ) -> None:
        """
        Send a user message and generate a response asynchronously.

        Callbacks are called from the background thread — use
        root.after(0, callback) if you need to update CTk widgets.
        """
        if not self.is_ready:
            on_error("Model not loaded. Please select a checkpoint in Settings.")
            return

        if self._is_generating:
            on_error("Already generating. Please wait.")
            return

        thread = threading.Thread(
            target=self._generate_worker,
            args=(user_text, attachments, on_token, on_done, on_error),
            daemon=True,
        )
        thread.start()

    def stop_generation(self) -> None:
        """Request early stop of the current generation."""
        self._stop_requested = True

    # ── Internal generation ────────────────────────────────────────────

    def _build_prompt_ids(self, history_subset=None) -> list:
        """
        Build the full token id sequence for the current conversation.
        Follows the same format as inference/chat.py.
        """
        ids = []
        messages = [{"role": "system", "content": self.system_prompt}] + (history_subset if history_subset is not None else self.history)
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            ids += self._tokenizer.encode(_ROLE_TOKENS[role])
            ids += self._tokenizer.encode(content)
            ids += self._tokenizer.encode(_END_TOKEN)
        # Prime model to respond as assistant
        ids += self._tokenizer.encode(_ROLE_TOKENS["assistant"])
        return ids

    def _trim_history_if_needed(self, prompt_ids: list, effective_max: int) -> list:
        """If the prompt is too long, drop the oldest turns from the prompt only (preserves self.history)."""
        ctx = self._model.cfg.context_length
        # Ensure we always reserve at least 16 tokens for the prompt
        max_allowed = max(ctx - effective_max - 10, 16)
        print(f"[ChatManager] max_allowed={max_allowed} (ctx={ctx}, effective_max={effective_max})", flush=True)

        local_history = list(self.history)
        prev_len = -1
        while len(prompt_ids) > max_allowed and len(local_history) > 2:
            # Guard against infinite loop when history can't be trimmed further
            if len(prompt_ids) == prev_len:
                print(f"[ChatManager] WARNING: Cannot trim history further, hard-truncating prompt from {len(prompt_ids)} to {max_allowed} tokens.", flush=True)
                # Keep the LAST max_allowed tokens (most recent context)
                prompt_ids = prompt_ids[-max_allowed:]
                break
            prev_len = len(prompt_ids)
            # Drop the oldest 2 messages (one user, one assistant exchange)
            local_history = local_history[2:]
            prompt_ids   = self._build_prompt_ids(local_history)

        # Final safety: never return an empty prompt
        if not prompt_ids:
            print("[ChatManager] ERROR: prompt_ids is empty after trimming! Rebuilding.", flush=True)
            local_history = self.history[-2:] if len(self.history) >= 2 else self.history
            prompt_ids = self._build_prompt_ids(local_history)

        return prompt_ids

    def _generate_worker(
        self,
        user_text:   str,
        attachments: List[str],
        on_token:    Callable[[str], None],
        on_done:     Callable[[str], None],
        on_error:    Callable[[str], None],
    ) -> None:
        """Background thread: runs generation and fires callbacks."""
        import time

        self._is_generating  = True
        self._stop_requested = False

        try:
            from .token_budget import (
                classify_task, allocate_token_budget, get_task_sampling,
                log_budget_decision, TaskType,
            )
            from .context_trimmer import trim_context_if_needed, estimate_history_tokens

            ctx = self._model.cfg.context_length

            # ── Step 1: Classify the task ────────────────────────────────
            has_tools = bool(attachments)
            task_type = classify_task(user_text, has_tools_active=has_tools)

            # ── Step 2: Add user turn to history ─────────────────────────
            msg_dict = {"role": "user", "content": user_text}
            if attachments:
                msg_dict["attachments"] = attachments
            self.history.append(msg_dict)
            print(f"[ChatManager] Generating response for: {user_text[:60]}...", flush=True)

            # ── Step 3: Smart context trimming ───────────────────────────
            budget_cfg = self._cfg.get("token_budget", {})
            max_ctx_tokens = budget_cfg.get("max_context_tokens", ctx - 100)
            trimmed_history, n_trimmed = trim_context_if_needed(
                self.history, max_ctx_tokens,
            )

            # ── Step 4: Build prompt from (potentially trimmed) history ──
            prompt_ids = self._build_prompt_ids(trimmed_history)
            print(f"[ChatManager] Prompt length: {len(prompt_ids)} tokens (trimmed {n_trimmed} msgs)", flush=True)

            # ── Step 5: Allocate dynamic token budget ────────────────────
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
                input_token_count=len(self._tokenizer.encode(user_text)),
                context_tokens=len(prompt_ids),
                model_ctx_length=ctx,
                verbosity=verbosity,
                custom_budgets=custom_budgets,
                hard_output_limit=hard_limit,
            )

            # Apply model profile budget multiplier (Pro/Planning = more tokens)
            if self._active_profile:
                effective_max = int(effective_max * self._active_profile.budget_multiplier)
                effective_max = max(30, min(effective_max, ctx - len(prompt_ids) - 10))

            # ── Step 6: Apply sampling overrides ─────────────────────────
            # Profile sampling takes priority, task sampling is the fallback
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

            # ── Step 7: Log the decision ─────────────────────────────────
            profile_label = self._active_profile.label if self._active_profile else "default"
            log_budget_decision(
                task_type=task_type,
                budget=effective_max,
                context_tokens=len(prompt_ids),
                model_ctx_length=ctx,
                verbosity=verbosity,
                trimmed_msgs=n_trimmed,
                sampling=sampling,
            )
            print(f"[budget] profile={profile_label}", flush=True)

            # Trim prompt if it still exceeds context window
            prompt_ids = self._trim_history_if_needed(prompt_ids, effective_max)

            # Build end-token set (stop generation at these ids)
            end_ids = [self._tokenizer.eos_id]
            for et in self._tokenizer.encode(_END_TOKEN):
                if et not in end_ids:
                    end_ids.append(et)

            # Limit PyTorch CPU threads to reduce GIL contention with Tkinter
            if self._device == "cpu":
                _torch.set_num_threads(2)

            ctx            = self._model.cfg.context_length
            idx            = _torch.tensor([prompt_ids], dtype=_torch.long,
                                           device=self._device)
            generated_ids  = []

            if idx.size(1) == 0:
                raise ValueError("Prompt is empty after trimming — try reducing max_new_tokens in Settings.")

            with _torch.no_grad():
                for step in range(effective_max):
                    if self._stop_requested:
                        print("[ChatManager] Stop requested, breaking.", flush=True)
                        break

                    input_ids = idx[:, -ctx:]

                    # AMP only on CUDA
                    if self._device == "cuda":
                        with _torch.amp.autocast(device_type="cuda",
                                                  dtype=_torch.bfloat16):
                            logits, _ = self._model(input_ids)
                    else:
                        logits, _ = self._model(input_ids)

                    logits = logits[:, -1, :].float()

                    # Repetition penalty (applied to recent 64 tokens)
                    if self.repetition_penalty != 1.0:
                        for tid in set(generated_ids[-64:]):
                            if 0 <= tid < logits.size(-1):
                                if logits[0, tid] > 0:
                                    logits[0, tid] /= self.repetition_penalty
                                else:
                                    logits[0, tid] *= self.repetition_penalty

                    # Temperature + top-k + top-p
                    if self.temperature == 0.0:
                        next_id = logits.argmax(dim=-1, keepdim=True)
                    else:
                        logits = logits / max(self.temperature, 1e-8)
                        if self.top_k > 0:
                            v, _ = _torch.topk(logits, min(self.top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = float("-inf")
                        if self.top_p < 1.0:
                            ps, si = _torch.sort(_F.softmax(logits, dim=-1), descending=True)
                            ps[ps.cumsum(-1) - ps > self.top_p] = 0.0
                            ps    /= ps.sum(-1, keepdim=True)
                            next_id = _torch.multinomial(ps, 1)
                            next_id = si.gather(-1, next_id)
                        else:
                            next_id = _torch.multinomial(_F.softmax(logits, dim=-1), 1)

                    tid = next_id.item()

                    if tid in end_ids:
                        print(f"[ChatManager] Hit end token at step {step}", flush=True)
                        break

                    generated_ids.append(tid)
                    idx = _torch.cat([idx, next_id], dim=1)

                    # Decode piece and stream it
                    piece = self._tokenizer.sp.id_to_piece(tid)
                    # Convert SentencePiece byte tokens (e.g. <0x0A> → \n)
                    if piece.startswith("<0x") and piece.endswith(">"):
                        try:
                            byte_val = int(piece[3:-1], 16)
                            piece = chr(byte_val)
                        except (ValueError, OverflowError):
                            pass
                    piece = piece.replace("▁", " ")
                    # Filter out special tokens leaking into output
                    if piece in ("<unk>", "⁇"):
                        continue
                    for special in list(_ROLE_TOKENS.values()) + [_END_TOKEN]:
                        piece = piece.replace(special, "")
                    if piece:
                        on_token(piece)

                    # Release GIL briefly so Tkinter can process UI events
                    time.sleep(0.001)

            # Full decoded response
            full_response = self._tokenizer.decode(generated_ids)
            for special in list(_ROLE_TOKENS.values()) + [_END_TOKEN]:
                full_response = full_response.replace(special, "")
            full_response = full_response.strip()

            print(f"[ChatManager] Generated {len(generated_ids)} tokens", flush=True)
            if full_response:
                print(f"[ChatManager] Response: {full_response[:100]}...", flush=True)
            else:
                print("[ChatManager] Response was EMPTY", flush=True)

            # Handle empty responses (model hit end token immediately)
            if not full_response:
                full_response = "(The model produced an empty response. Try rephrasing your question or adjusting sampling settings.)"

            # Append assistant turn to history
            self.history.append({"role": "assistant", "content": full_response})

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
            self._is_generating  = False
            self._stop_requested = False
            print("[ChatManager] Generation finished, _is_generating = False", flush=True)

    # ── Tool-augmented generation ──────────────────────────────────────

    def send_message_with_tools(
        self,
        augmented_text: str,
        deep_think:     bool,
        attachments:    List[str],
        on_token,
        on_done,
        on_error,
    ) -> None:
        """
        Send a message with tool-augmented context.

        If Deep Think is active, temporarily overrides generation params
        (higher max_new_tokens, lower temperature). Params are restored
        after generation completes.
        """
        if not deep_think:
            self.send_message(augmented_text, attachments, on_token, on_done, on_error)
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

        self.send_message(augmented_text, attachments, on_token, on_done_wrapper, on_error_wrapper)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
__all__ = ["ChatManager", "SYSTEM_PROMPT"]

