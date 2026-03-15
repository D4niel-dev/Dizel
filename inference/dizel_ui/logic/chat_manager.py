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
    "You are Dizel, a structured analytical AI model. "
    "Prioritize clarity, precision, and logical organization. "
    "Avoid unnecessary verbosity. "
    "If uncertain, explicitly state limitations."
)


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

        # Conversation state
        self.history:         List[Dict] = []
        self.system_prompt:   str        = SYSTEM_PROMPT
        self.session_id:      Optional[str] = None

        # Sampling defaults (can be changed at runtime)
        self.temperature         = 0.8
        self.top_k               = 50
        self.top_p               = 0.92
        self.repetition_penalty  = 1.15
        self.max_new_tokens      = 200

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
        model = _DizelLM(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        _report("Loading tokenizer…")
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

    def send_message(
        self,
        user_text:  str,
        on_token:   Callable[[str], None],
        on_done:    Callable[[str], None],
        on_error:   Callable[[str], None],
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
            args=(user_text, on_token, on_done, on_error),
            daemon=True,
        )
        thread.start()

    def stop_generation(self) -> None:
        """Request early stop of the current generation."""
        self._stop_requested = True

    # ── Internal generation ────────────────────────────────────────────

    def _build_prompt_ids(self) -> list:
        """
        Build the full token id sequence for the current conversation.
        Follows the same format as inference/chat.py.
        """
        ids = []
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            ids += self._tokenizer.encode(_ROLE_TOKENS[role])
            ids += self._tokenizer.encode(content)
            ids += self._tokenizer.encode(_END_TOKEN)
        # Prime model to respond as assistant
        ids += self._tokenizer.encode(_ROLE_TOKENS["assistant"])
        return ids

    def _trim_history_if_needed(self, prompt_ids: list) -> list:
        """If the prompt is too long, drop the oldest turns."""
        max_allowed = self._model.cfg.context_length - self.max_new_tokens - 10
        while len(prompt_ids) > max_allowed and len(self.history) > 2:
            self.history = self.history[-4:]
            prompt_ids   = self._build_prompt_ids()
        return prompt_ids

    def _generate_worker(
        self,
        user_text: str,
        on_token:  Callable[[str], None],
        on_done:   Callable[[str], None],
        on_error:  Callable[[str], None],
    ) -> None:
        """Background thread: runs generation and fires callbacks."""
        self._is_generating  = True
        self._stop_requested = False

        try:
            # Add user turn to history
            self.history.append({"role": "user", "content": user_text})

            prompt_ids  = self._build_prompt_ids()
            prompt_ids  = self._trim_history_if_needed(prompt_ids)

            # Build end-token set (stop generation at these ids)
            end_ids = [self._tokenizer.eos_id]
            for et in self._tokenizer.encode(_END_TOKEN):
                if et not in end_ids:
                    end_ids.append(et)

            ctx            = self._model.cfg.context_length
            idx            = _torch.tensor([prompt_ids], dtype=_torch.long,
                                           device=self._device)
            generated_ids  = []

            with _torch.no_grad():
                for _ in range(self.max_new_tokens):
                    if self._stop_requested:
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
                        break

                    generated_ids.append(tid)
                    idx = _torch.cat([idx, next_id], dim=1)

                    # Decode piece and stream it
                    piece = self._tokenizer.sp.id_to_piece(tid).replace("▁", " ")
                    # Filter out special tokens leaking into output
                    if piece in ("<unk>", "⁇"):
                        continue
                    for special in list(_ROLE_TOKENS.values()) + [_END_TOKEN]:
                        piece = piece.replace(special, "")
                    if piece:
                        on_token(piece)

            # Full decoded response
            full_response = self._tokenizer.decode(generated_ids)
            for special in list(_ROLE_TOKENS.values()) + [_END_TOKEN]:
                full_response = full_response.replace(special, "")
            full_response = full_response.strip()

            # Append assistant turn to history
            self.history.append({"role": "assistant", "content": full_response})

            on_done(full_response)

        except Exception as exc:
            on_error(f"Generation error: {exc}")
        finally:
            self._is_generating  = False
            self._stop_requested = False


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
__all__ = ["ChatManager", "SYSTEM_PROMPT"]
