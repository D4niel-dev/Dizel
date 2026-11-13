"""
dizel_ui/logic/model_backend.py
───────────────────────────────
Abstracts local inference across different model formats (.pt, .gguf, .safetensors).
"""

import os
import sys
import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Optional, Any

# Ensure project root is in sys.path. This module lives in dizel_core/, so the
# repo root is exactly one level up; resolving further up breaks checkpoints
# whose tokenizer path is stored relative to the repo.
_UI_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_UI_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Legacy v1.2 decoder: LayerNorm + standard FFN (GELU)
# Used to load old v1.2.x checkpoints where the model had LayerNorm norms
# and a 2-projection feed-forward instead of RMSNorm + SwiGLU.
# ---------------------------------------------------------------------------
class _LegacyFFN(nn.Module):
    """Old v1.2 standard FFN: fc1 → GELU → fc2."""
    def __init__(self, d_model, ffn_dim, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _LegacyDecoderV12(nn.Module):
    """Old v1.2 decoder layer: LayerNorm → MHA → LayerNorm → FFN."""
    def __init__(self, cfg):
        super().__init__()
        from model.architecture import GQAAttention
        d_model = cfg.d_model
        ffn_dim = int(d_model * cfg.ffn_mult)
        self.norm1 = nn.LayerNorm(d_model, bias=False)
        self.ln1   = self.norm1  # some old checkpoints may use "ln1" instead of "norm1"
        self.attn  = GQAAttention(cfg)
        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.ln2   = self.norm2  # old checkpoints may use "ln2"
        self.mlp   = _LegacyFFN(d_model, ffn_dim, cfg.bias)
        self.ffn   = self.mlp   # old checkpoints may use "ffn" instead of "mlp"

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _chat_contract(ckpt: dict, fmt: str, model_cfg, pretrain_only: bool = False):
    """Return ``(is_v16, chat_format, supports_system_prompt)``.

    v1.6 checkpoints predate the explicit metadata flags, so the format
    itself is the source of truth: ChatML and system turns unless a
    checkpoint says otherwise.  v1.5 post-training checkpoints inherit
    system-turn training from their SFT base, so they keep the raw
    ``<|role|>`` path but do accept a system turn.  A pretrain-only v1.5
    checkpoint never saw roles and must not receive one.
    """
    arch_version = str(ckpt.get("arch_version") or "").lower()
    num_experts = int(
        ckpt.get("num_experts", getattr(model_cfg, "num_experts", 1)) or 1
    )
    is_v16 = (
        str(fmt).startswith("1.6")
        or arch_version.startswith("dizel-1.6")
        or num_experts > 1
    )
    chat_format = str(
        ckpt.get("chat_format") or ("chatml" if is_v16 else "role_end")
    ).strip().lower()
    if chat_format not in ("chatml", "role_end"):
        chat_format = "chatml" if is_v16 else "role_end"
    declared = ckpt.get("supports_system_prompt")
    if declared is None:
        # Only v1.5 post-training inherited system turns from its SFT base.
        # v1.1/v1.2 saves predate role tokens and must not receive one.
        is_v15 = str(fmt).startswith("1.5")
        supports_system_prompt = is_v16 or (is_v15 and not pretrain_only)
    else:
        supports_system_prompt = bool(declared)
    return is_v16, chat_format, supports_system_prompt


# A v1.5 checkpoint is a 454M distillation student whose data is plain
# user/assistant pairs. Long system turns are out of distribution: measured on
# dizel-v15-distill-best, the GUI profile prompt turned "write a Python
# function" into a refusal, and the SFT-style prompt into "Okay, I understand.
# Please provide me with the task you asked for." A compact identity line is
# the one instruction it follows reliably, so long prompts collapse to it and
# short custom instructions keep it as a prefix.
_V15_IDENTITY = (
    "You are Dizel, an AI assistant created by the Dizel team. "
    "Answer the user's request directly and accurately."
)
_V15_SYSTEM_PROMPT_MAX_CHARS = 240


def _v15_system_prompt(system_prompt: str) -> str:
    """Compact a system turn to what a v1.5 checkpoint can follow."""
    text = " ".join(str(system_prompt or "").split())
    if not text:
        return ""
    if len(text) > _V15_SYSTEM_PROMPT_MAX_CHARS:
        return _V15_IDENTITY
    if "dizel" in text.lower():
        return text
    return f"{_V15_IDENTITY} {text}"


def _role_end_messages(
    messages: List[Dict],
    system_prompt: str,
    supports_system_prompt: bool,
    is_v15: bool,
) -> List[Dict[str, str]]:
    """Messages for the raw ``<|role|>`` path, shared by both backends.

    A checkpoint that never saw a system turn gets the system message dropped
    instead of injected; v1.5 post-training gets the compact form above.
    """
    out: List[Dict[str, str]] = []
    if system_prompt and supports_system_prompt:
        prompt = _v15_system_prompt(system_prompt) if is_v15 else system_prompt
        out.append({"role": "system", "content": prompt})
    for msg in messages:
        role = str(msg.get("role") or "user").strip().lower()
        if role == "system" and not supports_system_prompt:
            continue
        out.append({"role": role, "content": str(msg.get("content") or "")})
    return out


def _qwen_tokenizer_candidates(stored: Any) -> List[str]:
    """Ordered tokenizer sources to try for a v1.6 checkpoint.

    Training stores either a Hub id or a repo-relative path such as
    ``tokenizer/qwen2.5``.  ``AutoTokenizer.from_pretrained`` resolves a
    relative path against the process cwd, which is not the project root
    when the app is launched from elsewhere (for example Colab's
    ``/content``).  Resolve against the project root first, keep the stored
    Hub id as a fallback, and prefer the local Qwen copy shipped with the
    repo when the stored reference is stale.
    """
    from training.tokenizer_qwen import DEFAULT_MODEL_NAME

    name = str(stored or "").strip()
    candidates: List[str] = []
    hub_id = ""

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    if name:
        expanded = os.path.expanduser(name)
        if os.path.isabs(expanded):
            if os.path.isdir(expanded):
                add(os.path.normpath(expanded))
        else:
            for root in (_ROOT_DIR, os.getcwd()):
                local_path = os.path.join(root, expanded)
                if os.path.isdir(local_path):
                    add(os.path.normpath(local_path))

            normalized = expanded.replace("\\", "/")
            # ``tokenizer/qwen2.5`` is the training-side relative path, not
            # a Hub repo id. Passing it to the Hub produces a confusing
            # 404, so treat it as stale metadata instead.
            stale_local_path = (
                normalized.startswith(("./", "../", "~", "tokenizer/"))
                or "\\" in expanded
            )
            if not stale_local_path:
                hub_id = expanded

    # Prefer the local Qwen copy shipped with the repo: it is offline-safe
    # and its fingerprint is checked against the checkpoint below.
    local_qwen = os.path.join(_ROOT_DIR, "tokenizer", "qwen2.5")
    if os.path.isdir(local_qwen):
        add(local_qwen)
    if hub_id:
        add(hub_id)
    add(DEFAULT_MODEL_NAME)
    return candidates


class ModelBackend(ABC):
    """Base class for local model inference backends."""

    def close(self) -> None:
        """Release weights and native handles.

        Called when a different checkpoint replaces this backend. Implementations
        must be idempotent: a failed load can close a half-built backend and the
        caller may still drop it through the garbage collector afterwards.
        """

    @abstractmethod
    def load(self, checkpoint_path: str, device: str, on_progress: Callable[[str], None], **kwargs) -> None:
        pass
        
    @abstractmethod
    def generate(self, messages: List[Dict], system_prompt: str, sampling_params: Dict, max_tokens: int, on_token: Callable[[str], None]) -> str:
        """
        Run generation blockingly.
        `messages` is a list of {"role": str, "content": str}.
        `sampling_params` contains temperature, top_k, top_p, repetition_penalty.
        """
        pass
        
    @abstractmethod
    def stop(self) -> None:
        pass
        
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass
        
    @property
    @abstractmethod
    def context_length(self) -> int:
        pass
        
    @property
    @abstractmethod
    def model_info(self) -> dict:
        pass
        
    @property
    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @property
    def supports_system_prompt(self) -> bool:
        """Whether the loaded checkpoint was trained with a system turn.

        Dizel v1.5 SFT and distillation data contain only user/assistant
        turns. Sending a system message to those checkpoints puts the model
        out of distribution and visibly degrades generation, so the local
        v1.5 backends report False. Newer checkpoints can opt in via metadata.
        """
        return True


class NativeBackend(ModelBackend):
    """Original Dizel PyTorch (.pt) backend."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._checkpoint_path = ""
        self._stop_requested = False
        self._supports_system_prompt = False
        self._is_v16 = False
        self._is_v15 = False
        self._chat_format = "role_end"
        self._is_pretrain_only = False

    def close(self) -> None:
        """Drop the model and tokenizer, then return cached VRAM to the driver."""
        model, self._model = self._model, None
        tokenizer, self._tokenizer = self._tokenizer, None
        del model, tokenizer
        if self._device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def load(self, checkpoint_path: str, device: str, on_progress: Callable[[str], None], **kwargs) -> None:
        import torch
        from model.architecture import DizelLM
        from model.lora_merge import fold_lora_adapters, has_lora_adapters
        from config import CONFIG

        self._checkpoint_path = checkpoint_path
        self._device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        on_progress(f"Loading checkpoint from {checkpoint_path}…")
        # The desktop app only needs the weights, but a training checkpoint
        # also carries Adam state. A full torch.load() of a v1.6 checkpoint
        # reads ~5 GB before the model exists and can push a laptop into swap.
        # Memory-map the file so only the tensors load_state_dict() touches are
        # paged in, and keep it on CPU so a CUDA target gets a normal
        # host-to-device copy instead of a GPU memory spike.
        try:
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (RuntimeError, TypeError) as exc:
            if "mmap" not in str(exc).lower():
                raise
            on_progress("Checkpoint predates memory-mapped loading; loading in full…")
            ckpt = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        model_cfg = ckpt.get("model_cfg", CONFIG.model)
        state_dict = ckpt["model_state"]
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Drop training-only state so the optimizer copy does not stay
        # referenced while the model is built.
        for training_key in (
            "optim_state",
            "scaler_state",
            "scheduler",
            "rng_state",
            "dataset_state",
        ):
            ckpt.pop(training_key, None)

        # Distillation saves LoRA unmerged. Fold it before anything inspects
        # keys: the plain DizelLM has no LoRALinear wrappers, so loading the
        # raw dict would silently skip every projection matrix.
        if has_lora_adapters(cleaned):
            cleaned, lora_info = fold_lora_adapters(
                cleaned,
                lora_alpha=ckpt.get("lora_alpha"),
                lora_r=ckpt.get("lora_r"),
            )
            on_progress(
                f"Folded {lora_info['adapters']} LoRA adapters "
                f"(alpha={lora_info['alpha']}, r={lora_info['r']})"
            )

        # ── Detect checkpoint format version ──────────────────────────
        fmt = str(ckpt.get("_format_version", "") or "")
        if not fmt:
            # v1.6 checkpoints are identifiable before looking at tensor
            # names: the MoE shape and preview slug are explicit metadata.
            arch_version = str(ckpt.get("arch_version") or "").lower()
            declared_experts = int(
                ckpt.get("num_experts", getattr(model_cfg, "num_experts", 1))
                or 1
            )
            if arch_version.startswith("dizel-1.6") or declared_experts > 1:
                fmt = "1.6"
        if not fmt:
            ckpt_keys = set(cleaned.keys())
            # v1.1.0: learned positional embeddings
            if any(k.endswith("pos_emb.weight") for k in ckpt_keys):
                fmt = "1.1.0"
            # v1.5: SwiGLU gate_proj is exclusive to v1.5
            elif any(k.endswith("gate_proj.weight") for k in ckpt_keys):
                fmt = "1.5"
            # v1.5: RMSNorm "input_norm" naming is v1.5-specific
            elif any(k.endswith("input_norm.weight") for k in ckpt_keys):
                fmt = "1.5"
            # v1.2.x: LayerNorm "norm1/norm2" naming, no gate_proj present
            elif any(k.endswith("norm1.weight") for k in ckpt_keys) or \
                 any(k.endswith("norm2.weight") for k in ckpt_keys):
                fmt = "1.2.2"
            # v1.2.x: standard FFN fc1 (without SwiGLU), no gate_proj
            elif any(k.endswith("fc1.weight") for k in ckpt_keys):
                fmt = "1.2.2"
            else:
                fmt = "1.5"

        on_progress(f"Detected format: v{fmt}")

        # Pretraining checkpoints are raw-text models; post-training saves
        # declare their phase explicitly. Fall back to the filename for older
        # v1.5 files that predate the phase field.
        phase = str(ckpt.get("phase") or "").strip().lower()
        if phase:
            self._is_pretrain_only = phase in ("pretrain", "pre")
        else:
            name = os.path.basename(checkpoint_path).lower()
            self._is_pretrain_only = "pretrain" in name and "sft" not in name

        # v1.6 defaults to ChatML + Qwen + system turns even when an older
        # save lacks the explicit metadata flags. v1.5 post-training keeps
        # the raw <|role|> path but accepts the system turn it inherited
        # from its SFT base; a v1.5 pretrain checkpoint does not.
        self._is_v16, self._chat_format, self._supports_system_prompt = (
            _chat_contract(
                ckpt, fmt, model_cfg, pretrain_only=self._is_pretrain_only
            )
        )
        self._is_v15 = str(fmt).startswith("1.5")

        # ── Build model matching the checkpoint format ────────────────
        if fmt == "1.1.0":
            # Learned positional embeddings — needs custom forward pass
            on_progress("Building v1.1 legacy model (learned pos_emb)…")
            pos_emb_weight = cleaned["transformer.pos_emb.weight"]
            ctx_len, d_model = pos_emb_weight.shape
            self._model = DizelLM(model_cfg).to(self._device)
            self._model.transformer["pos_emb"] = torch.nn.Embedding(ctx_len, d_model).to(self._device)
            self._model.load_state_dict(cleaned, strict=False)

            # Replace forward with one that adds positional embeddings
            _orig_forward = self._model.forward
            def _legacy_forward(idx, targets=None, loss_mask=None):
                B, T = idx.shape
                tok = self._model.transformer["tok_emb"](idx)
                pos = self._model.transformer["pos_emb"](torch.arange(T, device=idx.device))
                x = self._model.transformer["emb_drop"](tok + pos)
                for block in self._model.transformer["blocks"]:
                    x = block(x)
                x = self._model.transformer["ln_f"](x)
                logits = self._model.lm_head(x)
                if targets is None:
                    return logits, None
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
                return logits, loss
            self._model.forward = _legacy_forward

        elif fmt == "1.2.2":
            # LayerNorm + standard 2-layer FFN (fc1 → GELU → fc2)
            on_progress("Building v1.2 legacy model (LayerNorm + standard FFN)…")
            self._model = DizelLM(model_cfg).to(self._device)
            # Replace RMSNorm ln_f with LayerNorm (v1.2 used LayerNorm, bias=False)
            self._model.transformer["ln_f"] = nn.LayerNorm(
                model_cfg.d_model, bias=False
            ).to(self._device)
            # Swap each v1.5 decoder block for a v1.2 legacy block
            for i in range(model_cfg.n_layers):
                legacy_block = _LegacyDecoderV12(model_cfg).to(self._device)
                self._model.transformer["blocks"][i] = legacy_block
            self._model.load_state_dict(cleaned, strict=False)

        else:
            # v1.5+ backbone: RMSNorm + SwiGLU + GQA. v1.6 replaces the dense
            # FFN with a dropless top-k MoE block, selected by model_cfg.
            if self._is_v16:
                on_progress(
                    "Building v1.6 MoE model (RMSNorm + SwiGLU + top-k experts)…"
                )
            else:
                on_progress("Building v1.5 model (RMSNorm + SwiGLU)…")
            self._model = DizelLM(model_cfg).to(self._device)
            on_progress("Copying checkpoint weights into the model…")
            missing, unexpected = self._model.load_state_dict(cleaned, strict=False)
            on_progress("Weights loaded")
            # A tied lm_head may be omitted by older exports; it shares the
            # token embedding, so that one key is harmless.
            missing = [k for k in missing if k != "lm_head.weight"]
            if missing or unexpected:
                raise RuntimeError(
                    "Checkpoint does not match the model architecture: "
                    f"{len(missing)} missing / {len(unexpected)} unexpected "
                    f"tensor(s). First missing: {missing[:3]}; first "
                    f"unexpected: {unexpected[:3]}. Distilled checkpoints must "
                    "have their LoRA adapters folded before loading."
                )

        self._model.eval()

        # ── Tokenizer ─────────────────────────────────────────────────
        on_progress("Loading tokenizer…")
        if self._is_v16 or self._chat_format == "chatml":
            # v1.6 uses the Qwen2.5 BPE tokenizer and stores only its name,
            # not the vocabulary bytes. The stored value may be a Hub id or
            # the training-side relative path tokenizer/qwen2.5, so resolve
            # it against the project root and fall back to the local copy or
            # the Hub instead of trusting the process cwd.
            from training.tokenizer_qwen import QwenTokenizer

            candidates = _qwen_tokenizer_candidates(ckpt.get("tokenizer_name"))
            expected_hash = str(ckpt.get("tokenizer_hash") or "")
            failures = []
            tokenizer = None
            for candidate in candidates:
                on_progress(f"Loading Qwen tokenizer from {candidate}…")
                try:
                    loaded = QwenTokenizer(model_name=candidate)
                except ImportError:
                    raise
                except Exception as exc:
                    failures.append(f"{candidate}: {exc}")
                    continue

                if expected_hash:
                    actual_hash = loaded.fingerprint()
                    if actual_hash != expected_hash:
                        failures.append(
                            f"{candidate}: fingerprint {actual_hash[:16]}… "
                            f"does not match checkpoint "
                            f"{expected_hash[:16]}…"
                        )
                        continue

                tokenizer = loaded
                break

            if tokenizer is None:
                expected = (
                    f"{expected_hash[:16]}…" if expected_hash else "(not recorded)"
                )
                raise RuntimeError(
                    "Could not load a Qwen tokenizer matching this v1.6 "
                    f"checkpoint. Expected fingerprint: {expected}. "
                    f"Tried: {candidates}. Install 'transformers' and either "
                    "stay online for the first load or place the tokenizer "
                    f"under tokenizer/qwen2.5. Failures: {failures}"
                )

            self._tokenizer = tokenizer
            if len(self._tokenizer) > int(model_cfg.vocab_size):
                raise RuntimeError(
                    f"Tokenizer vocabulary ({len(self._tokenizer):,}) is larger "
                    f"than the checkpoint embedding table "
                    f"({int(model_cfg.vocab_size):,})."
                )
            on_progress(f"Using Qwen tokenizer (vocab={len(self._tokenizer)})")
        elif "tokenizer_model" in ckpt:
            from training.dataset import Tokenizer

            tok_bytes = ckpt["tokenizer_model"]
            tok_tmp = os.path.join(os.path.dirname(checkpoint_path), "_embedded_tokenizer.model")
            with open(tok_tmp, "wb") as f:
                f.write(tok_bytes)
            self._tokenizer = Tokenizer(model_path=tok_tmp)
            on_progress(f"Using embedded tokenizer (vocab={len(self._tokenizer.sp)})")
        else:
            from training.dataset import Tokenizer

            tokenizer_path = os.path.join(_ROOT_DIR, CONFIG.tokenizer.model_path)
            self._tokenizer = Tokenizer(model_path=tokenizer_path)

        step = ckpt.get("step", "?")
        vl = ckpt.get("val_loss", float("inf"))
        on_progress(f"Ready  — step={step}  val_loss={vl:.4f}  params={self._model.num_parameters()/1e6:.2f}M  device={self._device}")

    def generate(self, messages: List[Dict], system_prompt: str, sampling_params: Dict, max_tokens: int, on_token: Callable[[str], None]) -> str:
        self._stop_requested = False
        import torch
        import torch.nn.functional as F
        import time

        # Build prompt IDs
        ids = []
        if self._is_pretrain_only:
            last_msg = messages[-1].get("content", "") if messages else ""
            ids = self._tokenizer.encode(last_msg)
        elif self._chat_format == "chatml":
            ids = self._build_chatml_prompt_ids(messages, system_prompt)
        else:
            from training.dataset import ROLE_TOKENS, END_TOKEN
            msgs = _role_end_messages(
                messages,
                system_prompt,
                self.supports_system_prompt,
                self._is_v15,
            )
            for msg in msgs:
                role = msg["role"]
                content = msg["content"]
                ids += self._tokenizer.encode(ROLE_TOKENS.get(role, ""))
                ids += self._tokenizer.encode(content)
                ids += self._tokenizer.encode(END_TOKEN)
            ids += self._tokenizer.encode(ROLE_TOKENS.get("assistant", ""))

        # Hard truncate prompt if needed
        ctx_len = self.context_length
        if len(ids) > ctx_len - max_tokens - 10:
            allowed = max(ctx_len - max_tokens - 10, 16)
            ids = ids[-allowed:]

        idx = torch.tensor([ids], dtype=torch.long, device=self._device)
        generated_ids = []
        prev_decoded = ""  # for streaming decode diff
        # Stop on the turn marker as well as EOS: a model that omits EOS
        # otherwise keeps generating a hallucinated next turn (and leaks
        # prompt-like text).
        end_ids = self._stop_token_ids()
        
        if self._device == "cpu":
            torch.set_num_threads(2)
            
        temp = sampling_params.get("temperature", 0.8)
        top_k = sampling_params.get("top_k", 50)
        top_p = sampling_params.get("top_p", 0.92)
        rep_pen = sampling_params.get("repetition_penalty", 1.15)
        
        with torch.no_grad():
            for step in range(max_tokens):
                if self._stop_requested:
                    break
                input_ids = idx[:, -ctx_len:]
                
                if self._device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, _ = self._model(input_ids)
                else:
                    logits, _ = self._model(input_ids)
                    
                logits = logits[:, -1, :].float()

                # v1.6 pads the embedding table to 151,936 rows while the
                # Qwen tokenizer emits 151,665 ids. The extra rows are never
                # training targets, so sampling them would emit random bytes.
                self._mask_padded_vocab(logits)

                if rep_pen != 1.0:
                    for tid in set(generated_ids[-128:]):
                        if 0 <= tid < logits.size(-1):
                            if logits[0, tid] > 0: logits[0, tid] /= rep_pen
                            else: logits[0, tid] *= rep_pen
                            
                if temp == 0.0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(temp, 1e-8)
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    if top_p < 1.0:
                        ps, si = torch.sort(F.softmax(logits, dim=-1), descending=True)
                        ps[ps.cumsum(-1) - ps > top_p] = 0.0
                        ps /= ps.sum(-1, keepdim=True)
                        next_id = torch.multinomial(ps, 1)
                        next_id = si.gather(-1, next_id)
                    else:
                        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                        
                tid = next_id.item()
                if tid in end_ids:
                    break
                    
                generated_ids.append(tid)
                idx = torch.cat([idx, next_id], dim=1)
                
                # Decode the full prefix each step and emit only the new
                # suffix. Byte-level BPE (Qwen) can leave a partial UTF-8
                # character at the end; hold it until the next token
                # completes it instead of emitting U+FFFD.
                current_decoded = self._decode_generated(generated_ids)
                if current_decoded.endswith("\ufffd"):
                    current_decoded = current_decoded[:-1]
                if current_decoded.startswith(prev_decoded):
                    new_text = current_decoded[len(prev_decoded):]
                else:
                    common = 0
                    for old_char, new_char in zip(prev_decoded, current_decoded):
                        if old_char != new_char:
                            break
                        common += 1
                    new_text = current_decoded[common:]
                prev_decoded = current_decoded
                if new_text:
                    on_token(new_text)
                time.sleep(0.001)

        return self._decode_generated(generated_ids)

    def _build_chatml_prompt_ids(
        self, messages: List[Dict], system_prompt: str
    ) -> List[int]:
        """ChatML ids matching ``training.posttrain_data_v16.format_chat_tokens``.

        Each turn is ``<|im_start|>{role}\\n{content}<|im_end|>\\n`` and the
        generation prefix is ``<|im_start|>assistant\\n``.  The trailing
        newline is formatting, exactly as it is during training.
        """
        system_text = str(system_prompt or "").strip()
        msgs: List[Dict[str, str]] = []
        if system_text and self.supports_system_prompt:
            msgs.append({"role": "system", "content": system_text})

        for msg in messages:
            role = str(msg.get("role") or "user").strip().lower()
            if role == "system":
                # A separate system_prompt wins; v1.5-style checkpoints that
                # cannot use a system turn drop it entirely.
                if not self.supports_system_prompt or system_text:
                    continue
            if role not in ("system", "user", "assistant"):
                role = "user"
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            msgs.append({"role": role, "content": content})

        ids: List[int] = []
        for msg in msgs:
            ids += self._tokenizer.encode(
                f"<|im_start|>{msg['role']}\n"
            )
            ids += self._tokenizer.encode(msg["content"])
            ids += self._tokenizer.encode("<|im_end|>")
            ids += self._tokenizer.encode("\n")
        ids += self._tokenizer.encode("<|im_start|>assistant\n")
        return ids

    def _tokenizer_vocab_size(self) -> int:
        """Rows of the tokenizer vocabulary, or 0 when it cannot be measured."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            return 0
        try:
            return int(len(tokenizer))
        except TypeError:
            sp = getattr(tokenizer, "sp", None)
            return int(len(sp)) if sp is not None else 0

    def _mask_padded_vocab(self, logits) -> None:
        """Never sample embedding rows the tokenizer cannot produce."""
        tokenizer_vocab = self._tokenizer_vocab_size()
        if 0 < tokenizer_vocab < logits.size(-1):
            logits[:, tokenizer_vocab:] = float("-inf")

    def _special_token_id(self, token: str) -> Optional[int]:
        """Resolve one control token to its id for either tokenizer family."""
        sp = getattr(self._tokenizer, "sp", None)
        if sp is not None:
            tid = sp.piece_to_id(token)
            if tid == getattr(self._tokenizer, "unk_id", -1):
                return None
            return int(tid)

        inner = getattr(self._tokenizer, "tokenizer", None)
        if inner is not None and hasattr(inner, "convert_tokens_to_ids"):
            tid = inner.convert_tokens_to_ids(token)
            if tid is None:
                return None
            tid = int(tid)
            if hasattr(inner, "convert_ids_to_tokens"):
                if inner.convert_ids_to_tokens(tid) != token:
                    return None
            return tid

        encoded = self._tokenizer.encode(token)
        return int(encoded[0]) if len(encoded) == 1 else None

    def _stop_token_ids(self) -> set:
        """Ids that end an assistant turn for the loaded chat format."""
        end_ids = {int(self._tokenizer.eos_id)}
        if self._chat_format == "chatml":
            specials = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
        else:
            from training.dataset import END_TOKEN, ROLE_TOKENS

            specials = (END_TOKEN, *ROLE_TOKENS.values())
        for special in specials:
            tid = self._special_token_id(special)
            if tid is not None:
                end_ids.add(tid)
        return end_ids

    def _decode_generated(self, ids: List[int]) -> str:
        text = self._tokenizer.decode(ids)
        if self._chat_format != "chatml":
            from training.dataset import END_TOKEN, ROLE_TOKENS

            for special in (*ROLE_TOKENS.values(), END_TOKEN):
                text = text.replace(special, "")
        # A generation cut mid-UTF-8 character must not surface U+FFFD.
        if text.endswith("\ufffd"):
            text = text[:-1]
        return text

    def stop(self) -> None:
        self._stop_requested = True
        
    def count_tokens(self, text: str) -> int:
        if not self._tokenizer: return len(text.split())
        return len(self._tokenizer.encode(text))
        
    @property
    def context_length(self) -> int:
        if not self._model: return 2048
        return self._model.cfg.context_length
        
    @property
    def model_info(self) -> dict:
        if not self._model: return {}
        return {
            "provider": "Local (.pt)",
            "params": f"{self._model.num_parameters()/1e6:.2f} M",
            "device": self._device,
            "d_model": self._model.cfg.d_model,
            "n_layers": self._model.cfg.n_layers,
            "n_heads": self._model.cfg.n_heads,
            "ctx_len": self._model.cfg.context_length,
            "vocab": self._model.cfg.vocab_size,
        }
        
    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def supports_system_prompt(self) -> bool:
        return self._supports_system_prompt


# ---------------------------------------------------------------------------
# GGUF metadata helpers (module-level, standalone).
# These never import anything heavy at module import time: the `gguf` PyPI
# package is imported lazily inside each function, and a minimal manual
# binary scan of the GGUF KV section is used as a fallback. That way this
# module still compiles / imports in environments without gguf or llama_cpp.
# ---------------------------------------------------------------------------

# GGUF value types (see gguf/ggml common.h / gguf_py constants).

def _reader_field_value(field, default=None):
    """Best-effort scalar extraction out of a ``gguf`` ReaderField."""
    if field is None:
        return default
    try:
        value = field.contents()
    except Exception:
        return default
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return value
    return value


def _build_gguf_reader(model_path):
    """Lazily construct ``gguf.GGUFReader``. Returns None on any failure."""
    try:
        import gguf
        return gguf.GGUFReader(model_path)
    except Exception:
        return None


_GGUF_SCALAR_STRUCTS = {
    0x00: ("B", 1),   # UINT8
    0x01: ("b", 1),   # INT8
    0x02: ("H", 2),   # UINT16
    0x03: ("h", 2),   # INT16
    0x04: ("I", 4),   # UINT32
    0x05: ("i", 4),   # INT32
    0x06: ("f", 4),   # FLOAT32
    0x07: ("?", 1),   # BOOL
    0x0A: ("Q", 8),   # UINT64
    0x0B: ("q", 8),   # INT64
    0x0C: ("d", 8),   # FLOAT64
}


def _read_gguf_metadata_fast(model_path) -> dict:
    """Read scalar/string GGUF metadata without materializing huge arrays.

    ``gguf.GGUFReader`` plus ``field.contents()`` on every field can take tens
    of seconds on multi-hundred-MB files because it eagerly builds every
    tensor/tokenizer array. We only need a handful of scalars and strings, so
    this parser walks the key/value section and skips arrays in-place.

    Returns {} when the file is not a readable GGUF.
    """
    import struct

    meta = {}
    try:
        with open(model_path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return {}
            fh.read(4)  # version
            fh.read(8)  # tensor_count
            (kv_count,) = struct.unpack("<Q", fh.read(8))
            if kv_count <= 0 or kv_count > 100_000:
                return {}

            def _read_string():
                (length,) = struct.unpack("<Q", fh.read(8))
                return fh.read(length).decode("utf-8", errors="replace")

            def _skip_array(elem_type, count):
                if elem_type == 8:  # string
                    for _ in range(count):
                        (length,) = struct.unpack("<Q", fh.read(8))
                        fh.seek(length, 1)
                elif elem_type == 9:  # nested array
                    for _ in range(count):
                        (sub_type,) = struct.unpack("<I", fh.read(4))
                        (sub_count,) = struct.unpack("<Q", fh.read(8))
                        _skip_array(sub_type, sub_count)
                else:
                    fmt_size = _GGUF_SCALAR_STRUCTS.get(elem_type)
                    if fmt_size is None:
                        raise ValueError("unknown GGUF array element type")
                    fh.seek(fmt_size[1] * count, 1)

            for _ in range(kv_count):
                key = _read_string()
                (value_type,) = struct.unpack("<I", fh.read(4))
                if value_type == 8:  # string
                    meta[key] = _read_string()
                elif value_type == 9:  # array: skip, never needed by callers
                    (elem_type,) = struct.unpack("<I", fh.read(4))
                    (count,) = struct.unpack("<Q", fh.read(8))
                    _skip_array(elem_type, count)
                else:
                    fmt_size = _GGUF_SCALAR_STRUCTS.get(value_type)
                    if fmt_size is None:
                        raise ValueError("unknown GGUF value type")
                    fmt, size = fmt_size
                    meta[key] = struct.unpack("<" + fmt, fh.read(size))[0]
    except Exception:
        return {}
    return meta


def _read_gguf_metadata_dict(model_path) -> dict:
    """Best-effort {key: value} dict of a GGUF file's metadata.

    1. Uses the ``gguf`` PyPI package when importable.
    2. Otherwise falls back to a minimal manual parser for the GGUF
       header + key/value section (per the GGUF spec), so context_length
       and architecture detection still work in constrained environments.
    Returns {} on any failure.
    """
    fast = _read_gguf_metadata_fast(model_path)
    if fast:
        return fast

    # Fallback for unusual files / older GGUF versions: the library reader.
    reader = _build_gguf_reader(model_path)
    if reader is None:
        return {}
    fields = getattr(reader, "fields", None)
    if not hasattr(fields, "items"):
        return {}
    meta = {}
    try:
        for name in fields:
            meta[name] = _reader_field_value(fields[name])
    except Exception:
        pass
    return meta


def _read_gguf_context_and_arch(model_path, meta: Optional[dict] = None):
    """Read the model context length and architecture from a GGUF file.

    Returns a ``(n_ctx, arch)`` tuple:
      * ``n_ctx`` — an int > 0 from ``<arch>.context_length`` (e.g.
        ``dizel.context_length`` or ``llama.context_length``) or an
        un-prefixed ``context_length`` key; ``None`` when unavailable.
      * ``arch``  — the lowercased ``general.architecture`` string, or None.

    Never raises: on an unreadable file / missing gguf package it returns
    ``(None, None)`` and callers fall back to llama-cpp's own defaults.
    """
    if meta is None:
        meta = _read_gguf_metadata_dict(model_path)

    arch = meta.get("general.architecture")
    if arch is not None:
        arch = str(arch).strip().lower() or None

    n_ctx = None
    candidates = ["context_length"]
    if arch:
        candidates.insert(0, "%s.context_length" % arch)
    for key in candidates:
        raw = meta.get(key)
        if raw is None:
            continue
        try:
            as_int = int(raw)
        except (TypeError, ValueError):
            continue
        if as_int > 0:
            n_ctx = as_int
            break
    return n_ctx, arch


def _gguf_has_chat_template(model_path, meta: Optional[dict] = None) -> bool:
    """True when the GGUF embeds a chat template in its tokenizer metadata."""
    if meta is None:
        meta = _read_gguf_metadata_dict(model_path)
    return any(
        meta.get(key) is not None
        for key in ("tokenizer.chat_template", "tokenizer.ggml.chat_template")
    )


def _estimate_gguf_params(meta: dict, arch) -> int:
    """Estimate total parameter count (M) from GGUF hyper-parameter metadata.

    Dizel v1.5 layout per layer: attn_norm(d) + q/k/v/o projections +
    ffn_norm(d) + SwiGLU gate/up/down, plus a tied token embedding
    (vocab*d, counted once) and the final output norm (d).

    q/o are d*d; k/v are (head_dim * kv_heads) * d each, where
    head_dim = d / head_count.
    """
    try:
        d = int(meta.get(f"{arch}.embedding_length") or meta.get("embedding_length") or 0)
        L = int(meta.get(f"{arch}.block_count") or meta.get("block_count") or 0)
        ffn = int(meta.get(f"{arch}.feed_forward_length") or meta.get("feed_forward_length") or 0)
        n_heads = int(meta.get(f"{arch}.attention.head_count") or meta.get("attention.head_count") or 0)
        kv_heads = int(meta.get(f"{arch}.attention.head_count_kv") or meta.get("attention.head_count_kv") or 0)
        vocab = int(meta.get(f"{arch}.vocab_size") or meta.get("vocab_size") or 0)
    except (TypeError, ValueError):
        return 0
    if not (d and L and ffn and n_heads and vocab):
        return 0
    kv_heads = kv_heads or n_heads
    head_dim = max(d // n_heads, 1)
    kv_dim = head_dim * kv_heads
    per_layer = (
        d                       # attn_norm
        + d * d                 # q_proj
        + 2 * (d * kv_dim)      # k_proj + v_proj
        + d * d                 # o_proj
        + d                     # ffn_norm
        + 3 * (d * ffn)         # gate/up/down
    )
    total = vocab * d + L * per_layer + d  # tok_emb (tied w/ lm_head, count once) + output_norm
    return total


_CHAT_TEMPLATE_ERROR_HINTS = (
    "chat template",
    "chat_template",
    "chat format",
    "chat_format",
    "chatformat",
    "chatml",
)


def _is_chat_template_error(exc: BaseException) -> bool:
    """Whether an exception raised by llama-cpp is about the chat template
    (missing / malformed) rather than a real bug."""
    exc_name = type(exc).__name__.lower()
    if "chat" in exc_name and "error" in exc_name:
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _CHAT_TEMPLATE_ERROR_HINTS)


def _llm_attr_or_call(obj, name, default=None):
    """Read ``obj.<name>`` regardless of whether it is a plain attribute or a
    method.

    llama-cpp-python exposes ``n_ctx()`` / ``n_vocab()`` / ``n_ctx_train()``
    as callable methods in some versions and as read-only properties in
    others (and omits ``n_ctx_train`` entirely in some, e.g. 0.3.23). This
    helper works with both shapes and returns ``default`` when the attribute
    is absent or reading it raises, so callers never crash across versions.
    """
    try:
        val = getattr(obj, name, None)
    except Exception:
        return default
    if val is None:
        return default
    try:
        return val() if callable(val) else val
    except Exception:
        return default


def _strip_chat_control(text: str) -> str:
    """Remove Dizel chat-control tokens from generated output."""
    for tok in ("<|end|>", "</s>", "<|assistant|>", "<|user|>", "<|system|>"):
        text = text.replace(tok, "")
    return text.strip()


class GGUFBackend(ModelBackend):
    """Backend for GGUF models via llama-cpp-python."""
    
    def __init__(self):
        self._llm = None
        self._stop_requested = False
        self._device = "cpu"
        # Bug 1: context length resolved from GGUF metadata (None = let
        # llama-cpp use its own default, never 0).
        self._n_ctx = None
        # Bug 2/3: whether the GGUF embeds a chat template (or an explicit
        # chat_format was requested). Default True, corrected in load().
        self._has_chat_template = True
        # Path + arch of the loaded GGUF, kept for model_info metadata reads.
        self._model_path = None
        self._gguf_arch = None
        self._gguf_meta = {}
        # v1.5 uses the SentencePiece tokenizer and the raw <|role|> format.
        # Its <|end|> is a single trained token that llama.cpp only emits
        # with special=True, so the raw path is built from token ids.
        self._is_v15_gguf = False
        # v1.5 GGUFs were trained with a system turn in their SFT base. A
        # future checkpoint can override with
        # `dizel.supports_system_prompt = false`.
        self._supports_system_prompt = False

    def close(self) -> None:
        """Free the llama.cpp model and context deterministically.

        Relying on ``Llama.__del__`` is fragile: 0.3.23 can raise from
        ``_internals.free_ctx`` during interpreter shutdown, and the free then
        happens on whichever thread triggers GC. The lifecycle owns the release.
        """
        llm, self._llm = self._llm, None
        if llm is None:
            return
        try:
            llm.close()
        except Exception:
            pass
        del llm

    def load(self, checkpoint_path: str, device: str, on_progress: Callable[[str], None], **kwargs) -> None:
        import time

        def _report(message: str) -> None:
            if on_progress:
                on_progress(message)

        # Importing llama.cpp loads its native library, and the first GPU probe
        # initializes the backend registry. On a CUDA build with a broken or
        # busy driver either step can block for a long time, so bracket them:
        # if a user reports a freeze, the trace names the step instead of
        # stopping at "Importing dependencies…".
        _report("Importing llama.cpp…")
        try:
            from llama_cpp import Llama
            import llama_cpp as _llama_cpp_cap
        except ImportError:
            raise ImportError("llama-cpp-python is not installed. Run: pip install llama-cpp-python")

        if device == "cuda":
            _report("Checking GPU offload support…")
            try:
                gpu_offload = bool(_llama_cpp_cap.llama_supports_gpu_offload())
            except Exception:
                gpu_offload = False
            if not gpu_offload:
                device = "cpu"
                _report(
                    "CUDA requested, but this llama.cpp build has no GPU "
                    "offload — using CPU."
                )

        # ── Defensive patch: llama-cpp-python <=0.3.23 GC destructor bug ──
        # LlamaModel.__del__ -> close() reads self.sampler / self.custom_samplers
        # which are only assigned AFTER a successful load. On a failed load the
        # garbage collector prints:
        #   "Exception ignored in: <function LlamaModel.__del__ ...>
        #    AttributeError: 'LlamaModel' object has no attribute 'sampler'"
        # Patch close() to be attribute-safe. No-op if the bug is already fixed.
        try:
            import llama_cpp as _llama_cpp_mod
            _internals = _llama_cpp_mod._internals
            _model_cls = _internals.LlamaModel
            if not getattr(_model_cls, "_dizel_gc_patch_applied", False):
                _orig_close = _model_cls.close

                def _safe_close(self):
                    if getattr(self, "sampler", None) is not None:
                        try:
                            for i, _ in reversed(getattr(self, "custom_samplers", ())):
                                _llama_cpp_mod.llama_sampler_chain_remove(self.sampler, i)
                            custom = getattr(self, "custom_samplers", None)
                            if custom is not None:
                                custom.clear()
                        except Exception:
                            pass
                    try:
                        _orig_close(self)
                    except Exception:
                        pass

                _model_cls.close = _safe_close
                _model_cls._dizel_gc_patch_applied = True
        except Exception:
            pass

        self._device = device
        self._gguf_meta = {}
        n_gpu_layers = -1 if device == "cuda" else 0
        
        if on_progress:
            on_progress(f"Loading GGUF model from {checkpoint_path}…")
        
        # Format can be overridden via kwargs, else None (auto-detect)
        chat_format = kwargs.get("chat_format", None)
        if chat_format == "auto":
            chat_format = None

        # ── Bug 1: resolve the real context length from GGUF metadata ─────
        # Passing a zero-valued n_ctx to llama-cpp does NOT auto-scale to the
        # model's window. Read `<arch>.context_length` (or bare
        # `context_length`) and pass that value explicitly.
        if on_progress:
            on_progress("Reading GGUF metadata…")
        gguf_meta = _read_gguf_metadata_dict(checkpoint_path)
        self._gguf_meta = gguf_meta
        n_ctx_val, arch = _read_gguf_context_and_arch(checkpoint_path, meta=gguf_meta)
        self._model_path = checkpoint_path
        self._gguf_arch = arch
        self._n_ctx = n_ctx_val if n_ctx_val else None

        # v1.5 is the only published GGUF family that uses the SentencePiece
        # tokenizer (GGUF tokenizer.ggml.model == "llama"). Its SFT base was
        # trained with a system turn, so accept one unless the file says
        # otherwise. A pretrain-only file has no role training at all.
        self._is_v15_gguf = (
            arch == "qwen2"
            and str(gguf_meta.get("tokenizer.ggml.model") or "") == "llama"
        )
        declared_system = gguf_meta.get("dizel.supports_system_prompt")
        if declared_system is None:
            name = os.path.basename(checkpoint_path).lower()
            is_pretrain = "pretrain" in name and "sft" not in name and "distill" not in name
            self._supports_system_prompt = self._is_v15_gguf and not is_pretrain
        else:
            self._supports_system_prompt = bool(declared_system)

        if on_progress:
            on_progress(f"GGUF detected: arch={arch or 'unknown'}, ctx={n_ctx_val or 'auto'}…")

        # ── Bug 2: the native 'dizel' arch is not loadable by llama.cpp ───
        if arch == "dizel":
            raise RuntimeError(
                "This GGUF uses the native 'dizel' architecture, which "
                "llama.cpp cannot load.\n"
                "Re-convert the checkpoint for llama.cpp with:\n"
                "    python scripts/convert_to_gguf.py --checkpoint <model.pt> "
                "--compat-arch qwen2\n"
                "or load the original .pt / .safetensors checkpoint instead."
            )

        # ── Chat-template detection (feeds the Bug 3 fallback) ────────────
        if self._is_v15_gguf:
            # The embedded v1.5 template renders `<|user|>content` without
            # the space the model was trained with, and llama.cpp splits
            # <|end|> unless it is requested as a special token. v1.5 always
            # uses the token-id raw path below.
            self._has_chat_template = False
        elif chat_format is None:
            # Auto mode: if the GGUF embeds a template, llama-cpp applies it.
            # Without one, generate() must fall back to raw completion.
            self._has_chat_template = bool(
                _gguf_has_chat_template(checkpoint_path, meta=gguf_meta)
            )
        else:
            # An explicit chat_format (e.g. "chatml") implies the chat path
            # is safe regardless of what the metadata contains.
            self._has_chat_template = True

        if on_progress:
            on_progress("Initializing llama.cpp engine…")

        if on_progress:
            on_progress("Allocating weights… (may take a while on CPU)")
        t0 = time.time()
        if n_ctx_val is not None:
            # Pass the metadata-derived context explicitly; if it could not be
            # read we omit the argument so llama-cpp picks its internal default.
            self._llm = Llama(
                model_path=checkpoint_path,
                n_ctx=n_ctx_val,
                n_gpu_layers=n_gpu_layers,
                chat_format=chat_format,
                verbose=False,
            )
        else:
            self._llm = Llama(
                model_path=checkpoint_path,
                n_gpu_layers=n_gpu_layers,
                chat_format=chat_format,
                verbose=False,
            )
        elapsed = time.time() - t0

        if on_progress:
            on_progress(f"Weights loaded in {elapsed:.1f}s")

        # When the metadata could not be read (llama-cpp used its own default),
        # recover the real trained context window afterwards if the binding
        # exposes it, so the `context_length` property reports a sane value.
        # n_ctx_train() is a method in some llama-cpp-python versions, a
        # property in others, and absent in yet others -- _llm_attr_or_call
        # handles all of them.
        if self._n_ctx is None:
            recovered = _llm_attr_or_call(self._llm, "n_ctx_train")
            if recovered is not None:
                try:
                    self._n_ctx = int(recovered)
                except (TypeError, ValueError):
                    self._n_ctx = None

        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        if on_progress:
            on_progress(f"Done loading {os.path.basename(checkpoint_path)} ({size_mb:.1f} MB) ✔")

    def generate(self, messages: List[Dict], system_prompt: str, sampling_params: Dict, max_tokens: int, on_token: Callable[[str], None]) -> str:
        self._stop_requested = False
        
        api_msgs = _role_end_messages(
            messages,
            system_prompt,
            self.supports_system_prompt,
            self._is_v15_gguf,
        )

        sampling = {
            "max_tokens": max_tokens,
            "temperature": sampling_params.get("temperature", 0.8),
            "top_k": sampling_params.get("top_k", 50),
            "top_p": sampling_params.get("top_p", 0.92),
            "repeat_penalty": sampling_params.get("repetition_penalty", 1.15),
            # v1.5 assistant turns end with <|end|> before </s>. Without these
            # stops a model that skips EOS continues into a fake next turn,
            # which the control-token stripper would leak as prompt text.
            "stop": ["<|end|>", "<|user|>", "<|system|>", "<|assistant|>"],
        }

        # The embedded v1.5 template renders `<|user|>content` without the
        # space the model was trained with, and llama.cpp splits <|end|>
        # unless it is requested as a special token. v1.5 therefore always
        # uses the token-id raw path below; newer system-aware checkpoints
        # keep the template path.
        use_chat_template = self._has_chat_template and self.supports_system_prompt
        if use_chat_template:
            # ── Bug 3: template-aware path ────────────────────────────────
            try:
                stream = self._llm.create_chat_completion(
                    messages=api_msgs,
                    stream=True,
                    **sampling
                )
                return self._stream_chat(stream, on_token)
            except Exception as exc:
                # A GGUF may lack / mis-embed a template despite the flag; fall
                # back to the raw path instead of hard-crashing the GUI.
                if not _is_chat_template_error(exc):
                    raise

        # ── Raw completion fallback (no usable chat template) ────────────
        prompt = self._build_v15_prompt_ids(api_msgs)
        stream = self._llm.create_completion(
            prompt=prompt,
            stream=True,
            **sampling
        )
        full_text = []
        for chunk in stream:
            if self._stop_requested:
                break
            choices = chunk.get("choices", [])
            if not choices:
                continue
            # create_completion streams raw text under "text", not "delta".
            text = choices[0].get("text", "")
            if text:
                full_text.append(text)
                on_token(text)
        return _strip_chat_control("".join(full_text))

    def _stream_chat(self, stream, on_token: Callable[[str], None]) -> str:
        """Consume a create_chat_completion streaming iterator with the same
        on_token(content)-per-chunk contract expected by the caller."""
        full_text = []
        for chunk in stream:
            if self._stop_requested:
                break
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if "content" in delta:
                content = delta["content"]
                if content:
                    full_text.append(content)
                    on_token(content)
        return _strip_chat_control("".join(full_text))

    def _v15_special_id(self, token: str) -> int:
        """Resolve a v1.5 control token to its single trained id."""
        ids = self._llm.tokenize(
            token.encode("utf-8"), add_bos=False, special=True
        )
        if len(ids) != 1:
            raise RuntimeError(
                f"GGUF tokenizer does not map {token!r} to one token "
                f"(got {ids}). This file cannot be prompted safely."
            )
        return int(ids[0])

    def _build_v15_prompt_ids(self, messages: List[Dict]) -> List[int]:
        """Token ids for the Dizel v1.5 ``<|role|>`` prompt.

        Built from ids, not text: llama.cpp splits ``<|end|>`` into five
        ordinary pieces unless ``special=True`` is used, and it adds its own
        dummy prefix around special tokens.  The model was trained with
        ``<|end|>`` as one token, so a text prompt silently fed it a
        different sequence.  The layout mirrors ``NativeBackend``'s
        per-chunk encode (a ``▁`` boundary before every role marker, content
        and ``<|end|>``) so both backends send identical ids.
        """
        # `tokenize(" ")` returns two ▁ pieces (llama.cpp's dummy prefix plus
        # the literal space); the first is the boundary token.
        space_ids = self._llm.tokenize(b" ", add_bos=False)
        if not space_ids:
            raise RuntimeError("GGUF tokenizer did not produce a space token.")
        space_id = int(space_ids[0])
        end_id = self._v15_special_id("<|end|>")
        assistant_id = self._v15_special_id("<|assistant|>")

        ids: List[int] = []
        for msg in messages:
            role = str(msg.get("role") or "user").strip().lower()
            if role not in ("system", "user", "assistant"):
                role = "user"
            content = str(msg.get("content") or "").strip()
            ids.append(space_id)
            ids.append(self._v15_special_id(f"<|{role}|>"))
            ids.extend(
                int(tid)
                for tid in self._llm.tokenize(
                    content.encode("utf-8"), add_bos=False
                )
            )
            ids.append(space_id)
            ids.append(end_id)
        ids.append(space_id)
        ids.append(assistant_id)
        return ids

    def stop(self) -> None:
        self._stop_requested = True
        
    def count_tokens(self, text: str) -> int:
        if not self._llm: return len(text.split())
        return len(self._llm.tokenize(text.encode("utf-8")))
        
    @property
    def context_length(self) -> int:
        # Prefer the GGUF-metadata context resolved at load time; otherwise
        # fall back to llama-cpp's own view (n_ctx_train when exposed, then
        # n_ctx) instead of the old hard-coded 2048. Both are read through
        # _llm_attr_or_call so method- and property-style bindings work.
        if self._llm is None:
            return self._n_ctx if self._n_ctx else 2048
        if self._n_ctx:
            return self._n_ctx
        train_ctx = _llm_attr_or_call(self._llm, "n_ctx_train")
        if train_ctx is not None:
            try:
                return int(train_ctx)
            except (TypeError, ValueError):
                pass
        ctx = _llm_attr_or_call(self._llm, "n_ctx")
        if ctx is not None:
            try:
                return int(ctx)
            except (TypeError, ValueError):
                pass
        return 2048

    @property
    def model_info(self) -> dict:
        if not self._llm: return {}
        meta = self._gguf_meta
        if not meta and self._model_path:
            meta = _read_gguf_metadata_dict(self._model_path)
        arch = self._gguf_arch

        def _meta(key, default=None):
            """Read a metadata value with arch-prefixed then bare key."""
            if arch:
                v = meta.get(f"{arch}.{key}")
                if v is not None:
                    return v
            return meta.get(key, default)

        d_model = _llm_attr_or_call(self._llm, "n_embd") or _meta("embedding_length")
        n_layers = _llm_attr_or_call(self._llm, "n_layer") or _meta("block_count")
        n_heads = _llm_attr_or_call(self._llm, "n_head") or _meta("attention.head_count")
        vocab = _llm_attr_or_call(self._llm, "n_vocab") or _meta("vocab_size")
        info = {
            "provider": "llama.cpp (GGUF)",
            "device": self._device,
            "ctx_len": self.context_length,
            "vocab": vocab,
        }
        if d_model:  info["d_model"] = int(d_model)
        if n_layers: info["n_layers"] = int(n_layers)
        if n_heads:  info["n_heads"] = int(n_heads)
        total = _estimate_gguf_params(meta, arch)
        if total:
            info["params"] = f"{total/1e6:.2f} M"
        return info
        
    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    @property
    def supports_system_prompt(self) -> bool:
        return self._supports_system_prompt


class SafeTensorsBackend(ModelBackend):
    """Backend for HuggingFace SafeTensors models."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._stop_requested = False
        self._streamer = None

    def close(self) -> None:
        model, self._model = self._model, None
        tokenizer, self._tokenizer = self._tokenizer, None
        self._streamer = None
        del model, tokenizer
        if self._device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def load(self, checkpoint_path: str, device: str, on_progress: Callable[[str], None], **kwargs) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("HuggingFace transformers is not installed. Run: pip install transformers torch safetensors")
            
        self._device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        on_progress(f"Loading SafeTensors model from {checkpoint_path}…")
        
        model_dir = os.path.dirname(checkpoint_path)
        # Assuming the directory contains the full HF model (config.json, tokenizer.json, etc.)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            device_map=self._device,
            torch_dtype="auto",
            trust_remote_code=True
        )
        self._model.eval()
        
        on_progress("SafeTensors model ready.")

    def generate(self, messages: List[Dict], system_prompt: str, sampling_params: Dict, max_tokens: int, on_token: Callable[[str], None]) -> str:
        self._stop_requested = False
        import torch
        from transformers import TextIteratorStreamer
        
        api_msgs = []
        if system_prompt:
            api_msgs.append({"role": "system", "content": system_prompt})
        for msg in messages:
            api_msgs.append({"role": msg["role"], "content": msg["content"]})
            
        prompt_text = self._tokenizer.apply_chat_template(api_msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        
        self._streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = {
            "inputs": inputs.input_ids,
            "max_new_tokens": max_tokens,
            "temperature": sampling_params.get("temperature", 0.8),
            "top_k": sampling_params.get("top_k", 50),
            "top_p": sampling_params.get("top_p", 0.92),
            "repetition_penalty": sampling_params.get("repetition_penalty", 1.15),
            "do_sample": sampling_params.get("temperature", 0.8) > 0,
            "streamer": self._streamer
        }
        
        # Run generation in a background thread so we can iterate the streamer here
        import threading
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        
        full_text = []
        for text in self._streamer:
            if self._stop_requested:
                break
            full_text.append(text)
            if text:
                on_token(text)
        
        # Wait for the generation thread to finish to prevent resource leaks
        thread.join(timeout=30)
                
        return "".join(full_text)

    def stop(self) -> None:
        self._stop_requested = True
        
    def count_tokens(self, text: str) -> int:
        if not self._tokenizer: return len(text.split())
        return len(self._tokenizer.encode(text))
        
    @property
    def context_length(self) -> int:
        if not self._model: return 2048
        return getattr(self._model.config, "max_position_embeddings", 2048)
        
    @property
    def model_info(self) -> dict:
        if not self._model: return {}
        return {
            "provider": "HuggingFace (SafeTensors)",
            "device": self._device,
            "ctx_len": self.context_length,
            "vocab": self._model.config.vocab_size,
        }
        
    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None
