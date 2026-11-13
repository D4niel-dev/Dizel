"""
training/tokenizer_qwen.py — Qwen2.5 tokenizer wrapper for Dizel 1.6.

The v1.6 model keeps Qwen2.5's padded 151,936-row embedding table so that
future token-level distillation from ``Qwen/Qwen2.5-0.5B-Instruct`` needs no
vocabulary migration.  The tokenizer itself emits 151,665 ids; the extra
model rows are never targets. This module is deliberately thin: it exposes the same
``encode``/``decode``/``eos_id`` surface that ``training/dataset.py`` and
``training/tokenizer_utils.py`` expect, plus a vocabulary fingerprint used to
keep token caches and checkpoints tied to the exact tokenizer.

``transformers`` is imported lazily inside :class:`QwenTokenizer`, so this
module can be imported (and the pretraining checkpoint helpers tested) on a
machine without transformers installed.
"""

import hashlib
import os
import sys
from collections.abc import Iterable
from typing import List, Optional

__all__ = ["QwenTokenizer", "verify_tokenizer"]

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

VERIFY_SAMPLES = {
    "english": (
        "The quick brown fox jumps over the lazy dog. "
        "Dizel is a small language model built for experiments."
    ),
    "vietnamese": (
        "Xin chào, tôi là Dizel. Hôm nay trời rất đẹp và chúng ta sẽ "
        "cùng nhau học cách huấn luyện một mô hình ngôn ngữ nhỏ."
    ),
    "code": (
        "def add(a: int, b: int) -> int:\n"
        "    # Return the sum of two integers.\n"
        "    return a + b\n"
    ),
}


class QwenTokenizer:
    """Thin wrapper around ``transformers.AutoTokenizer``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "QwenTokenizer requires 'transformers>=4.40.0'. "
                "Install it with: pip install -r requirements.txt"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model_name = model_name
        self.vocab_size = len(self.tokenizer)

        # Discover special ids instead of hardcoding them.
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.unk_id = self.tokenizer.unk_token_id

        if self.eos_id is None:
            raise ValueError(
                f"Tokenizer '{model_name}' has no EOS token; a causal LM "
                "training run cannot define its stop token."
            )
        if self.pad_id is None:
            # Qwen2.5 ships without a dedicated pad token. Using EOS as pad is
            # the standard choice; pretraining windows never contain padding,
            # this only matters for future SFT batching.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_id = self.tokenizer.pad_token_id

        self.chat_template = getattr(self.tokenizer, "chat_template", None)

        self._fingerprint: Optional[str] = None
        self._cache_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, Iterable) and not isinstance(ids, (list, tuple)):
            ids = list(ids)
        return self.tokenizer.decode(
            ids, skip_special_tokens=skip_special_tokens
        )

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def fingerprint(self) -> str:
        """Stable SHA-256 digest of the ordered (id, token) vocabulary."""
        if self._fingerprint is None:
            vocab = self.tokenizer.get_vocab()
            digest = hashlib.sha256()
            for token, token_id in sorted(vocab.items(), key=lambda kv: kv[1]):
                digest.update(str(int(token_id)).encode("ascii"))
                digest.update(b"\x00")
                digest.update(token.encode("utf-8"))
                digest.update(b"\n")
            self._fingerprint = digest.hexdigest()
        return self._fingerprint

    def cache_key(self) -> str:
        """Short tokenizer-specific key used to namespace token caches."""
        if self._cache_key is None:
            self._cache_key = f"qwen-{self.fingerprint()[:16]}"
        return self._cache_key

    # ------------------------------------------------------------------
    # Persistence / chat
    # ------------------------------------------------------------------
    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        """Delegate to the Qwen chat template (exposed for future SFT)."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


# ---------------------------------------------------------------------------
# Verification helper (no network calls)
# ---------------------------------------------------------------------------
def _safe_preview(text: str) -> str:
    """Keep the report printable without hiding Unicode on UTF-8 terminals."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return str(text).encode(encoding, "replace").decode(encoding)
    except (LookupError, UnicodeError):
        return str(text).encode("ascii", "replace").decode("ascii")


def verify_tokenizer(tokenizer) -> None:
    """
    Check the tokenizer against the v1.6 plan §5 checklist.

    Fails only on a missing EOS, an empty vocabulary, or a sample that
    tokenizes to zero tokens. Everything else is reported for inspection.
    """
    model_name = getattr(tokenizer, "model_name", "<unknown>")
    vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    eos_id = getattr(tokenizer, "eos_id", None)

    print("Tokenizer verification")
    print(f"  model      : {model_name}")
    print(f"  vocab_size : {vocab_size:,}")
    print(
        "  special ids: "
        f"pad={getattr(tokenizer, 'pad_id', None)} "
        f"bos={getattr(tokenizer, 'bos_id', None)} "
        f"eos={eos_id} "
        f"unk={getattr(tokenizer, 'unk_id', None)}"
    )

    if eos_id is None:
        raise RuntimeError("Tokenizer verification failed: EOS token is missing")
    if vocab_size <= 0:
        raise RuntimeError("Tokenizer verification failed: vocabulary is empty")

    for label, text in VERIFY_SAMPLES.items():
        ids = tokenizer.encode(text)
        if not ids:
            raise RuntimeError(
                f"Tokenizer verification failed: {label} sample tokenized "
                "to zero tokens"
            )
        decoded = tokenizer.decode(ids)
        preview = _safe_preview(decoded[:60].replace("\n", " "))
        print(f"  {label:<10} : {len(ids):>4} tokens | {preview!r}")

    for marker in ("<think>", "</think>"):
        ids = tokenizer.encode(marker)
        decoded = tokenizer.decode(ids)
        print(
            f"  {marker:<10} : {ids} -> "
            f"{_safe_preview(decoded)!r}"
        )

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        print("  chat template: not present")
        return

    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        print(f"  chat template: present but failed to render ({exc})")
        return

    preview = _safe_preview(str(rendered)[:80].replace("\n", " "))
    print(f"  chat template: present | {preview!r}")
