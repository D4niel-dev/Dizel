"""CPU-only regression tests for the local inference backends.

Run from the repository root:

    python scripts/test_inference_backend.py

Covers the two failure modes that produced malformed desktop output:

* distilled checkpoints are saved with unmerged LoRA adapters, and
* the v1.5 prompt must be built from the trained token ids and keep the
  system turn its SFT base learned.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from model.lora_merge import fold_lora_adapters, has_lora_adapters
from model.naming import model_display_name


def _load_backend_module():
    path = os.path.join(ROOT, "dizel_core", "model_backend.py")
    spec = importlib.util.spec_from_file_location("dizel_model_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lora_fold():
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    a = torch.randn(2, 4)
    b = torch.randn(3, 2)
    state = {
        "blocks.0.attn.q_proj.linear.weight": weight,
        "blocks.0.attn.q_proj.lora_A": a,
        "blocks.0.attn.q_proj.lora_B": b,
        "blocks.0.attn.q_proj.linear.bias": torch.ones(3),
    }
    assert has_lora_adapters(state)

    folded, info = fold_lora_adapters(state, lora_alpha=4, lora_r=2)
    assert info == {"adapters": 1, "alpha": 4, "r": 2}
    expected = weight + (b @ a) * 2.0
    assert torch.allclose(folded["blocks.0.attn.q_proj.weight"], expected)
    assert torch.equal(folded["blocks.0.attn.q_proj.bias"], torch.ones(3))
    assert "blocks.0.attn.q_proj.linear.weight" not in folded
    assert "blocks.0.attn.q_proj.lora_A" not in folded
    assert "blocks.0.attn.q_proj.lora_B" not in folded

    # No metadata: r comes from lora_A, alpha follows alpha = 2 * r.
    folded2, info2 = fold_lora_adapters(state)
    assert info2 == {"adapters": 1, "alpha": 4, "r": 2}
    assert torch.allclose(folded2["blocks.0.attn.q_proj.weight"], expected)


def test_lora_noop():
    weight = torch.zeros(3, 4)
    folded, info = fold_lora_adapters({"plain.weight": weight})
    assert info["adapters"] == 0
    assert list(folded) == ["plain.weight"]
    assert not has_lora_adapters(folded)


class _FakeV15Llama:
    """Minimal llama.cpp tokenizer surface for prompt-id tests."""

    _IDS = {
        b" ": [91683, 91683],
        b"<|end|>": [8],
        b"<|system|>": [6],
        b"<|user|>": [4],
        b"<|assistant|>": [5],
        b"Hello": [24887],
    }

    def tokenize(self, text, add_bos=True, special=False):
        assert add_bos is False
        return list(self._IDS[text])


def test_v15_prompt_boundaries():
    module = _load_backend_module()
    backend = module.GGUFBackend()
    backend._llm = _FakeV15Llama()

    user_only = backend._build_v15_prompt_ids(
        [{"role": "user", "content": "Hello"}]
    )
    assert user_only == [91683, 4, 24887, 91683, 8, 91683, 5], user_only

    with_system = backend._build_v15_prompt_ids(
        [
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "Hello"},
        ]
    )
    assert with_system == [
        91683, 6, 24887, 91683, 8,
        91683, 4, 24887, 91683, 8,
        91683, 5,
    ], with_system

    # The .pt backend encodes each chunk separately. Both backends must send
    # the same ids for the same conversation.
    from config import V15_CONFIG
    from training.dataset import END_TOKEN, ROLE_TOKENS, Tokenizer

    tokenizer = Tokenizer(
        model_path=os.path.join(ROOT, V15_CONFIG.tokenizer.model_path)
    )
    native_ids = []
    for role in ("system", "user"):
        native_ids += tokenizer.encode(ROLE_TOKENS[role])
        native_ids += tokenizer.encode("Hello")
        native_ids += tokenizer.encode(END_TOKEN)
    native_ids += tokenizer.encode(ROLE_TOKENS["assistant"])
    assert with_system == native_ids, (with_system, native_ids)


def test_v15_system_prompt_contract():
    module = _load_backend_module()
    cfg = SimpleNamespace(num_experts=1)

    # v1.5 post-training inherits system turns from its SFT base.
    assert module._chat_contract({}, "1.5", cfg)[2] is True
    assert module._chat_contract({"phase": "sft"}, "1.5", cfg)[2] is True
    # A pretrain-only checkpoint never saw role or system tokens.
    assert (
        module._chat_contract(
            {"phase": "pretrain"}, "1.5", cfg, pretrain_only=True
        )[2]
        is False
    )
    # An explicit checkpoint flag always wins.
    assert (
        module._chat_contract(
            {"supports_system_prompt": False}, "1.5", cfg
        )[2]
        is False
    )
    # v1.1/v1.2 predate role tokens entirely.
    assert module._chat_contract({}, "1.2.2", cfg)[2] is False
    assert module._chat_contract({}, "1.1.0", cfg)[2] is False


def test_v15_system_prompt_compaction():
    module = _load_backend_module()

    # The long GUI/profile prompt is out of distribution and derails the
    # model, so it collapses to the compact identity line.
    long_prompt = (
        "You are Dizel. You are not any other AI model or assistant. "
        "You are a highly capable, intelligent, and helpful AI assistant. "
        "You answer thoughtfully, concisely, and accurately. "
        "You use Markdown formatting to organize your thoughts and provide "
        "clear, structured text. Adapt your tone to the context."
    )
    assert len(long_prompt) > module._V15_SYSTEM_PROMPT_MAX_CHARS
    assert module._v15_system_prompt(long_prompt) == module._V15_IDENTITY

    # A short instruction survives, and always keeps the identity.
    assert module._v15_system_prompt("Answer in French.") == (
        f"{module._V15_IDENTITY} Answer in French."
    )
    # A short prompt that already sets identity is left alone.
    assert module._v15_system_prompt("You are Dizel.") == "You are Dizel."
    # No system prompt means no system turn at all.
    assert module._v15_system_prompt("   ") == ""

    msgs = module._role_end_messages(
        [{"role": "user", "content": "Hi"}], long_prompt, True, True
    )
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == module._V15_IDENTITY

    # A pretrain-only checkpoint gets no system turn and keeps no system
    # message from history either.
    msgs = module._role_end_messages(
        [
            {"role": "system", "content": "history system"},
            {"role": "user", "content": "Hi"},
        ],
        long_prompt,
        False,
        True,
    )
    assert msgs == [{"role": "user", "content": "Hi"}]

    # Non-v1.5 keeps the prompt verbatim.
    msgs = module._role_end_messages(
        [{"role": "user", "content": "Hi"}], long_prompt, True, False
    )
    assert msgs[0]["content"] == long_prompt


def test_display_names():
    cases = {
        "dizel-1.6-flash-preview-m1.pt": "Dizel 1.6 Flash Preview",
        "dizel-1.6-flash-preview-m1.gguf": "Dizel 1.6 Flash Preview",
        "dizel-1.6-preview-m1.pt": "Dizel 1.6 Preview",
        "dizel-1.6-flash-435m-preview-m1.pt": "Dizel 1.6 Flash 435M Preview",
        "dizel-v15-distill-best.pt": "Dizel 1.5 Distill",
        "Dizel-Distill.gguf": "Dizel Distill",
        "dizel.pt": "Dizel",
        "dizel-1.6-flash-preview-m1-step5000-q4_k_m.gguf": (
            "Dizel 1.6 Flash Preview"
        ),
        "dizel-1.6-preview-m1-milestone-step000500.pt": "Dizel 1.6 Preview",
        "Qwen2.5-7B-Instruct.gguf": "Qwen2.5 7B Instruct",
        "gemma-2-9b-it.gguf": "Gemma 2 9B IT",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": (
            "Meta Llama 3.1 8B Instruct Q4_K_M"
        ),
    }
    for filename, expected in cases.items():
        actual = model_display_name(filename)
        assert actual == expected, (filename, actual, expected)


def test_v16_chat_contract_defaults():
    module = _load_backend_module()

    is_v16, chat_format, system = module._chat_contract(
        {"_format_version": "1.6"}, "1.6", SimpleNamespace(num_experts=4)
    )
    assert (is_v16, chat_format, system) == (True, "chatml", True)

    # Older saves may omit every v1.6 flag; the MoE shape is enough.
    is_v16, chat_format, system = module._chat_contract(
        {}, "1.5", SimpleNamespace(num_experts=4)
    )
    assert (is_v16, chat_format, system) == (True, "chatml", True)

    # v1.5 post-training stays on the raw path but keeps the system turn
    # inherited from its SFT base.
    assert module._chat_contract(
        {"_format_version": "1.5"}, "1.5", SimpleNamespace(num_experts=1)
    ) == (False, "role_end", True)

    # A v1.5 checkpoint can opt out explicitly.
    assert module._chat_contract(
        {"supports_system_prompt": False},
        "1.5",
        SimpleNamespace(num_experts=1),
    )[2] is False


class _StubQwenTokenizer:
    """Character-level stand-in exposing the QwenTokenizer surface."""

    eos_id = 0
    unk_id = None

    class tokenizer:
        _IDS = {"<|im_end|>": 1, "<|endoftext|>": 0, "<|im_start|>": 2}

        @classmethod
        def convert_tokens_to_ids(cls, token):
            return cls._IDS.get(token)

        @classmethod
        def convert_ids_to_tokens(cls, token_id):
            for token, value in cls._IDS.items():
                if value == token_id:
                    return token
            return None

    def encode(self, text, add_bos=False, add_eos=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token_id)) for token_id in ids)


def test_chatml_prompt():
    module = _load_backend_module()
    backend = module.NativeBackend()
    backend._tokenizer = _StubQwenTokenizer()
    backend._chat_format = "chatml"
    backend._supports_system_prompt = True

    ids = backend._build_chatml_prompt_ids(
        [{"role": "user", "content": "Hello"}], "You are Dizel."
    )
    text = backend._tokenizer.decode(ids)
    expected = (
        "<|im_start|>system\nYou are Dizel.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert text == expected, repr(text)

    # A separate system_prompt wins over a system turn in history.
    ids = backend._build_chatml_prompt_ids(
        [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "Hi"},
        ],
        "new",
    )
    text = backend._tokenizer.decode(ids)
    assert text.count("<|im_start|>system") == 1
    assert "old" not in text and "new" in text

    # Without system support the system turn is dropped.
    backend._supports_system_prompt = False
    ids = backend._build_chatml_prompt_ids(
        [
            {"role": "system", "content": "dropped"},
            {"role": "user", "content": "Hi"},
        ],
        "ignored",
    )
    text = backend._tokenizer.decode(ids)
    assert "<|im_start|>system" not in text
    assert "dropped" not in text and "ignored" not in text


def test_v16_stop_ids():
    module = _load_backend_module()
    backend = module.NativeBackend()
    backend._tokenizer = _StubQwenTokenizer()
    backend._chat_format = "chatml"
    assert backend._stop_token_ids() == {0, 1, 2}


class _VocabStub:
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size


def test_padded_vocab_is_masked():
    module = _load_backend_module()
    backend = module.NativeBackend()

    # v1.6: 151,936 embedding rows, 151,665 tokenizer ids.
    backend._tokenizer = _VocabStub(8)
    logits = torch.zeros(1, 10)
    backend._mask_padded_vocab(logits)
    assert torch.isinf(logits[0, 8:]).all()
    assert (logits[0, 8:] < 0).all()
    assert torch.equal(logits[0, :8], torch.zeros(8))

    # A tokenizer that covers the whole table leaves it untouched.
    backend._tokenizer = _VocabStub(10)
    logits = torch.zeros(1, 10)
    backend._mask_padded_vocab(logits)
    assert torch.equal(logits, torch.zeros(1, 10))


def test_tokenizer_candidates():
    module = _load_backend_module()
    default = "Qwen/Qwen2.5-0.5B-Instruct"

    # The training-side relative path must never be sent to the Hub.
    stale = module._qwen_tokenizer_candidates("tokenizer/qwen2.5")
    assert "tokenizer/qwen2.5" not in stale
    assert stale[-1] == default

    # A real Hub id stays available so custom tokenizers still work.
    hub = module._qwen_tokenizer_candidates("Qwen/Qwen2.5-1.5B-Instruct")
    assert "Qwen/Qwen2.5-1.5B-Instruct" in hub
    assert hub[-1] == default

    with tempfile.TemporaryDirectory() as tmp:
        assert module._qwen_tokenizer_candidates(tmp)[0] == tmp

    missing = os.path.join(tempfile.gettempdir(), "dizel-missing-tokenizer")
    candidates = module._qwen_tokenizer_candidates(missing)
    assert not any(path.endswith("dizel-missing-tokenizer") for path in candidates)
    assert candidates[-1] == default

    assert module._qwen_tokenizer_candidates("")[-1] == default


if __name__ == "__main__":
    test_lora_fold()
    test_lora_noop()
    test_v15_prompt_boundaries()
    test_v15_system_prompt_contract()
    test_v15_system_prompt_compaction()
    test_display_names()
    test_v16_chat_contract_defaults()
    test_chatml_prompt()
    test_v16_stop_ids()
    test_padded_vocab_is_masked()
    test_tokenizer_candidates()
    print("inference backend tests: OK")
