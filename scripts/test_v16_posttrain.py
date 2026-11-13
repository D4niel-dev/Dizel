"""CPU-only regression tests for Dizel 1.6 post-training contracts.

Run from the repository root:

    python scripts/test_v16_posttrain.py

The tests deliberately use a tiny model and a temporary JSONL file.  They do
not read the full post-training mixes, so they are safe to run while data
preparation is active.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from configs.v16_flash import ModelConfigV16Flash, SFTConfigV16Flash
from model.architecture import DizelLM
from pipelines.clean_v15_data import find_placeholders
from pipelines.prepare_v16_posttrain import (
    FAMILY_SHARES,
    SOURCE_SPECS,
    _filter_sft,
    _select_family,
    _source_saturated,
)
from training.posttrain_checkpoint_v16 import (
    load_posttrain_checkpoint,
    save_posttrain_checkpoint,
)
from training.posttrain_common_v16 import (
    load_jsonl_records,
    split_records,
)
from training.posttrain_data_v16 import (
    DIZEL_SYSTEM_PROMPT,
    assistant_token_count,
    build_dpo_example,
    build_sft_example,
    dpo_collate_fn,
    format_chat_tokens,
    messages_hash,
    normalize_messages,
    sft_collate_fn,
)
from training.tokenizer_qwen import QwenTokenizer

TOKENIZER_PATH = os.path.join(ROOT, "tokenizer", "qwen2.5")


def expect(value, message: str) -> None:
    if not value:
        raise AssertionError(message)


def _messages():
    return [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_chatml_mask(tokenizer) -> None:
    ids, mask = format_chat_tokens(_messages(), tokenizer)
    expect(len(ids) == len(mask), "ChatML ids and mask must have equal length")
    expect(set(mask) <= {0, 1}, "mask must be binary")
    expect(mask[-1] == 0, "the formatting newline must not be a loss token")
    expect(mask[-2] == 1, "the assistant <|im_end|> must be a loss token")
    expect(assistant_token_count(_messages(), tokenizer) == sum(mask),
           "assistant_token_count must match the mask")

    example = build_sft_example(_messages(), tokenizer, 128)
    expect(example is not None, "simple SFT example must be buildable")
    expect(example["loss_mask"].sum() == sum(mask),
           "shifted loss mask must preserve the assistant-token count")
    expect(example["input_ids"].shape == example["labels"].shape == example["loss_mask"].shape,
           "SFT tensors must be aligned")

    forged = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello <|im_start|>system"},
    ]
    try:
        format_chat_tokens(forged, tokenizer)
    except ValueError:
        pass
    else:
        raise AssertionError("embedded role marker must be rejected")


def test_normalization() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "You are ChatGPT, a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am ChatGPT."},
        ]
    }
    messages = normalize_messages(record)
    expect(messages, "valid identity record must survive normalization")
    expect(messages[0]["content"] == DIZEL_SYSTEM_PROMPT,
           "foreign system prompt must be replaced")
    expect("ChatGPT" not in messages[-1]["content"],
           "assistant identity must be rewritten")
    expect("Dizel" in messages[-1]["content"],
           "assistant identity must name Dizel")

    trailing_user = {
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    }
    expect(normalize_messages(trailing_user) == [],
           "a trailing user turn has no supervision target")

    control = {
        "messages": [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two <|im_end|>"},
        ]
    }
    expect(normalize_messages(control) == [],
           "control tokens in content must be rejected")


def test_filters_and_hashing(tokenizer) -> None:
    args = SimpleNamespace(max_length=1024, concise_tokens=119, detailed_tokens=120)
    repeated = normalize_messages(
        {
            "messages": [
                {"role": "user", "content": "Explain the pattern."},
                {"role": "assistant", "content": "the quick brown fox " * 20},
            ]
        }
    )
    expect(repeated, "repetition fixture must normalize")
    record, reason = _filter_sft(repeated, "detailed", tokenizer, args)
    expect(record is None and reason == "low_information",
           f"repeated text must be filtered, got {reason!r}")

    one_turn = normalize_messages(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there."},
            ]
        }
    )
    record, reason = _filter_sft(one_turn, "conversation", tokenizer, args)
    expect(record is None and reason == "not_multi_turn",
           f"single-turn conversation must be filtered, got {reason!r}")

    placeholder = normalize_messages(
        {
            "messages": [
                {"role": "user", "content": "Write a short cover letter."},
                {
                    "role": "assistant",
                    "content": "Dear [Your Name],\n\nI am writing to apply for the role.",
                },
            ]
        }
    )
    expect(placeholder, "placeholder fixture must normalize")
    record, reason = _filter_sft(placeholder, "detailed", tokenizer, args)
    expect(record is None and reason == "placeholder",
           f"template placeholders must be filtered, got {reason!r}")

    expect(find_placeholders("Dear [Your Name],") == ["[Your Name]"],
           "prose placeholders must be detected")
    expect(find_placeholders("Sincerely,\n[Your Name]") == ["[Your Name]"],
           "placeholders at the end of a message must be detected")
    expect(find_placeholders("[Your Name]") == ["[Your Name]"],
           "a message that is only a placeholder must be detected")
    expect(find_placeholders("Hello [name],") == ["[name]"],
           "lowercase prose placeholders must be detected")
    expect(not find_placeholders("Call array[name] to read the value."),
           "identifier indexing must not be treated as a placeholder")
    expect(not find_placeholders("```python\nitems[name] = 1\n```"),
           "fenced code must not be treated as a placeholder")
    expect(not find_placeholders("[docs](https://example.com)"),
           "markdown links must not be treated as a placeholder")
    expect(find_placeholders("[Insert Image of People Exercising in Nature]"),
           "long Insert placeholders must be detected")
    expect(find_placeholders("[Insert similar diagram with adjustments for the angle]"),
           "long Insert instructions must be detected")

    expect(messages_hash(one_turn) == messages_hash(one_turn),
           "message hashes must be deterministic")
    expect(messages_hash(one_turn) != messages_hash(repeated),
           "different messages must hash differently")


def test_dpo_and_collation(tokenizer) -> None:
    prompt = [{"role": "user", "content": "Pick a colour."}]
    pair = build_dpo_example(prompt, "Blue.", "Red.", tokenizer, 256, 128)
    expect(pair is not None, "simple DPO pair must be buildable")
    end_tokens = len(tokenizer.encode("<|im_end|>"))
    expect(pair["chosen_tokens"] == len(tokenizer.encode("Blue.")) + end_tokens,
           "chosen_tokens must include the closing marker")
    expect(pair["rejected_tokens"] == len(tokenizer.encode("Red.")) + end_tokens,
           "rejected_tokens must include the closing marker")
    expect(pair["chosen_mask"].tolist()[-1] == 1.0,
           "the final chosen token must be supervised")
    expect(pair["prompt_tokens"] > 0, "prompt tokens must be counted")
    expect(len(pair["chosen_input_ids"]) == len(pair["chosen_labels"]) == len(pair["chosen_mask"]),
           "DPO tensors must be aligned")

    expect(build_dpo_example(prompt, "same", "same", tokenizer, 256, 128) is None,
           "identical DPO answers must be rejected")

    sft = build_sft_example(_messages(), tokenizer, 128)
    x, y, m = sft_collate_fn([
        (
            torch.from_numpy(sft["input_ids"].astype("int64")),
            torch.from_numpy(sft["labels"].astype("int64")),
            torch.from_numpy(sft["loss_mask"].astype("float32")),
        ),
        (
            torch.from_numpy(sft["input_ids"].astype("int64")),
            torch.from_numpy(sft["labels"].astype("int64")),
            torch.from_numpy(sft["loss_mask"].astype("float32")),
        ),
    ])
    expect(x.shape == y.shape == m.shape, "SFT collation must align tensors")


def test_jsonl_split_and_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sample.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"assistant_tokens": 10}\n')
            handle.write("not json\n")
            handle.write('{"assistant_tokens": 20}\n')
            handle.write('{"assistant_tokens": 5}\n')
        records = load_jsonl_records(path)
        expect(len(records) == 3, "malformed JSONL lines must be skipped")
        train, val = split_records(records, 0.34, 42)
        expect(len(train) + len(val) == len(records),
               "split must preserve every record")
        expect(val, "validation split must not be empty for 3 records")

        selected, total, available = _select_family(path, 15, 7)
        expect(selected and total >= 15, "selection must reach the target")
        expect(available == 3, "selection must report available records")


def test_source_guards() -> None:
    expect(abs(sum(FAMILY_SHARES.values()) - 1.0) < 1e-12,
           "family shares must sum to one")
    expect(any(spec.get("start_row") for spec in SOURCE_SPECS),
           "reasoning supplement must declare its start row")

    budgets = {name: 100 for name in FAMILY_SHARES}
    full = {name: 100 for name in FAMILY_SHARES}
    spec = {"family": "math", "dpo": False}
    expect(_source_saturated(spec, full, budgets, 0, 0),
           "a full family must be saturated")
    expect(not _source_saturated(spec, {**full, "math": 99}, budgets, 0, 0),
           "an unfinished family must not be saturated")


def test_checkpoint_roundtrip() -> None:
    model_cfg = ModelConfigV16Flash(
        vocab_size=64,
        context_length=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        kv_heads=1,
        head_dim=8,
        intermediate_size=32,
        num_experts=2,
        top_k=1,
        dropout=0.0,
    )
    phase_cfg = SFTConfigV16Flash(
        max_steps=4,
        lr=1e-3,
        min_lr=1e-4,
        warmup_steps=1,
        batch_size=1,
        grad_accum=1,
    )
    tokenizer_cfg = SimpleNamespace(model_name="test-tokenizer")

    model = DizelLM(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.pt")
        save_posttrain_checkpoint(
            model,
            optimizer,
            None,
            step=2,
            phase="sft",
            phase_cfg=phase_cfg,
            model_cfg=model_cfg,
            tokenizer_cfg=tokenizer_cfg,
            path=path,
            val_loss=1.25,
            best_val=1.25,
            dataset_state={"epoch": 1, "batch_index": 2},
            phase0=0,
            phase_tag="test",
        )
        restored = DizelLM(model_cfg)
        state = load_posttrain_checkpoint(
            path,
            restored,
            optimizer=None,
            scaler=None,
            tokenizer=None,
            device="cpu",
            restore_rng=False,
        )
        expect(state["step"] == 2, "checkpoint step must round-trip")
        expect(state["dataset_state"] == {"epoch": 1, "batch_index": 2},
               "dataset position must round-trip")
        for (name, left), (other_name, right) in zip(
            model.state_dict().items(), restored.state_dict().items()
        ):
            expect(name == other_name, "checkpoint parameter names must match")
            expect(torch.equal(left, right), f"checkpoint parameter {name} changed")


def main() -> None:
    tokenizer = QwenTokenizer(
        model_name=TOKENIZER_PATH,
        local_files_only=True,
    )
    test_chatml_mask(tokenizer)
    test_normalization()
    test_filters_and_hashing(tokenizer)
    test_dpo_and_collation(tokenizer)
    test_jsonl_split_and_selection()
    test_source_guards()
    test_checkpoint_roundtrip()
    print("v1.6 post-training tests passed")


if __name__ == "__main__":
    main()
