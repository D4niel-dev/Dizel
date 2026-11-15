"""CPU-only regression tests for the Skywork reward-model DPO filter.

Run from the repository root:

    python scripts/test_reward_filter_v16.py

The tests use a tiny randomly initialized Qwen3 sequence classifier and the
local Qwen2.5 tokenizer, so they do not download the 1.7B Skywork weights.
They verify the parts that can silently corrupt a run: system-prompt stripping,
batched-vs-single scoring, margin filtering, split consistency, and partial
score-file recovery.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from transformers import (
        AutoTokenizer,
        Qwen3Config,
        Qwen3ForSequenceClassification,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "These tests require 'transformers>=4.51.0' for Qwen3. "
        "Install it with: pip install 'transformers>=4.51.0'"
    ) from exc

from scripts.score_dpo_pairs_v16 import (
    ReportContext,
    append_scored_rows,
    build_rm_messages,
    filter_scored,
    format_conversation,
    load_scored_rows,
    make_scored_rows,
    percentile,
    prepare_pairs,
    score_pairs,
    split_indices,
    summarize,
)
from training.posttrain_common_v16 import split_records

TOKENIZER_PATH = os.path.join(ROOT, "tokenizer", "qwen2.5")


def expect(value, message: str) -> None:
    if not value:
        raise AssertionError(message)


def _record(system, user, chosen, rejected, source="test"):
    prompt = [{"role": "user", "content": user}]
    if system is not None:
        prompt.insert(0, {"role": "system", "content": system})
    return {
        "prompt_messages": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
    }


def _tiny_rm(tokenizer):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
    )
    model = Qwen3ForSequenceClassification(config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.requires_grad_(False)
    return model


def _direct_score(model, tokenizer, messages, response, max_length=512):
    ids = format_conversation(
        tokenizer,
        messages + [{"role": "assistant", "content": response}],
        max_length,
    )
    expect(ids is not None, "direct-score conversation must fit")
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    return float(logits[0, 0])


def test_system_prompt_is_removed() -> None:
    messages = build_rm_messages(
        _record("Be concise.", "What is 2+2?", "4", "5")
    )
    expect(messages is not None, "a valid prompt must normalize")
    expect(
        all(message["role"] != "system" for message in messages),
        "Dizel's system prompt must not reach the Skywork template",
    )
    expect(messages[-1]["role"] == "user", "the prompt must end on a user turn")

    trailing_assistant = {
        "prompt_messages": [
            {"role": "user", "content": "Pick one."},
            {"role": "assistant", "content": "Earlier answer."},
        ],
        "chosen": "A",
        "rejected": "B",
    }
    messages = build_rm_messages(trailing_assistant)
    expect(messages is not None, "assistant history before the final user must survive")
    expect(messages[-1]["role"] == "user", "trailing assistant history is removed")


def test_batched_scores_match_direct(tokenizer) -> None:
    model = _tiny_rm(tokenizer)
    records = [
        _record(
            "Be concise.",
            "What is 2+2?",
            "The answer is 4.",
            "The answer is 5.",
        ),
        _record(None, "Say hello.", "Hello!", "Goodbye."),
        _record("Be kind.", "Name a colour.", "Blue.", "Red."),
    ]
    pairs, skipped = prepare_pairs(records, tokenizer, 512)
    expect(len(pairs) == 3, "all three pairs must tokenize")
    expect(not skipped, f"unexpected skips: {dict(skipped)}")

    scores = score_pairs(
        pairs,
        model,
        int(tokenizer.pad_token_id),
        2,
        torch.device("cpu"),
    )
    rows = make_scored_rows(pairs, scores, "tiny-rm")
    expect([row["rm_index"] for row in rows] == [0, 1, 2],
           "scored rows must preserve input order")
    for pair, row in zip(pairs, rows):
        direct_chosen = _direct_score(model, tokenizer, pair.messages, pair.chosen)
        direct_rejected = _direct_score(
            model, tokenizer, pair.messages, pair.rejected
        )
        expect(
            abs(row["rm_chosen"] - direct_chosen) < 1e-3,
            "batched chosen score must match a single-sequence forward",
        )
        expect(
            abs(row["rm_rejected"] - direct_rejected) < 1e-3,
            "batched rejected score must match a single-sequence forward",
        )
        expect(
            abs(row["rm_margin"] - (row["rm_chosen"] - row["rm_rejected"])) < 1e-9,
            "rm_margin must be chosen minus rejected",
        )


def test_prepare_skips_and_length_limit(tokenizer) -> None:
    records = [
        _record(None, "Hi", "", "x"),
        _record(None, "Hi", "a", "b"),
    ]
    pairs, skipped = prepare_pairs(records, tokenizer, 4096)
    expect(len(pairs) == 1, "the empty chosen completion must be skipped")
    expect(skipped["invalid_response"] == 1, "the skip reason must be recorded")

    pairs, skipped = prepare_pairs(records, tokenizer, 1)
    expect(not pairs, "a one-token limit must reject the conversations")
    expect(
        skipped["chosen_too_long_or_empty"] == 1,
        "the chosen-length skip must be recorded",
    )
    expect(
        skipped["invalid_response"] == 1,
        "the invalid record must still be counted",
    )


def test_margin_filter_and_report() -> None:
    rows = [
        {
            "rm_index": 0,
            "rm_model": "tiny",
            "rm_chosen": 0.0,
            "rm_rejected": 1.0,
            "rm_margin": -1.0,
            "source": "a",
        },
        {
            "rm_index": 1,
            "rm_model": "tiny",
            "rm_chosen": 1.0,
            "rm_rejected": 1.0,
            "rm_margin": 0.0,
            "source": "a",
        },
        {
            "rm_index": 2,
            "rm_model": "tiny",
            "rm_chosen": 1.5,
            "rm_rejected": 1.0,
            "rm_margin": 0.5,
            "source": "b",
        },
        {
            "rm_index": 3,
            "rm_model": "tiny",
            "rm_chosen": 3.0,
            "rm_rejected": 1.0,
            "rm_margin": 2.0,
            "source": "b",
        },
    ]
    kept, dropped = filter_scored(rows, 0.0)
    expect(len(kept) == 3, "ties must be kept at min_margin=0")
    expect(dropped == 1, "the negative-margin pair must be dropped")

    train_rows = rows[:3]
    val_rows = rows[3:]
    kept_train, _ = filter_scored(train_rows, 0.0)
    context = ReportContext(
        reward_model="tiny",
        dtype="float32",
        device="cpu",
        input_path="input.jsonl",
        scored_path="scored.jsonl",
        output_path="train.jsonl",
        val_output_path="val.jsonl",
        input_records=4,
        input_sha256="input",
        scored_sha256="scored",
        output_sha256="output",
        val_sha256="val",
        min_margin=0.0,
        max_length=512,
        batch_size=2,
        limit=None,
        val_fraction=0.25,
        seed=42,
        elapsed_seconds=1.0,
    )
    report = summarize(
        rows,
        train_rows,
        kept_train,
        val_rows,
        Counter(),
        context,
    )
    expect(report["counts"]["kept_train_pairs"] == 2,
           "report must count kept training pairs")
    expect(report["counts"]["dropped_train_pairs"] == 1,
           "report must count dropped training pairs")
    expect(report["train"]["agreement_rate"] == 1 / 3,
           "train agreement rate must use margin > 0")
    expect(report["val"]["agreement_rate"] == 1.0,
           "the held-out pair must be counted separately")
    expect(report["by_source"]["a"]["train_pairs"] == 2,
           "per-source counts must respect the split")
    expect(report["by_source"]["a"]["val_pairs"] == 0,
           "the val index must not leak into the source train count")
    expect(percentile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5,
           "percentile interpolation must be linear")


def test_partial_score_file_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scores.jsonl")
        first = {
            "rm_index": 0,
            "rm_model": "tiny",
            "rm_chosen": 1.0,
            "rm_rejected": 0.0,
            "rm_margin": 1.0,
        }
        second = {
            "rm_index": 1,
            "rm_model": "tiny",
            "rm_chosen": 0.0,
            "rm_rejected": 1.0,
            "rm_margin": -1.0,
        }
        append_scored_rows(path, [first, second])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"rm_index": 2, "rm_model": "tiny"')

        loaded = load_scored_rows(path, "tiny")
        expect(len(loaded) == 2, "an incomplete final line must be discarded")
        with open(path, "rb") as handle:
            expect(handle.read().endswith(b"\n"),
                   "the repaired score file must end on a newline")

        third = {
            "rm_index": 2,
            "rm_model": "tiny",
            "rm_chosen": 2.0,
            "rm_rejected": 1.0,
            "rm_margin": 1.0,
        }
        append_scored_rows(path, [third])
        loaded = load_scored_rows(path, "tiny")
        expect(len(loaded) == 3, "scoring must resume after a partial line")


def test_line_separators_in_text_survive_reload() -> None:
    # Real DPO text contains U+2028 (and U+2029).  ``json.dumps`` leaves those
    # unescaped, and ``str.splitlines()`` would split the JSON object in half.
    # Regression: a 9,975-pair run died on rm_index 5674 for exactly this.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scores.jsonl")
        rows = [
            {
                "rm_index": 0,
                "rm_model": "tiny",
                "rm_chosen": 1.0,
                "rm_rejected": 0.0,
                "rm_margin": 1.0,
                "chosen": "first\u2028second\u2029third",
            },
            {
                "rm_index": 1,
                "rm_model": "tiny",
                "rm_chosen": 0.0,
                "rm_rejected": 1.0,
                "rm_margin": -1.0,
                "chosen": "plain",
            },
        ]
        append_scored_rows(path, rows)
        loaded = load_scored_rows(path, "tiny")
        expect(len(loaded) == 2, "U+2028/U+2029 must not split a scored row")
        expect(
            loaded[0]["chosen"] == "first\u2028second\u2029third",
            "separator characters must round-trip unchanged",
        )


def test_split_indices_matches_records() -> None:
    records = [{"i": index} for index in range(13)]
    train, val = split_records(records, 0.3, 7)
    train_indices, val_indices = split_indices(len(records), 0.3, 7)
    expect(train == [records[index] for index in train_indices],
           "split_records must use split_indices")
    expect(val == [records[index] for index in val_indices],
           "the validation split must match split_indices")
    expect(len(val) == 3, "13 records at 30% must reserve 3 validation rows")


def test_format_length_limit(tokenizer) -> None:
    messages = [{"role": "user", "content": "Hello"}]
    expect(format_conversation(tokenizer, messages, 4096) is not None,
           "a short conversation must format")
    expect(format_conversation(tokenizer, messages, 1) is None,
           "a one-token limit must reject the template markers")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    test_system_prompt_is_removed()
    test_batched_scores_match_direct(tokenizer)
    test_prepare_skips_and_length_limit(tokenizer)
    test_margin_filter_and_report()
    test_partial_score_file_recovery()
    test_line_separators_in_text_survive_reload()
    test_split_indices_matches_records()
    test_format_length_limit(tokenizer)
    print("v1.6 reward-filter tests passed")


if __name__ == "__main__":
    main()
