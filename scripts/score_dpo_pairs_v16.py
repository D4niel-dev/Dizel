"""
scripts/score_dpo_pairs_v16.py — Offline Skywork RM scoring for Dizel 1.6 DPO.

DPO's training objective does not contain an external reward model.  The loss
is computed from the policy's chosen/rejected log-probabilities and the frozen
reference's log-probabilities.  The Skywork reward model is used here, offline,
to:

  * check whether the chosen completion scores above the rejected one,
  * filter noisy training pairs before DPO, and
  * write an unfiltered held-out split for honest checkpoint evaluation.

The scorer follows the Skywork-Reward-V2 model card: Qwen3 chat template with
no system prompt, scalar Bradley-Terry head (``num_labels=1``), and a
16,384-token limit.  Dizel's system prompt is stripped before scoring.  Reward
scores are relative, not calibrated probabilities.

Usage:
    python scripts/score_dpo_pairs_v16.py \
        --input data/v16/posttrain/dpo.jsonl \
        --scored_output data/v16/posttrain/dpo_rm_scored.jsonl \
        --output data/v16/posttrain/dpo_rm_filtered.jsonl \
        --val_output data/v16/posttrain/dpo_rm_val.jsonl \
        --report data/v16/posttrain/dpo_rm_report.json

    python training/dpo_v16.py \
        --base_checkpoint checkpoints/dizel-1.6-preview-m1-distill/dizel-1.6-preview-m1-distill-final.pt \
        --data data/v16/posttrain/dpo_rm_filtered.jsonl \
        --val_data data/v16/posttrain/dpo_rm_val.jsonl

On a T4 use ``--dtype float16``; on Ampere or newer the default is bfloat16.
The model card requires ``transformers>=4.51`` for Qwen3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.display_utils import (
    config_table,
    error,
    info,
    ok,
    rule,
    section,
    warning,
)
from training.posttrain_common_v16 import (
    load_hf_token_from_dotenv,
    split_indices,
)
from training.posttrain_data_v16 import normalize_prompt_messages

DEFAULT_REWARD_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
DEFAULT_INPUT = "data/v16/posttrain/dpo.jsonl"
DEFAULT_SCORED_OUTPUT = "data/v16/posttrain/dpo_rm_scored.jsonl"
DEFAULT_FILTERED_OUTPUT = "data/v16/posttrain/dpo_rm_filtered.jsonl"
DEFAULT_VAL_OUTPUT = "data/v16/posttrain/dpo_rm_val.jsonl"
DEFAULT_REPORT = "data/v16/posttrain/dpo_rm_report.json"
SKYWORK_MAX_LENGTH = 16_384


@dataclass
class PreparedPair:
    """One DPO pair tokenized for both reward-model completions."""

    index: int
    record: dict
    messages: list[dict]
    chosen: str
    rejected: str
    chosen_ids: list[int]
    rejected_ids: list[int]


@dataclass
class ReportContext:
    """Run-level provenance for the JSON report."""

    reward_model: str
    dtype: str
    device: str
    input_path: str
    scored_path: str
    output_path: str
    val_output_path: str
    input_records: int
    input_sha256: str
    scored_sha256: str
    output_sha256: str
    val_sha256: str
    min_margin: float
    max_length: int
    batch_size: int
    limit: Optional[int]
    val_fraction: float
    seed: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Input and formatting
# ---------------------------------------------------------------------------
def load_records(path: str) -> list[dict]:
    """Read a DPO JSONL file strictly; malformed data is a hard failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DPO data not found: {path!r}. Run "
            "`python pipelines/prepare_v16_posttrain.py` first."
        )
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"DPO data is empty: {path}")
    return records


def build_rm_messages(record: dict) -> Optional[list[dict]]:
    """Normalize a DPO prompt and remove every system message.

    The Skywork model card explicitly says not to include system prompts.  The
    stored prompt is normalized with the same helper the trainer uses, so the
    scorer sees the same user/assistant history that DPO would see.
    """
    prompt = record.get("prompt_messages")
    if not isinstance(prompt, list):
        return None
    normalized = normalize_prompt_messages({"messages": prompt})
    if not normalized:
        return None
    messages = [message for message in normalized if message.get("role") != "system"]
    return messages or None


def format_conversation(
    tokenizer,
    messages: list[dict],
    max_length: int,
) -> Optional[list[int]]:
    """Format one conversation exactly as the Skywork card recommends.

    The card formats to text first, strips a leading BOS token if the template
    already emitted it, then tokenizes.  This avoids a duplicate BOS.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    bos = getattr(tokenizer, "bos_token", None)
    if bos and text.startswith(bos):
        text = text[len(bos):]
    encoded = tokenizer(text, add_special_tokens=True)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids or len(ids) > max_length:
        return None
    return [int(token) for token in ids]


def prepare_pairs(
    records: list[dict],
    tokenizer,
    max_length: int,
    limit: Optional[int] = None,
):
    """Tokenize valid pairs and account for every skipped record."""
    pairs: list[PreparedPair] = []
    skipped = Counter()
    for index, record in enumerate(records):
        messages = build_rm_messages(record)
        if not messages:
            skipped["invalid_prompt"] += 1
            continue
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            skipped["invalid_response"] += 1
            continue
        chosen = chosen.strip()
        rejected = rejected.strip()
        if not chosen or not rejected or chosen == rejected:
            skipped["invalid_response"] += 1
            continue

        chosen_ids = format_conversation(
            tokenizer,
            messages + [{"role": "assistant", "content": chosen}],
            max_length,
        )
        if chosen_ids is None:
            skipped["chosen_too_long_or_empty"] += 1
            continue
        rejected_ids = format_conversation(
            tokenizer,
            messages + [{"role": "assistant", "content": rejected}],
            max_length,
        )
        if rejected_ids is None:
            skipped["rejected_too_long_or_empty"] += 1
            continue

        pairs.append(
            PreparedPair(
                index=index,
                record=record,
                messages=messages,
                chosen=chosen,
                rejected=rejected,
                chosen_ids=chosen_ids,
                rejected_ids=rejected_ids,
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs, skipped


# ---------------------------------------------------------------------------
# Reward-model inference
# ---------------------------------------------------------------------------
def score_pairs(
    pairs: list[PreparedPair],
    model,
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[tuple[int, str], float]:
    """Score chosen and rejected completions in length-sorted batches.

    Right padding is used deliberately: decoder-only reward heads select the
    rightmost non-pad token, so real tokens keep the same positions they would
    have in a single-sequence forward pass.
    """
    if not pairs:
        return {}
    jobs = []
    for pair in pairs:
        jobs.append((pair.index, "chosen", pair.chosen_ids))
        jobs.append((pair.index, "rejected", pair.rejected_ids))

    order = sorted(range(len(jobs)), key=lambda index: len(jobs[index][2]))
    scores: dict[tuple[int, str], float] = {}
    for start in range(0, len(order), batch_size):
        batch = [jobs[index] for index in order[start:start + batch_size]]
        max_len = max(len(ids) for _, _, ids in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, (_, _, ids) in enumerate(batch):
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :len(ids)] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        values = logits[:, 0].float().cpu().tolist()
        for (pair_index, side, _), value in zip(batch, values):
            scores[(pair_index, side)] = float(value)
    return scores


def make_scored_rows(
    pairs: list[PreparedPair],
    scores: dict[tuple[int, str], float],
    reward_model: str,
) -> list[dict]:
    """Attach RM metadata to the original DPO records without rewriting them."""
    rows = []
    for pair in pairs:
        chosen_score = scores[(pair.index, "chosen")]
        rejected_score = scores[(pair.index, "rejected")]
        row = dict(pair.record)
        row["rm_model"] = reward_model
        row["rm_chosen"] = chosen_score
        row["rm_rejected"] = rejected_score
        row["rm_margin"] = chosen_score - rejected_score
        row["rm_index"] = pair.index
        rows.append(row)
    return rows


def filter_scored(rows: list[dict], min_margin: float):
    """Keep rows whose chosen reward is at least ``min_margin`` above rejected."""
    kept = []
    for row in rows:
        margin = row.get("rm_margin")
        if not isinstance(margin, (int, float)):
            raise ValueError(
                f"Scored row {row.get('rm_index')} has no numeric rm_margin"
            )
        if float(margin) >= min_margin:
            kept.append(row)
    return kept, len(rows) - len(kept)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def percentile(sorted_values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile for an already sorted list."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def _margin_summary(rows: list[dict]) -> dict:
    margins = sorted(float(row["rm_margin"]) for row in rows)
    if not margins:
        return {
            "count": 0,
            "agreement_rate": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": len(margins),
        "agreement_rate": sum(1 for margin in margins if margin > 0) / len(margins),
        "mean": statistics.mean(margins),
        "median": statistics.median(margins),
        "stdev": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
        "min": margins[0],
        "max": margins[-1],
        "p10": percentile(margins, 0.10),
        "p25": percentile(margins, 0.25),
        "p75": percentile(margins, 0.75),
        "p90": percentile(margins, 0.90),
    }


def summarize(
    scored: list[dict],
    train_rows: list[dict],
    kept_train: list[dict],
    val_rows: list[dict],
    skipped: Counter,
    ctx: ReportContext,
) -> dict:
    """Build the machine-readable filtering report."""
    train_indices = {int(row["rm_index"]) for row in train_rows}
    val_indices = {int(row["rm_index"]) for row in val_rows}
    kept_indices = {int(row["rm_index"]) for row in kept_train}

    by_source: dict[str, dict] = {}
    for row in scored:
        source = str(row.get("source") or "unknown")
        entry = by_source.setdefault(
            source,
            {
                "pairs": 0,
                "train_pairs": 0,
                "val_pairs": 0,
                "kept_train": 0,
                "agreed": 0,
                "_margins": [],
            },
        )
        index = int(row["rm_index"])
        margin = float(row["rm_margin"])
        entry["pairs"] += 1
        entry["agreed"] += int(margin > 0)
        entry["_margins"].append(margin)
        if index in train_indices:
            entry["train_pairs"] += 1
        if index in val_indices:
            entry["val_pairs"] += 1
        if index in kept_indices:
            entry["kept_train"] += 1

    for entry in by_source.values():
        margins = sorted(entry.pop("_margins"))
        entry["agreement_rate"] = (
            entry["agreed"] / entry["pairs"] if entry["pairs"] else None
        )
        entry["margin_mean"] = statistics.mean(margins) if margins else None

    overall = _margin_summary(scored)
    train_summary = _margin_summary(train_rows)
    train_summary["kept"] = len(kept_train)
    train_summary["dropped"] = len(train_rows) - len(kept_train)
    val_summary = _margin_summary(val_rows)

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reward_model": ctx.reward_model,
        "reward_model_license": "Apache-2.0",
        "dtype": ctx.dtype,
        "device": ctx.device,
        "score_semantics": (
            "Scalar Bradley-Terry reward. Margins are relative, not calibrated "
            "probabilities."
        ),
        "system_prompt_policy": (
            "Dizel system prompts are stripped before scoring, following the "
            "Skywork card's 'do not include system prompts' rule."
        ),
        "dpo_note": (
            "DPO training uses chosen/rejected text only. The rm_* fields are "
            "metadata and are ignored by training/dpo_v16.py."
        ),
        "settings": {
            "max_length": ctx.max_length,
            "batch_size": ctx.batch_size,
            "min_margin": ctx.min_margin,
            "limit": ctx.limit,
            "val_fraction": ctx.val_fraction,
            "seed": ctx.seed,
        },
        "paths": {
            "input": ctx.input_path,
            "scored_output": ctx.scored_path,
            "filtered_train": ctx.output_path,
            "unfiltered_val": ctx.val_output_path,
        },
        "hashes": {
            "input_sha256": ctx.input_sha256,
            "scored_sha256": ctx.scored_sha256,
            "filtered_train_sha256": ctx.output_sha256,
            "unfiltered_val_sha256": ctx.val_sha256,
        },
        "counts": {
            "input_records": ctx.input_records,
            "scored_pairs": len(scored),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "kept_train_pairs": len(kept_train),
            "dropped_train_pairs": len(train_rows) - len(kept_train),
            "skipped_records": sum(skipped.values()),
            "unprocessed_records": max(
                0,
                ctx.input_records - len(scored) - sum(skipped.values()),
            ),
        },
        "agreement_rate": overall["agreement_rate"],
        "overall": overall,
        "train": train_summary,
        "val": val_summary,
        "by_source": {
            source: by_source[source] for source in sorted(by_source)
        },
        "skipped": dict(sorted(skipped.items())),
        "elapsed_seconds": round(ctx.elapsed_seconds, 2),
    }


# ---------------------------------------------------------------------------
# File I/O and resume
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: str, payload: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def write_jsonl(path: str, rows: list[dict]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_scored_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def load_scored_rows(path: str, expected_model: str) -> list[dict]:
    """Load a partial/final score file and repair an incomplete final line."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Score file not found: {path}")
    with open(path, "rb") as handle:
        data = handle.read()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        data = data[:cut]
        with open(path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        warning(f"Truncated an incomplete final line in {path}")

    rows = []
    seen = set()
    # Split on "\n" only.  ``str.splitlines()`` also breaks on U+2028 and
    # U+2029, which ``json.dumps`` leaves unescaped inside response text, so a
    # single JSON object could be cut into invalid fragments.
    for line_no, raw in enumerate(data.decode("utf-8").split("\n"), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        index = row.get("rm_index")
        if not isinstance(index, int):
            raise ValueError(f"{path}:{line_no}: missing integer rm_index")
        if index in seen:
            raise ValueError(f"{path}:{line_no}: duplicate rm_index {index}")
        if row.get("rm_model") != expected_model:
            raise ValueError(
                f"{path}:{line_no}: rm_model is {row.get('rm_model')!r}, "
                f"expected {expected_model!r}"
            )
        for key in ("rm_chosen", "rm_rejected", "rm_margin"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{path}:{line_no}: {key} is not finite")
        seen.add(index)
        rows.append(row)
    return rows


def check_meta(path: str, expected: dict) -> dict:
    """Reject a resume whose input, model, length, dtype, or limit changed."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Resume metadata not found: {path}. Start a new run without --resume."
        )
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    for key in ("input_sha256", "reward_model", "max_length", "dtype"):
        if meta.get(key) != expected.get(key):
            raise ValueError(
                f"Cannot resume: {key} changed "
                f"({meta.get(key)!r} -> {expected.get(key)!r}). "
                "Use a new --scored_output path."
            )
    old_limit = meta.get("limit")
    new_limit = expected.get("limit")
    if old_limit is None and new_limit is not None:
        raise ValueError(
            "Cannot resume a full run with --limit. Use a new --scored_output path."
        )
    if old_limit is not None and new_limit is not None and new_limit < old_limit:
        raise ValueError(
            f"Cannot resume: --limit decreased ({old_limit} -> {new_limit}). "
            "Use a new --scored_output path."
        )
    return meta


# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping[name]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        warning(
            "The reward model is running in reduced precision on CPU; "
            "float32 is safer and usually faster."
        )
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        warning("This GPU does not advertise bf16 support; use --dtype float16 on a T4.")
    return dtype


def load_tokenizer(model_name: str, local_files_only: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The Skywork reward model requires 'transformers>=4.51.0'. "
            "Install it with: pip install 'transformers>=4.51.0'"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load the reward tokenizer {model_name!r}. "
            "Install transformers>=4.51.0, verify HF access/HF_TOKEN, or omit "
            "--local_files_only when the model is not cached."
        ) from exc
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"Reward tokenizer {model_name!r} has no chat template; "
            "expected a Qwen3 tokenizer."
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Reward tokenizer has neither a pad nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_reward_model(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    pad_id: int,
    local_files_only: bool,
):
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "The Skywork reward model requires 'transformers>=4.51.0'. "
            "Install it with: pip install 'transformers>=4.51.0'"
        ) from exc
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not load the reward model {model_name!r}. Install "
            "transformers>=4.51.0, verify HF access/HF_TOKEN, or omit "
            "--local_files_only when the model is not cached."
        ) from exc
    model.config.pad_token_id = int(pad_id)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Score Dizel 1.6 DPO pairs with Skywork-Reward-V2 and write an "
            "RM-filtered training split plus an unfiltered validation split."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--scored_output", default=DEFAULT_SCORED_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_FILTERED_OUTPUT)
    parser.add_argument("--val_output", default=DEFAULT_VAL_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--reward_model", default=DEFAULT_REWARD_MODEL)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help=f"Token limit per conversation (Skywork recommends <= {SKYWORK_MAX_LENGTH}).",
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.0,
        help="Keep a pair when rm_chosen - rm_rejected >= this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score at most N prepared pairs (0 = all); useful for a smoke test.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Held-out fraction; match the DPO config when not using --val_data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partial --scored_output after verifying its .meta.json.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.max_length < 1:
        raise ValueError("--max_length must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if not math.isfinite(args.min_margin):
        raise ValueError("--min_margin must be finite")
    if not 0.0 <= args.val_fraction <= 0.5:
        raise ValueError("--val_fraction must be between 0 and 0.5")
    if args.max_length > SKYWORK_MAX_LENGTH:
        warning(
            f"Skywork-Reward-V2 was trained on <= {SKYWORK_MAX_LENGTH:,} tokens; "
            f"--max_length {args.max_length:,} is outside that contract."
        )

    run_started = time.time()
    load_hf_token_from_dotenv()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    dtype_name = str(dtype).replace("torch.", "")

    rule("Dizel 1.6 - Skywork reward-model filter (DPO)")
    section("Input")
    info(f"Reward model : {args.reward_model}")
    info(f"Device       : {device} ({dtype_name})")
    records = load_records(args.input)
    input_sha = sha256_file(args.input)
    info(f"Input        : {args.input} ({len(records):,} records)")

    tokenizer = load_tokenizer(args.reward_model, args.local_files_only)
    pad_id = int(tokenizer.pad_token_id)

    section("Prepare")
    limit = args.limit or None
    pairs, skipped = prepare_pairs(records, tokenizer, args.max_length, limit)
    if not pairs:
        raise ValueError(
            "No scorable DPO pairs after formatting. Check prompt_messages, "
            "chosen/rejected fields, and --max_length."
        )
    info(f"Prepared     : {len(pairs):,} pairs")
    if skipped:
        warning(f"Skipped      : {dict(sorted(skipped.items()))}")

    limit_value = args.limit or None
    meta = {
        "input_sha256": input_sha,
        "reward_model": args.reward_model,
        "max_length": args.max_length,
        "dtype": dtype_name,
        "limit": limit_value,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.resume:
        if not os.path.exists(args.scored_output):
            raise FileNotFoundError(
                f"Resume score file not found: {args.scored_output}. "
                "Omit --resume to start a new run."
            )
        check_meta(args.scored_output + ".meta.json", meta)
        existing = load_scored_rows(args.scored_output, args.reward_model)
        done = {int(row["rm_index"]) for row in existing}
        prepared_indices = {pair.index for pair in pairs}
        extra = done - prepared_indices
        if extra:
            raise ValueError(
                f"{len(extra):,} scored pairs are outside the current "
                "--limit/--max_length set. Use a new --scored_output path."
            )
        ok(f"Resumed      : {len(done):,}/{len(pairs):,} pairs already scored")
    else:
        if os.path.exists(args.scored_output):
            raise FileExistsError(
                f"{args.scored_output} already exists. Pass --resume to continue "
                "it or choose a new --scored_output."
            )
        os.makedirs(os.path.dirname(args.scored_output) or ".", exist_ok=True)
        write_json(args.scored_output + ".meta.json", meta)
        with open(args.scored_output, "w", encoding="utf-8"):
            pass
        done = set()
        ok(f"Created      : {args.scored_output}")

    remaining = [pair for pair in pairs if pair.index not in done]
    added = 0
    if remaining:
        section("Score")
        model = load_reward_model(
            args.reward_model,
            dtype,
            device,
            pad_id,
            args.local_files_only,
        )
        info(f"Loaded       : {type(model).__name__}")
        chunk_size = max(args.batch_size * 16, 32)
        started = time.time()
        try:
            for start in range(0, len(remaining), chunk_size):
                chunk = remaining[start:start + chunk_size]
                scores = score_pairs(
                    chunk,
                    model,
                    pad_id,
                    args.batch_size,
                    device,
                )
                rows = make_scored_rows(chunk, scores, args.reward_model)
                append_scored_rows(args.scored_output, rows)
                done.update(int(row["rm_index"]) for row in rows)
                added += len(rows)
                elapsed = max(time.time() - started, 1e-9)
                info(
                    f"Scored {len(done):,}/{len(pairs):,} pairs "
                    f"({added / elapsed:.1f} pairs/s this run)"
                )
        except KeyboardInterrupt:
            warning(
                "Interrupted. Partial scores are safe; rerun with --resume "
                f"--scored_output {args.scored_output}"
            )
            raise
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        ok("All pairs already scored; skipped reward-model load")

    scored_rows = load_scored_rows(args.scored_output, args.reward_model)
    scored_rows.sort(key=lambda row: int(row["rm_index"]))
    scored_indices = {int(row["rm_index"]) for row in scored_rows}
    prepared_indices = {pair.index for pair in pairs}
    missing = prepared_indices - scored_indices
    if missing:
        raise RuntimeError(
            f"{len(missing):,} prepared pairs are missing from "
            f"{args.scored_output}. Re-run scoring or use a new output path."
        )
    extra = scored_indices - prepared_indices
    if extra:
        raise RuntimeError(
            f"{len(extra):,} scored pairs are outside the prepared set. "
            "Use a new --scored_output path."
        )

    train_indices, val_indices = split_indices(
        len(records), args.val_fraction, args.seed
    )
    scored_by_index = {
        int(row["rm_index"]): row for row in scored_rows
    }
    train_rows = sorted(
        (scored_by_index[index] for index in train_indices if index in scored_by_index),
        key=lambda row: int(row["rm_index"]),
    )
    val_rows = sorted(
        (scored_by_index[index] for index in val_indices if index in scored_by_index),
        key=lambda row: int(row["rm_index"]),
    )
    kept_train, _ = filter_scored(train_rows, args.min_margin)

    section("Write")
    write_jsonl(args.scored_output, scored_rows)
    write_jsonl(args.output, kept_train)
    write_jsonl(args.val_output, val_rows)

    ctx = ReportContext(
        reward_model=args.reward_model,
        dtype=dtype_name,
        device=str(device),
        input_path=args.input,
        scored_path=args.scored_output,
        output_path=args.output,
        val_output_path=args.val_output,
        input_records=len(records),
        input_sha256=input_sha,
        scored_sha256=sha256_file(args.scored_output),
        output_sha256=sha256_file(args.output),
        val_sha256=sha256_file(args.val_output),
        min_margin=args.min_margin,
        max_length=args.max_length,
        batch_size=args.batch_size,
        limit=limit_value,
        val_fraction=args.val_fraction,
        seed=args.seed,
        elapsed_seconds=time.time() - run_started,
    )
    report = summarize(
        scored_rows,
        train_rows,
        kept_train,
        val_rows,
        skipped,
        ctx,
    )
    write_json(args.report, report)

    section("Summary")
    overall = report["overall"]
    agreement = overall["agreement_rate"]
    config_table(
        {
            "reward model": args.reward_model,
            "scored pairs": f"{len(scored_rows):,}",
            "train / val": f"{len(train_rows):,} / {len(val_rows):,}",
            "RM agreement": f"{agreement:.1%}" if agreement is not None else "n/a",
            "margin median": (
                f"{overall['median']:+.4f}"
                if overall["median"] is not None
                else "n/a"
            ),
            "kept train": (
                f"{len(kept_train):,} "
                f"(dropped {len(train_rows) - len(kept_train):,})"
            ),
            "min margin": f"{args.min_margin:+.4f}",
        }
    )
    ok(f"Scored  : {args.scored_output}")
    ok(f"Train   : {args.output} ({len(kept_train):,} pairs)")
    ok(f"Val     : {args.val_output} ({len(val_rows):,} pairs)")
    ok(f"Report  : {args.report}")

    if not val_rows:
        warning(
            "No validation rows were scored; omit --limit or increase it before "
            "passing --val_data to DPO."
        )
    if not kept_train:
        error("No training pairs passed the reward-margin filter.")
        warning(
            "Lower --min_margin or inspect the report. The unfiltered val file "
            "is still valid for evaluation."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
