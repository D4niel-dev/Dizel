"""
training/posttrain_data_v16.py — Shared data helpers for Dizel 1.6 post-training.

The v1.6 model uses the Qwen2.5 tokenizer and Qwen ChatML special tokens.
This module keeps the SFT, distillation and DPO paths on one formatting
contract:

    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    ...
    <|im_end|>
    <|im_start|>assistant
    ...
    <|im_end|>

Only assistant content and the assistant's closing ``<|im_end|>`` are loss
tokens.  Prompt tokens are masked.  Keeping this logic in one place prevents
the three post-training entry points from drifting apart.

The module deliberately has no training-loop code.  It is safe to import from
the CPU-only data-preparation pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Reuse the already-tested v1.5 content filters.  These functions do not
# depend on the old tokenizer.
from pipelines.clean_v15_data import (
    check_circular,
    check_low_information,
    check_quality,
    fix_newlines,
    normalize_unicode,
)
from pipelines.identity_guard import check_identity, rewrite_identity

__all__ = [
    "DIZEL_SYSTEM_PROMPT",
    "CONTROL_TOKEN_RE",
    "normalize_messages",
    "normalize_prompt_messages",
    "format_chat_tokens",
    "assistant_token_count",
    "build_sft_example",
    "build_dpo_example",
    "looks_like_dpo_record",
    "ChatSFTDataset",
    "DPODataset",
    "sft_collate_fn",
    "distill_collate_fn",
    "dpo_collate_fn",
    "compute_log_probs",
]

DIZEL_SYSTEM_PROMPT = (
    "You are Dizel, a structured analytical AI model. Prioritize clarity, "
    "precision, and logical organization. Use structured formatting when "
    "appropriate. Avoid unnecessary verbosity. If uncertain, explicitly "
    "state limitations."
)

# Any embedded role marker can forge a new turn.  Reject such records in the
# data pipeline instead of silently teaching the model a prompt-injection path.
CONTROL_TOKEN_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|system|user|assistant|end)\|>",
    re.IGNORECASE,
)

_FOREIGN_SYSTEM_RE = re.compile(
    r"\b(?:qwen|chatgpt|openai|anthropic|claude|gemini|llama|deepseek|"
    r"alibaba|created\s+by\s+(?:openai|anthropic|alibaba|google|meta))\b",
    re.IGNORECASE,
)

_ROLE_ALIASES = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}

_END_MARKER = "<|im_end|>"


def _text(value) -> str:
    """Coerce a dataset value into clean text without losing newlines."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        # Some preference datasets store content as a list of parts.
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        value = "".join(parts)
    if not isinstance(value, str):
        value = str(value)
    value = normalize_unicode(value).replace("\x00", "")
    value = fix_newlines(value)
    return value.strip()


def _canonical_role(value) -> Optional[str]:
    if value is None:
        return None
    return _ROLE_ALIASES.get(str(value).strip().lower())


def _message_content(item: dict) -> str:
    for key in ("content", "value", "text", "response"):
        if key in item:
            return _text(item[key])
    return ""


def _extract_messages_from_record(record: dict) -> list[dict]:
    """Accept the common instruction-dataset shapes used by the v1.6 mix."""
    raw = None
    if isinstance(record.get("messages"), list):
        raw = record["messages"]
    elif isinstance(record.get("conversations"), list):
        raw = record["conversations"]
    elif isinstance(record.get("prompt"), list):
        raw = record["prompt"]
    else:
        prompt = record.get("prompt", record.get("instruction", ""))
        output = (
            record.get("output")
            or record.get("response")
            or record.get("chosen")
            or record.get("answer")
            or ""
        )
        inp = record.get("input", "")
        prompt = _text(prompt)
        if inp and _text(inp) and _text(inp) not in prompt:
            prompt = prompt + "\n\n" + _text(inp)
        if prompt and output:
            raw = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output},
            ]

    if not raw:
        return []

    messages = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = _canonical_role(item.get("role", item.get("from")))
        if role is None:
            continue
        content = _message_content(item)
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _sanitize_system(text: str) -> str:
    text = _text(text)
    if not text:
        return DIZEL_SYSTEM_PROMPT
    if _FOREIGN_SYSTEM_RE.search(text):
        return DIZEL_SYSTEM_PROMPT
    return text


def normalize_messages(record: dict) -> list[dict]:
    """Return a clean ChatML message list with a Dizel system prompt.

    The returned list is suitable for both training and token counting.  A
    record is rejected if it has no user/assistant pair, if it ends on a user
    turn (no target), or if a message contains an embedded role marker.
    """
    messages = _extract_messages_from_record(record)
    if not messages:
        return []

    systems = [m["content"] for m in messages if m["role"] == "system"]
    messages = [m for m in messages if m["role"] != "system"]
    if not messages:
        return []

    # Join multiple system messages; the first one usually carries the
    # dataset's behaviour instructions.  Foreign identities are replaced.
    system = _sanitize_system("\n\n".join(systems))

    # Assistant content is where the model learns identity.  Rewrite clear
    # foreign self-identifications; drop records that still look unsafe or
    # incoherent after the rewrite.
    cleaned = []
    for message in messages:
        content = _text(message["content"])
        if not content:
            return []
        if CONTROL_TOKEN_RE.search(content):
            return []
        if message["role"] == "assistant":
            content = rewrite_identity(content).strip()
            if not content:
                return []
            if not check_identity(content, is_code=False):
                return []
        cleaned.append({"role": message["role"], "content": content})

    cleaned.insert(0, {"role": "system", "content": system})

    # A trailing user turn has no supervision target.  Drop it rather than
    # training the model to predict a role marker with no answer.
    if cleaned[-1]["role"] != "assistant":
        return []
    if not any(m["role"] == "user" for m in cleaned):
        return []
    if not any(m["role"] == "assistant" for m in cleaned):
        return []
    return cleaned


def normalize_prompt_messages(record: dict) -> list[dict]:
    """Normalize a DPO prompt, which intentionally ends on a user turn.

    Assistant history is allowed, but the trailing assistant turn (if any) is
    removed so the chosen/rejected completion starts at a generation point.
    """
    messages = _extract_messages_from_record(record)
    if not messages:
        return []
    systems = [m["content"] for m in messages if m["role"] == "system"]
    messages = [m for m in messages if m["role"] != "system"]
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages or messages[-1]["role"] != "user":
        return []

    cleaned = []
    for message in messages:
        content = _text(message["content"])
        if not content or CONTROL_TOKEN_RE.search(content):
            return []
        if message["role"] == "assistant":
            content = rewrite_identity(content).strip()
            if not content or not check_identity(content, is_code=False):
                return []
        cleaned.append({"role": message["role"], "content": content})
    system = _sanitize_system("\n\n".join(systems))
    cleaned.insert(0, {"role": "system", "content": system})
    return cleaned if any(m["role"] == "user" for m in cleaned) else []


def messages_hash(messages: Iterable[dict]) -> str:
    blob = json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def format_chat_tokens(
    messages: list[dict],
    tokenizer,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Tokenize ChatML and return ``(token_ids, assistant_loss_mask)``.

    The mask marks the assistant content and its closing ``<|im_end|>``.  The
    assistant role prefix and the trailing newline are not loss tokens.
    """
    ids: list[int] = []
    mask: list[int] = []

    for message in messages:
        role = _canonical_role(message.get("role"))
        if role is None:
            raise ValueError(f"Unsupported chat role: {message.get('role')!r}")
        content = _text(message.get("content", ""))
        if not content:
            raise ValueError("Empty chat message")
        if CONTROL_TOKEN_RE.search(content):
            raise ValueError("Chat content contains a control token")

        prefix_ids = tokenizer.encode(f"<|im_start|>{role}\n")
        content_ids = tokenizer.encode(content)
        end_ids = tokenizer.encode(_END_MARKER)

        is_assistant = role == "assistant"
        ids.extend(prefix_ids)
        mask.extend([0] * len(prefix_ids))
        ids.extend(content_ids)
        mask.extend([1 if is_assistant else 0] * len(content_ids))
        ids.extend(end_ids)
        mask.extend([1 if is_assistant else 0] * len(end_ids))
        # The newline after <|im_end|> is formatting, not a learning target.
        newline_ids = tokenizer.encode("\n")
        ids.extend(newline_ids)
        mask.extend([0] * len(newline_ids))

    if add_generation_prompt:
        prefix_ids = tokenizer.encode("<|im_start|>assistant\n")
        ids.extend(prefix_ids)
        mask.extend([0] * len(prefix_ids))

    return ids, mask


def assistant_token_count(messages: list[dict], tokenizer) -> int:
    """Number of assistant content + closing-marker tokens in a conversation."""
    total = 0
    for message in messages:
        if _canonical_role(message.get("role")) != "assistant":
            continue
        content = _text(message.get("content", ""))
        if content:
            total += len(tokenizer.encode(content)) + len(tokenizer.encode(_END_MARKER))
    return total


def build_sft_example(
    messages: list[dict],
    tokenizer,
    max_length: int,
) -> Optional[dict]:
    """Build an SFT/distillation example, or return None if it is unusable."""
    try:
        ids, mask = format_chat_tokens(messages, tokenizer)
    except ValueError:
        return None
    if len(ids) < 4 or len(ids) > max_length:
        return None
    input_ids = ids[:-1]
    labels = ids[1:]
    loss_mask = mask[1:]
    if not any(loss_mask):
        return None
    return {
        "input_ids": np.asarray(input_ids, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int32),
        "loss_mask": np.asarray(loss_mask, dtype=np.bool_),
        "assistant_tokens": int(sum(mask)),
        "total_tokens": int(len(ids)),
    }


def build_dpo_example(
    prompt_messages: list[dict],
    chosen: str,
    rejected: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
) -> Optional[dict]:
    """Build a DPO pair using the same ChatML prompt contract as SFT."""
    chosen = rewrite_identity(_text(chosen)).strip()
    rejected = _text(rejected).strip()
    if not chosen or not rejected or chosen == rejected:
        return None
    if CONTROL_TOKEN_RE.search(chosen) or CONTROL_TOKEN_RE.search(rejected):
        return None

    try:
        prompt_ids, _ = format_chat_tokens(
            prompt_messages, tokenizer, add_generation_prompt=True
        )
    except ValueError:
        return None
    if not prompt_ids or len(prompt_ids) > max_prompt_length:
        return None

    end_ids = tokenizer.encode(_END_MARKER)
    chosen_ids = tokenizer.encode(chosen) + end_ids
    rejected_ids = tokenizer.encode(rejected) + end_ids
    if not chosen_ids or not rejected_ids:
        return None
    if len(prompt_ids) + len(chosen_ids) > max_length:
        return None
    if len(prompt_ids) + len(rejected_ids) > max_length:
        return None

    def pack(answer_ids):
        all_ids = prompt_ids + answer_ids
        input_ids = all_ids[:-1]
        labels = all_ids[1:]
        mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
        return (
            np.asarray(input_ids, dtype=np.int32),
            np.asarray(labels, dtype=np.int32),
            np.asarray(mask[1:], dtype=np.bool_),
        )

    c_input, c_labels, c_mask = pack(chosen_ids)
    r_input, r_labels, r_mask = pack(rejected_ids)
    return {
        "chosen_input_ids": c_input,
        "chosen_labels": c_labels,
        "chosen_mask": c_mask,
        "rejected_input_ids": r_input,
        "rejected_labels": r_labels,
        "rejected_mask": r_mask,
        "prompt_tokens": int(len(prompt_ids)),
        "chosen_tokens": int(len(chosen_ids)),
        "rejected_tokens": int(len(rejected_ids)),
    }


def looks_like_dpo_record(record: dict) -> bool:
    return bool(record.get("chosen") and record.get("rejected"))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ChatSFTDataset(Dataset):
    """A compact in-memory map-style dataset for SFT or distillation."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        prefix: str = "",
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        skipped = 0
        for record in records:
            messages = record.get("messages")
            if not isinstance(messages, list):
                skipped += 1
                continue
            # Normalize again so hand-edited JSONL cannot bypass the filter.
            messages = normalize_messages({"messages": messages})
            if not messages:
                skipped += 1
                continue
            example = build_sft_example(messages, tokenizer, max_length)
            if example is None:
                skipped += 1
                continue
            example["family"] = record.get("family", "unknown")
            example["source"] = record.get("source", "")
            self.samples.append(example)
        if skipped:
            print(f"  [posttrain] skipped {skipped:,} unusable {prefix or 'SFT'} records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            torch.from_numpy(sample["input_ids"].astype(np.int64)),
            torch.from_numpy(sample["labels"].astype(np.int64)),
            torch.from_numpy(sample["loss_mask"].astype(np.float32)),
        )


class DPODataset(Dataset):
    """Preference pairs whose prompt uses the Qwen ChatML template."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        max_prompt_length: int,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        skipped = 0
        for record in records:
            prompt_messages = record.get("prompt_messages")
            if not isinstance(prompt_messages, list):
                skipped += 1
                continue
            prompt_messages = normalize_prompt_messages(
                {"messages": prompt_messages}
            )
            if not prompt_messages:
                skipped += 1
                continue
            example = build_dpo_example(
                prompt_messages,
                record.get("chosen", ""),
                record.get("rejected", ""),
                tokenizer,
                max_length,
                max_prompt_length,
            )
            if example is None:
                skipped += 1
                continue
            self.samples.append(example)
        if skipped:
            print(f"  [posttrain] skipped {skipped:,} unusable DPO pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return tuple(
            torch.from_numpy(sample[key].astype(dtype))
            for key, dtype in (
                ("chosen_input_ids", np.int64),
                ("chosen_labels", np.int64),
                ("chosen_mask", np.float32),
                ("rejected_input_ids", np.int64),
                ("rejected_labels", np.int64),
                ("rejected_mask", np.float32),
            )
        )


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
def _pad_sequences(sequences, pad_value, dtype):
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    masks = []
    for seq in sequences:
        n = seq.size(0)
        padded.append(
            torch.cat(
                [seq, torch.full((max_len - n,), pad_value, dtype=dtype)]
            )
        )
        masks.append(
            torch.cat(
                [torch.ones(n, dtype=torch.long), torch.zeros(max_len - n, dtype=torch.long)]
            )
        )
    return torch.stack(padded), torch.stack(masks)


def sft_collate_fn(batch, pad_id: int = 0):
    input_ids, labels, masks = zip(*batch)
    x, _ = _pad_sequences(input_ids, pad_id, torch.long)
    y, _ = _pad_sequences(labels, -1, torch.long)
    m, _ = _pad_sequences(masks, 0.0, torch.float)
    return x, y, m


def distill_collate_fn(batch, pad_id: int = 0):
    input_ids, labels, masks = zip(*batch)
    x, attention = _pad_sequences(input_ids, pad_id, torch.long)
    y, _ = _pad_sequences(labels, -1, torch.long)
    m, _ = _pad_sequences(masks, 0.0, torch.float)
    return x, y, m, attention


def dpo_collate_fn(batch, pad_id: int = 0):
    c_ids, c_labels, c_mask, r_ids, r_labels, r_mask = zip(*batch)
    c_x, _ = _pad_sequences(c_ids, pad_id, torch.long)
    c_y, _ = _pad_sequences(c_labels, -1, torch.long)
    c_m, _ = _pad_sequences(c_mask, 0.0, torch.float)
    r_x, _ = _pad_sequences(r_ids, pad_id, torch.long)
    r_y, _ = _pad_sequences(r_labels, -1, torch.long)
    r_m, _ = _pad_sequences(r_mask, 0.0, torch.float)
    return c_x, c_y, c_m, r_x, r_y, r_m


def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample average completion log-probability for DPO."""
    logits, _ = model(input_ids)
    # Avoid materialising a (B,T,V) float32 log-softmax.  Gather first, then
    # normalise per row in chunks.
    logits = logits.float()
    safe_labels = labels.clamp(min=0)
    gathered = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    log_z = torch.logsumexp(logits, dim=-1)
    per_token = (gathered - log_z) * loss_mask
    token_count = loss_mask.sum(dim=-1).clamp(min=1)
    return per_token.sum(dim=-1) / token_count
