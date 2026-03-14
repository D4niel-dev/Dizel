"""
training/dataset.py — Dataset utilities for pre-training and SFT.

Pre-training Dataset
--------------------
Tokenises the entire corpus into a flat array of ids, then yields
overlapping windows of length `context_length + 1`.  The +1 is because
the label for token i is token i+1 (next-token prediction).

SFT Dataset
-----------
Reads JSONL files where each line is:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Each conversation is formatted with special role tokens and tokenised.
Loss is only computed on assistant tokens (loss_mask = 0 elsewhere).

JSON Output Support
-------------------
If a user message contains the keyword [json], the assistant target is
expected to be valid JSON.  The SFT data can include such examples to
teach the model to produce structured output.
"""

import json
import math
import random
import re
import sys
import os

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG, ModelConfig, TokenizerConfig


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------
class Tokenizer:
    """Thin wrapper around SentencePiece for encode/decode."""

    def __init__(self, model_path: str = None) -> None:
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")

        path = model_path or CONFIG.tokenizer.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Tokenizer model not found at '{path}'. "
                "Run: python tokenizer/train_tokenizer.py"
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(path)

        self.pad_id = CONFIG.tokenizer.pad_id
        self.bos_id = CONFIG.tokenizer.bos_id
        self.eos_id = CONFIG.tokenizer.eos_id
        self.unk_id = CONFIG.tokenizer.unk_id
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Filter special tokens before decoding
        filtered = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.decode(filtered)

    def __len__(self) -> int:
        return self.vocab_size


# ---------------------------------------------------------------------------
# Pre-Training Dataset
# ---------------------------------------------------------------------------
class PretrainDataset(Dataset):
    """
    Flat sliding-window dataset for next-token language modelling.

    Given a tokenised corpus [t0, t1, t2, …, tN], each sample is:
        x = tokens[i : i + context_length]
        y = tokens[i+1 : i + context_length + 1]

    Stride < context_length creates overlap, giving the model more
    training examples from a small corpus (helps with overfitting).
    """

    def __init__(
        self,
        token_ids: list,           # full flat list of integer token ids
        context_length: int,
        stride: int = None,        # step between windows; default = context_length // 2
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.ctx     = context_length
        self.stride  = stride if stride is not None else context_length // 2
        self.shuffle = shuffle
        self.seed    = seed

        # Build index of window start positions
        self._make_index()

    def _make_index(self) -> None:
        """Rebuild the list of start indices (called each epoch for reshuffling)."""
        n = len(self.data)
        starts = list(range(0, n - self.ctx, self.stride))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(starts)
        self._starts = starts

    def reshuffle(self, new_seed: int = None) -> None:
        """Call at the end of each epoch to re-randomise window order."""
        if new_seed is not None:
            self.seed = new_seed
        self._make_index()

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        start = self._starts[idx]
        x = self.data[start     : start + self.ctx]
        y = self.data[start + 1 : start + self.ctx + 1]
        return x, y


def build_pretrain_loaders(
    data_path: str,
    tokenizer: Tokenizer,
    context_length: int,
    batch_size: int,
    train_split: float = 0.90,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Tokenise the corpus, split into train/val, return DataLoaders.

    Returns
    -------
    train_loader, val_loader, train_dataset
    (train_dataset is returned so you can call reshuffle() each epoch)
    """
    # ── Read and tokenise ───────────────────────────────────────────────
    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read()

    print(f"[dataset] Tokenising corpus ({len(raw):,} chars) ...")
    all_ids = tokenizer.encode(raw, add_bos=True)
    print(f"[dataset] Total tokens: {len(all_ids):,}")

    # ── Train / val split ───────────────────────────────────────────────
    split_at  = int(len(all_ids) * train_split)
    train_ids = all_ids[:split_at]
    val_ids   = all_ids[split_at:]

    print(f"[dataset] Train tokens : {len(train_ids):,}")
    print(f"[dataset] Val   tokens : {len(val_ids):,}")

    if len(val_ids) < context_length + 1:
        # Small corpus: use last 10% of train as val
        fallback_split = int(len(train_ids) * 0.90)
        val_ids        = train_ids[fallback_split:]
        train_ids      = train_ids[:fallback_split]
        print("[dataset] ⚠ Val set too small — using 10% of train as val")

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = PretrainDataset(
        train_ids, context_length,
        stride=context_length // 2,
        shuffle=True, seed=seed,
    )
    val_ds = PretrainDataset(
        val_ids, context_length,
        stride=context_length,       # no overlap for val (cleaner estimate)
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=False,               # PretrainDataset shuffles internally
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, train_ds


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------
ROLE_TOKENS = {
    "system":    "<|system|>",
    "user":      "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"


def format_conversation(messages: list, tokenizer: Tokenizer) -> tuple:
    """
    Convert a list of message dicts into a flat token sequence with a
    binary loss mask (1 = compute loss, 0 = ignore).

    Format per turn:
        <|system|>  {content} <|end|>
        <|user|>    {content} <|end|>
        <|assistant|> {content} <|end|> </s>

    Returns (input_ids, target_ids, loss_mask) as Python lists.
    """
    input_ids  = []
    target_ids = []
    loss_mask  = []

    for msg in messages:
        role    = msg["role"]
        content = msg["content"]

        role_prefix  = tokenizer.encode(ROLE_TOKENS[role])
        content_ids  = tokenizer.encode(content)
        end_ids      = tokenizer.encode(END_TOKEN)

        # For the last assistant turn, append EOS so the model learns to stop
        is_last_assistant = (
            role == "assistant" and msg is messages[-1]
        )
        if is_last_assistant:
            end_ids = end_ids + [tokenizer.eos_id]

        turn_ids = role_prefix + content_ids + end_ids
        is_assistant = (role == "assistant")

        # Input is everything; targets shifted by one
        # (handled by taking x=ids[:-1], y=ids[1:])
        input_ids  += turn_ids
        target_ids += turn_ids
        loss_mask  += [1 if is_assistant else 0] * len(turn_ids)

    # Shift: x = ids[:-1], y = ids[1:]
    return input_ids[:-1], target_ids[1:], loss_mask[1:]


class SFTDataset(Dataset):
    """
    Reads a JSONL file of chat conversations and returns tokenised
    (input, target, loss_mask) triples truncated to context_length.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        context_length: int,
        mask_prompt_loss: bool = True,
    ) -> None:
        self.tokenizer        = tokenizer
        self.ctx              = context_length
        self.mask_prompt_loss = mask_prompt_loss
        self.samples          = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if not msgs:
                    continue
                x, y, mask = format_conversation(msgs, tokenizer)
                # Truncate to context_length
                x    = x[:context_length]
                y    = y[:context_length]
                mask = mask[:context_length]
                if len(x) < 2:
                    continue
                self.samples.append((x, y, mask))

        print(f"[dataset] SFT samples loaded: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, mask = self.samples[idx]
        return (
            torch.tensor(x,    dtype=torch.long),
            torch.tensor(y,    dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
        )


def sft_collate_fn(batch, pad_id: int = 0):
    """Pad sequences in a batch to the same length."""
    xs, ys, masks = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    def pad(tensors, value):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.size(0),), value)])
            for t in tensors
        ])

    return pad(xs, pad_id), pad(ys, -1), pad(masks, 0)


def build_sft_loaders(
    jsonl_path: str,
    tokenizer: Tokenizer,
    context_length: int,
    batch_size: int,
    val_frac: float = 0.10,
    seed: int = 42,
    num_workers: int = 0,
):
    full = SFTDataset(jsonl_path, tokenizer, context_length)

    n_val   = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    collate = lambda b: sft_collate_fn(b, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
