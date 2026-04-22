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
import numpy as np
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
        token_ids,                 # numpy array or list of integer token ids
        context_length: int,
        stride: int = None,        # step between windows; default = context_length // 2
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if isinstance(token_ids, np.ndarray):
            self.data = torch.from_numpy(token_ids.astype(np.int64))
        else:
            self.data = torch.tensor(token_ids, dtype=torch.long)
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
    shard_size_mb: int = 100,
    cache_dir: str = ".cache/tokenized",
    resume_enabled: bool = True,
):
    """
    Tokenise the corpus, split into train/val, return DataLoaders.

    v1.2 pipeline upgrades:
      - Large shards (~100MB) instead of 5MB micro-chunks
      - Per-shard .pt caching for instant resume
      - Batch tokenization (2MB text batches to SentencePiece)

    Returns
    -------
    train_loader, val_loader, train_dataset
    (train_dataset is returned so you can call reshuffle() each epoch)
    """
    from training.shard_utils import shard_corpus, read_shard
    from training.cache_utils import (
        get_cache_dir, load_manifest, save_manifest,
        is_shard_cached, save_shard_tokens, load_shard_tokens,
        load_all_cached_tokens,
    )
    from training.tokenizer_utils import tokenize_shard_batched

    file_size = os.path.getsize(data_path)
    file_mb = file_size / (1024 * 1024)
    corpus_name = os.path.splitext(os.path.basename(data_path))[0]
    c_dir = get_cache_dir(cache_dir, corpus_name)

    # ── Compute shard boundaries ────────────────────────────────────────
    boundaries = shard_corpus(data_path, shard_size_mb=shard_size_mb)
    num_shards = len(boundaries)
    print(f"[dataset] Corpus: {file_mb:.1f} MB → {num_shards} shards ({shard_size_mb} MB each)")

    # ── Try loading entire cache first ──────────────────────────────────
    if resume_enabled:
        cached = load_all_cached_tokens(c_dir, num_shards)
        if cached is not None:
            print(f"[dataset] ✓ Loaded {len(cached):,} tokens from cache ({c_dir})")
            all_ids = cached
        else:
            all_ids = None
    else:
        all_ids = None

    # ── Tokenize shards (with per-shard resume) ─────────────────────────
    if all_ids is None:
        manifest = load_manifest(c_dir)
        shard_arrays = []
        for idx, (start, end) in enumerate(boundaries):
            shard_mb = (end - start) / (1024 * 1024)

            if resume_enabled and is_shard_cached(c_dir, idx, manifest):
                ids = load_shard_tokens(c_dir, idx)
                print(f"  shard {idx+1}/{num_shards} ({shard_mb:.0f} MB) → cached ({len(ids):,} tokens)")
            else:
                text = read_shard(data_path, start, end)
                ids = tokenize_shard_batched(
                    text,
                    encode_fn=tokenizer.encode,
                    add_bos_first=(idx == 0),
                )
                ids = np.array(ids, dtype=np.int32)
                # Persist for future runs
                if resume_enabled:
                    save_shard_tokens(c_dir, idx, ids)
                    manifest.setdefault("completed_shards", []).append(idx)
                    manifest["total_tokens"] = manifest.get("total_tokens", 0) + len(ids)
                    save_manifest(c_dir, manifest)
                print(f"  shard {idx+1}/{num_shards} ({shard_mb:.0f} MB) → tokenized ({len(ids):,} tokens)")

            shard_arrays.append(ids if isinstance(ids, np.ndarray) else np.array(ids, dtype=np.int32))
            del ids  # free shard memory immediately

        all_ids = np.concatenate(shard_arrays)
        del shard_arrays  # free the list of arrays

    print(f"[dataset] Total tokens: {len(all_ids):,}")

    # ── Train / val split ───────────────────────────────────────────────
    split_at  = int(len(all_ids) * train_split)
    train_ids = all_ids[:split_at]
    val_ids   = all_ids[split_at:]

    print(f"[dataset] Train tokens : {len(train_ids):,}")
    print(f"[dataset] Val   tokens : {len(val_ids):,}")

    if len(val_ids) < context_length + 1:
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
        stride=context_length,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=False,
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

        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  [warn] Skipping bad JSON on line {line_num}: {e}")
                    continue
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
        if skipped:
            print(f"  [warn] Skipped {skipped} bad JSON lines total")

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


# ---------------------------------------------------------------------------
# Mixed SFT Dataset (v1.2 — multiple JSONL files with weighted sampling)
# ---------------------------------------------------------------------------
class MixedSFTDataset(Dataset):
    """
    Loads multiple JSONL files and assigns per-sample weights for
    weighted random sampling during training.

    Each dataset source gets its weight from the DatasetMixConfig.
    Samples from high-weight sources appear more frequently per epoch.
    """

    def __init__(
        self,
        mix_config,
        tokenizer: Tokenizer,
        context_length: int,
        mask_prompt_loss: bool = True,
    ) -> None:
        self.tokenizer        = tokenizer
        self.ctx              = context_length
        self.mask_prompt_loss = mask_prompt_loss
        self.samples          = []
        self.sample_weights   = []

        for entry in mix_config.datasets:
            if not os.path.exists(entry.path):
                print(f"  [mix] ⚠ Skipping missing dataset: {entry.name} ({entry.path})")
                continue

            ds_samples = []
            skipped = 0
            with open(entry.path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    msgs = obj.get("messages", [])
                    if not msgs:
                        continue
                    x, y, mask = format_conversation(msgs, tokenizer)
                    x    = x[:context_length]
                    y    = y[:context_length]
                    mask = mask[:context_length]
                    if len(x) < 2:
                        continue
                    ds_samples.append((x, y, mask))

                    if entry.max_samples > 0 and len(ds_samples) >= entry.max_samples:
                        break

            if skipped:
                print(f"  [mix] {entry.name}: skipped {skipped} bad JSON lines")

            print(f"  [mix] {entry.name}: {len(ds_samples)} samples (weight={entry.weight})")
            self.samples.extend(ds_samples)
            self.sample_weights.extend([entry.weight] * len(ds_samples))

        # Normalize weights
        if self.sample_weights:
            total = sum(self.sample_weights)
            self.sample_weights = [w / total for w in self.sample_weights]

        print(f"[dataset] Mixed SFT total: {len(self.samples)} samples from {len(mix_config.datasets)} sources")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, mask = self.samples[idx]
        return (
            torch.tensor(x,    dtype=torch.long),
            torch.tensor(y,    dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
        )


def build_mixed_sft_loaders(
    mix_config,
    tokenizer: Tokenizer,
    context_length: int,
    batch_size: int,
    val_frac: float = 0.10,
    seed: int = 42,
    num_workers: int = 0,
):
    """Build train/val DataLoaders from a mixed dataset config with weighted sampling."""
    from torch.utils.data import WeightedRandomSampler

    full = MixedSFTDataset(mix_config, tokenizer, context_length)

    n_val   = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # Build weighted sampler for training set only
    train_weights = [full.sample_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_ds), replacement=True)

    collate = lambda b: sft_collate_fn(b, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
