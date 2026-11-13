"""
training/pretokenized.py -- Loader for laptop-packed Dizel 1.6 shards.

The v1.6 Phase A corpus is prepared once on a laptop by
``pipelines/prepare_v16_data.py``.  It is stored as flat ``uint32`` shards of
fixed ``context_length + 1`` token rows, so a Colab session never has to
tokenize raw text.  This module validates that artifact and exposes the same
small dataset surface the training loop already uses:

* ``__len__`` / ``__getitem__`` returning ``(x, y)``
* ``get_state`` / ``load_state`` for exact resume
* ``reshuffle`` at epoch boundaries

The manifest is the contract between the laptop and the trainer.  A missing
shard, a wrong sequence length, or a tokenizer fingerprint mismatch fails
loudly instead of silently training on the wrong bytes.
"""

from __future__ import annotations

import bisect
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FORMAT_VERSION = "dizel-pretokenized-v1"
SHARD_DTYPE = np.uint32
SHARD_ITEMSIZE = np.dtype(SHARD_DTYPE).itemsize


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PackedTokenDataset(Dataset):
    """A shuffled collection of fixed-length rows spread over ``.bin`` shards.

    Each row contains ``seq_len = context_length + 1`` tokens.  ``x`` is the
    first ``context_length`` tokens and ``y`` is the same window shifted by
    one, exactly like the raw-text pretraining dataset.
    """

    def __init__(
        self,
        shard_paths: List[str],
        seq_len: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if not shard_paths:
            raise ValueError("PackedTokenDataset requires at least one shard")

        self.shard_paths = [os.path.abspath(path) for path in shard_paths]
        self.seq_len = int(seq_len)
        if self.seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {self.seq_len}")

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._rows: List[int] = []
        self._offsets: List[int] = [0]
        self._maps: Dict[int, np.memmap] = {}
        self._starts: Optional[np.ndarray] = None
        self._current_pos = 0

        row_bytes = self.seq_len * SHARD_ITEMSIZE
        total = 0
        for path in self.shard_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Packed shard not found: {path!r}")
            size = os.path.getsize(path)
            if size % row_bytes != 0:
                raise ValueError(
                    f"Shard {path!r} is {size:,} bytes, not a multiple of "
                    f"{row_bytes:,} ({self.seq_len} tokens x "
                    f"{SHARD_ITEMSIZE} bytes)."
                )
            rows = size // row_bytes
            if rows == 0:
                raise ValueError(f"Packed shard is empty: {path!r}")
            self._rows.append(rows)
            total += rows
            self._offsets.append(total)

        self.total_rows = total
        self._make_index()

    # -- internals ------------------------------------------------------
    def _open_shard(self, shard_idx: int) -> np.memmap:
        cached = self._maps.get(shard_idx)
        if cached is None:
            cached = np.memmap(
                self.shard_paths[shard_idx],
                dtype=SHARD_DTYPE,
                mode="r",
                shape=(self._rows[shard_idx], self.seq_len),
            )
            self._maps[shard_idx] = cached
        return cached

    def _make_index(self) -> None:
        starts = np.arange(self.total_rows, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(starts)
        self._starts = starts
        self._current_pos = 0

    def __getstate__(self):
        # Workers are spawned on Windows and may be forked on Linux.  Keep
        # open memmaps out of the pickled state; each worker opens lazily.
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._maps = {}

    # -- Dataset surface -------------------------------------------------
    def __len__(self) -> int:
        return int(self._starts.size)

    def __getitem__(self, idx: int):
        self._current_pos = int(idx)
        global_row = int(self._starts[idx])
        shard_idx = bisect.bisect_right(self._offsets, global_row) - 1
        local_row = global_row - self._offsets[shard_idx]
        row = self._open_shard(shard_idx)[local_row]

        x = np.asarray(row[:-1], dtype=np.int64)
        y = np.asarray(row[1:], dtype=np.int64)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    @property
    def current_index(self) -> int:
        return self._current_pos

    def get_state(self) -> dict:
        return {
            "starts": np.array(self._starts, dtype=np.int64, copy=True),
            "seed": self.seed,
        }

    def load_state(self, state: dict) -> None:
        starts = state["starts"]
        if isinstance(starts, torch.Tensor):
            starts = starts.detach().cpu().numpy()
        starts = np.asarray(starts, dtype=np.int64)
        if starts.shape != (self.total_rows,):
            raise ValueError(
                f"Dataset state has {starts.size:,} rows, expected "
                f"{self.total_rows:,}; was the corpus rebuilt?"
            )
        self._starts = starts
        self.seed = int(state.get("seed", self.seed))

    def reshuffle(self, new_seed: Optional[int] = None) -> None:
        if new_seed is not None:
            self.seed = int(new_seed)
        self._make_index()


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------
def load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Pre-tokenized manifest not found: {path!r}. Run "
            "`python pipelines/prepare_v16_data.py --stage all` on the laptop "
            "and upload the resulting data directory."
        )
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported pre-tokenized format {manifest.get('format')!r}; "
            f"expected {FORMAT_VERSION!r}."
        )
    return manifest


def _validate_tokenizer(manifest: dict, tokenizer) -> None:
    info = manifest.get("tokenizer") or {}
    expected_vocab = int(info.get("vocab_size", 0))
    expected_fp = info.get("fingerprint")
    if tokenizer is None:
        return

    actual_vocab = len(tokenizer)
    if expected_vocab and actual_vocab != expected_vocab:
        raise ValueError(
            f"Tokenizer vocabulary mismatch: manifest has {expected_vocab:,} "
            f"tokens, runtime tokenizer has {actual_vocab:,}."
        )

    fingerprint = getattr(tokenizer, "fingerprint", None)
    if expected_fp and callable(fingerprint):
        actual_fp = fingerprint()
        if actual_fp != expected_fp:
            raise ValueError(
                "Tokenizer fingerprint mismatch: the pre-tokenized corpus was "
                f"built with {expected_fp[:16]}..., runtime tokenizer is "
                f"{actual_fp[:16]}.... Rebuild the corpus or use the matching "
                "tokenizer."
            )


def _shard_entries(manifest: dict, split: str, data_dir: str):
    entries = []
    for shard in manifest.get("shards", []):
        if shard.get("split") != split:
            continue
        rel = shard.get("file")
        if not rel:
            raise ValueError(f"Manifest shard missing 'file' in split {split!r}")
        path = os.path.normpath(os.path.join(data_dir, rel))
        entries.append((path, shard))
    return entries


def build_pretokenized_loaders(
    data_dir: str,
    context_length: int,
    batch_size: int,
    tokenizer=None,
    seed: int = 42,
    num_workers: int = 2,
    val_num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, PackedTokenDataset, dict]:
    """Validate a packed corpus and return train/val loaders plus metadata."""
    data_dir = os.path.abspath(data_dir)
    manifest = load_manifest(data_dir)

    seq_len = int(manifest.get("seq_len", 0))
    expected_seq_len = int(context_length) + 1
    if seq_len != expected_seq_len:
        raise ValueError(
            f"Corpus seq_len is {seq_len}, but context_length={context_length} "
            f"requires {expected_seq_len} tokens per row. Rebuild the corpus "
            "with the matching --context-length."
        )
    if manifest.get("dtype") != "uint32":
        raise ValueError(
            f"Unsupported shard dtype {manifest.get('dtype')!r}; expected uint32."
        )
    if manifest.get("byte_order") not in (None, "little"):
        raise ValueError(
            f"Unsupported byte order {manifest.get('byte_order')!r}."
        )
    if sys.byteorder != "little":
        raise RuntimeError("Packed uint32 shards require a little-endian host")

    _validate_tokenizer(manifest, tokenizer)

    train_entries = _shard_entries(manifest, "train", data_dir)
    val_entries = _shard_entries(manifest, "val", data_dir)
    if not train_entries:
        raise ValueError(f"No train shards in {data_dir!r}")
    if not val_entries:
        raise ValueError(f"No val shards in {data_dir!r}")

    train_tokens = 0
    for path, shard in train_entries:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Train shard missing: {path!r}")
        rows = int(shard.get("sequences", 0))
        if rows <= 0:
            raise ValueError(f"Train shard has no sequences: {path!r}")
        expected_size = rows * seq_len * SHARD_ITEMSIZE
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise ValueError(
                f"Train shard size mismatch for {path!r}: {actual_size:,} "
                f"bytes, expected {expected_size:,}."
            )
        train_tokens += int(shard.get("tokens", rows * seq_len))

    val_tokens = 0
    for path, shard in val_entries:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Val shard missing: {path!r}")
        rows = int(shard.get("sequences", 0))
        if rows <= 0:
            raise ValueError(f"Val shard has no sequences: {path!r}")
        expected_size = rows * seq_len * SHARD_ITEMSIZE
        actual_size = os.path.getsize(path)
        if actual_size != expected_size:
            raise ValueError(
                f"Val shard size mismatch for {path!r}: {actual_size:,} "
                f"bytes, expected {expected_size:,}."
            )
        val_tokens += int(shard.get("tokens", rows * seq_len))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds = PackedTokenDataset(
        [path for path, _ in train_entries],
        seq_len=seq_len,
        shuffle=True,
        seed=seed,
    )
    val_ds = PackedTokenDataset(
        [path for path, _ in val_entries],
        seq_len=seq_len,
        shuffle=False,
        seed=seed,
    )

    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": True,
    }
    if num_workers > 0:
        # Do NOT enable persistent_workers.  Workers get a private copy of the
        # dataset (fork on Linux, spawn on Windows), and the trainer calls
        # train_ds.reshuffle() then re-iterates this loader.  Persistent
        # workers keep the stale copy, ignore every reshuffle, and replay the
        # same segment forever.  A fresh iterator spawns workers that see the
        # current _starts, including the state restored on resume.
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **loader_kwargs)

    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(val_num_workers),
        pin_memory=bool(pin_memory),
        drop_last=True,
    )

    text_bytes = int(manifest.get("text_bytes", 0))
    if text_bytes <= 0:
        raise ValueError(
            "Manifest has no positive text_bytes; the BPB proxy cannot be "
            "computed. Rebuild the corpus with prepare_v16_data.py."
        )

    meta = {
        "data_dir": data_dir,
        "manifest": manifest,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "text_bytes": text_bytes,
        "bytes_per_token": text_bytes / max(train_tokens + val_tokens, 1),
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "train_shards": len(train_entries),
        "val_shards": len(val_entries),
    }
    return train_loader, val_loader, train_ds, meta


def replay_loader_position(
    dataset: PackedTokenDataset,
    loader: DataLoader,
    state: dict,
    start_step: int,
    batch_size: int,
    reshuffle_every_n_steps: int,
    seed: int,
) -> Tuple[Iterator, int, int, bool]:
    """Restore a packed dataset and position its loader for a resumed step.

    ``state["pos"]`` counts windows consumed since the last reshuffle, not an
    absolute dataset position.  Restore the saved window order first, then:

    * at a reshuffle boundary the trainer reshuffled after writing the
      checkpoint, so the saved segment is finished -- start a fresh segment
      with ``seed + start_step`` instead of replaying it;
    * otherwise replay ``pos // batch_size`` batches so the next batch
      continues the saved order exactly.

    Returns ``(iterator, batches_since_reshuffle, saved_pos, at_boundary)``.
    """
    dataset.load_state(state)
    saved_pos = int(state.get("pos", 0))
    at_boundary = (
        reshuffle_every_n_steps > 0
        and start_step > 0
        and start_step % reshuffle_every_n_steps == 0
    )

    if at_boundary:
        dataset.reshuffle(new_seed=seed + start_step)
        return iter(loader), 0, saved_pos, True

    batch_num = 0
    data_iter = iter(loader)
    for _ in range(saved_pos // batch_size):
        try:
            next(data_iter)
            batch_num += 1
        except StopIteration:
            dataset.reshuffle(new_seed=seed + start_step)
            data_iter = iter(loader)
            batch_num = 0
            break
    return data_iter, batch_num, saved_pos, False
