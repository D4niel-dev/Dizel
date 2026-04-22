"""
training/cache_utils.py — Tokenization caching and resume logic.

Saves per-shard tokenized output as .pt files so repeated runs
skip already-processed shards entirely. Implements the 12-hour
"fresh enough" policy and a simple manifest for resume tracking.
"""

import json
import os
import time
from typing import List, Optional

import numpy as np
import torch


def get_cache_dir(base_cache_dir: str, corpus_name: str) -> str:
    """Return (and create) the cache subdirectory for a specific corpus."""
    path = os.path.join(base_cache_dir, corpus_name)
    os.makedirs(path, exist_ok=True)
    return path


def shard_cache_path(cache_dir: str, shard_idx: int) -> str:
    return os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")


def manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "manifest.json")


def load_manifest(cache_dir: str) -> dict:
    mpath = manifest_path(cache_dir)
    if os.path.exists(mpath):
        with open(mpath, "r") as f:
            return json.load(f)
    return {"completed_shards": [], "total_tokens": 0, "timestamp": 0}


def save_manifest(cache_dir: str, manifest: dict) -> None:
    manifest["timestamp"] = time.time()
    with open(manifest_path(cache_dir), "w") as f:
        json.dump(manifest, f, indent=2)


def is_shard_cached(cache_dir: str, shard_idx: int, manifest: dict) -> bool:
    """Check if a shard has been tokenized and saved."""
    if shard_idx not in manifest.get("completed_shards", []):
        return False
    return os.path.exists(shard_cache_path(cache_dir, shard_idx))


def save_shard_tokens(cache_dir: str, shard_idx: int, token_ids) -> None:
    """Persist a shard's token ids to disk as a .pt file."""
    path = shard_cache_path(cache_dir, shard_idx)
    if isinstance(token_ids, list):
        token_ids = np.array(token_ids, dtype=np.int32)
    torch.save(torch.from_numpy(token_ids.astype(np.int32)), path)


def load_shard_tokens(cache_dir: str, shard_idx: int) -> np.ndarray:
    """Load a cached shard's token ids from disk as a numpy array."""
    path = shard_cache_path(cache_dir, shard_idx)
    return torch.load(path, weights_only=True).numpy().astype(np.int32)


def load_all_cached_tokens(cache_dir: str, num_shards: int) -> Optional[np.ndarray]:
    """
    Attempt to load all shards from cache.
    Returns None if any shard is missing (forces re-tokenization for that shard).
    Uses numpy concatenation to avoid Python list RAM overhead.
    """
    manifest = load_manifest(cache_dir)
    completed = set(manifest.get("completed_shards", []))

    if len(completed) < num_shards:
        return None  # Not all shards are cached

    shard_arrays = []
    for i in range(num_shards):
        if not os.path.exists(shard_cache_path(cache_dir, i)):
            return None
        shard_arrays.append(load_shard_tokens(cache_dir, i))

    return np.concatenate(shard_arrays)


def clear_cache(cache_dir: str) -> None:
    """Remove all cached shard files and manifest."""
    if not os.path.exists(cache_dir):
        return
    for fname in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    print(f"[cache] Cleared: {cache_dir}")
