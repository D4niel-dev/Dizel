"""
training/cache_utils.py — Tokenization caching and resume logic.

v1.3: Memory-mapped approach. Instead of loading all shards into RAM,
tokens are appended to a flat binary file (tokens.bin) and accessed
via np.memmap.  Total RAM used for token storage: ~0 bytes.
"""

import json
import os
import time
from typing import Optional

import numpy as np
import torch


def get_cache_dir(base_cache_dir: str, corpus_name: str) -> str:
    """Return (and create) the cache subdirectory for a specific corpus."""
    path = os.path.join(base_cache_dir, corpus_name)
    os.makedirs(path, exist_ok=True)
    return path


def shard_cache_path(cache_dir: str, shard_idx: int) -> str:
    return os.path.join(cache_dir, f"shard_{shard_idx:04d}.pt")


def tokens_bin_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "tokens.bin")


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


# ---------------------------------------------------------------------------
# Memory-mapped token storage (v1.3)
# ---------------------------------------------------------------------------
def append_tokens_to_bin(cache_dir: str, token_ids: np.ndarray) -> int:
    """
    Append token_ids (int32) to the flat binary file.
    Returns the number of tokens written.
    """
    path = tokens_bin_path(cache_dir)
    arr = token_ids.astype(np.int32)
    with open(path, "ab") as f:
        f.write(arr.tobytes())
        f.flush()
        os.fsync(f.fileno())
    return len(arr)


def build_memmap(cache_dir: str, total_tokens: int) -> np.ndarray:
    """
    Open the flat binary file as a read-only memory-mapped int32 array.
    This uses virtually ZERO RAM — the OS pages data from disk on demand.
    
    If the file is smaller than expected (e.g. disk pressure truncated it),
    we clamp to the actual token count derived from the file size.
    """
    path = tokens_bin_path(cache_dir)
    actual_bytes = os.path.getsize(path)
    expected_bytes = total_tokens * 4  # int32 = 4 bytes
    if actual_bytes < expected_bytes:
        actual_tokens = actual_bytes // 4
        print(f"[cache] ⚠ tokens.bin is {actual_bytes:,} bytes but expected {expected_bytes:,}")
        print(f"[cache]   Clamping from {total_tokens:,} → {actual_tokens:,} tokens")
        total_tokens = actual_tokens
    return np.memmap(path, dtype=np.int32, mode="r", shape=(total_tokens,))


def is_bin_complete(cache_dir: str, num_shards: int) -> Optional[int]:
    """
    Check if tokens.bin exists and the manifest says all shards are done.
    Returns total_tokens if complete, None otherwise.
    """
    manifest = load_manifest(cache_dir)
    completed = set(manifest.get("completed_shards", []))
    if len(completed) < num_shards:
        return None
    bin_path = tokens_bin_path(cache_dir)
    if not os.path.exists(bin_path):
        return None
    total = manifest.get("total_tokens", 0)
    if total == 0:
        return None
    # Verify file size matches
    expected_bytes = total * 4  # int32 = 4 bytes
    actual_bytes = os.path.getsize(bin_path)
    if actual_bytes != expected_bytes:
        print(f"[cache] ⚠ tokens.bin size mismatch ({actual_bytes} vs {expected_bytes}), rebuilding...")
        return None
    return total


# ---------------------------------------------------------------------------
# Legacy compatibility
# ---------------------------------------------------------------------------
def load_all_cached_tokens(cache_dir: str, num_shards: int) -> Optional[np.ndarray]:
    """
    Legacy loader — only used if tokens.bin doesn't exist but .pt shards do.
    Builds tokens.bin from existing .pt shards without loading all into RAM.
    """
    manifest = load_manifest(cache_dir)
    completed = set(manifest.get("completed_shards", []))

    if len(completed) < num_shards:
        return None

    # Check if all .pt files exist
    for i in range(num_shards):
        if not os.path.exists(shard_cache_path(cache_dir, i)):
            return None

    # Stream .pt shards → tokens.bin (one at a time, no full RAM load)
    bin_path = tokens_bin_path(cache_dir)
    total_tokens = 0
    print("[cache] Converting .pt shards → tokens.bin (streaming, low RAM)...")
    with open(bin_path, "wb") as f:
        for i in range(num_shards):
            shard = load_shard_tokens(cache_dir, i)
            f.write(shard.astype(np.int32).tobytes())
            total_tokens += len(shard)
            del shard  # free immediately
            print(f"  shard {i+1}/{num_shards} written ({total_tokens:,} tokens so far)")

    manifest["total_tokens"] = total_tokens
    save_manifest(cache_dir, manifest)
    print(f"[cache] ✓ tokens.bin built: {total_tokens:,} tokens")

    return build_memmap(cache_dir, total_tokens)


def clear_cache(cache_dir: str) -> None:
    """Remove all cached shard files, tokens.bin, and manifest."""
    if not os.path.exists(cache_dir):
        return
    for fname in os.listdir(cache_dir):
        fpath = os.path.join(cache_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
    print(f"[cache] Cleared: {cache_dir}")
