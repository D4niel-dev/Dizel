"""
training/shard_utils.py — Corpus sharding for efficient tokenization.

Splits a large raw text file into memory-friendly shards (default 100MB)
instead of the old 5MB micro-chunks, reducing file-handling overhead
and improving tokenization throughput on Colab T4.
"""

import os
from typing import List, Tuple


def shard_corpus(
    data_path: str,
    shard_size_mb: int = 100,
    encoding: str = "utf-8",
) -> List[Tuple[int, int]]:
    """
    Compute byte-offset boundaries for sharding a text file.

    Instead of physically splitting the file, returns a list of
    (start_byte, end_byte) tuples so each shard can be read lazily.
    Shard boundaries are snapped to newline characters to avoid
    splitting a line across two shards.

    Returns
    -------
    List of (start_byte, end_byte) pairs.
    """
    file_size = os.path.getsize(data_path)
    shard_bytes = shard_size_mb * 1024 * 1024
    boundaries: List[Tuple[int, int]] = []

    with open(data_path, "rb") as f:
        start = 0
        while start < file_size:
            end = min(start + shard_bytes, file_size)
            if end < file_size:
                # Snap to a newline so we don't chop mid-line
                f.seek(end)
                remainder = f.readline()  # read to end of current line
                end = f.tell()
            boundaries.append((start, end))
            start = end

    return boundaries


def read_shard(data_path: str, start: int, end: int, encoding: str = "utf-8") -> str:
    """Read a single shard from disk by byte offsets."""
    with open(data_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    return raw.decode(encoding, errors="replace")
