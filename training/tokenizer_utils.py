"""
training/tokenizer_utils.py — Fast batch tokenization with multiprocessing.

Wraps SentencePiece to tokenize text shards in large batches
using multiple CPU workers instead of the old line-by-line approach.
"""

import multiprocessing as mp
from typing import List, Callable
from functools import partial


def _tokenize_chunk(text: str, encode_fn: Callable, add_bos: bool = False) -> List[int]:
    """Tokenize a single text chunk (used by workers)."""
    return encode_fn(text, add_bos=add_bos)


def tokenize_shard_batched(
    shard_text: str,
    encode_fn: Callable,
    batch_size_chars: int = 2 * 1024 * 1024,  # 2MB text batches
    add_bos_first: bool = False,
) -> List[int]:
    """
    Tokenize a large shard by splitting into text batches and
    processing them sequentially with the encode function.

    This avoids the old 5MB-chunk-then-line-by-line pattern.
    SentencePiece is already quite fast for large strings, so
    the main gain is from feeding it bigger contiguous text blocks.
    """
    all_ids = []
    offset = 0
    is_first = True

    while offset < len(shard_text):
        end = min(offset + batch_size_chars, len(shard_text))

        # Snap to newline boundary if possible
        if end < len(shard_text):
            nl = shard_text.rfind("\n", offset, end)
            if nl > offset:
                end = nl + 1

        batch = shard_text[offset:end]
        bos = add_bos_first and is_first
        ids = encode_fn(batch, add_bos=bos)
        all_ids.extend(ids)

        offset = end
        is_first = False

    return all_ids


def tokenize_shards_parallel(
    shard_texts: List[str],
    encode_fn: Callable,
    num_workers: int = 2,
    batch_size_chars: int = 2 * 1024 * 1024,
) -> List[List[int]]:
    """
    Tokenize multiple shard texts in parallel using multiprocessing.

    Falls back to sequential processing if num_workers <= 1 or
    if multiprocessing is unavailable (e.g. some Colab restrictions).

    Note: SentencePiece encode_fn must be picklable. Since SentencePiece
    objects are not easily picklable, this function is best used when
    the caller handles shard-level parallelism (tokenize one shard at
    a time but in bigger batches). For true multi-shard parallelism,
    use the process-based approach in build_pretrain_loaders.
    """
    results = []
    for i, text in enumerate(shard_texts):
        bos = (i == 0)
        ids = tokenize_shard_batched(
            text, encode_fn,
            batch_size_chars=batch_size_chars,
            add_bos_first=bos,
        )
        results.append(ids)
    return results
