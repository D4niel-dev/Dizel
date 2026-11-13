"""Regression tests for packed-corpus reshuffle and resume replay.

The v1.6 trainer reshuffles ``PackedTokenDataset`` every
``reshuffle_every_n_steps`` and re-iterates the DataLoader.  DataLoader
workers get a private copy of the dataset (fork on Linux, spawn on Windows),
so persistent workers keep the stale ``_starts`` order and silently replay
the same segment.  These tests pin two invariants:

* a re-iterated loader sees ``dataset.reshuffle()`` in its workers;
* replaying a checkpoint reproduces the uninterrupted window order, both at
  a reshuffle boundary and mid-segment.
"""

import gc
import json
import multiprocessing
import os
import sys
import tempfile
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.pretokenized import (
    build_pretokenized_loaders,
    replay_loader_position,
)

SEQ_LEN = 4
CONTEXT_LENGTH = SEQ_LEN - 1
BATCH_SIZE = 2
GRAD_ACCUM = 4
RESHUFFLE_EVERY = 5
SEED = 42
TOTAL_STEPS = 15


def _write_shard(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.repeat(np.arange(rows, dtype=np.uint32), SEQ_LEN)
    data.reshape(rows, SEQ_LEN).tofile(path)


def _make_corpus(directory, train_rows=200, val_rows=20):
    _write_shard(os.path.join(directory, "train", "shard_00000.bin"), train_rows)
    _write_shard(os.path.join(directory, "val", "shard_00000.bin"), val_rows)
    manifest = {
        "format": "dizel-pretokenized-v1",
        "seq_len": SEQ_LEN,
        "context_length": CONTEXT_LENGTH,
        "dtype": "uint32",
        "byte_order": "little",
        "text_bytes": train_rows * SEQ_LEN * 4,
        "shards": [
            {
                "file": "train/shard_00000.bin",
                "split": "train",
                "sequences": train_rows,
                "tokens": train_rows * SEQ_LEN,
            },
            {
                "file": "val/shard_00000.bin",
                "split": "val",
                "sequences": val_rows,
                "tokens": val_rows * SEQ_LEN,
            },
        ],
    }
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)


@contextmanager
def _corpus(directory, num_workers, train_rows=200, val_rows=20):
    _make_corpus(directory, train_rows, val_rows)
    loader, val_loader, dataset, _ = build_pretokenized_loaders(
        directory,
        context_length=CONTEXT_LENGTH,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=num_workers,
        val_num_workers=0,
        pin_memory=False,
    )
    try:
        yield loader, dataset
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        loader._iterator = None
        for memmap in getattr(dataset, "_maps", {}).values():
            try:
                memmap._mmap.close()
            except (AttributeError, ValueError):
                pass
        dataset._maps.clear()
        del val_loader
        gc.collect()
        # Non-persistent iterators are not stored on the loader; the test's
        # local still holds the last one while this cleanup runs.  Stop any
        # remaining workers so Windows can release the shard files.
        for child in multiprocessing.active_children():
            child.terminate()
            child.join(timeout=5)


def _consume(iterator, micro_batches):
    rows = []
    for _ in range(micro_batches):
        x, _ = next(iterator)
        rows.extend(x[:, 0].tolist())
    return rows


def _continuous_run(dataset, loader):
    """Mirror the trainer: save state, then reshuffle at segment boundaries."""
    rows = {}
    states = {}
    iterator = iter(loader)
    batch_num = 0
    for step in range(1, TOTAL_STEPS + 1):
        rows[step] = _consume(iterator, GRAD_ACCUM)
        batch_num += GRAD_ACCUM

        state = dataset.get_state()
        state["pos"] = batch_num * BATCH_SIZE
        states[step] = state

        if step % RESHUFFLE_EVERY == 0 and step < TOTAL_STEPS:
            dataset.reshuffle(new_seed=SEED + step)
            iterator = iter(loader)
            batch_num = 0
    return rows, states


def _resumed_run(dataset, loader, state, start_step):
    iterator, _batch_num, saved_pos, at_boundary = replay_loader_position(
        dataset,
        loader,
        state,
        start_step=start_step,
        batch_size=BATCH_SIZE,
        reshuffle_every_n_steps=RESHUFFLE_EVERY,
        seed=SEED,
    )
    rows = {}
    for step in range(start_step + 1, TOTAL_STEPS + 1):
        rows[step] = _consume(iterator, GRAD_ACCUM)
        if step % RESHUFFLE_EVERY == 0 and step < TOTAL_STEPS:
            dataset.reshuffle(new_seed=SEED + step)
            iterator = iter(loader)
    return rows, saved_pos, at_boundary


def test_reshuffle_reaches_workers():
    with tempfile.TemporaryDirectory() as directory:
        with _corpus(directory, num_workers=2, train_rows=40, val_rows=8) as (
            loader,
            dataset,
        ):
            expected_new = (
                np.random.default_rng(10042).permutation(40)[:BATCH_SIZE].tolist()
            )
            expected_old = (
                np.random.default_rng(SEED).permutation(40)[:BATCH_SIZE].tolist()
            )
            assert expected_new != expected_old

            iterator = iter(loader)
            for _ in range(5):
                next(iterator)

            dataset.reshuffle(new_seed=10042)
            iterator = iter(loader)
            x, _ = next(iterator)
            got = x[:, 0].tolist()
            assert got == expected_new, (
                "DataLoader workers still hold the pre-reshuffle window order; "
                f"expected {expected_new}, got {got}"
            )


def test_resume_at_boundary_matches_continuous_run():
    with tempfile.TemporaryDirectory() as directory:
        with _corpus(directory, num_workers=0) as (loader, dataset):
            rows, states = _continuous_run(dataset, loader)

        # The step-10 checkpoint was written before the end-of-step reshuffle.
        assert states[10]["seed"] == SEED + 5
        assert np.array_equal(
            states[10]["starts"],
            np.random.default_rng(SEED + 5).permutation(200),
        )

        with _corpus(directory, num_workers=0) as (loader, dataset):
            resumed, saved_pos, at_boundary = _resumed_run(
                dataset, loader, states[10], 10
            )
            assert at_boundary, "step 10 is a reshuffle boundary"
            assert saved_pos == 5 * GRAD_ACCUM * BATCH_SIZE
            for step in range(11, TOTAL_STEPS + 1):
                assert resumed[step] == rows[step], (
                    f"boundary resume diverged at step {step}"
                )


def test_resume_mid_segment_matches_continuous_run():
    with tempfile.TemporaryDirectory() as directory:
        with _corpus(directory, num_workers=0) as (loader, dataset):
            rows, states = _continuous_run(dataset, loader)

        with _corpus(directory, num_workers=0) as (loader, dataset):
            resumed, saved_pos, at_boundary = _resumed_run(
                dataset, loader, states[7], 7
            )
            assert not at_boundary, "step 7 is inside the step-5 segment"
            assert saved_pos == 2 * GRAD_ACCUM * BATCH_SIZE
            for step in range(8, TOTAL_STEPS + 1):
                assert resumed[step] == rows[step], (
                    f"mid-segment resume diverged at step {step}"
                )


def main():
    test_reshuffle_reaches_workers()
    test_resume_at_boundary_matches_continuous_run()
    test_resume_mid_segment_matches_continuous_run()
    print("pretrain resume tests passed")


if __name__ == "__main__":
    main()
