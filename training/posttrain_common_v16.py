"""
training/posttrain_common_v16.py — Small shared runtime for v1.6 post-training.

Only mechanics live here: JSONL loading, deterministic train/val splits,
optimizer construction, a resumable batch stream, and checkpoint side effects.
The phase-specific losses stay in sft_v16.py, distill_v16.py and dpo_v16.py.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from training.posttrain_checkpoint_v16 import (
    save_posttrain_checkpoint,
    sync_checkpoint_to_drive,
)

__all__ = [
    "load_jsonl_records",
    "split_records",
    "build_optimizer",
    "make_amp_context",
    "ResumableBatchStream",
    "save_and_sync",
    "load_hf_token_from_dotenv",
]


def load_hf_token_from_dotenv() -> bool:
    """Load ``HF_TOKEN`` from the repository .env without printing it.

    Colab shell exports are usually enough, but local runs should not require
    the token on the command line.  Existing environment values win, so this
    never overwrites an explicitly supplied token.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return False
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
    ]
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[len("export "):].strip()
                    if key not in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
                        continue
                    value = value.strip().strip('"').strip("'")
                    if value:
                        os.environ["HF_TOKEN"] = value
                        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", value)
                        return True
        except OSError:
            continue
    return False


def load_jsonl_records(path: str, limit: Optional[int] = None) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Post-training data not found: {path!r}. Run "
            "`python pipelines/prepare_v16_posttrain.py` first."
        )
    records = []
    bad = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
            if limit is not None and len(records) >= limit:
                break
    if bad:
        print(f"  [posttrain] skipped {bad:,} malformed JSONL records in {path}")
    return records


def split_records(records: list[dict], val_fraction: float, seed: int):
    """Deterministic split; always leaves at least one training example."""
    if not records:
        raise ValueError("Cannot split an empty post-training dataset")
    order = list(range(len(records)))
    rng = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(order), generator=rng).tolist()
    shuffled = [records[i] for i in order]
    n_val = int(len(shuffled) * max(0.0, min(0.5, float(val_fraction))))
    if len(shuffled) > 1:
        n_val = max(1, n_val)
    else:
        n_val = 0
    n_train = len(shuffled) - n_val
    return shuffled[:n_train], shuffled[n_train:]


def build_optimizer(model, cfg) -> optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg.lr),
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def make_amp_context(cfg, device):
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    amp_dtype = torch.float32
    if use_amp:
        # ``is_bf16_supported()`` includes software emulation by default in
        # recent PyTorch, which is unusably slow on pre-Ampere GPUs (T4).
        native_bf16 = False
        if str(cfg.amp_dtype) == "bfloat16" and torch.cuda.is_bf16_supported():
            try:
                native_bf16 = torch.cuda.is_bf16_supported(
                    including_emulation=False
                )
            except TypeError:
                major, _ = torch.cuda.get_device_capability(device)
                native_bf16 = major >= 8
        amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()
    scaler = GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    return use_amp, amp_dtype, amp_ctx, scaler


class ResumableBatchStream:
    """Deterministic, resumable epoch stream for map-style datasets.

    Each epoch uses ``seed + epoch`` for its DataLoader generator.  On resume,
    the stream skips the number of already-consumed batches in the saved epoch.
    The model RNG is restored separately by the checkpoint loader.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        collate_fn: Callable,
        seed: int,
        num_workers: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.collate_fn = collate_fn
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.batch_index = 0
        self._iterator = None
        self._skip = 0

    def state(self) -> dict:
        return {"epoch": int(self.epoch), "batch_index": int(self.batch_index)}

    def load_state(self, state: Optional[dict]) -> None:
        if not state:
            return
        self.epoch = int(state.get("epoch", 0))
        self.batch_index = int(state.get("batch_index", 0))
        self._skip = self.batch_index
        self._iterator = None

    def _make_loader(self, epoch: int):
        generator = torch.Generator().manual_seed(self.seed + int(epoch))
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=self.drop_last,
        )

    def _new_iterator(self):
        iterator = iter(self._make_loader(self.epoch))
        if self._skip:
            for _ in range(self._skip):
                try:
                    next(iterator)
                except StopIteration:
                    break
            self._skip = 0
        self._iterator = iterator

    def next_batch(self):
        if self._iterator is None:
            self._new_iterator()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batch_index = 0
            self._skip = 0
            self._iterator = iter(self._make_loader(self.epoch))
            batch = next(self._iterator)
        self.batch_index += 1
        return batch


def save_and_sync(
    path: str,
    model,
    optimizer,
    scaler,
    step: int,
    phase: str,
    phase_cfg,
    model_cfg,
    tokenizer_cfg,
    tokenizer,
    val_loss: Optional[float],
    best_val: Optional[float],
    dataset_state: Optional[dict],
    phase0: Optional[int],
    phase_tag: Optional[str],
    base_checkpoint: Optional[str] = None,
    reference_checkpoint: Optional[str] = None,
    drive_dir: str = "",
):
    save_posttrain_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        phase=phase,
        phase_cfg=phase_cfg,
        model_cfg=model_cfg,
        tokenizer_cfg=tokenizer_cfg,
        path=path,
        tokenizer=tokenizer,
        val_loss=val_loss,
        best_val=best_val,
        dataset_state=dataset_state,
        phase0=phase0,
        phase_tag=phase_tag,
        base_checkpoint=base_checkpoint,
        reference_checkpoint=reference_checkpoint,
    )
    if drive_dir:
        sync_checkpoint_to_drive(path, drive_dir)
    return path
