"""
scripts/eval_holdout_bpb.py - fixed byte-level holdout BPB for Dizel 1.5/1.6.

The training loop reports a corpus-wide BPB proxy using the packed validation
split.  That proxy is useful for monitoring, but it is not a fixed benchmark:
the validation mixture and its bytes-per-token ratio change when the corpus
changes.  This script evaluates a checkpoint on the frozen Wikipedia holdout
under ``data/v16/holdout`` and reports bits per UTF-8 byte.

Usage:
    python scripts/eval_holdout_bpb.py \
        --checkpoint checkpoints/dizel-1.6-preview-m1-best.pt \
        --holdout data/v16/holdout/wikipedia.jsonl.gz \
        --tokenizer tokenizer/qwen2.5/tokenizer.json

    # dense v1.5 control (its distillation checkpoint stores unmerged LoRA)
    python scripts/eval_holdout_bpb.py \
        --checkpoint checkpoints/dizel15/dizel-v15-distill-best.pt \
        --holdout data/v16/holdout/wikipedia.jsonl.gz

The holdout is concatenated as ``text + <|endoftext|>`` per document, matching
the packed corpus contract.  For v1.5 the document separator is SentencePiece
EOS (id 2); for v1.6 it is Qwen ``<|endoftext|>`` (id 151643).  Windows overlap
by one token, so every holdout token except the first is scored exactly once.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from typing import Iterable, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.architecture import DizelLM
from pipelines.prepare_v16_data import open_tokenizer, sha256_file
from training.pretrain_v16 import load_checkpoint_v16


class TokenizerIdentity:
    """The small surface ``load_checkpoint_v16`` needs from a tokenizer."""

    def __init__(self, tokenizer, info: dict) -> None:
        self.tokenizer = tokenizer
        self.info = info

    def __len__(self) -> int:
        return int(self.info["vocab_size"])

    def fingerprint(self) -> str:
        return str(self.info["fingerprint"])


def iter_holdout_texts(path: str, max_docs: int = 0) -> Iterable[str]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_docs and index >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"Holdout row {index:,} has no string 'text' field"
                )
            yield text


def verify_holdout(path: str) -> None:
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(path)), "manifest.json")
    if not os.path.isfile(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = manifest.get("gzip_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"Holdout SHA-256 mismatch: {path!r}")


def encode_ids(tokenizer, text: str, checkpoint_format: str) -> List[int]:
    if checkpoint_format == "v16":
        return [
            int(token)
            for token in tokenizer.encode(
                text, add_special_tokens=False
            ).ids
        ]
    return [int(token) for token in tokenizer.encode(text)]


def build_sequence(
    path: str,
    tokenizer,
    doc_sep_id: int,
    checkpoint_format: str,
    max_docs: int = 0,
) -> tuple[List[int], int, int]:
    ids: List[int] = []
    text_bytes = 0
    documents = 0
    for text in iter_holdout_texts(path, max_docs=max_docs):
        encoded = encode_ids(tokenizer, text, checkpoint_format)
        ids.extend(encoded)
        ids.append(int(doc_sep_id))
        text_bytes += len(text.encode("utf-8"))
        documents += 1
    return ids, text_bytes, documents


def load_v15_state(model: DizelLM, checkpoint: dict) -> None:
    """Load and fold the LoRA adapters stored in a v1.5 distillation checkpoint."""
    from training.distill import attach_lora, merge_lora_weights

    state = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state"].items()
    }
    r = int(checkpoint.get("lora_r", 32))
    alpha = int(checkpoint.get("lora_alpha", 2 * r))
    attach_lora(
        model,
        r=r,
        alpha=alpha,
        dropout=0.0,
        target_names=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model.load_state_dict(state, strict=True)
    merge_lora_weights(model)


def evaluate(
    model: DizelLM,
    ids: List[int],
    context_length: int,
    device: torch.device,
    use_amp: bool,
    progress: bool,
) -> dict:
    total_tokens = len(ids)
    if total_tokens < 2:
        raise ValueError(f"Holdout produced only {total_tokens} tokens")

    amp_dtype = None
    if use_amp:
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    total_nll = 0.0
    predicted = 0
    windows = 0
    started = time.time()

    def forward(chunk: torch.Tensor) -> float:
        nonlocal predicted, windows
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        with torch.inference_mode():
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, targets=y)
            else:
                _, loss = model(x, targets=y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Non-finite holdout loss")
        count = int(y.numel())
        predicted += count
        windows += 1
        return float(loss.detach().cpu()) * count

    # Windows overlap by one token (stride == context_length).  That scores
    # every token except the first exactly once.
    for start in range(0, total_tokens - 1, context_length):
        window_ids = ids[start : start + context_length + 1]
        if len(window_ids) < 2:
            break
        chunk = torch.tensor(
            window_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        total_nll += forward(chunk)
        if progress and windows % 50 == 0:
            print(
                f"  scored {predicted:,}/{total_tokens - 1:,} tokens "
                f"in {windows:,} windows",
                flush=True,
            )

    elapsed = time.time() - started
    return {
        "documents": None,
        "text_bytes": None,
        "tokens": total_tokens,
        "predicted_tokens": predicted,
        "windows": windows,
        "nll_nats": total_nll,
        "bits_per_token": total_nll / math.log(2) / max(predicted, 1),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Dizel checkpoint on the fixed holdout BPB"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--holdout",
        default=os.path.join("data", "v16", "holdout", "wikipedia.jsonl.gz"),
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help=(
            "Tokenizer path. Defaults to tokenizer/qwen2.5/tokenizer.json for "
            "v1.6 and tokenizer/dizel_v15.model for v1.5."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["auto", "v15", "v16"],
        default="auto",
        help="Checkpoint format; auto reads _format_version.",
    )
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint!r}")
    if not os.path.isfile(args.holdout):
        raise FileNotFoundError(f"Holdout not found: {args.holdout!r}")
    if not args.no_verify:
        verify_holdout(args.holdout)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_cfg")
    if model_cfg is None:
        raise ValueError(f"Checkpoint {args.checkpoint!r} has no model_cfg")

    version = str(ckpt.get("_format_version", ""))
    if version == "1.6":
        checkpoint_format = "v16"
    elif version in ("1.5", "1.2.2"):
        checkpoint_format = "v15"
    else:
        checkpoint_format = args.format
        if checkpoint_format == "auto":
            raise ValueError(
                f"Cannot infer checkpoint format from {version!r}; "
                "pass --format v15 or --format v16."
            )
    if args.format != "auto" and args.format != checkpoint_format:
        raise ValueError(
            f"Checkpoint is {checkpoint_format}, but --format {args.format} "
            "was requested."
        )

    if checkpoint_format == "v16":
        tokenizer_path = args.tokenizer or os.path.join(
            "tokenizer", "qwen2.5", "tokenizer.json"
        )
        tokenizer, info = open_tokenizer(tokenizer_path)
        identity = TokenizerIdentity(tokenizer, info)
        model = DizelLM(model_cfg)
        load_checkpoint_v16(
            args.checkpoint,
            model,
            tokenizer=identity,
            device=args.device,
            restore_rng=False,
        )
        doc_sep_id = int(info["doc_sep_id"])
        tokenizer_fingerprint = str(info["fingerprint"])
    else:
        tokenizer_path = args.tokenizer or os.path.join(
            "tokenizer", "dizel_v15.model"
        )
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(
                f"v1.5 tokenizer not found: {tokenizer_path!r}"
            )
        from training.dataset import Tokenizer as V15Tokenizer

        tokenizer = V15Tokenizer(tokenizer_path)
        model = DizelLM(model_cfg)
        load_v15_state(model, ckpt)
        doc_sep_id = int(tokenizer.eos_id)
        tokenizer_fingerprint = sha256_file(tokenizer_path)

    ckpt.pop("model_state", None)
    ckpt.pop("optim_state", None)
    device = torch.device(args.device)
    model.to(device).eval()

    ids, text_bytes, documents = build_sequence(
        args.holdout,
        tokenizer,
        doc_sep_id,
        checkpoint_format,
        max_docs=args.max_docs,
    )
    context_length = int(getattr(model_cfg, "context_length", 0))
    if context_length < 2:
        raise ValueError(f"Invalid model context_length: {context_length}")

    print(
        f"Holdout: {documents:,} docs, {text_bytes:,} UTF-8 bytes, "
        f"{len(ids):,} tokens, context={context_length:,}, "
        f"format={checkpoint_format}",
        flush=True,
    )
    result = evaluate(
        model,
        ids,
        context_length,
        device,
        use_amp=(device.type == "cuda" and not args.no_amp),
        progress=args.progress,
    )
    result.update(
        {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_format": checkpoint_format,
            "checkpoint_step": int(ckpt.get("step", 0)),
            "tokens_seen": int(ckpt.get("tokens_seen", 0)),
            "checkpoint_val_loss": ckpt.get("val_loss"),
            "config_hash": ckpt.get("config_hash"),
            "holdout": os.path.abspath(args.holdout),
            "documents": documents,
            "text_bytes": text_bytes,
            "tokens_per_byte": len(ids) / max(text_bytes, 1),
            "bits_per_byte": result["nll_nats"]
            / math.log(2)
            / max(text_bytes, 1),
            "tokenizer": os.path.abspath(tokenizer_path),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "model_context_length": context_length,
        }
    )

    print(
        f"BPB: {result['bits_per_byte']:.6f} | "
        f"bits/token: {result['bits_per_token']:.6f} | "
        f"scored: {result['predicted_tokens']:,}/{len(ids) - 1:,} tokens | "
        f"{result['elapsed_seconds']:.1f}s",
        flush=True,
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
