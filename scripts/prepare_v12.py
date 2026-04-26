"""
scripts/prepare_v12.py — Data preparation script for Dizel v1.2 training.

Downloads, processes, and organizes training datasets for:
  - Pretraining (raw text corpus)
  - SFT (instruction-following JSONL from multiple sources)

Supports the weighted dataset mixing system in training/data_mixing.py.

Usage:
    python scripts/prepare_v12.py --phase pretrain    # Prepare pretrain corpus
    python scripts/prepare_v12.py --phase sft         # Prepare SFT datasets
    python scripts/prepare_v12.py --phase all         # Both
    python scripts/prepare_v12.py --check             # Verify data exists
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SFT_DIR = os.path.join(PROJECT_ROOT, "sft_data")
PRETRAIN_OUT = os.path.join(DATA_DIR, "pretrain_v12.txt")


# ---------------------------------------------------------------------------
# Dataset registry — what we need for v1.2
# ---------------------------------------------------------------------------
SFT_DATASETS = {
    # ── Dizel Identity (weight 5.0) ───────────────────────────────────
    "chat_expanded": {
        "local": True,  # Already exists in sft_data/
        "path": "sft_data/chat_expanded.jsonl",
        "max_samples": -1,
        "format": "messages",
        "weight": 5.0,
    },
    # ── Code & Reasoning (weight 2.5–3.0) ─────────────────────────────
    "codealpaca": {
        "hf_id": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "max_samples": 20000,
        "format": "alpaca",
        "weight": 3.0,
    },
    "codefeedback": {
        "hf_id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "split": "train",
        "max_samples": 15000,
        "format": "messages",
        "weight": 2.5,
    },
    # ── Conversational (weight 2.5) ───────────────────────────────────
    "oasst2": {
        "hf_id": "OpenAssistant/oasst2",
        "split": "train",
        "max_samples": 30000,
        "format": "tree",
        "weight": 2.5,
    },
    # ── Instruction following (weight 1.5–2.0) ────────────────────────
    "alpaca_gpt4": {
        "hf_id": "vicgalle/alpaca-gpt4",
        "split": "train",
        "max_samples": 20000,
        "format": "alpaca",
        "weight": 2.0,
    },
    "dolly": {
        "hf_id": "databricks/databricks-dolly-15k",
        "split": "train",
        "max_samples": 15000,
        "format": "dolly",
        "weight": 1.5,
    },
    # ── Editing (weight 1.0) ──────────────────────────────────────────
    "coedit": {
        "hf_id": "grammarly/coedit",
        "split": "train",
        "max_samples": 8000,
        "format": "src_tgt",
        "weight": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Format converters — normalize each dataset to the standard messages format
# ---------------------------------------------------------------------------
def _to_messages_alpaca(row):
    """Convert Alpaca-format (instruction, input, output) to messages."""
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", "")
    user_msg = f"{instruction}\n{inp}".strip() if inp else instruction
    if not user_msg or not output:
        return None
    return {"messages": [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]}


def _to_messages_dolly(row):
    """Convert Dolly format to messages."""
    instruction = row.get("instruction", "")
    context = row.get("context", "")
    response = row.get("response", "")
    user_msg = f"{instruction}\n\nContext: {context}".strip() if context else instruction
    if not user_msg or not response:
        return None
    return {"messages": [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response},
    ]}


def _to_messages_qa(row):
    """Convert Q&A format to messages."""
    question = row.get("question", "")
    answer = row.get("correct_answer", row.get("answer", ""))
    support = row.get("support", "")
    if not question or not answer:
        return None
    full_answer = f"{answer}\n\n{support}".strip() if support else answer
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_answer},
    ]}


def _to_messages_src_tgt(row):
    """Convert source/target format (like CoEdit) to messages."""
    src = row.get("src", row.get("source", ""))
    tgt = row.get("tgt", row.get("target", ""))
    if not src or not tgt:
        return None
    return {"messages": [
        {"role": "user", "content": src},
        {"role": "assistant", "content": tgt},
    ]}


def _to_messages_pass(row):
    """Already in messages format — just validate."""
    msgs = row.get("messages", row.get("conversations", []))
    if not msgs or len(msgs) < 2:
        return None
    # Normalize role names
    normalized = []
    for m in msgs:
        role = m.get("role", m.get("from", "")).lower()
        content = m.get("content", m.get("value", ""))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        if content:
            normalized.append({"role": role, "content": content})
    if len(normalized) < 2:
        return None
    return {"messages": normalized}


FORMAT_MAP = {
    "alpaca": _to_messages_alpaca,
    "dolly": _to_messages_dolly,
    "qa": _to_messages_qa,
    "src_tgt": _to_messages_src_tgt,
    "messages": _to_messages_pass,
    "conversations": _to_messages_pass,
    "tree": _to_messages_pass,
}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def process_dataset(name: str, info: dict):
    """Download and convert a single dataset to JSONL."""
    # Local datasets (e.g. chat_expanded) — already processed, just verify
    if info.get("local"):
        local_path = info["path"]
        if os.path.exists(local_path):
            count = sum(1 for _ in open(local_path, "r", encoding="utf-8"))
            print(f"  [local] {name}: {count} samples ({local_path})")
            return local_path
        else:
            print(f"  [error] Local dataset not found: {local_path}")
            return None

    out_path = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
    if os.path.exists(out_path):
        count = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
        print(f"  [skip] {name}: already exists ({count} samples)")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError:
        print(f"  [error] 'datasets' library not installed. Run: pip install datasets")
        return None

    print(f"  [download] {name} from {info['hf_id']}...")
    try:
        ds = load_dataset(info["hf_id"], split=info["split"], trust_remote_code=True)
    except Exception as e:
        print(f"  [error] Failed to load {name}: {e}")
        return None

    converter = FORMAT_MAP.get(info["format"], _to_messages_pass)
    max_samples = info.get("max_samples", -1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            if max_samples > 0 and count >= max_samples:
                break
            converted = converter(row)
            if converted is None:
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [done] {name}: {count} samples → {out_path}")
    return out_path


def check_data():
    """Check which datasets exist and report status."""
    print("\n=== Dizel v1.2 Data Status ===\n")

    # Pretrain
    if os.path.exists(PRETRAIN_OUT):
        size_mb = os.path.getsize(PRETRAIN_OUT) / (1024 * 1024)
        print(f"  ✅ Pretrain corpus: {size_mb:.1f} MB")
    else:
        print(f"  ❌ Pretrain corpus: MISSING ({PRETRAIN_OUT})")

    # SFT datasets
    print(f"\n  SFT datasets ({PROCESSED_DIR}):")
    total = 0
    for name, info in SFT_DATASETS.items():
        path = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
        if os.path.exists(path):
            count = sum(1 for _ in open(path, "r", encoding="utf-8"))
            total += count
            print(f"    ✅ {name:16s}  {count:>6,} samples  (weight={info['weight']})")
        else:
            print(f"    ❌ {name:16s}  MISSING")

    print(f"\n  Total SFT samples: {total:,}")


def prepare_sft():
    """Download and process all SFT datasets."""
    print("\n=== Preparing SFT Datasets ===\n")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for name, info in SFT_DATASETS.items():
        process_dataset(name, info)

    print("\n=== SFT preparation complete ===")
    check_data()


def prepare_pretrain():
    """Prepare pretrain corpus (placeholder — requires manual data collection)."""
    print("\n=== Pretrain Corpus ===\n")
    if os.path.exists(PRETRAIN_OUT):
        size_mb = os.path.getsize(PRETRAIN_OUT) / (1024 * 1024)
        print(f"  Pretrain corpus already exists: {size_mb:.1f} MB")
    else:
        print(f"  Pretrain corpus not found at: {PRETRAIN_OUT}")
        print(f"  To create it, concatenate your text sources into:")
        print(f"    {PRETRAIN_OUT}")
        print(f"\n  Recommended sources for v1.2:")
        print(f"    - Wikipedia dump (cleaned)")
        print(f"    - OpenWebText subset")
        print(f"    - BookCorpus or similar")
        print(f"    - Code (GitHub subset)")
        print(f"\n  Target: ~500MB - 1GB of cleaned text")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Dizel v1.2 data preparation")
    p.add_argument("--phase", choices=["pretrain", "sft", "all"], default="all",
                   help="Which phase to prepare")
    p.add_argument("--check", action="store_true",
                   help="Only check data status, don't download")
    args = p.parse_args()

    if args.check:
        check_data()
        return

    if args.phase in ("pretrain", "all"):
        prepare_pretrain()
    if args.phase in ("sft", "all"):
        prepare_sft()


if __name__ == "__main__":
    main()
