"""
scripts/format_v122_data.py — Convert cleaned data into unified training format.

Converts all cleaned JSONL into two output formats:
  1. Pretrain format:  {"text": "..."} — plain text for next-token prediction
  2. SFT format:       {"instruction": "...", "input": "...", "output": "..."} — for instruction tuning

Also handles mixing/sampling for final training corpus.

Usage:
    python scripts/format_v122_data.py --input data/cleaned/ --mode pretrain --output data/formatted/pretrain.jsonl
    python scripts/format_v122_data.py --input data/cleaned/ --mode sft --output data/formatted/sft.jsonl
    python scripts/format_v122_data.py --input data/cleaned/ --mode pretrain --target dizel
    python scripts/format_v122_data.py --input data/cleaned/ --mode pretrain --target mila
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mixture Configs ───────────────────────────────────────────────────────

DIZEL_PRETRAIN_MIX = {
    "fineweb-edu":          {"weight": 0.40, "max_samples": 400_000},
    "finemath":             {"weight": 0.12, "max_samples": 80_000},
    "stack-edu-python":     {"weight": 0.08, "max_samples": 60_000},
    "stack-edu-javascript": {"weight": 0.04, "max_samples": 30_000},
    "stack-edu-typescript": {"weight": 0.03, "max_samples": 20_000},
    "cosmopedia":           {"weight": 0.10, "max_samples": 60_000},
    "oasst2":               {"weight": 0.05, "max_samples": 15_000},
    "codealpaca":           {"weight": 0.08, "max_samples": 15_000},
    "codefeedback":         {"weight": 0.05, "max_samples": 15_000},
    "ultrachat":            {"weight": 0.05, "max_samples": 30_000},
}

MILA_PRETRAIN_MIX = {
    "fineweb-edu":          {"weight": 0.40, "max_samples": 200_000},
    "cosmopedia":           {"weight": 0.15, "max_samples": 50_000},
    "oasst2":               {"weight": 0.15, "max_samples": 20_000},
    "ultrachat":            {"weight": 0.15, "max_samples": 40_000},
    "stack-edu-python":     {"weight": 0.05, "max_samples": 15_000},
    "stack-edu-javascript": {"weight": 0.05, "max_samples": 15_000},
    "dolly":                {"weight": 0.05, "max_samples": 10_000},
}

DIZEL_SFT_MIX = {
    "chat_expanded":{"weight": 5.0,  "max_samples": -1},  # Identity anchor
    "codealpaca":   {"weight": 3.0,  "max_samples": -1},
    "codefeedback": {"weight": 2.5,  "max_samples": 15_000},
    "oasst2":       {"weight": 2.5,  "max_samples": -1},
    "alpaca_gpt4":  {"weight": 2.0,  "max_samples": 20_000},
    "dolly":        {"weight": 1.5,  "max_samples": -1},
    "coedit":       {"weight": 1.0,  "max_samples": 8_000},
}

MILA_SFT_MIX = {
    "oasst2":       {"weight": 3.0,  "max_samples": -1},
    "alpaca_gpt4":  {"weight": 2.0,  "max_samples": 15_000},
    "dolly":        {"weight": 2.0,  "max_samples": -1},
    "coedit":       {"weight": 1.0,  "max_samples": 5_000},
}

MIXTURES = {
    "dizel_pretrain": DIZEL_PRETRAIN_MIX,
    "mila_pretrain":  MILA_PRETRAIN_MIX,
    "dizel_sft":      DIZEL_SFT_MIX,
    "mila_sft":       MILA_SFT_MIX,
}


# ── Format Converters ────────────────────────────────────────────────────

def to_pretrain_format(record: dict) -> dict:
    """Convert any record to pretrain format {"text": "..."}."""
    text = (record.get("text") or record.get("output") or
            record.get("response") or record.get("content") or "")

    # If it has instruction + output, combine them
    instruction = record.get("instruction", "")
    inp = record.get("input", "")
    output = record.get("output", "")

    if instruction and output and not text:
        parts = [instruction]
        if inp:
            parts.append(inp)
        parts.append(output)
        text = "\n\n".join(parts)

    if not text.strip():
        return None

    return {"text": text.strip()}


def to_sft_format(record: dict) -> dict:
    """Convert any record to SFT format {"instruction", "input", "output"}."""
    instruction = record.get("instruction", "")
    inp = record.get("input", "")
    output = record.get("output", "")

    # Handle different source formats
    if not instruction and "text" in record:
        # Plain text — skip for SFT
        return None

    if not instruction and "prompt" in record:
        instruction = record["prompt"]
        output = record.get("response", record.get("output", ""))

    if not instruction or not output:
        return None

    return {
        "instruction": instruction.strip(),
        "input": inp.strip() if inp else "",
        "output": output.strip(),
    }


def to_pretrain_text_file(records: list, output_path: str):
    """Write records as a plain text file for pretraining (one doc per double-newline)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            text = r.get("text", "")
            if text:
                f.write(text + "\n\n")
    print(f"  [write] Pretrain text file: {output_path} ({len(records):,} docs)")


# ── Sampling Logic ────────────────────────────────────────────────────────

def load_and_sample(input_dir: str, mixture: dict, seed: int = 42) -> list:
    """Load JSONL files according to mixture config with weighted oversampling."""
    rng = random.Random(seed)
    loaded_sources = {}

    for source_name, cfg in mixture.items():
        max_samples = cfg.get("max_samples", -1)

        # Try multiple paths
        candidates = [
            os.path.join(input_dir, f"{source_name}.jsonl"),
            os.path.join(input_dir, source_name, f"{source_name}.jsonl"),
        ]

        # Also check sft_data/ for identity files
        sft_candidates = [
            os.path.join("sft_data", f"{source_name}.jsonl"),
            os.path.join("data", "processed", f"{source_name}.jsonl"),
        ]
        candidates.extend(sft_candidates)

        found_path = None
        for path in candidates:
            if os.path.exists(path):
                found_path = path
                break

        if not found_path:
            print(f"  [skip] {source_name}: file not found")
            continue

        # Load records
        records = []
        with open(found_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Sample if needed
        if max_samples > 0 and len(records) > max_samples:
            rng.shuffle(records)
            records = records[:max_samples]

        loaded_sources[source_name] = records
        print(f"  [load] {source_name}: {len(records):,} samples (weight={cfg['weight']})")

    # Apply weighted oversampling
    if not loaded_sources:
        return []

    weights = {name: mixture[name]["weight"] for name in loaded_sources}
    min_weight = min(weights.values())

    all_records = []
    for source_name, records in loaded_sources.items():
        repeat = max(1, round(weights[source_name] / min_weight))
        all_records.extend(records * repeat)
        if repeat > 1:
            print(f"  [oversample] {source_name}: {repeat}x -> {len(records) * repeat:,} effective samples")

    rng.shuffle(all_records)
    return all_records


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Format Dizel v1.2.2 training data")
    parser.add_argument("--input", required=True, help="Input directory with cleaned JSONL files")
    parser.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--target", choices=["dizel", "mila"], default="dizel")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--txt", action="store_true", help="Also output as plain .txt for pretrain")
    args = parser.parse_args()

    # Select mixture
    mix_key = f"{args.target}_{args.mode}"
    if mix_key not in MIXTURES:
        print(f"ERROR: Unknown mixture '{mix_key}'")
        sys.exit(1)

    mixture = MIXTURES[mix_key]

    # Default output path
    if args.output is None:
        os.makedirs("data/formatted", exist_ok=True)
        args.output = f"data/formatted/{args.target}_{args.mode}.jsonl"

    print(f"\n{'='*60}")
    print(f"  Dizel v1.2.2 Data Formatter")
    print(f"  Target: {args.target}")
    print(f"  Mode: {args.mode}")
    print(f"  Sources: {len(mixture)}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # Load and sample
    records = load_and_sample(args.input, mixture, args.seed)
    print(f"\n  Total loaded: {len(records):,} records")

    # Convert format
    converter = to_pretrain_format if args.mode == "pretrain" else to_sft_format
    formatted = []
    dropped = 0
    for r in records:
        result = converter(r)
        if result:
            formatted.append(result)
        else:
            dropped += 1

    print(f"  Formatted: {len(formatted):,} ({dropped:,} dropped during conversion)")

    # Write JSONL
    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in formatted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Written: {args.output} ({size_mb:.1f} MB, {len(formatted):,} records)")

    # Optionally write as plain text for pretrain
    if args.txt and args.mode == "pretrain":
        txt_path = args.output.replace(".jsonl", ".txt")
        to_pretrain_text_file(formatted, txt_path)
        txt_size = os.path.getsize(txt_path) / (1024 * 1024)
        print(f"  Written: {txt_path} ({txt_size:.1f} MB)")

    # Stats
    if args.mode == "pretrain":
        total_chars = sum(len(r.get("text", "")) for r in formatted)
        est_tokens = total_chars / 4  # rough estimate: ~4 chars per token
        print(f"\n  Estimated tokens: ~{est_tokens/1e6:.0f}M")
    elif args.mode == "sft":
        avg_inst = sum(len(r.get("instruction", "")) for r in formatted) / max(1, len(formatted))
        avg_out = sum(len(r.get("output", "")) for r in formatted) / max(1, len(formatted))
        print(f"\n  Avg instruction length: {avg_inst:.0f} chars")
        print(f"  Avg output length: {avg_out:.0f} chars")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
