"""
scripts/download_v122_data.py — Download training datasets for Dizel v1.2.2.

Downloads from HuggingFace using streaming to handle Colab disk constraints.
Saves raw data as JSONL to data/raw/{source_name}/.

Usage (Colab):
    !pip install datasets
    !python scripts/download_v122_data.py --target dizel
    !python scripts/download_v122_data.py --target mila
    !python scripts/download_v122_data.py --target all
    !python scripts/download_v122_data.py --target dizel --source fineweb-edu
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Dataset Registry ──────────────────────────────────────────────────────

DATASETS = {
    # ─── Pretrain Sources ────────────────────────────────────────────
    "fineweb-edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "streaming": True,
        "max_samples": 500_000,  # ~800M tokens at ~1600 tok/sample avg
        "text_field": "text",
        "filter_fn": lambda x: x.get("score", 0) >= 3 and 100 < len(x.get("text", "")) < 32000,
        "targets": ["dizel", "mila"],
        "domain": "knowledge",
        "description": "High-quality educational web text (score≥3)",
    },
    "finemath": {
        "hf_path": "HuggingFaceTB/finemath",
        "hf_name": "finemath-4plus",
        "split": "train",
        "streaming": True,
        "max_samples": 100_000,  # ~100M tokens
        "text_field": "text",
        "filter_fn": lambda x: len(x.get("text", "")) > 50,
        "targets": ["dizel"],
        "domain": "math",
        "description": "Mathematical reasoning and problem solving",
    },
    "stack-edu-python": {
        "hf_path": "HuggingFaceTB/stack-edu",
        "hf_name": "Python",
        "split": "train",
        "streaming": True,
        "max_samples": 80_000,
        "text_field": "text",
        "filter_fn": lambda x: (
            x.get("max_score", 0) >= 3
            and 50 < len(x.get("text", "")) < 32000
        ),
        "targets": ["dizel", "mila"],
        "domain": "code",
        "description": "Educational Python code (score≥3)",
    },
    "stack-edu-javascript": {
        "hf_path": "HuggingFaceTB/stack-edu",
        "hf_name": "JavaScript",
        "split": "train",
        "streaming": True,
        "max_samples": 40_000,
        "text_field": "text",
        "filter_fn": lambda x: (
            x.get("max_score", 0) >= 3
            and 50 < len(x.get("text", "")) < 32000
        ),
        "targets": ["dizel", "mila"],
        "domain": "code",
        "description": "Educational JavaScript code (score≥3)",
    },
    "stack-edu-typescript": {
        "hf_path": "HuggingFaceTB/stack-edu",
        "hf_name": "TypeScript",
        "split": "train",
        "streaming": True,
        "max_samples": 30_000,
        "text_field": "text",
        "filter_fn": lambda x: (
            x.get("max_score", 0) >= 3
            and 50 < len(x.get("text", "")) < 32000
        ),
        "targets": ["dizel"],
        "domain": "code",
        "description": "Educational TypeScript code (score≥3)",
    },
    "cosmopedia": {
        "hf_path": "HuggingFaceTB/cosmopedia",
        "hf_name": "web_samples_v2",
        "split": "train",
        "streaming": True,
        "max_samples": 80_000,  # ~100M tokens
        "text_field": "text",
        "filter_fn": lambda x: 200 < len(x.get("text", "")) < 16000,
        "targets": ["dizel", "mila"],
        "domain": "synthetic_knowledge",
        "description": "Synthetic textbook-style knowledge",
    },

    # ─── SFT / Instruction Sources ───────────────────────────────────
    "ultrafeedback": {
        "hf_path": "openbmb/UltraFeedback",
        "split": "train",
        "streaming": True,
        "max_samples": 30_000,
        "text_field": None,  # multi-field
        "extract_fn": "ultrafeedback",
        "targets": ["dizel"],
        "domain": "preference",
        "description": "Preference pairs for DPO (chosen/rejected)",
    },
    "helpsteer2": {
        "hf_path": "nvidia/HelpSteer2",
        "split": "train",
        "streaming": True,
        "max_samples": 10_000,
        "text_field": None,
        "extract_fn": "helpsteer",
        "targets": ["dizel"],
        "domain": "preference",
        "description": "NVIDIA helpfulness/safety preference data",
    },
}


# ── Extraction Functions ──────────────────────────────────────────────────

def extract_text_default(sample, text_field="text"):
    """Default: just grab the text field."""
    text = sample.get(text_field, "")
    if isinstance(text, str) and text.strip():
        return {"text": text.strip()}
    return None


def extract_ultrafeedback(sample):
    """Extract chosen/rejected pairs from UltraFeedback."""
    instruction = sample.get("instruction", "")
    completions = sample.get("completions", [])
    if not instruction or len(completions) < 2:
        return None

    # Sort by overall_score descending
    scored = []
    for c in completions:
        annotations = c.get("annotations", {})
        scores = []
        for ann in annotations.values():
            if isinstance(ann, dict) and "Rating" in ann:
                try:
                    scores.append(float(ann["Rating"]))
                except (ValueError, TypeError):
                    pass
        if scores:
            scored.append((c.get("response", ""), sum(scores) / len(scores)))

    if len(scored) < 2:
        return None

    scored.sort(key=lambda x: -x[1])
    return {
        "instruction": instruction,
        "chosen": scored[0][0],
        "rejected": scored[-1][0],
        "chosen_score": scored[0][1],
        "rejected_score": scored[-1][1],
    }


def extract_helpsteer(sample):
    """Extract from HelpSteer2 format."""
    prompt = sample.get("prompt", "")
    response = sample.get("response", "")
    helpfulness = sample.get("helpfulness", 0)
    if not prompt or not response:
        return None
    return {
        "instruction": prompt,
        "output": response,
        "helpfulness_score": helpfulness,
    }


EXTRACTORS = {
    "ultrafeedback": extract_ultrafeedback,
    "helpsteer": extract_helpsteer,
}


# ── Download Logic ────────────────────────────────────────────────────────

def download_dataset(name: str, config: dict, output_dir: str):
    """Download a single dataset using HF streaming."""
    from datasets import load_dataset

    out_path = os.path.join(output_dir, name)
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{name}.jsonl")

    # Check if already downloaded
    if os.path.exists(out_file):
        existing_lines = sum(1 for _ in open(out_file, "r", encoding="utf-8"))
        if existing_lines >= config["max_samples"]:
            print(f"  [skip] {name}: already has {existing_lines:,} samples")
            return existing_lines
        print(f"  [resume] {name}: has {existing_lines:,}, need {config['max_samples']:,}")
        # Continue from where we left off
        skip_count = existing_lines
    else:
        skip_count = 0

    print(f"  [download] {name}: {config['description']}")
    print(f"    Source: {config['hf_path']}")
    print(f"    Target: {config['max_samples']:,} samples")

    # Load with streaming
    load_kwargs = {
        "path": config["hf_path"],
        "split": config.get("split", "train"),
        "streaming": config.get("streaming", True),
    }
    if "hf_name" in config:
        load_kwargs["name"] = config["hf_name"]

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"  [ERROR] Failed to load {name}: {e}")
        return 0

    # Extract and save
    extract_fn_name = config.get("extract_fn")
    text_field = config.get("text_field", "text")
    filter_fn = config.get("filter_fn")
    max_samples = config["max_samples"]

    count = 0
    skipped = 0
    filtered = 0
    t0 = time.time()

    mode = "a" if skip_count > 0 else "w"
    with open(out_file, mode, encoding="utf-8") as f:
        for sample in ds:
            # Skip already-saved samples on resume
            if skipped < skip_count:
                skipped += 1
                continue

            # Apply filter
            if filter_fn and not filter_fn(sample):
                filtered += 1
                continue

            # Extract
            if extract_fn_name and extract_fn_name in EXTRACTORS:
                record = EXTRACTORS[extract_fn_name](sample)
            elif text_field:
                record = extract_text_default(sample, text_field)
            else:
                record = None

            if record is None:
                filtered += 1
                continue

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 10_000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                print(f"    ... {count + skip_count:,}/{max_samples:,} saved ({rate:.0f} samples/s, {filtered:,} filtered)")

            if count + skip_count >= max_samples:
                break

    elapsed = time.time() - t0
    total = count + skip_count
    print(f"  [done] {name}: {total:,} samples saved ({elapsed:.1f}s, {filtered:,} filtered out)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Download v1.2.2 training datasets")
    parser.add_argument("--target", choices=["dizel", "mila", "all"], default="all",
                        help="Which model to download data for")
    parser.add_argument("--source", default=None,
                        help="Download only a specific source (e.g., fineweb-edu)")
    parser.add_argument("--output", default="data/raw",
                        help="Output directory")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for name, cfg in DATASETS.items():
            targets = ", ".join(cfg["targets"])
            print(f"  {name:20s}  {cfg['domain']:20s}  {cfg['max_samples']:>10,} samples  [{targets}]")
            print(f"    {cfg['description']}")
        return

    # Filter datasets by target model
    to_download = {}
    for name, cfg in DATASETS.items():
        if args.source and name != args.source:
            continue
        if args.target == "all" or args.target in cfg["targets"]:
            to_download[name] = cfg

    if not to_download:
        print(f"No datasets match target='{args.target}'" +
              (f", source='{args.source}'" if args.source else ""))
        return

    print(f"\n{'='*60}")
    print(f"  Dizel v1.2.2 Dataset Downloader")
    print(f"  Target: {args.target}")
    print(f"  Datasets: {len(to_download)}")
    print(f"  Output: {args.output}/")
    print(f"{'='*60}\n")

    results = {}
    for name, cfg in to_download.items():
        try:
            count = download_dataset(name, cfg, args.output)
            results[name] = count
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results[name] = 0

    # Summary
    print(f"\n{'='*60}")
    print(f"  Download Summary")
    print(f"{'='*60}")
    total = 0
    for name, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name:20s}  {count:>10,} samples")
        total += count
    print(f"  {'─'*40}")
    print(f"    Total: {total:,} samples")
    print()


if __name__ == "__main__":
    main()
