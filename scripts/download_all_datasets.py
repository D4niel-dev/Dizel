"""
scripts/download_all_datasets.py — Download ALL missing HuggingFace datasets for migration.

Run this before migrate_datasets.py. It checks data/raw/ for each dataset,
and downloads any that are missing or empty.

Usage:
    cd /content/drive/MyDrive/Dizel
    python scripts/download_all_datasets.py
"""

import os
import sys
from datasets import load_dataset

PROJECT = os.getcwd()
RAW_DIR = os.path.join(PROJECT, "data", "raw")

# ── Dataset manifest ─────────────────────────────────────────────────────────
# Each entry: (folder, hf_name, split, filename, description, max_samples)
# max_samples=0 means no cap
DATASETS = [
    # Phase 2: Instruction
    ("alpaca_gpt4",  "tatsu-lab/alpaca",                "train",      "train.parquet",  "Alpaca (52k instructions)",       0),
    ("codealpaca",   "sahil2801/CodeAlpaca-20k",        "train",      "train.parquet",  "CodeAlpaca (20k code instructions)", 0),
    ("coedit",       "grammarly/coedit",                 "train",      "train.parquet",  "CoEdIT (69k text editing)",       0),
    ("dolly",        "databricks/databricks-dolly-15k",  "train",      "train.parquet",  "Dolly (15k instruction-following)", 0),

    # Phase 3: Conversational
    ("oasst2",       "OpenAssistant/oasst2",             "train",      "train.parquet",  "OASST2 (conversational trees)",   0),
    ("ultrachat",    "HuggingFaceH4/ultrachat_200k",     "train_sft",  "train.parquet",  "UltraChat (200k multi-turn)",     0),

    # Phase 4: Complex
    ("openorca",     "Open-Orca/OpenOrca",               "train",      "train.parquet",  "OpenOrca (capped to 200k)",       200_000),
    ("codefeedback", "HuggingFaceH4/Code-Feedback",       "train_sft",  "train.parquet",  "CodeFeedback (code quality)",     0),
]


def dataset_size(path: str) -> int:
    """Return number of parquet rows without loading full dataset."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return pf.metadata.num_rows
    except Exception:
        return -1


def needs_download(folder: str) -> bool:
    """Check if the dataset folder is missing or has no valid parquet files."""
    path = os.path.join(RAW_DIR, folder)
    if not os.path.isdir(path):
        return True
    # Look for any .parquet files directly inside
    for fname in os.listdir(path):
        if fname.endswith(".parquet") and not fname.startswith("."):
            fpath = os.path.join(path, fname)
            rows = dataset_size(fpath)
            if rows > 0:
                print(f"  ✓ {folder} — {rows:,} rows ({fname})")
                return False
    return True


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Dizel Dataset Downloader")
    print(f"Raw dir: {RAW_DIR}")
    print(f"{'='*60}")

    pending = []
    for folder, hf_name, split, fname, desc, max_samples in DATASETS:
        dirpath = os.path.join(RAW_DIR, folder)
        os.makedirs(dirpath, exist_ok=True)

        if needs_download(folder):
            pending.append((folder, hf_name, split, fname, desc, max_samples))
        else:
            pass  # already has data

    if not pending:
        print("\n✅ All datasets already downloaded. Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"Need to download {len(pending)} dataset(s):")
    for folder, hf_name, _, _, desc, max_samples in pending:
        cap = f" (max {max_samples:,})" if max_samples else ""
        print(f"  • {folder:15s} → {desc}{cap}")
    print(f"{'='*60}\n")

    for folder, hf_name, split, fname, desc, max_samples in pending:
        out_path = os.path.join(RAW_DIR, folder, fname)
        print(f"\n[{folder}] Downloading {hf_name} ({split})...")
        # Try configured split, then fallbacks
        split_candidates = [split, "train", "train_sft"]
        ds = None
        for sp in split_candidates:
            try:
                ds = load_dataset(hf_name, split=sp, trust_remote_code=True)
                if ds is not None and len(ds) > 0:
                    print(f"  Using split '{sp}'")
                    break
            except Exception as e:
                print(f"  ⚠ Split '{sp}' failed: {e}")
        if ds is None or len(ds) == 0:
            print(f"  ✗ Could not load any split for {hf_name}")
            continue

        # Cap at max_samples if set (e.g. OpenOrca 4M → 200k)
        if max_samples > 0 and len(ds) > max_samples:
            ds = ds.select(range(max_samples))
            print(f"  Capped to {max_samples:,} rows")

        ds.to_parquet(out_path)
        print(f"  ✓ {desc} — {len(ds):,} rows → {out_path}")

    print(f"\n{'='*60}")
    print("Done. Now run:")
    print("  python scripts/migrate_datasets.py --base_dir .")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
