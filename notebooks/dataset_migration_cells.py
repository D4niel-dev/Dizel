"""
notebooks/dataset_migration_cells.py

Copy each section (marked with # %% [markdown] or # %%) into a Google Colab
notebook cell. This file is structured to be Colab-compatible.

Run order: Cell 1 -> Cell 2 -> ... -> Cell 8
"""

# %% [markdown]
# # Dizel Dataset Migration Pipeline
#
# Converts 8 HuggingFace datasets + existing handcrafted SFT data into a
# single unified JSONL format for SFT training.
#
# **Requirements**: Run on Google Colab with Google Drive mounted.

# %% Cell 1: Install Dependencies
# !pip install pandas pyarrow tqdm datasketch

# %% Cell 2: Mount Drive & Setup
import sys
import os

from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/Dizel'
sys.path.insert(0, BASE)

# Verify directory exists
assert os.path.isdir(BASE), f"Dizel project not found at {BASE}"
assert os.path.isdir(os.path.join(BASE, "data", "raw")), "data/raw/ not found"

# Create output directories
for d in ['data/processed', 'data/exports']:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)

print(f"Base: {BASE}")
print(f"Raw datasets: {os.listdir(os.path.join(BASE, 'data', 'raw'))}")
print("Setup complete!")

# %% Cell 3: Import Pipeline (always reloads latest from Drive)
import importlib
sys.path.insert(0, os.path.join(BASE, "scripts"))
try:
    import migrate_datasets
    importlib.reload(migrate_datasets)
except ModuleNotFoundError:
    pass
from migrate_datasets import *
print("Pipeline loaded ✓")

# %% Cell 4: Run Full Pipeline
# This runs the entire migration end-to-end.
# For large datasets (openorca, ultrachat) this may take 30-60 minutes.

all_samples = run_full_pipeline(BASE)

# %% Cell 5: (Optional) Interactive Inspection
# Use this cell to inspect specific datasets or debug issues.

import json

# Read and preview the exported file
exports_dir = os.path.join(BASE, "data", "exports")

for fname in ["train.jsonl", "valid.jsonl"]:
    fpath = os.path.join(exports_dir, fname)
    if os.path.exists(fpath):
        count = sum(1 for _ in open(fpath))
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"{fname}: {count:,} samples ({size_mb:.1f} MB)")

# Preview 5 random samples from train
print("\n--- Sample Preview ---")
with open(os.path.join(exports_dir, "train.jsonl")) as f:
    lines = f.readlines()

import random
random.seed(42)
for line in random.sample(lines, min(5, len(lines))):
    obj = json.loads(line)
    n_turns = len(obj["messages"])
    user_preview = ""
    asst_preview = ""
    for m in obj["messages"]:
        if m["role"] == "user" and not user_preview:
            user_preview = m["content"][:80]
        if m["role"] == "assistant" and not asst_preview:
            asst_preview = m["content"][:80]
    print(f"  [{n_turns} turns] User: {user_preview}...")
    print(f"            Asst: {asst_preview}...")
    print()

# %% Cell 6: (Optional) Verify Compatibility with SFTDataset
# Smoke test to confirm the output works with training/dataset.py

sys.path.insert(0, BASE)
from training.dataset import Tokenizer, SFTDataset

tok = Tokenizer()
train_path = os.path.join(exports_dir, "train.jsonl")
ds = SFTDataset(train_path, tok, context_length=2048)

print(f"SFTDataset loaded: {len(ds)} samples")
x, y, mask = ds[0]
print(f"Sample 0: x.shape={x.shape}, y.shape={y.shape}, mask.shape={mask.shape}")
print(f"Decoded x[:50]: {tok.decode(x[:50])}")

# %% Cell 7: (Optional) Category/Source Statistics
from collections import Counter

# Reload with metadata
all_data = []
with open(os.path.join(exports_dir, "dizel_migrated.jsonl")) as f:
    for line in f:
        all_data.append(json.loads(line))

# Since we strip metadata in exports, re-count from processed files
processed_dir = os.path.join(BASE, "data", "processed")
if os.path.isdir(processed_dir):
    for fname in os.listdir(processed_dir):
        if fname.endswith(".jsonl"):
            count = sum(1 for _ in open(os.path.join(processed_dir, fname)))
            print(f"  {fname}: {count:,}")

print(f"\nTotal exported: {len(all_data):,}")

# %% Cell 8: Done!
print("=" * 50)
print("  MIGRATION COMPLETE!")
print(f"  Output: {exports_dir}")
print(f"  Files:")
for f in os.listdir(exports_dir):
    size = os.path.getsize(os.path.join(exports_dir, f)) / (1024*1024)
    print(f"    {f}: {size:.1f} MB")
print("=" * 50)
print("\nNext step: Run SFT training with:")
print(f"  python training/diz-sft.py --base_checkpoint <your-pretrain-ckpt>")
print(f"\nUpdate config.py sft_data_path to point to:")
print(f"  {os.path.join(exports_dir, 'train.jsonl')}")
