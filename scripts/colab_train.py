"""
scripts/colab_train.py — Google Colab training script for Dizel v1.2 & Mila.

Copy-paste each section into a Colab cell.
Assumes: T4 GPU, Google Drive mounted at /content/drive.

Your Google Drive should have:
    MyDrive/Dizel/           ← the repo
    MyDrive/Dizel/sft_data/  ← your train.jsonl and valid.jsonl

Usage:
    1. Open Google Colab (GPU runtime → T4)
    2. Copy each cell below into separate Colab cells
    3. Run them in order
"""

# =============================================================================
# CELL 1: Setup — Mount Drive, Install Dependencies
# =============================================================================
CELL_1 = '''
# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Set project path ---
import os
PROJECT = "/content/drive/MyDrive/Dizel"  # adjust if different
os.chdir(PROJECT)
print(f"Working directory: {os.getcwd()}")

# --- Install dependencies ---
!pip install -q torch sentencepiece

# --- Verify GPU ---
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
'''

# =============================================================================
# CELL 2: Verify Config & Architecture
# =============================================================================
CELL_2 = '''
# --- Quick sanity check ---
!python config.py
!python model/registry.py
'''

# =============================================================================
# CELL 3: Prepare SFT Data (merge train.jsonl + valid.jsonl if needed)
# =============================================================================
CELL_3 = '''
import os

PROJECT = "/content/drive/MyDrive/Dizel"
SFT_DIR = os.path.join(PROJECT, "sft_data")

train_path = os.path.join(SFT_DIR, "train.jsonl")
valid_path = os.path.join(SFT_DIR, "valid.jsonl")
combined_path = os.path.join(SFT_DIR, "chat_v12.jsonl")

# The SFT loader expects a single file and splits internally.
# Merge train + valid into one file (the loader will re-split 90/10).
if os.path.exists(train_path):
    import shutil
    with open(combined_path, "w") as out:
        for src in [train_path, valid_path]:
            if os.path.exists(src):
                with open(src, "r") as f:
                    count = 0
                    for line in f:
                        line = line.strip()
                        if line:
                            out.write(line + "\\n")
                            count += 1
                print(f"  Merged {src}: {count} samples")

    total = sum(1 for _ in open(combined_path))
    print(f"\\nCombined SFT data: {total} samples -> {combined_path}")
else:
    print(f"Looking for data at: {train_path}")
    print("Files in sft_data/:")
    for f in os.listdir(SFT_DIR):
        size = os.path.getsize(os.path.join(SFT_DIR, f)) / 1024
        print(f"  {f} ({size:.1f} KB)")
'''

# =============================================================================
# CELL 4A: DIZEL v1.2 — Pretraining (skip if you already have a pretrain checkpoint)
# =============================================================================
CELL_4A = '''
# --- Dizel v1.2 Pretraining ---
# Skip this cell if you already have a pretrained checkpoint.
# This takes ~30-40 hours on T4 for 30k steps.

!python training/pretrain.py \\
    --data_path data/pretrain_v12.txt \\
    --batch_size 2 \\
    --grad_accum 32 \\
    --max_steps 30000 \\
    --lr 3e-5 \\
    --context_length 4096
'''

# =============================================================================
# CELL 4A-resume: Resume Dizel pretraining (if Colab disconnected)
# =============================================================================
CELL_4A_RESUME = '''
!python training/pretrain.py \\
    --resume checkpoints/dizel-v12-pretrain-best.pt \\
    --batch_size 2 \\
    --grad_accum 32 \\
    --max_steps 30000 \\
    --lr 3e-5
'''

# =============================================================================
# CELL 4B: DIZEL v1.2 — SFT Training
# =============================================================================
CELL_4B = '''
# --- Dizel v1.2 SFT ---
# Uses single combined file (--no_mix) since data is already merged.
# batch_size=1 + grad_accum=64 = effective batch of 64 (tight on T4 with 4096 ctx)

!python training/sft.py \\
    --base_checkpoint checkpoints/dizel-v12-pretrain-best.pt \\
    --no_mix \\
    --lr 2e-5 \\
    --max_steps 20000 \\
    --no_amp
'''

# =============================================================================
# CELL 4B-AMP: DIZEL v1.2 — SFT with mixed precision (faster, uses less VRAM)
# =============================================================================
CELL_4B_AMP = '''
# --- Dizel v1.2 SFT with AMP (recommended for T4) ---
# float16 AMP cuts VRAM usage ~40%, enabling larger effective batches.

import sys, os
sys.path.insert(0, os.getcwd())

# Override AMP settings before running
from config import CONFIG
CONFIG.sft.use_amp = True
CONFIG.sft.amp_dtype = "float16"
CONFIG.sft.batch_size = 1
CONFIG.sft.grad_accum = 64

!python training/sft.py \\
    --base_checkpoint checkpoints/dizel-v12-pretrain-best.pt \\
    --no_mix \\
    --lr 2e-5 \\
    --max_steps 20000
'''

# =============================================================================
# CELL 4B-resume: Resume Dizel SFT (if Colab disconnected)
# =============================================================================
CELL_4B_RESUME = '''
!python training/sft.py \\
    --resume checkpoints/dizel-v12-sft-best.pt \\
    --no_mix \\
    --max_steps 20000
'''

# =============================================================================
# CELL 5: MILA — Train Tokenizer (required before pretraining)
# =============================================================================
CELL_5 = '''
# --- Mila Tokenizer ---
# Only run this once. Needs mila_pretrain.txt corpus.
# If you don't have Mila's corpus yet, skip to the "Mila SFT Only" section.

!python tokenizer/train_mila_tokenizer.py \\
    --input data/mila_pretrain.txt \\
    --vocab_size 32000
'''

# =============================================================================
# CELL 6A: MILA — Pretraining
# =============================================================================
CELL_6A = '''
# --- Mila Pretraining ---
# ~19 hours on T4 for 20k steps (110M is lighter than Dizel)

!python training/mila_pretrain.py \\
    --batch_size 4 \\
    --grad_accum 16 \\
    --max_steps 20000 \\
    --lr 2e-4
'''

# =============================================================================
# CELL 6A-resume: Resume Mila pretraining
# =============================================================================
CELL_6A_RESUME = '''
!python training/mila_pretrain.py \\
    --resume checkpoints/mila-pretrain-best.pt \\
    --max_steps 20000
'''

# =============================================================================
# CELL 6B: MILA — SFT Training
# =============================================================================
CELL_6B = '''
# --- Mila SFT ---
# Shorter training (5k steps) since Mila's SFT set is smaller.
# batch_size=2 is fine for 110M model on T4.

!python training/mila_sft.py \\
    --base_checkpoint checkpoints/mila-pretrain-best.pt \\
    --lr 2e-5 \\
    --max_steps 5000
'''

# =============================================================================
# CELL 6B-resume: Resume Mila SFT
# =============================================================================
CELL_6B_RESUME = '''
!python training/mila_sft.py \\
    --resume checkpoints/mila-sft-best.pt \\
    --max_steps 5000
'''

# =============================================================================
# CELL 7: Verify checkpoints
# =============================================================================
CELL_7 = '''
import os
ckpt_dir = "checkpoints"
print("\\n=== Checkpoints ===\\n")
for f in sorted(os.listdir(ckpt_dir)):
    if f.endswith(".pt"):
        size_mb = os.path.getsize(os.path.join(ckpt_dir, f)) / (1024*1024)
        print(f"  {f:45s}  {size_mb:>8.1f} MB")
'''

# =============================================================================
# CELL 8: Quick test — generate text
# =============================================================================
CELL_8 = '''
import torch, sys, os
sys.path.insert(0, os.getcwd())
from config import CONFIG
from model.architecture import DizelLM
from training.dataset import Tokenizer

# Load model
tokenizer = Tokenizer()
CONFIG.model.vocab_size = len(tokenizer)
model = DizelLM(CONFIG.model)

# Load checkpoint
ckpt = torch.load("checkpoints/dizel-v12-sft-best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Generate
prompt = "Hello, how are you?"
ids = tokenizer.encode(prompt)
input_ids = torch.tensor([ids], dtype=torch.long)
output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
print(f"Prompt: {prompt}")
print(f"Response: {tokenizer.decode(output[0].tolist())}")
'''

# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Dizel v1.2 + Mila — Colab Training Commands")
    print("=" * 60)
    print()
    print("Copy each CELL into a separate Google Colab cell.")
    print("Run them in order. Use the -resume variants if Colab disconnects.")
    print()
    print("Training time estimates (T4 GPU):")
    print("  Dizel v1.2 pretrain : ~30-40 hours (30k steps)")
    print("  Dizel v1.2 SFT     : ~10-15 hours (20k steps)")
    print("  Mila pretrain       : ~19 hours (20k steps)")
    print("  Mila SFT            : ~2-3 hours (5k steps)")
    print()
    print("Total: ~60-75 hours across multiple Colab sessions")
    print("Use --resume to continue after disconnects.")
