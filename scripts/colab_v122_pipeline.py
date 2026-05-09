"""
scripts/colab_v122_pipeline.py — Full v1.2.2 training pipeline for Google Colab T4.

Run this in a Colab cell after cloning the repo:

    !git clone https://github.com/d4niel-dev/dizel.git
    %cd dizel
    !pip install sentencepiece datasets torch
    !python scripts/colab_v122_pipeline.py --phase A
    !python scripts/colab_v122_pipeline.py --phase B
    !python scripts/colab_v122_pipeline.py --phase C

Phases:
    A — Data Foundation: download, clean, format
    B — Pretraining: Stage 1 (general) + Stage 2 (emphasis)
    C — SFT: Instruction tuning
    D — DPO: Preference optimization (future)
"""

import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PYTHON = sys.executable


def run(cmd: str, desc: str = ""):
    """Run a shell command with live output."""
    print(f"\n{'='*60}")
    print(f"  {desc}" if desc else f"  Running: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        print(f"  Command: {cmd}")
        return False
    return True


def check_gpu():
    """Check GPU availability and print info."""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[GPU] {gpu_name} ({vram:.1f} GB VRAM)")
        return True
    else:
        print("[GPU] No CUDA GPU detected. Training will be very slow on CPU.")
        return False


def check_disk():
    """Check available disk space."""
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"[Disk] {free_gb:.1f} GB free")
    if free_gb < 5:
        print("[WARNING] Less than 5 GB free. Consider cleaning up data/raw/openorca/")
    return free_gb


# ── Phase A: Data Foundation ──────────────────────────────────────────────

def phase_a(target: str = "dizel"):
    """Download, clean, and format training data."""
    print("\n" + "=" * 60)
    print("  PHASE A — DATA FOUNDATION")
    print("=" * 60)

    check_disk()

    # Step 1: Tokenizer audit
    print("\n[Phase A.1] Tokenizer Audit")
    run(f"{PYTHON} scripts/tokenizer_audit.py", "Running tokenizer audit...")

    # Step 2: Download datasets
    print("\n[Phase A.2] Downloading datasets")
    run(f"{PYTHON} scripts/download_v122_data.py --target {target}", "Downloading training data...")

    # Step 3: Clean data
    print("\n[Phase A.3] Cleaning data")
    run(f"{PYTHON} scripts/clean_v122_data.py --input data/raw/ --all --output data/cleaned/",
        "Cleaning and filtering...")

    # Step 4: Format for training
    print("\n[Phase A.4] Formatting data")
    run(f"{PYTHON} scripts/format_v122_data.py --input data/cleaned/ --mode pretrain --target {target} --txt",
        f"Formatting pretrain data for {target}...")
    run(f"{PYTHON} scripts/format_v122_data.py --input data/cleaned/ --mode sft --target {target}",
        f"Formatting SFT data for {target}...")

    print("\n[Phase A] COMPLETE!")
    print("  Next: python scripts/colab_v122_pipeline.py --phase B")


# ── Phase B: Pretraining ─────────────────────────────────────────────────

def phase_b(target: str = "dizel", resume: str = None):
    """Run pretraining: Stage 1 (general) + Stage 2 (emphasis)."""
    print("\n" + "=" * 60)
    print("  PHASE B — PRETRAINING")
    print("=" * 60)

    has_gpu = check_gpu()
    check_disk()

    if not has_gpu:
        print("[ERROR] GPU required for pretraining. Run this in Colab with T4.")
        return

    # Check that formatted data exists
    pretrain_data = f"data/formatted/{target}_pretrain.txt"
    if not os.path.exists(pretrain_data):
        print(f"[ERROR] Pretrain data not found at {pretrain_data}")
        print("  Run Phase A first: python scripts/colab_v122_pipeline.py --phase A")
        return

    data_size_mb = os.path.getsize(pretrain_data) / (1024 * 1024)
    print(f"[Phase B] Pretrain data: {pretrain_data} ({data_size_mb:.1f} MB)")

    # Stage 1: General pretraining
    print("\n[Phase B.1] Stage 1 — General Pretraining")
    resume_flag = f"--resume {resume}" if resume else ""
    amp_flag = "--no_amp" if not has_gpu else ""

    cmd = (
        f"{PYTHON} training/pretrain.py "
        f"--data_path {pretrain_data} "
        f"--max_steps 15000 "
        f"--lr 3e-5 "
        f"--batch_size 4 "
        f"--grad_accum 16 "
        f"{resume_flag} {amp_flag}"
    )
    run(cmd, "Stage 1: General pretraining (15K steps)...")

    # Stage 2: Reasoning/Code emphasis (continued from Stage 1)
    best_ckpt = "checkpoints/dizel-pretrain-best.pt"
    if os.path.exists(best_ckpt):
        print("\n[Phase B.2] Stage 2 — Reasoning & Code Emphasis")
        # For Stage 2, we'd use a resampled corpus with more code/math
        # For now, continue training with lower LR
        cmd = (
            f"{PYTHON} training/pretrain.py "
            f"--data_path {pretrain_data} "
            f"--max_steps 20000 "
            f"--lr 1e-5 "
            f"--batch_size 4 "
            f"--grad_accum 16 "
            f"--resume {best_ckpt}"
        )
        run(cmd, "Stage 2: Emphasis pass (5K more steps, lower LR)...")
    else:
        print(f"[WARN] No best checkpoint found at {best_ckpt}, skipping Stage 2")

    print("\n[Phase B] COMPLETE!")
    print("  Next: python scripts/colab_v122_pipeline.py --phase C")


# ── Phase C: SFT ─────────────────────────────────────────────────────────

def phase_c(target: str = "dizel", base_checkpoint: str = None):
    """Run Supervised Fine-Tuning."""
    print("\n" + "=" * 60)
    print("  PHASE C — SUPERVISED FINE-TUNING (SFT)")
    print("=" * 60)

    has_gpu = check_gpu()
    if not has_gpu:
        print("[ERROR] GPU required. Run this in Colab with T4.")
        return

    # Find base checkpoint
    if base_checkpoint is None:
        candidates = [
            "checkpoints/dizel-pretrain-best.pt",
            "checkpoints/dizel-pretrain-final.pt",
            "checkpoints/dizel-lite-best.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                base_checkpoint = c
                break

    if not base_checkpoint or not os.path.exists(base_checkpoint):
        print("[ERROR] No base checkpoint found. Run Phase B first.")
        return

    # Check SFT data
    sft_data = f"data/formatted/{target}_sft.jsonl"
    if not os.path.exists(sft_data):
        # Fall back to existing SFT data
        sft_data = "sft_data/chat_expanded.jsonl"

    if not os.path.exists(sft_data):
        print(f"[ERROR] SFT data not found. Run Phase A first.")
        return

    data_size_mb = os.path.getsize(sft_data) / (1024 * 1024)
    print(f"[Phase C] SFT data: {sft_data} ({data_size_mb:.1f} MB)")
    print(f"[Phase C] Base checkpoint: {base_checkpoint}")

    cmd = (
        f"{PYTHON} training/sft.py "
        f"--sft_data {sft_data} "
        f"--base_checkpoint {base_checkpoint} "
        f"--max_steps 10000 "
        f"--lr 2e-5 "
        f"--batch_size 2 "
        f"--grad_accum 16"
    )
    run(cmd, "SFT training (10K steps)...")

    print("\n[Phase C] COMPLETE!")
    print("  Your fine-tuned model is in checkpoints/")
    print("  Test it: python inference/dizel_ui/main.py --checkpoint checkpoints/dizel-sft-best.pt")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dizel v1.2.2 Training Pipeline")
    parser.add_argument("--phase", choices=["A", "B", "C", "D", "all"], required=True,
                        help="Which phase to run")
    parser.add_argument("--target", choices=["dizel", "mila"], default="dizel",
                        help="Which model to train")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    t0 = time.time()

    print(f"\n{'#'*60}")
    print(f"  DIZEL v1.2.2 TRAINING PIPELINE")
    print(f"  Phase: {args.phase}")
    print(f"  Target: {args.target}")
    print(f"{'#'*60}")

    if args.phase == "A" or args.phase == "all":
        phase_a(args.target)

    if args.phase == "B" or args.phase == "all":
        phase_b(args.target, args.resume)

    if args.phase == "C" or args.phase == "all":
        phase_c(args.target, args.resume)

    elapsed = time.time() - t0
    hours = elapsed / 3600
    print(f"\n[Pipeline] Phase {args.phase} completed in {hours:.1f} hours ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
