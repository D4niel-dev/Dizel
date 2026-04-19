"""
training/mila_sft.py — Supervised Fine-Tuning for Mila chat format.

Reuses the same SFT infrastructure as Dizel (sft.py) but with
Mila-specific config (MilaConfig) and data paths.

Mila's SFT data emphasizes:
  - Conversational / friendly data (35%)
  - General knowledge in warm tone (20%)
  - Coding help with encouragement (20%)
  - Creative / brainstorming (15%)
  - Identity anchors (10%, hand-written)

Usage:
    python training/mila_sft.py --base_checkpoint checkpoints/mila-pretrain-best.pt
    python training/mila_sft.py --resume checkpoints/mila-sft-step500.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MILA_CONFIG
from training.sft import sft_train


def parse_args():
    p = argparse.ArgumentParser(description="Mila SFT")
    p.add_argument("--base_checkpoint", type=str, required=False, default="",
                   help="Path to Mila pretrained checkpoint (.pt)")
    p.add_argument("--resume", type=str, default="",
                   help="Path to a Mila SFT checkpoint (.pt) to resume from")
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--max_steps", type=int,   default=None)
    p.add_argument("--no_amp",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MILA_CONFIG

    if not args.base_checkpoint and not args.resume:
        print("Error: Must provide either --base_checkpoint or --resume")
        sys.exit(1)

    if args.lr        is not None: cfg.sft.lr        = args.lr
    if args.max_steps is not None: cfg.sft.max_steps = args.max_steps
    if args.no_amp:                cfg.sft.use_amp   = False
    if args.base_checkpoint:       cfg.sft.base_checkpoint = args.base_checkpoint

    # Disable mixed dataset loading for Mila — uses single sft file
    cfg._use_mix = False

    print(f"[mila-sft] Using Mila config (d_model={cfg.model.d_model}, "
          f"layers={cfg.model.n_layers}, ctx={cfg.model.context_length})")
    print(f"[mila-sft] SFT data: {cfg.sft.sft_data_path}")

    target_ckpt = args.resume if args.resume else args.base_checkpoint
    sft_train(cfg, target_ckpt)


if __name__ == "__main__":
    main()
