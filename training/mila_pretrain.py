"""
training/mila_pretrain.py — Next-token pretraining for Mila.

Reuses the same training infrastructure as Dizel (pretrain.py) but with
Mila-specific config (MilaConfig) and paths.

Usage:
    python training/mila_pretrain.py
    python training/mila_pretrain.py --resume checkpoints/mila-pretrain-step5000.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MILA_CONFIG, MilaConfig
from model.architecture import DizelLM  # Mila uses the same architecture class
from training.dataset import Tokenizer
from training.pretrain import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mila pre-training")
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--min_lr",         type=float, default=None)
    p.add_argument("--max_steps",      type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--grad_accum",     type=int,   default=None)
    p.add_argument("--data_path",      type=str,   default=None)
    p.add_argument("--resume",         type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_amp",         action="store_true",
                   help="Disable mixed precision")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = MILA_CONFIG

    # Apply CLI overrides
    if args.lr         is not None: cfg.pretrain.lr         = args.lr
    if args.min_lr     is not None: cfg.pretrain.min_lr     = args.min_lr
    if args.max_steps  is not None: cfg.pretrain.max_steps  = args.max_steps
    if args.batch_size is not None: cfg.pretrain.batch_size = args.batch_size
    if args.grad_accum is not None: cfg.pretrain.grad_accum = args.grad_accum
    if args.data_path  is not None: cfg.pretrain.data_path  = args.data_path
    if args.no_amp:                 cfg.pretrain.use_amp    = False

    # Override tokenizer path for Mila
    print(f"[mila] Using Mila config (d_model={cfg.model.d_model}, "
          f"layers={cfg.model.n_layers}, ctx={cfg.model.context_length})")
    print(f"[mila] Tokenizer: {cfg.tokenizer.model_path}")
    print(f"[mila] Data: {cfg.pretrain.data_path}")

    # Reuse Dizel's train() — it reads cfg.model and cfg.pretrain
    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
