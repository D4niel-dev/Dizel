#!/usr/bin/env python3
"""Run the fixed core benchmark suite against a local checkpoint.

    python scripts/run_scorecard.py checkpoints/Dizel-1.5-SFT.gguf
    python scripts/run_scorecard.py <checkpoint> --device cuda --json

The scorecard is stored under data/scorecards/ and shown in both clients for
as long as the checkpoint file is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dizel_core import scorecards  # noqa: E402
from evaluation import suite  # noqa: E402


def backend_for(checkpoint: str):
    from dizel_core.model_backend import GGUFBackend, NativeBackend, SafeTensorsBackend

    lowered = checkpoint.lower()
    if lowered.endswith(".gguf"):
        return GGUFBackend()
    if lowered.endswith(".safetensors"):
        return SafeTensorsBackend()
    return NativeBackend()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a .gguf, .safetensors, or .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Generation ceiling per case (default: 128)",
    )
    parser.add_argument("--json", action="store_true", help="Print the record as JSON")
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        print(f"Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    backend = backend_for(checkpoint)
    try:
        backend.load(
            checkpoint,
            args.device,
            lambda stage: print(f"[load] {stage}", file=sys.stderr),
        )
    except Exception as error:
        print(f"Could not load the checkpoint: {error}", file=sys.stderr)
        return 1

    def handler(prompt: str) -> str:
        return backend.generate(
            [{"role": "user", "content": prompt}],
            "",
            {},
            args.max_tokens,
            lambda token: None,
        )

    try:
        card = suite.run_core_suite(handler, checkpoint=os.path.basename(checkpoint))
        record = scorecards.save(checkpoint, card)
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(scorecards.report(record))
        print(f"\nSaved to {os.path.join(scorecards.SCORECARDS_DIR, record['fingerprint'] + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
