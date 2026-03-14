"""
utils/test_model.py — Task 9: Automated evaluation against target prompts.

Tests the model on the exact prompts from the specification and checks:
  - Response is not empty
  - Response has no excessive character repetition (gibberish)
  - Response does not repeat words excessively
  - Response ends before hitting max_new_tokens (model stopped cleanly)
  - Response length is reasonable (not too short, not too long)

Usage
-----
    # Test SFT checkpoint (recommended)
    python utils/test_model.py --checkpoint checkpoints/dizel-v1-sft-best.pt

    # Test pretrain checkpoint
    python utils/test_model.py --checkpoint checkpoints/dizel-v1-pretrain-best.pt

    # Adjust sampling
    python utils/test_model.py --checkpoint ... --temperature 0.5 --top_k 30

    # Verbose: show full response
    python utils/test_model.py --checkpoint ... --verbose
"""

import argparse
import math
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG, SPECIAL
from model.architecture import DizelLM
from training.dataset import Tokenizer
from utils.data_cleaner import filter_generated_output


# ---------------------------------------------------------------------------
# Task 9: target prompts and minimum expected content
# ---------------------------------------------------------------------------
TEST_CASES = [
    # (user_prompt, [keywords expected in a good response])
    ("Hi",                     []),
    ("Hello",                  []),
    ("Hey",                    []),
    ("Good morning",           []),
    ("How are you?",           []),
    ("What is programming?",   ["program", "code", "computer", "language", "instruction"]),
    ("Explain Python in simple terms.",
                               ["python", "language", "program", "simple", "code", "web", "data"]),
    ("What is machine learning?",
                               ["learn", "data", "model", "pattern", "ai", "train"]),
    ("What is a neural network?",
                               ["neuron", "layer", "learn", "model", "network"]),
    ("What is the speed of light?",
                               ["light", "speed", "kilometer", "second"]),
    ("Who are you?",           ["dizel", "ai", "assistant", "language"]),
    ("What can you do?",       ["help", "answer", "question", "assist"]),
    ("Thank you",              ["welcome"]),
    ("Goodbye",                ["goodbye", "bye", "day"]),
]

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def has_gibberish(text: str, max_char_run: int = 5, max_word_run: int = 3) -> bool:
    """Return True if the text appears to contain gibberish."""
    if re.search(r"(.)\1{" + str(max_char_run) + r",}", text):
        return True
    words = text.lower().split()
    for i in range(len(words) - max_word_run + 1):
        if len(set(words[i: i + max_word_run])) == 1:
            return True
    return False


def keywords_found(text: str, keywords: list) -> bool:
    if not keywords:
        return True
    text_lower = text.lower()
    # At least half the keywords must appear
    found = sum(1 for k in keywords if k in text_lower)
    return found >= max(1, len(keywords) // 2)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        print(f"[test] ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("model_cfg", CONFIG.model)
    model = DizelLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    step = ckpt.get("step", "?")
    vl   = ckpt.get("val_loss", float("inf"))
    print(f"[test] Checkpoint : {ckpt_path}  (step={step}, val_loss={vl:.4f})")
    print(f"[test] Model      : {model}\n")
    return model


def build_prompt(user_msg: str, tokenizer: Tokenizer) -> list:
    """Build a single-turn prompt in the v1 chat format."""
    text = (
        SPECIAL.USER_START + "\n" +
        user_msg + "\n" +
        SPECIAL.USER_END +
        SPECIAL.ASST_START + "\n"
    )
    return [tokenizer.bos_id] + tokenizer.encode(text)


def get_eos_ids(tokenizer: Tokenizer) -> list:
    unk = tokenizer.unk_id
    ids = [tokenizer.eos_id]
    for sym in [SPECIAL.EOS, SPECIAL.EOS_TAG, SPECIAL.ASST_END]:
        tid = tokenizer.sp.piece_to_id(sym)
        if tid != unk:
            ids.append(tid)
    return list(set(ids))


@torch.no_grad()
def generate_response(
    model:      DizelLM,
    tokenizer:  Tokenizer,
    prompt:     str,
    device:     str,
    max_new:    int,
    temperature: float,
    top_k:      int,
    top_p:      float,
    rep_pen:    float,
    eos_ids:    list,
) -> tuple:
    """Return (response_text, stopped_cleanly, n_tokens, elapsed_s)."""
    ids  = build_prompt(prompt, tokenizer)
    idx  = torch.tensor([ids], dtype=torch.long, device=device)
    t0   = time.time()
    out  = model.generate(
        idx,
        max_new_tokens     = max_new,
        temperature        = temperature,
        top_k              = top_k,
        top_p              = top_p,
        repetition_penalty = rep_pen,
        eos_ids            = eos_ids,
    )
    elapsed = time.time() - t0
    gen_ids = out[0, len(ids):].tolist()

    # Check if model stopped cleanly (shorter than max)
    stopped_cleanly = len(gen_ids) < max_new

    raw = tokenizer.decode(gen_ids)
    # Strip format tokens
    for tag in [SPECIAL.ASST_END, SPECIAL.ASST_START, SPECIAL.USER_START,
                SPECIAL.USER_END, SPECIAL.EOS_TAG]:
        raw = raw.replace(tag, "")

    # Apply output filter
    filtered = filter_generated_output(raw.strip())
    return filtered, stopped_cleanly, len(gen_ids), elapsed


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(args: argparse.Namespace) -> None:
    device    = args.device
    model     = load_model(args.checkpoint, device)
    tokenizer = Tokenizer()
    eos_ids   = get_eos_ids(tokenizer)

    print(f"[test] EOS ids    : {eos_ids}")
    print(f"[test] Temperature: {args.temperature}")
    print(f"[test] Top-k      : {args.top_k}   Top-p: {args.top_p}")
    print(f"[test] Rep. penalty: {args.repetition_penalty}")
    print(f"[test] max_new_tokens: {args.max_new_tokens}")
    print(f"\n{'='*70}")
    print(f"  Running {len(TEST_CASES)} test prompts")
    print(f"{'='*70}\n")

    results   = []
    passed    = 0
    failed    = 0
    warnings  = 0

    for i, (prompt, keywords) in enumerate(TEST_CASES, 1):
        response, stopped_cleanly, n_tok, elapsed = generate_response(
            model, tokenizer, prompt, device,
            max_new    = args.max_new_tokens,
            temperature = args.temperature,
            top_k       = args.top_k,
            top_p       = args.top_p,
            rep_pen     = args.repetition_penalty,
            eos_ids     = eos_ids,
        )

        # ── Checks ────────────────────────────────────────────────────
        is_empty       = len(response.strip()) < 3
        has_junk       = has_gibberish(response)
        kw_ok          = keywords_found(response, keywords)
        too_short      = len(response.split()) < 2
        too_long       = n_tok >= args.max_new_tokens

        issues = []
        if is_empty:       issues.append("EMPTY_RESPONSE")
        if has_junk:       issues.append("GIBBERISH_DETECTED")
        if not kw_ok:      issues.append(f"KEYWORDS_MISSING({','.join(keywords[:3])})")
        if too_short:      issues.append("TOO_SHORT")
        if too_long:       issues.append("HIT_MAX_TOKENS (may be truncated)")

        if not issues:
            status = "PASS"
            passed += 1
        elif "HIT_MAX_TOKENS" in " ".join(issues) and len(issues) == 1:
            status = "WARN"
            warnings += 1
        else:
            status = "FAIL"
            failed += 1

        results.append((prompt, response, status, issues, n_tok, elapsed))

        # ── Print result ─────────────────────────────────────────────
        status_tag = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
        print(f"{i:2d}. {status_tag} Prompt: {prompt!r}")

        if args.verbose or status != "PASS":
            # Show response (truncated for readability)
            display = response[:200] + ("..." if len(response) > 200 else "")
            print(f"       Response ({n_tok} tok, {elapsed:.2f}s): {display!r}")

        if issues:
            print(f"       Issues: {', '.join(issues)}")
        if not args.verbose and status == "PASS":
            short = response[:80] + ("..." if len(response) > 80 else "")
            print(f"       {short!r}")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    total = len(TEST_CASES)
    print("="*70)
    print(f"  Results: {passed}/{total} PASS  |  {warnings} WARN  |  {failed} FAIL")
    print("="*70)

    if failed == 0 and warnings == 0:
        print("\n  All tests passed! The model produces coherent, clean responses.\n")
    elif failed == 0:
        print(f"\n  {warnings} warning(s): some responses hit max_new_tokens. "
              "Consider longer max_new_tokens or check if the model stops cleanly.\n")
    else:
        print(f"\n  {failed} test(s) FAILED. Review the issues above.")
        print("  Common fixes:")
        print("    GIBBERISH: lower temperature, increase repetition_penalty")
        print("    EMPTY    : model not trained enough; add more SFT data")
        print("    KEYWORDS : pretrain longer or add more Q&A to SFT data\n")

    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ic = CONFIG.inference
    p  = argparse.ArgumentParser(description="Dizel-v1 model evaluation (Task 9)")
    p.add_argument("--checkpoint",          type=str,   required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--max_new_tokens",      type=int,   default=ic.max_new_tokens)
    p.add_argument("--temperature",         type=float, default=ic.temperature)
    p.add_argument("--top_k",               type=int,   default=ic.top_k)
    p.add_argument("--top_p",               type=float, default=ic.top_p)
    p.add_argument("--repetition_penalty",  type=float, default=ic.repetition_penalty)
    p.add_argument("--device",              type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose",             action="store_true",
                   help="Always print full response text")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    passed = run_tests(args)
    sys.exit(0 if passed else 1)
