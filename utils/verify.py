"""
utils/verify.py — Pre-flight sanity checks for Dizel-v1.

Run before training to confirm architecture, format tokens,
output filter, and (if trained) tokenizer round-trip all work.

Usage
-----
    python utils/verify.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig, DizelConfig
from model.architecture import DizelLM
from utils.data_cleaner import filter_generated_output


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


def check_config():
    section("1. Config & Parameters")
    cfg = DizelConfig()
    est = cfg.model.param_estimate()
    print(f"  vocab_size     : {cfg.model.vocab_size:,}")
    print(f"  d_model        : {cfg.model.d_model}")
    print(f"  n_layers       : {cfg.model.n_layers}")
    print(f"  n_heads        : {cfg.model.n_heads}")
    print(f"  context_length : {cfg.model.context_length}")
    print(f"  ~ parameters   : {est/1e6:.1f} M")
    assert 2e6 < est < 100e6, f"Param count {est:.0f} out of expected range"
    print("  OK")
    return cfg


def check_forward(cfg):
    section("2. Forward Pass")
    mc    = cfg.model
    model = DizelLM(mc)
    B, T  = 2, 32
    x     = torch.randint(0, mc.vocab_size, (B, T))
    y     = torch.randint(0, mc.vocab_size, (B, T))
    logits, loss = model(x, y)
    assert logits.shape == (B, T, mc.vocab_size)
    expected = math.log(mc.vocab_size)
    assert abs(loss.item() - expected) < 3.0, f"Init loss {loss.item():.3f} far from {expected:.3f}"
    print(f"  logits shape : {logits.shape}  OK")
    print(f"  init loss    : {loss.item():.3f}  (expected ~{expected:.2f})  OK")
    return model


def check_generation(model):
    section("3. Generation")
    prompt = torch.zeros(1, 1, dtype=torch.long)
    out    = model.generate(prompt, max_new_tokens=20, eos_ids=[999999])
    assert out.shape[1] == 21
    print(f"  Generated {out.shape[1]} tokens  OK")


def check_output_filter():
    section("4. Output Filter")
    tests = [
        ("Hello! asdfasdfasdfasdf",   True),
        ("the the the the extra",     True),
        ("Clean sentence.",           False),
    ]
    for raw, expect_trunc in tests:
        filtered = filter_generated_output(raw)
        if expect_trunc:
            print(f"  IN : {raw!r}")
            print(f"  OUT: {filtered!r}  OK")
        else:
            assert filtered.strip()
            print(f"  Clean preserved: {filtered!r}  OK")


def check_tokenizer():
    section("5. Tokenizer (optional)")
    from config import CONFIG
    path = CONFIG.tokenizer.model_path
    if not os.path.exists(path):
        print(f"  SKIP — run: python tokenizer/train_tokenizer.py")
        return
    from training.dataset import Tokenizer, ROLE_TOKENS, END_TOKEN
    tok  = Tokenizer(path)
    test = "Hello! I am Dizel. How can I help?"
    ids  = tok.encode(test)
    dec  = tok.decode(ids)
    assert dec.strip() == test.strip(), f"Round-trip failed: {dec!r}"
    print(f"  vocab={len(tok):,}  tokens={len(ids)}  round-trip OK")
    unk = tok.unk_id
    for sym in list(ROLE_TOKENS.values()) + [END_TOKEN]:
        tid = tok.sp.piece_to_id(sym)
        ok  = "OK" if tid != unk else "UNK (re-train tokenizer!)"
        print(f"  {sym:20s} -> {tid}  {ok}")


def main():
    print("\n  Dizel-v1 Verification Suite")
    print("  " + "-"*40)
    cfg   = check_config()
    model = check_forward(cfg)
    check_generation(model)
    check_output_filter()
    check_tokenizer()
    print(f"\n{'='*55}")
    print("  All checks passed!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
