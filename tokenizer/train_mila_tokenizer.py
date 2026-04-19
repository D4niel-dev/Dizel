"""
tokenizer/train_mila_tokenizer.py -- Train SentencePiece BPE tokenizer for Mila.

Mila needs her own tokenizer because her corpus is conversational (not code-heavy).
BPE merges differ from Dizel's: casual phrases get better tokenization,
while code tokens are less frequent.

IMPORTANT: Preserves newlines and uses the same special token IDs as Dizel
so the chat format is compatible between models.

Usage:
    python tokenizer/train_mila_tokenizer.py --input data/mila_pretrain.txt
    python tokenizer/train_mila_tokenizer.py --input data/mila_pretrain.txt --vocab_size 32000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Chat format tokens shared with Dizel (constructed to avoid escaping issues)
_L = chr(60)  # <
_R = chr(62)  # >
_P = chr(124) # |
CHAT_TOKENS = [
    f"{_L}{_P}system{_P}{_R}",
    f"{_L}{_P}user{_P}{_R}",
    f"{_L}{_P}assistant{_P}{_R}",
    f"{_L}{_P}end{_P}{_R}",
    f"{_L}/s{_R}",
]


def train_tokenizer(
    input_file: str,
    model_prefix: str = "tokenizer/mila",
    vocab_size: int = 32000,
    character_coverage: float = 1.0,
):
    """Train a SentencePiece BPE tokenizer for Mila."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Error: pip install sentencepiece")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print(f"  Prepare Mila's pretrain corpus first.")
        sys.exit(1)

    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"[mila-tok] Training tokenizer on: {input_file} ({file_size_mb:.1f} MB)")
    print(f"[mila-tok] Vocab size: {vocab_size}")
    print(f"[mila-tok] Output: {model_prefix}.model")
    print(f"[mila-tok] Chat tokens: {CHAT_TOKENS}")

    os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        # Preserve whitespace and newlines (critical for conversation format)
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        # Special tokens -- MUST match Dizel's IDs for chat compatibility
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        # User-defined tokens for chat format
        user_defined_symbols=CHAT_TOKENS,
        # BPE training params
        num_threads=os.cpu_count() or 4,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
    )

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")
    print(f"\n[mila-tok] Done! Vocab size: {sp.GetPieceSize()}")
    print(f"[mila-tok] PAD={sp.pad_id()}, BOS={sp.bos_id()}, EOS={sp.eos_id()}, UNK={sp.unk_id()}")

    # Test encode a sample Mila-style message
    test_text = "Heyy! What's going on?"
    tokens = sp.Encode(test_text, out_type=str)
    ids = sp.Encode(test_text)
    print(f"\n[mila-tok] Test: '{test_text}'")
    print(f"  tokens: {tokens}")
    print(f"  ids:    {ids}")
    print(f"  decode: '{sp.Decode(ids)}'")

    # Verify chat tokens
    print(f"\n[mila-tok] Chat token IDs:")
    for tok in CHAT_TOKENS:
        tid = sp.PieceToId(tok)
        print(f"  {tok} -> {tid}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Mila tokenizer")
    p.add_argument("--input", type=str, default="data/mila_pretrain.txt",
                   help="Path to Mila pretrain corpus")
    p.add_argument("--output", type=str, default="tokenizer/mila",
                   help="Output model prefix (creates .model and .vocab)")
    p.add_argument("--vocab_size", type=int, default=32000)
    return p.parse_args()


def main():
    args = parse_args()
    train_tokenizer(
        input_file=args.input,
        model_prefix=args.output,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
