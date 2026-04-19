"""
tokenizer/train_tokenizer.py -- Train a BPE SentencePiece tokenizer on the corpus.

Usage:
    python tokenizer/train_tokenizer.py

Outputs:
    tokenizer/dizel.model
    tokenizer/dizel.vocab
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG


def extract_text_from_markdown(path: str) -> str:
    """Strip Markdown headers and return plain text."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    text = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_plain_text(text: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[tokenizer] Wrote plain text to {out_path}  "
          f"({len(text):,} chars, {len(text.split()):,} words)")


def train_tokenizer(plain_text_path: str, model_prefix: str, vocab_size: int) -> None:
    """
    Train a BPE SentencePiece model with newline preservation.

    Special token IDs (must match TokenizerConfig):
      0 = <pad>
      1 = <s>   (BOS)
      2 = </s>  (EOS)
      3 = <unk>
    """
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("SentencePiece not installed. Run: pip install sentencepiece")

    spm.SentencePieceTrainer.train(
        input=plain_text_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=CONFIG.tokenizer.character_coverage,
        # Reserve first 4 ids for special tokens
        pad_id=CONFIG.tokenizer.pad_id,
        bos_id=CONFIG.tokenizer.bos_id,
        eos_id=CONFIG.tokenizer.eos_id,
        unk_id=CONFIG.tokenizer.unk_id,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        # Chat role tokens
        user_defined_symbols=["<|user|>", "<|assistant|>", "<|system|>", "<|json|>", "<|end|>"],
        # Whitespace handling — preserve newlines!
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        split_digits=False,
        byte_fallback=True,
    )
    print(f"[tokenizer] Trained BPE model  ->  {model_prefix}.model")
    print(f"[tokenizer] Vocabulary size     :  {vocab_size:,} tokens")


def verify_tokenizer(model_path: str) -> None:
    """Quick sanity-check: encode/decode a test sentence."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    test = "Hello! How are you doing today? I am Dizel, a small language model."
    ids = sp.encode(test)
    decoded = sp.decode(ids)

    print(f"[tokenizer] Test encode -> {len(ids)} tokens")
    print(f"[tokenizer] Tokens      : {sp.id_to_piece(ids[:12])} ...")
    print(f"[tokenizer] Decoded     : {decoded[:80]}")
    assert decoded.strip() == test.strip(), (
        f"Round-trip mismatch!\n  original : {test}\n  decoded  : {decoded}"
    )
    print("[tokenizer] Round-trip OK")

    # Verify newline preservation
    nl_ids = sp.encode("\n")
    print(f"[tokenizer] Newline encode test: {nl_ids}")
    if nl_ids:
        print(f"[tokenizer] Newline token: {repr(sp.id_to_piece(nl_ids[0]))}")
    else:
        print("[tokenizer] WARNING: Newline still not tokenized!")


def main() -> None:
    cfg = CONFIG.tokenizer
    data_path   = CONFIG.pretrain.data_path
    plain_path  = "tokenizer/corpus.txt"
    model_prefix = cfg.model_path.replace(".model", "")

    os.makedirs("tokenizer", exist_ok=True)

    print(f"[tokenizer] Reading corpus from {data_path} ...")
    if data_path.endswith(".txt"):
        import shutil
        shutil.copy(data_path, plain_path)
        size_mb = os.path.getsize(plain_path) / (1024 * 1024)
        print(f"[tokenizer] Copied plain text to {plain_path}  ({size_mb:.1f} MB)")
    else:
        text = extract_text_from_markdown(data_path)
        write_plain_text(text, plain_path)

    print(f"[tokenizer] Training SentencePiece BPE  "
          f"(vocab={cfg.vocab_size:,}) ...")
    train_tokenizer(plain_path, model_prefix, cfg.vocab_size)

    verify_tokenizer(cfg.model_path)
    print("[tokenizer] Done! Files written:")
    print(f"   {cfg.model_path}")
    print(f"   {model_prefix}.vocab")


if __name__ == "__main__":
    main()
