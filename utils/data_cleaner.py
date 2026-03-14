"""
utils/data_cleaner.py — Dataset cleaning pipeline for Dizel (Task 1).

Cleans and normalises the raw training corpus before tokeniser training
and pre-training. Run this before train_tokenizer.py.

Checks performed
----------------
1. UTF-8 normalisation (NFC) and BOM removal
2. Empty / whitespace-only line removal
3. Minimum token count filter (removes too-short lines)
4. Repeated-character run detection  (e.g. "aaaaaaa", "########")
5. Symbol/punctuation ratio filter   (e.g. "$$$$$" or "asdf!@#$")
6. Repeated-word/phrase detection    (e.g. "hello hello hello hello hello")
7. Exact duplicate line removal
8. Basic encoding-artifact removal   (e.g. â€™, Ã©)

Usage
-----
    python utils/data_cleaner.py                          # default paths
    python utils/data_cleaner.py --input data/english.md --output data/english_clean.md
    python utils/data_cleaner.py --verbose                # show every rejected line
"""

import argparse
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CONFIG, DataCleanConfig


# ---------------------------------------------------------------------------
# Individual cleaning checks
# ---------------------------------------------------------------------------

def normalise_unicode(text: str) -> str:
    """NFC normalisation + BOM removal."""
    text = text.replace("\ufeff", "")          # BOM
    text = unicodedata.normalize("NFC", text)
    return text


def has_encoding_artifacts(line: str) -> bool:
    """
    Detect common Latin-1 / Windows-1252 bytes decoded as UTF-8 incorrectly.
    Patterns like â€™ (should be '), Ã© (should be é), etc.
    """
    artifact_patterns = [
        r"â€[™\x99\x9c\x9d]",   # curly quotes decoded wrong
        r"Ã[©®°±]",              # accented chars decoded wrong
        r"\x00",                  # null bytes
        r"[\x80-\x9f]",          # C1 control characters
    ]
    for pat in artifact_patterns:
        if re.search(pat, line):
            return True
    return False


def has_excessive_char_repeat(line: str, max_repeat: int) -> bool:
    """
    Returns True if any single character repeats more than max_repeat times
    consecutively.  E.g. "aaaaaaa" or "######".
    """
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return bool(re.search(pattern, line))


def has_high_symbol_ratio(line: str, max_ratio: float) -> bool:
    """
    Returns True if the fraction of non-alphanumeric, non-whitespace
    characters exceeds max_ratio.
    """
    if not line:
        return False
    non_alnum = sum(1 for c in line if not c.isalnum() and not c.isspace())
    return non_alnum / len(line) > max_ratio


def has_repeated_words(line: str, max_consecutive: int = 4) -> bool:
    """
    Returns True if any word repeats more than max_consecutive times in a row.
    E.g. "hello hello hello hello hello".
    """
    words = line.lower().split()
    if len(words) < max_consecutive:
        return False
    for i in range(len(words) - max_consecutive + 1):
        if len(set(words[i : i + max_consecutive])) == 1:
            return True
    return False


def is_too_short(line: str, min_tokens: int) -> bool:
    """Rough word-count check (not actual tokenisation)."""
    return len(line.split()) < min_tokens


def strip_markdown_syntax(text: str) -> str:
    """
    Convert Markdown to clean plain text:
    - Remove heading markers (#)
    - Remove bold/italic markers (* _)
    - Remove code fences (```)
    - Collapse excessive blank lines
    """
    # Headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    # Code fences (remove the line with backticks, keep content)
    text = re.sub(r"```[^\n]*\n", "", text)
    text = re.sub(r"```", "", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # HTML tags (basic)
    text = re.sub(r"<[^>]+>", "", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

@dataclass
class CleanStats:
    total_lines:     int = 0
    kept_lines:      int = 0
    removed_empty:   int = 0
    removed_short:   int = 0
    removed_repeat_char: int = 0
    removed_symbols: int = 0
    removed_repeat_word: int = 0
    removed_artifacts:   int = 0
    removed_duplicates:  int = 0

    def report(self) -> str:
        removed = self.total_lines - self.kept_lines
        pct = 100 * removed / max(1, self.total_lines)
        return (
            f"\n  Total lines        : {self.total_lines:>7,}\n"
            f"  Kept               : {self.kept_lines:>7,}\n"
            f"  Removed total      : {removed:>7,}  ({pct:.1f}%)\n"
            f"    └ empty/whitespace: {self.removed_empty:>7,}\n"
            f"    └ too short       : {self.removed_short:>7,}\n"
            f"    └ char repeat     : {self.removed_repeat_char:>7,}\n"
            f"    └ symbol ratio    : {self.removed_symbols:>7,}\n"
            f"    └ word repeat     : {self.removed_repeat_word:>7,}\n"
            f"    └ encoding artif. : {self.removed_artifacts:>7,}\n"
            f"    └ duplicates      : {self.removed_duplicates:>7,}\n"
        )


def clean_corpus(
    raw_text: str,
    cfg: DataCleanConfig,
    verbose: bool = False,
) -> Tuple[str, CleanStats]:
    """
    Clean a raw text corpus string and return (clean_text, stats).

    The cleaning operates at the paragraph / sentence level (split on
    double newlines first, then per-line within each paragraph).
    """
    stats = CleanStats()

    # 1. Unicode normalisation
    if cfg.normalise_unicode:
        raw_text = normalise_unicode(raw_text)

    # 2. Strip Markdown syntax
    raw_text = strip_markdown_syntax(raw_text)

    # 3. Split into lines
    lines = raw_text.splitlines()
    stats.total_lines = len(lines)

    seen:  set  = set()
    clean: List[str] = []

    def reject(reason: str, line: str) -> None:
        if verbose:
            preview = line[:60].replace("\n", " ")
            print(f"  [REJECT:{reason:14s}] {preview!r}")

    for line in lines:
        line = line.strip()

        # Empty
        if not line:
            stats.removed_empty += 1
            clean.append("")          # preserve paragraph breaks
            continue

        # Encoding artifacts
        if has_encoding_artifacts(line):
            stats.removed_artifacts += 1
            reject("artifact", line)
            continue

        # Too short
        if is_too_short(line, cfg.min_tokens_per_line):
            stats.removed_short += 1
            reject("too_short", line)
            continue

        # Repeated characters
        if has_excessive_char_repeat(line, cfg.max_char_repeat):
            stats.removed_repeat_char += 1
            reject("char_repeat", line)
            continue

        # Symbol ratio
        if has_high_symbol_ratio(line, cfg.max_symbol_ratio):
            stats.removed_symbols += 1
            reject("symbol_ratio", line)
            continue

        # Repeated words
        if has_repeated_words(line):
            stats.removed_repeat_word += 1
            reject("word_repeat", line)
            continue

        # Duplicates
        if cfg.remove_duplicates:
            key = line.lower().strip()
            if key in seen:
                stats.removed_duplicates += 1
                reject("duplicate", line)
                continue
            seen.add(key)

        clean.append(line)
        stats.kept_lines += 1

    # Collapse consecutive blank lines introduced by removals
    clean_text = "\n".join(clean)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

    return clean_text, stats


# ---------------------------------------------------------------------------
# Output filter for inference (Task 6)
# ---------------------------------------------------------------------------

def filter_generated_output(
    text: str,
    max_char_repeat: int  = 4,
    max_word_repeat: int  = 3,
    filter_punct_spam: bool = True,
) -> str:
    """
    Post-process a model generation to remove trailing gibberish (Task 6).

    Strategy
    --------
    1. Truncate at the first excessive character repeat
       e.g. "Hello! asdfasdf" → "Hello!"
    2. Truncate at the first excessive word repeat run
       e.g. "Hi there the the the" → "Hi there"
    3. Remove lines that are entirely punctuation spam
    4. Clean trailing whitespace / punctuation
    """
    if not text:
        return text

    # ── 1. Char repeat truncation ──────────────────────────────────────
    pat_char = re.compile(r"(.)\1{" + str(max_char_repeat) + r",}")
    m = pat_char.search(text)
    if m:
        text = text[: m.start()].rstrip()

    # ── 2. Word repeat truncation ──────────────────────────────────────
    words = text.split()
    cut   = len(words)
    pat_n = max_word_repeat
    for i in range(len(words) - pat_n + 1):
        if len(set(w.lower() for w in words[i : i + pat_n])) == 1:
            cut = i
            break
    text = " ".join(words[:cut]).rstrip()

    # ── 3. Punctuation spam filter ─────────────────────────────────────
    if filter_punct_spam:
        lines = text.splitlines()
        clean = []
        for ln in lines:
            stripped = ln.strip()
            if stripped and has_high_symbol_ratio(stripped, max_ratio=0.60):
                break    # stop at first spam line
            clean.append(ln)
        text = "\n".join(clean)

    # ── 4. Clean trailing artifacts ───────────────────────────────────
    # Remove isolated special characters at the end
    text = re.sub(r"[\s\W]+$", "", text)
    # Remove dangling role tokens that may have leaked
    for tag in ["<user>", "</user>", "<assistant>", "</assistant>", "<eos>", "<s>", "</s>"]:
        text = text.replace(tag, "")

    return text.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dizel dataset cleaner")
    p.add_argument("--input",   default="data/english.md",
                   help="Input raw corpus file (Markdown or plain text)")
    p.add_argument("--output",  default="data/english_clean.md",
                   help="Output cleaned file")
    p.add_argument("--verbose", action="store_true",
                   help="Print every rejected line")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = CONFIG.data_clean

    if not os.path.exists(args.input):
        print(f"[cleaner] ERROR: input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    print(f"[cleaner] Input  : {args.input}  ({len(raw):,} chars)")
    print(f"[cleaner] Cleaning...")

    clean_text, stats = clean_corpus(raw, cfg, verbose=args.verbose)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"[cleaner] Output : {args.output}  ({len(clean_text):,} chars)")
    print(stats.report())

    # Quick test of the inference filter
    test_cases = [
        "Hello! How can I help you today? asdfasdfasdf",
        "Sure, Python is great! the the the the extra",
        "Good morning! ########################################",
    ]
    print("[cleaner] Output filter tests:")
    for tc in test_cases:
        filtered = filter_generated_output(tc)
        print(f"  IN : {tc!r}")
        print(f"  OUT: {filtered!r}\n")


if __name__ == "__main__":
    main()
