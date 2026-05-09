"""
scripts/clean_v122_data.py — Clean and filter downloaded datasets for Dizel v1.2.2.

Pipeline stages:
  1. Unicode normalization (NFKC)
  2. Length filtering (min/max chars)
  3. Quality heuristics (symbol ratio, char repeats)
  4. Circular answer detection
  5. Deduplication (exact + near-duplicate via MinHash)
  6. Stats reporting

Usage:
    python scripts/clean_v122_data.py --input data/raw/fineweb-edu/fineweb-edu.jsonl
    python scripts/clean_v122_data.py --input data/raw/ --all
    python scripts/clean_v122_data.py --input data/raw/ --all --report reports/data_cleaning_v122.md
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Cleaning Functions ────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFKC normalization + clean up common encoding artifacts."""
    text = unicodedata.normalize("NFKC", text)
    # Fix common artifacts
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u200b", "")   # Zero-width space
    text = text.replace("\ufeff", "")   # BOM
    return text


def check_length(text: str, min_chars: int = 50, max_chars: int = 32000) -> bool:
    """Reject samples that are too short or too long."""
    length = len(text)
    return min_chars <= length <= max_chars


def check_quality(text: str, max_symbol_ratio: float = 0.40, max_char_repeat: int = 5) -> bool:
    """Reject low-quality text based on symbol density and character repetition."""
    if not text.strip():
        return False

    # Symbol ratio check
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    ratio = symbols / max(1, len(text))
    if ratio > max_symbol_ratio:
        return False

    # Character repetition check (e.g., "aaaaaa" or "!!!!!!")
    for match in re.finditer(r'(.)\1{' + str(max_char_repeat) + r',}', text):
        return False

    # Line repetition check (same line repeated 3+ times)
    lines = text.strip().split("\n")
    if len(lines) > 3:
        line_counts = Counter(lines)
        most_common_count = line_counts.most_common(1)[0][1]
        if most_common_count >= 3 and most_common_count / len(lines) > 0.3:
            return False

    return True


def check_circular(text: str) -> bool:
    """Detect and reject circular or tautological answers."""
    text_lower = text.lower().strip()

    # Pattern: "X is X" or "X refers to X"
    circular_patterns = [
        r"(.{10,50})\s+is\s+\1",
        r"(.{10,50})\s+refers?\s+to\s+\1",
        r"(.{10,50})\s+means?\s+\1",
    ]
    for pattern in circular_patterns:
        if re.search(pattern, text_lower):
            return False

    # "As an AI language model" rejection
    ai_disclaimers = [
        "as an ai language model",
        "as a large language model",
        "i cannot provide",
        "i'm just an ai",
        "as an artificial intelligence",
    ]
    for disclaimer in ai_disclaimers:
        if disclaimer in text_lower:
            return False

    return True


def check_low_information(text: str) -> bool:
    """Reject text with very low information density."""
    words = text.split()
    if len(words) < 5:
        return False

    # Unique word ratio — very low means repetitive content
    unique = len(set(w.lower() for w in words))
    ratio = unique / len(words)
    if ratio < 0.15 and len(words) > 20:
        return False

    return True


def normalize_whitespace(text: str) -> str:
    """Clean up excessive whitespace while preserving code indentation."""
    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # Remove trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


# ── Deduplication ─────────────────────────────────────────────────────────

class ExactDeduplicator:
    """Fast exact deduplication using content hashes."""

    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False


class NearDeduplicator:
    """Near-duplicate detection using character n-gram shingling."""

    def __init__(self, n: int = 5, threshold: float = 0.8):
        self.n = n
        self.threshold = threshold
        self.seen_shingles = []
        self.max_compare = 1000  # Only compare against last N for speed

    def _shingle(self, text: str) -> set:
        """Create character n-gram shingle set."""
        text = text.lower().replace(" ", "")
        if len(text) < self.n:
            return {text}
        return {text[i:i+self.n] for i in range(len(text) - self.n + 1)}

    def is_near_duplicate(self, text: str) -> bool:
        shingles = self._shingle(text)
        if not shingles:
            return False

        # Compare against recent entries
        compare_set = self.seen_shingles[-self.max_compare:]
        for prev in compare_set:
            if not prev:
                continue
            intersection = len(shingles & prev)
            union = len(shingles | prev)
            if union > 0 and intersection / union >= self.threshold:
                return True

        self.seen_shingles.append(shingles)
        return False


# ── Main Pipeline ─────────────────────────────────────────────────────────

def clean_file(input_path: str, output_path: str, config: dict = None) -> dict:
    """Clean a single JSONL file. Returns stats."""
    config = config or {}
    min_chars = config.get("min_chars", 50)
    max_chars = config.get("max_chars", 32000)
    max_symbol_ratio = config.get("max_symbol_ratio", 0.40)
    max_char_repeat = config.get("max_char_repeat", 5)
    do_dedup = config.get("dedup", True)
    do_near_dedup = config.get("near_dedup", False)  # Slower, optional

    stats = {
        "input_count": 0,
        "output_count": 0,
        "dropped_unicode": 0,
        "dropped_length": 0,
        "dropped_quality": 0,
        "dropped_circular": 0,
        "dropped_low_info": 0,
        "dropped_exact_dup": 0,
        "dropped_near_dup": 0,
        "total_chars_in": 0,
        "total_chars_out": 0,
    }

    exact_dedup = ExactDeduplicator() if do_dedup else None
    near_dedup = NearDeduplicator() if do_near_dedup else None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            stats["input_count"] += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped_quality"] += 1
                continue

            # Find the text field
            text = (record.get("text") or record.get("output") or
                    record.get("response") or record.get("content") or "")
            if not text:
                stats["dropped_length"] += 1
                continue

            stats["total_chars_in"] += len(text)

            # Stage 1: Unicode normalization
            text = normalize_unicode(text)

            # Stage 2: Length filter
            if not check_length(text, min_chars, max_chars):
                stats["dropped_length"] += 1
                continue

            # Stage 3: Quality filter
            if not check_quality(text, max_symbol_ratio, max_char_repeat):
                stats["dropped_quality"] += 1
                continue

            # Stage 4: Circular answer detection
            if not check_circular(text):
                stats["dropped_circular"] += 1
                continue

            # Stage 5: Low information detection
            if not check_low_information(text):
                stats["dropped_low_info"] += 1
                continue

            # Stage 6: Whitespace normalization
            text = normalize_whitespace(text)

            # Stage 7: Exact deduplication
            if exact_dedup and exact_dedup.is_duplicate(text):
                stats["dropped_exact_dup"] += 1
                continue

            # Stage 8: Near deduplication (optional, slow)
            if near_dedup and near_dedup.is_near_duplicate(text):
                stats["dropped_near_dup"] += 1
                continue

            # Write cleaned record
            # Update the text field in the record
            if "text" in record:
                record["text"] = text
            elif "output" in record:
                record["output"] = text
            elif "response" in record:
                record["response"] = text
            else:
                record["text"] = text

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["output_count"] += 1
            stats["total_chars_out"] += len(text)

            if stats["input_count"] % 50_000 == 0:
                elapsed = time.time() - t0
                rate = stats["input_count"] / elapsed
                kept_pct = stats["output_count"] / max(1, stats["input_count"]) * 100
                print(f"    ... {stats['input_count']:,} processed, "
                      f"{stats['output_count']:,} kept ({kept_pct:.1f}%), "
                      f"{rate:.0f} samples/s")

    stats["elapsed"] = time.time() - t0
    return stats


def generate_report(all_stats: dict, output_path: str):
    """Generate a cleaning report as markdown."""
    lines = ["# Data Cleaning Report — Dizel v1.2.2\n"]

    total_in = sum(s["input_count"] for s in all_stats.values())
    total_out = sum(s["output_count"] for s in all_stats.values())
    total_chars_in = sum(s["total_chars_in"] for s in all_stats.values())
    total_chars_out = sum(s["total_chars_out"] for s in all_stats.values())

    lines.append(f"**Total Input:** {total_in:,} samples ({total_chars_in/1e6:.1f}M chars)")
    lines.append(f"**Total Output:** {total_out:,} samples ({total_chars_out/1e6:.1f}M chars)")
    lines.append(f"**Kept:** {total_out/max(1,total_in)*100:.1f}%\n")

    lines.append("## Per-Source Breakdown\n")
    lines.append("| Source | Input | Output | Kept% | Length | Quality | Circular | Low-Info | Dedup |")
    lines.append("|--------|-------|--------|-------|--------|---------|----------|---------|-------|")

    for name, s in all_stats.items():
        kept_pct = s["output_count"] / max(1, s["input_count"]) * 100
        lines.append(
            f"| {name} | {s['input_count']:,} | {s['output_count']:,} | "
            f"{kept_pct:.0f}% | {s['dropped_length']:,} | {s['dropped_quality']:,} | "
            f"{s['dropped_circular']:,} | {s['dropped_low_info']:,} | "
            f"{s['dropped_exact_dup']:,} |"
        )

    lines.append("\n---")
    lines.append("*Generated by `scripts/clean_v122_data.py`*")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[clean] Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean Dizel v1.2.2 training data")
    parser.add_argument("--input", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output", default="data/cleaned", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Process all JSONL files in input dir")
    parser.add_argument("--report", default="reports/data_cleaning_v122.md", help="Report output path")
    parser.add_argument("--near-dedup", action="store_true", help="Enable near-duplicate detection (slow)")
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=32000)
    args = parser.parse_args()

    config = {
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "dedup": True,
        "near_dedup": args.near_dedup,
    }

    # Find files to process
    if args.all and os.path.isdir(args.input):
        files = []
        for root, dirs, filenames in os.walk(args.input):
            for fname in filenames:
                if fname.endswith(".jsonl"):
                    files.append(os.path.join(root, fname))
    elif os.path.isfile(args.input):
        files = [args.input]
    else:
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    if not files:
        print("No JSONL files found.")
        return

    print(f"\n{'='*60}")
    print(f"  Dizel v1.2.2 Data Cleaning Pipeline")
    print(f"  Files: {len(files)}")
    print(f"  Output: {args.output}/")
    print(f"  Near-dedup: {'ON' if args.near_dedup else 'OFF'}")
    print(f"{'='*60}\n")

    all_stats = {}
    for fpath in files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(args.output, f"{name}.jsonl")
        print(f"[clean] Processing: {name}")
        stats = clean_file(fpath, out_path, config)
        all_stats[name] = stats
        kept_pct = stats["output_count"] / max(1, stats["input_count"]) * 100
        print(f"  -> {stats['output_count']:,}/{stats['input_count']:,} kept ({kept_pct:.1f}%) in {stats['elapsed']:.1f}s\n")

    generate_report(all_stats, args.report)

    # Summary
    total_in = sum(s["input_count"] for s in all_stats.values())
    total_out = sum(s["output_count"] for s in all_stats.values())
    print(f"\nTotal: {total_out:,}/{total_in:,} samples kept ({total_out/max(1,total_in)*100:.1f}%)")


if __name__ == "__main__":
    main()
