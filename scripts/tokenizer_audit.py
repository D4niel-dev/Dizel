"""
scripts/tokenizer_audit.py — Audit the Dizel tokenizer for quality issues.

Checks:
  1. Fertility score (tokens/word ratio)
  2. Code token fragmentation
  3. Math expression handling
  4. Common word splitting
  5. Special token coverage
  6. Chat template tokens

Usage:
    python scripts/tokenizer_audit.py
    python scripts/tokenizer_audit.py --model tokenizer/dizel.model
    python scripts/tokenizer_audit.py --model tokenizer/dizel.model --report reports/tokenizer_audit_v122.md
"""

import argparse
import os
import sys
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_tokenizer(model_path: str):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


# ── Test Samples ──────────────────────────────────────────────────────────

ENGLISH_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require large amounts of training data to generalize well.",
    "In a distributed system, consistency, availability, and partition tolerance cannot all be guaranteed simultaneously.",
    "The Transformer architecture uses self-attention mechanisms to process sequential data in parallel.",
    "Gradient descent optimizes the loss function by iteratively updating model parameters.",
    "Natural language processing enables computers to understand and generate human language.",
    "Reinforcement learning agents learn optimal policies through trial and error interactions with an environment.",
    "The backpropagation algorithm computes gradients efficiently using the chain rule of calculus.",
    "Convolutional neural networks are particularly effective for image classification tasks.",
    "Transfer learning allows pre-trained models to be fine-tuned for specific downstream tasks.",
]

CODE_SAMPLES = [
    'def hello_world():\n    print("Hello, World!")\n    return True',
    'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self, d_model=768):\n        super().__init__()\n        self.linear = nn.Linear(d_model, d_model)',
    'for i in range(10):\n    if i % 2 == 0:\n        result.append(i * 2)\n    else:\n        result.append(i + 1)',
    'async function fetchData(url) {\n    const response = await fetch(url);\n    const data = await response.json();\n    return data;\n}',
    'SELECT u.name, COUNT(o.id) AS order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.active = 1\nGROUP BY u.name\nHAVING COUNT(o.id) > 5;',
    '#include <stdio.h>\nint main() {\n    int arr[5] = {1, 2, 3, 4, 5};\n    for (int i = 0; i < 5; i++) {\n        printf("%d\\n", arr[i]);\n    }\n    return 0;\n}',
]

MATH_SAMPLES = [
    "x^2 + y^2 = z^2",
    "f(x) = 3x^2 + 2x - 5",
    "∫ x² dx = x³/3 + C",
    "lim(x→0) sin(x)/x = 1",
    "P(A|B) = P(B|A) * P(A) / P(B)",
    "E = mc²",
    "∑(i=1 to n) i = n(n+1)/2",
    "√(a² + b²)",
    "log₂(1024) = 10",
    "∂f/∂x = 2xy + 3",
]

REASONING_SAMPLES = [
    "Step 1: First, let's identify the key variables in this problem.\nStep 2: We know that the total distance is 100km and the speed is 50km/h.\nStep 3: Using the formula time = distance / speed, we get time = 100/50 = 2 hours.\nTherefore, the journey takes 2 hours.",
    "Let's think about this carefully. If there are 12 eggs in a dozen, and we need 3 dozen, that means we need 12 × 3 = 36 eggs total. Since each carton holds 6 eggs, we need 36 / 6 = 6 cartons.",
]

CHAT_TEMPLATE_SAMPLES = [
    "<|user|>\nWhat is machine learning?\n<|end|>\n<|assistant|>\nMachine learning is a subset of artificial intelligence.\n<|end|>",
    "<|system|>\nYou are a helpful assistant.\n<|end|>\n<|user|>\nExplain Python decorators.\n<|end|>",
]

COMMON_WORDS = [
    "function", "variable", "algorithm", "implementation", "optimization",
    "database", "architecture", "configuration", "authentication", "performance",
    "transformer", "attention", "embedding", "gradient", "backpropagation",
    "inference", "checkpoint", "tokenizer", "parameter", "vocabulary",
    "JavaScript", "TypeScript", "Python", "TensorFlow", "PyTorch",
    "the", "and", "is", "are", "was", "have", "been", "will", "with", "this",
]

STRUCTURAL_TOKENS = [
    "```python", "```json", "```", "---", "###", "- [ ]", "- [x]",
    '{"key": "value"}', "[1, 2, 3]", "| Column | Value |",
]


def compute_fertility(sp, texts: list) -> tuple:
    """Compute tokens-per-word ratio (fertility). Lower = better."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        tokens = sp.encode(text)
        words = text.split()
        total_tokens += len(tokens)
        total_words += len(words)
    fertility = total_tokens / max(1, total_words)
    return fertility, total_tokens, total_words


def check_word_fragmentation(sp, words: list) -> list:
    """Check if common words are split into multiple tokens."""
    issues = []
    for word in words:
        pieces = sp.encode(word, out_type=str)
        if len(pieces) > 2:
            issues.append((word, pieces, len(pieces)))
    return sorted(issues, key=lambda x: -x[2])


def check_code_fragmentation(sp, samples: list) -> dict:
    """Analyze code token efficiency."""
    results = {
        "total_chars": 0,
        "total_tokens": 0,
        "samples": [],
    }
    for sample in samples:
        tokens = sp.encode(sample)
        pieces = sp.encode(sample, out_type=str)
        results["total_chars"] += len(sample)
        results["total_tokens"] += len(tokens)
        results["samples"].append({
            "chars": len(sample),
            "tokens": len(tokens),
            "ratio": len(tokens) / max(1, len(sample.split())),
        })
    return results


def check_special_tokens(sp) -> dict:
    """Check if all required special tokens exist."""
    required = {
        "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        "<|user|>": None, "<|assistant|>": None, "<|system|>": None,
        "<|json|>": None, "<|end|>": None,
    }
    results = {}
    for token, expected_id in required.items():
        piece_id = sp.piece_to_id(token)
        exists = piece_id != sp.unk_id() or token == "<unk>"
        correct_id = expected_id is None or piece_id == expected_id
        results[token] = {
            "exists": exists,
            "id": piece_id,
            "expected_id": expected_id,
            "correct": exists and correct_id,
        }
    return results


def check_roundtrip(sp, samples: list) -> list:
    """Check encode→decode roundtrip fidelity."""
    failures = []
    for text in samples:
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        if decoded.strip() != text.strip():
            failures.append({
                "original": text[:80],
                "decoded": decoded[:80],
                "token_count": len(ids),
            })
    return failures


def generate_report(sp, model_path: str) -> str:
    """Generate the full audit report as markdown."""
    lines = []
    lines.append("# Tokenizer Audit Report — Dizel v1.2.2\n")
    lines.append(f"**Model:** `{model_path}`")
    lines.append(f"**Vocab Size:** {sp.get_piece_size():,}")
    lines.append(f"**Type:** SentencePiece BPE")
    lines.append("")

    # ── 1. Fertility Scores ──────────────────────────────────────────
    lines.append("## 1. Fertility Score (tokens/word)")
    lines.append("")
    lines.append("| Domain | Fertility | Tokens | Words | Verdict |")
    lines.append("|--------|-----------|--------|-------|---------|")

    en_f, en_t, en_w = compute_fertility(sp, ENGLISH_SAMPLES)
    code_f, code_t, code_w = compute_fertility(sp, CODE_SAMPLES)
    math_f, math_t, math_w = compute_fertility(sp, MATH_SAMPLES)
    reason_f, reason_t, reason_w = compute_fertility(sp, REASONING_SAMPLES)
    chat_f, chat_t, chat_w = compute_fertility(sp, CHAT_TEMPLATE_SAMPLES)

    def verdict(f):
        if f < 1.4: return "✅ Good"
        if f < 1.7: return "⚠️ Moderate"
        return "❌ High fragmentation"

    lines.append(f"| English | {en_f:.2f} | {en_t} | {en_w} | {verdict(en_f)} |")
    lines.append(f"| Code | {code_f:.2f} | {code_t} | {code_w} | {verdict(code_f)} |")
    lines.append(f"| Math | {math_f:.2f} | {math_t} | {math_w} | {verdict(math_f)} |")
    lines.append(f"| Reasoning | {reason_f:.2f} | {reason_t} | {reason_w} | {verdict(reason_f)} |")
    lines.append(f"| Chat Templates | {chat_f:.2f} | {chat_t} | {chat_w} | {verdict(chat_f)} |")

    avg_fertility = statistics.mean([en_f, code_f, math_f, reason_f])
    lines.append(f"\n**Overall Average Fertility: {avg_fertility:.2f}**\n")

    # ── 2. Common Word Fragmentation ─────────────────────────────────
    lines.append("## 2. Common Word Fragmentation")
    lines.append("")
    issues = check_word_fragmentation(sp, COMMON_WORDS)
    if issues:
        lines.append(f"**{len(issues)} words split into 3+ pieces:**\n")
        lines.append("| Word | Pieces | Count |")
        lines.append("|------|--------|-------|")
        for word, pieces, count in issues[:20]:
            pieces_str = " + ".join(f"`{p}`" for p in pieces)
            lines.append(f"| {word} | {pieces_str} | {count} |")
    else:
        lines.append("✅ All common words tokenize into ≤2 pieces.\n")

    # ── 3. Code Token Efficiency ─────────────────────────────────────
    lines.append("\n## 3. Code Token Efficiency")
    lines.append("")
    code_results = check_code_fragmentation(sp, CODE_SAMPLES)
    code_ratio = code_results["total_tokens"] / max(1, code_results["total_chars"])
    lines.append(f"- Total chars: {code_results['total_chars']:,}")
    lines.append(f"- Total tokens: {code_results['total_tokens']:,}")
    lines.append(f"- Chars/token: {1/code_ratio:.1f}")
    lines.append("")

    # Check specific code constructs
    lines.append("**Key code construct tokenization:**\n")
    lines.append("| Construct | Pieces | Count |")
    lines.append("|-----------|--------|-------|")
    for construct in STRUCTURAL_TOKENS:
        pieces = sp.encode(construct, out_type=str)
        lines.append(f"| `{construct[:30]}` | {' '.join(f'`{p}`' for p in pieces[:6])} | {len(pieces)} |")

    # ── 4. Special Token Coverage ────────────────────────────────────
    lines.append("\n## 4. Special Token Coverage")
    lines.append("")
    special = check_special_tokens(sp)
    lines.append("| Token | Exists | ID | Expected | Status |")
    lines.append("|-------|--------|-----|----------|--------|")
    all_ok = True
    for token, info in special.items():
        status = "✅" if info["correct"] else "❌"
        if not info["correct"]:
            all_ok = False
        exp = info["expected_id"] if info["expected_id"] is not None else "any"
        lines.append(f"| `{token}` | {'Yes' if info['exists'] else 'No'} | {info['id']} | {exp} | {status} |")

    # ── 5. Roundtrip Fidelity ────────────────────────────────────────
    lines.append("\n## 5. Roundtrip Fidelity")
    lines.append("")
    all_samples = ENGLISH_SAMPLES + CODE_SAMPLES + MATH_SAMPLES + REASONING_SAMPLES
    failures = check_roundtrip(sp, all_samples)
    if failures:
        lines.append(f"**{len(failures)} roundtrip failures out of {len(all_samples)} samples:**\n")
        for f in failures[:5]:
            lines.append(f"- Original: `{f['original']}...`")
            lines.append(f"  Decoded:  `{f['decoded']}...`\n")
    else:
        lines.append(f"✅ All {len(all_samples)} samples pass encode→decode roundtrip.\n")

    # ── 6. Recommendation ────────────────────────────────────────────
    lines.append("## 6. Recommendation")
    lines.append("")

    problems = 0
    if avg_fertility > 1.7:
        problems += 2
    elif avg_fertility > 1.4:
        problems += 1
    if len(issues) > 5:
        problems += 1
    if not all_ok:
        problems += 2
    if failures:
        problems += 1

    if problems == 0:
        lines.append("### ✅ KEEP — Tokenizer is healthy")
        lines.append("No critical issues found. Proceed with current tokenizer.")
    elif problems <= 2:
        lines.append("### ⚠️ PATCH — Minor issues detected")
        lines.append("Consider adding missing merges or user-defined symbols.")
        lines.append("A full retrain is NOT necessary unless corpus changes drastically.")
    else:
        lines.append("### ❌ RETRAIN — Significant issues detected")
        lines.append("Retrain on the final intended corpus (combined pretrain + SFT text).")
        lines.append("Target same 32K vocab with `byte_fallback=True`.")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by `scripts/tokenizer_audit.py`*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit Dizel tokenizer quality")
    parser.add_argument("--model", default="tokenizer/dizel.model", help="Path to .model file")
    parser.add_argument("--report", default="reports/tokenizer_audit_v122.md", help="Output report path")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Tokenizer model not found at {args.model}")
        sys.exit(1)

    print(f"[audit] Loading tokenizer from {args.model}...")
    sp = load_tokenizer(args.model)
    print(f"[audit] Vocab size: {sp.get_piece_size():,}")

    print("[audit] Running audit...")
    report = generate_report(sp, args.model)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[audit] Report written to {args.report}")
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
