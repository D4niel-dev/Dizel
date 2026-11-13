"""
scripts/prepare_dpo_data.py — Prepare DPO preference pairs from downloaded data.

Converts raw ultrafeedback data (from download_v122_data.py) into the JSONL
format expected by training/dpo.py:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Also supports the HuggingFaceH4/ultrafeedback_binarized dataset format
and argilla/distilabel-intel-orca-dpo-pairs if available.

Usage (Colab):
    # After downloading ultrafeedback:
    python scripts/prepare_dpo_data.py --target dizel

    # From HuggingFace directly (downloads + processes):
    python scripts/prepare_dpo_data.py --target dizel --source hf

    # Custom input file:
    python scripts/prepare_dpo_data.py --input data/raw/ultrafeedback/ultrafeedback.jsonl
"""

import argparse
import json
import os
import sys
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_raw_ultrafeedback(path: str, min_score_gap: float = 1.0):
    """
    Load preference pairs from raw UltraFeedback format (openbmb/UltraFeedback).

    The raw format has:
        {"instruction": "...", "chosen": "...", "rejected": "...",
         "chosen_score": 4.5, "rejected_score": 2.0}

    Filters by minimum score gap to ensure clear preferences.
    """
    pairs = []
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt = obj.get("instruction", obj.get("prompt", ""))
            chosen = obj.get("chosen", "")
            rejected = obj.get("rejected", "")
            chosen_score = obj.get("chosen_score", 5.0)
            rejected_score = obj.get("rejected_score", 1.0)

            if not prompt or not chosen or not rejected:
                skipped += 1
                continue

            # Filter: require minimum score gap
            if chosen_score - rejected_score < min_score_gap:
                skipped += 1
                continue

            # Filter: skip very short or near-identical responses
            if len(chosen.strip()) < 20 or len(rejected.strip()) < 20:
                skipped += 1
                continue

            if chosen.strip() == rejected.strip():
                skipped += 1
                continue

            pairs.append({
                "prompt": prompt.strip(),
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
            })

    if skipped:
        print(f"  [filter] Skipped {skipped} pairs (low gap, too short, or bad JSON)")

    return pairs


def load_hf_ultrafeedback_binarized(max_samples: int = 20000):
    """
    Download and load from HuggingFaceH4/ultrafeedback_binarized.

    This is a pre-processed version with clear chosen/rejected splits.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] pip install datasets")
        return []

    print("  [hf] Loading HuggingFaceH4/ultrafeedback_binarized...")
    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        streaming=True,
    )

    pairs = []
    count = 0
    for sample in ds:
        if count >= max_samples:
            break

        chosen_msgs = sample.get("chosen", [])
        rejected_msgs = sample.get("rejected", [])

        if not chosen_msgs or not rejected_msgs:
            continue

        # Extract prompt (first user message)
        prompt = ""
        chosen_response = ""
        rejected_response = ""

        for msg in chosen_msgs:
            if msg["role"] == "user":
                prompt = msg["content"]
            elif msg["role"] == "assistant":
                chosen_response = msg["content"]

        for msg in rejected_msgs:
            if msg["role"] == "assistant":
                rejected_response = msg["content"]

        if not prompt or not chosen_response or not rejected_response:
            continue

        if len(chosen_response.strip()) < 20 or len(rejected_response.strip()) < 20:
            continue

        pairs.append({
            "prompt": prompt.strip(),
            "chosen": chosen_response.strip(),
            "rejected": rejected_response.strip(),
        })
        count += 1

        if count % 5000 == 0:
            print(f"    ... {count}/{max_samples} pairs loaded")

    return pairs


# ── Domain classification ──────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    "programming": {
        "python", "javascript", "typescript", "java", "c++", "rust", "golang",
        "function", "class", "def ", "code", "algorithm", "program",
        "implement", "debug", "error", "html", "css", "api", "database",
        "sql", "json", "react", "node", "variable", "loop", "array",
        "compile", "runtime", "syntax", "exception", "import", "```",
        "return ", "print(", "console.log", "git ", "docker", "deploy",
        "refactor", "test case", "unit test", "bug", "fix ", "library",
        "framework", "endpoint", "server", "async", "await", "callback",
        "promise", "http", "rest", "graphql", "npm", "pip ", "install",
        "terminal", "command", "script", "module", "package", "dependency",
        "recursion", "iteration", "inheritance", "polymorphism",
        "null pointer", "stack trace", "traceback", "exception handling",
        "code review", "pull request", "merge", "branch", "repository",
        "stdlib", "type hint", "annotation", "decorator", "generator",
        "lambda", "list comprehension", "regex", "multithreading",
    },
    "math": {
        "equation", "solve", "calculate", "derivative", "integral",
        "matrix", "vector", "algebra", "geometry", "theorem", "proof",
        "probability", "statistics", "linear regression", "calculus",
        "differential", "summation", "factorial", "permutation",
        "combination", "trigonometry", "sine", "cosine", "logarithm",
        "exponential", "polynomial", "quadratic", "fraction", "decimal",
        "x =", "y =", "f(x)", "compute", "formula", "arithmetic",
        "prime number", "gcd", "lcm", "modulo", "binomial",
    },
    "reasoning": {
        "logic", "puzzle", "riddle", "brain teaser", "critical thinking",
        "deduction", "induction", "syllogism", "inference", "fallacy",
        "assume", "hypothesis", "reason", "argument",
        "contradiction", "paradox", "thought experiment",
        "trolley problem", "ethical dilemma", "decision",
        "problem solving", "analyze", "evaluate", "compare and contrast",
        "what is the difference", "explain why", "why is",
        "step by step", "think through", "consider",
        "implication", "conclusion", "premise", "counterexample",
        "plausible", "valid", "sound", "necessary", "sufficient",
        "logical", "reasoning", "deduce", "infer", "conclude",
        "therefore", "thus", "hence", "syllogism", "if all", "then ",
        "necessarily", "follows that", "it follows", "prove that",
    },
    "writing": {
        "write", "essay", "poem", "story", "paragraph", "article",
        "blog", "content", "creative writing", "edit", "rewrite",
        "proofread", "grammar", "style", "tone", "narrative",
        "describe", "compose", "draft", "outline", "thesis",
        "introduction", "conclusion", "synopsis", "summary",
        "headline", "title", "caption", "dialogue", "script",
        "genre", "fiction", "non-fiction", "prose", "verse",
        "metaphor", "simile", "imagery", "character", "plot",
        "setting", "theme", "voice", "audience",
        "persuasive", "expository", "narrative", "descriptive",
        "review", "critique", "feedback",
    },
    "science": {
        "physics", "chemistry", "biology", "astronomy", "geology",
        "molecule", "atom", "element", "reaction", "chemical",
        "cell", "dna", "rna", "protein", "evolution", "species",
        "ecosystem", "organism", "photosynthesis", "mitosis",
        "gravity", "velocity", "acceleration", "force", "energy",
        "thermodynamics", "quantum", "relativity", "electromagnetism",
        "newton", "einstein", "darwin", "experiment", "laboratory",
        "hypothesis", "scientific method", "observation",
        "gene", "chromosome", "bacteria", "virus", "vaccine",
        "climate", "temperature", "pressure", "density", "mass",
        "volume", "wavelength", "frequency", "amplitude",
        "planet", "star", "galaxy", "solar system", "orbit",
        "acid", "base", "ph", "enzyme", "catalyst",
    },
    "translation": {
        "translate", "translation", "translator", "transliterate",
        "translate to", "translate from", "in french", "in spanish",
        "in german", "in chinese", "in japanese", "in russian",
        "in arabic", "in italian", "in portuguese", "in korean",
        "in hindi", "into french", "into spanish", "into german",
        "into chinese", "into japanese",
        "what is the word for", "how do you say",
        "meaning in", "equivalent in", "langue", "idioma",
        "english to", "french to", "spanish to",
    },
    "conversation": {
        "hello", "hi ", "hey", "how are you", "what's up",
        "good morning", "good evening", "nice to meet",
        "thank", "thanks", "you're welcome", "sorry",
        "excuse me", "please", "yes", "no", "maybe",
        "opinion", "think about", "what do you", "do you think",
        "favorite", "prefer", "recommend", "suggest",
        "tell me about yourself", "what can you do",
        "help me with", "i need", "can you", "could you",
        "would you", "i'm looking for", "i want to",
        "fun", "interesting", "cool", "awesome", "great",
        "joke", "funny", "humor", "story about",
        "weather", "hobby", "music", "movie", "book", "food",
        "travel", "sport", "game", "news", "life",
        "chat", "talk", "discuss", "conversation",
        "random", "anything", "general",
    },
}


def classify_domain(prompt: str) -> str:
    """
    Classify a prompt into one of seven domains using keyword analysis.

    Returns one of: 'programming', 'math', 'reasoning', 'writing',
                    'conversation', 'science', 'translation'
    """
    text = prompt.lower()

    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        scores[domain] = score

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    # Tie-break: conversation is the fallback if nothing matched
    if best_score == 0:
        return "conversation"

    return best_domain


DEFAULT_TARGET_RATIOS = {
    "programming": 0.40,
    "math": 0.10,
    "reasoning": 0.10,
    "science": 0.10,
    "writing": 0.10,
    "translation": 0.05,
    "conversation": 0.15,
}

DIZEL_TARGET_RATIOS = {
    "programming": 0.55,
    "math": 0.12,
    "reasoning": 0.12,
    "science": 0.08,
    "writing": 0.05,
    "translation": 0.03,
    "conversation": 0.05,
}


def rebalance_by_domain(pairs: list, target_ratios: dict = None) -> list:
    """
    Classify every pair by domain and rebalance to achieve target ratios.

    1. Classifies every pair by domain
    2. Prints domain distribution stats
    3. Downsamples overrepresented domains, keeps all underrepresented ones
    4. Returns a shuffled balanced list

    If *target_ratios* is None, uses DEFAULT_TARGET_RATIOS.
    """
    if target_ratios is None:
        target_ratios = DEFAULT_TARGET_RATIOS

    domain_pairs = {d: [] for d in target_ratios}
    domain_unknown = []

    for pair in pairs:
        domain = classify_domain(pair["prompt"])
        if domain in domain_pairs:
            domain_pairs[domain].append(pair)
        else:
            domain_unknown.append(pair)

    total = len(pairs)
    print(f"\n  [domain] Distribution ({total} pairs):")
    for domain in sorted(target_ratios.keys()):
        count = len(domain_pairs[domain])
        pct = (count / total * 100) if total else 0.0
        target_pct = target_ratios[domain] * 100
        bar_len = max(1, int(pct / 2))
        bar = "▓" * bar_len
        flag = " ▼ OVERSAMPLED" if pct > target_pct * 1.2 else \
               " ▲ UNDERSAMPLED" if pct < target_pct * 0.8 else ""
        print(f"    {domain:15s} {count:5d}  ({pct:5.1f}%)  target {target_pct:5.1f}%  {bar}{flag}")

    if domain_unknown:
        print(f"    {'unknown':15s} {len(domain_unknown):5d}")

    # Calculate target counts, keeping total close to original
    rng = random.Random(42)
    result = []

    for domain, ratio in target_ratios.items():
        target_count = int(total * ratio)
        available = len(domain_pairs[domain])

        if available <= target_count:
            # Underrepresented — keep all
            result.extend(domain_pairs[domain])
            remaining = ""
        else:
            # Overrepresented — downsample
            sampled = rng.sample(domain_pairs[domain], target_count)
            result.extend(sampled)
            remaining = f" (dropped {available - target_count})"

        print(f"    {domain:15s}: {len(domain_pairs[domain])} available → "
              f"{min(available, target_count)} kept{remaining}")

    rng.shuffle(result)

    print(f"\n  [rebalanced] {len(result)} pairs after domain rebalancing")
    return result


def save_dpo_data(pairs: list, output_path: str):
    """Save DPO pairs to JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[saved] {len(pairs)} DPO pairs -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO training data")
    parser.add_argument("--target", choices=["dizel", "mila"], default="dizel")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to raw ultrafeedback JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for DPO JSONL")
    parser.add_argument("--source", choices=["local", "hf"], default="local",
                        help="'local' = use downloaded data, 'hf' = download from HuggingFace")
    parser.add_argument("--max_samples", type=int, default=20000,
                        help="Maximum DPO pairs to generate")
    parser.add_argument("--min_score_gap", type=float, default=1.0,
                        help="Minimum score difference between chosen/rejected")
    parser.add_argument("--rebalance", action="store_true",
                        help="Enable domain rebalancing (replaces old --coding_ratio)")
    parser.add_argument("--dizel_mode", action="store_true",
                        help="Use Dizel-optimized ratios (more coding, less chit-chat)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip all filtering/rebalancing (overrides --rebalance)")
    args = parser.parse_args()

    output_path = args.output or f"data/formatted/{args.target}_dpo.jsonl"

    print(f"\n{'='*60}")
    print(f"  DPO Data Preparation")
    print(f"  Target: {args.target}")
    print(f"  Source: {args.source}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    all_pairs = []

    if args.source == "hf":
        # Download directly from HuggingFace
        pairs = load_hf_ultrafeedback_binarized(max_samples=args.max_samples)
        all_pairs.extend(pairs)
        print(f"[hf] Loaded {len(pairs)} pairs from HuggingFace")

    else:
        # Use locally downloaded data
        input_path = args.input or "data/raw/ultrafeedback/ultrafeedback.jsonl"
        if not os.path.exists(input_path):
            print(f"[ERROR] Input file not found: {input_path}")
            print(f"  Options:")
            print(f"    1. Download first: python scripts/download_v122_data.py --target dizel --source ultrafeedback")
            print(f"    2. Use HF directly: python scripts/prepare_dpo_data.py --source hf")
            sys.exit(1)

        pairs = load_raw_ultrafeedback(input_path, min_score_gap=args.min_score_gap)
        all_pairs.extend(pairs)
        print(f"[local] Loaded {len(pairs)} pairs from {input_path}")

    if not all_pairs:
        print("[ERROR] No valid DPO pairs found!")
        sys.exit(1)

    # Cap at max samples
    if len(all_pairs) > args.max_samples:
        rng = random.Random(42)
        all_pairs = rng.sample(all_pairs, args.max_samples)
        print(f"[cap] Capped to {args.max_samples} pairs")

    # Apply domain rebalancing
    if not args.no_filter and (args.rebalance or args.target == "dizel"):
        ratios = DIZEL_TARGET_RATIOS if args.dizel_mode else None
        all_pairs = rebalance_by_domain(all_pairs, target_ratios=ratios)

    # Stats
    avg_prompt_len = sum(len(p["prompt"]) for p in all_pairs) / len(all_pairs)
    avg_chosen_len = sum(len(p["chosen"]) for p in all_pairs) / len(all_pairs)
    avg_rejected_len = sum(len(p["rejected"]) for p in all_pairs) / len(all_pairs)

    print(f"\n[stats]")
    print(f"  Total pairs    : {len(all_pairs)}")
    print(f"  Avg prompt len : {avg_prompt_len:.0f} chars")
    print(f"  Avg chosen len : {avg_chosen_len:.0f} chars")
    print(f"  Avg rejected len: {avg_rejected_len:.0f} chars")

    save_dpo_data(all_pairs, output_path)

    print(f"\n[done] DPO data ready at {output_path}")
    print(f"  Next: python training/dpo.py --base_checkpoint checkpoints/dizel-sft-best.pt")


if __name__ == "__main__":
    main()
