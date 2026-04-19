"""
data/prepare_v11.py — Combine all HuggingFace datasets into a single SFT JSONL
for Dizel v1.1 training.

Target distribution:
    40% conversation   — UltraChat, OASST2
    25% knowledge      — Dolly, OpenOrca
    25% programming    — CodeAlpaca, CodeFeedback
    10% grammar/writing — Alpaca-GPT4, CoEdIT

Output:
    sft_data/chat_v11.jsonl   — unified SFT data
    data/pretrain_v11.txt     — plain text for pretraining (all responses concatenated)

Usage:
    python data/prepare_v11.py
    python data/prepare_v11.py --max-total 500000    # cap total samples
    python data/prepare_v11.py --dry-run              # just print counts

NOTE: Designed to run on machines with as little as 4 GB RAM.
      All parquet files are read in small row-group batches.
"""

import argparse
import gc
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW      = Path("data/raw")
OUT_SFT  = Path("sft_data/chat_v11.jsonl")
OUT_PRE  = Path("data/pretrain_v11.txt")

SYSTEM_PROMPT = (
    "You are Dizel, a structured analytical AI model. "
    "Prioritize clarity, precision, and logical organization. "
    "Use structured formatting when appropriate. "
    "Avoid unnecessary verbosity. "
    "If uncertain, explicitly state limitations."
)


def make_msg(role: str, content: str) -> Dict:
    return {"role": role, "content": content.strip()}


def make_sample(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> Optional[Dict]:
    """Create a single SFT sample in Dizel format."""
    user = user.strip()
    assistant = assistant.strip()
    if not user or not assistant:
        return None
    if len(assistant) < 5:
        return None
    msgs = [make_msg("system", system), make_msg("user", user), make_msg("assistant", assistant)]
    return {"messages": msgs}


def make_multi_turn(turns: List[Dict], system: str = SYSTEM_PROMPT) -> Optional[Dict]:
    """Create a multi-turn SFT sample."""
    if not turns or len(turns) < 2:
        return None
    msgs = [make_msg("system", system)]
    for t in turns:
        role = t.get("role", "")
        content = t.get("content", "")
        if role in ("user", "assistant") and content.strip():
            msgs.append(make_msg(role, content))
    if len(msgs) < 3:
        return None
    return {"messages": msgs}


def _read_parquet_batched(path, columns=None, batch_size=2000):
    """
    Memory-efficient parquet reader. Yields rows as dicts, reading
    one row-group at a time via pyarrow (never loads full file).
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        tbl = batch.to_pydict()
        n = len(next(iter(tbl.values())))
        for i in range(n):
            yield {k: tbl[k][i] for k in tbl}


# ===========================================================================
# Dataset loaders
# ===========================================================================

# ── 1. UltraChat (parquet with "messages" column in data/ subdir) ─────────
def load_ultrachat(max_samples: int = 80_000) -> List[Dict]:
    """40% conversation — multi-turn dialogues."""
    print("  Loading UltraChat...", end=" ", flush=True)
    folder = RAW / "ultrachat" / "data"
    if not folder.exists():
        folder = RAW / "ultrachat"

    # Use the SFT split files specifically
    parquets = sorted(folder.glob("train_sft*.parquet"))
    if not parquets:
        parquets = sorted(folder.glob("*.parquet"))
    if not parquets:
        print(f"SKIP (no parquet files)")
        return []

    samples = []
    for pq_path in parquets:
        try:
            for row in _read_parquet_batched(pq_path, columns=["messages"]):
                turns = row.get("messages", [])
                if isinstance(turns, list) and len(turns) >= 2:
                    sample = make_multi_turn(turns)
                    if sample:
                        samples.append(sample)
                if len(samples) >= max_samples:
                    break
        except Exception as e:
            print(f"\n    Warning: {pq_path.name}: {e}", flush=True)
            continue
        gc.collect()
        if len(samples) >= max_samples:
            break

    print(f"{len(samples):,} samples")
    return samples[:max_samples]


# ── 2. OASST2 (parquet in data/ subdir) ───────────────────────────────────
def load_oasst2(max_samples: int = 30_000) -> List[Dict]:
    """40% conversation — real human/AI multi-turn convos."""
    print("  Loading OASST2...", end=" ", flush=True)

    # Try parquet first (simpler, in data/ subdir)
    pq_dir = RAW / "oasst2" / "data"
    parquets = sorted(pq_dir.glob("*.parquet")) if pq_dir.exists() else []

    if parquets:
        return _load_oasst2_parquet(parquets, max_samples)

    # Fallback: gzipped trees
    folder = RAW / "oasst2"
    ready = folder / "2023-11-05_oasst2_ready.trees.jsonl.gz"
    if not ready.exists():
        gz_files = list(folder.glob("*.trees.jsonl.gz"))
        if not gz_files:
            print(f"SKIP (no data)")
            return []
        ready = gz_files[0]

    return _load_oasst2_trees(ready, max_samples)


def _load_oasst2_parquet(parquets, max_samples):
    """Load from parquet — each row has text, role, parent_id, message_tree_id."""
    # Build conversation trees from flat messages
    trees = {}  # message_tree_id -> list of messages
    for pq_path in parquets:
        try:
            for row in _read_parquet_batched(pq_path):
                tree_id = row.get("message_tree_id", "")
                if tree_id not in trees:
                    trees[tree_id] = []
                role = "user" if row.get("role") == "prompter" else "assistant"
                trees[tree_id].append({
                    "role": role,
                    "content": row.get("text", ""),
                    "parent_id": row.get("parent_id"),
                    "message_id": row.get("message_id"),
                    "rank": row.get("rank") if row.get("rank") is not None else 999,
                })
        except Exception as e:
            print(f"\n    Warning: {pq_path.name}: {e}", flush=True)

    # Extract best conversation path from each tree
    samples = []
    for tree_id, messages in trees.items():
        # Find root (no parent)
        roots = [m for m in messages if not m.get("parent_id")]
        if not roots:
            continue
        root = roots[0]
        # Build conversation by following best-ranked replies
        conv = _build_oasst_conv(root, messages)
        if len(conv) >= 2:
            sample = make_multi_turn(conv)
            if sample:
                samples.append(sample)
        if len(samples) >= max_samples:
            break

    gc.collect()
    print(f"{len(samples):,} samples")
    return samples[:max_samples]


def _build_oasst_conv(node, all_messages):
    """Build a linear conversation from a tree by picking the best-ranked reply."""
    conv = [{"role": node["role"], "content": node["content"]}]
    msg_id = node.get("message_id")

    while True:
        # Find children of current message
        children = [m for m in all_messages if m.get("parent_id") == msg_id]
        if not children:
            break
        # Pick best ranked
        best = min(children, key=lambda m: m.get("rank") if m.get("rank") is not None else 999)
        conv.append({"role": best["role"], "content": best["content"]})
        msg_id = best.get("message_id")

    return conv


def _load_oasst2_trees(gz_path, max_samples):
    """Fallback: load from gzipped tree JSONL."""
    samples = []
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                tree = json.loads(line)
                conv = _oasst_extract_best_path(tree)
                if conv and len(conv) >= 2:
                    sample = make_multi_turn(conv)
                    if sample:
                        samples.append(sample)
                if len(samples) >= max_samples:
                    break
    except Exception as e:
        print(f"ERROR ({e})")
        return []
    print(f"{len(samples):,} samples")
    return samples[:max_samples]


def _oasst_extract_best_path(node: dict) -> List[Dict]:
    """DFS through OASST2 tree, picking the highest-ranked reply."""
    result = []
    role = "user" if node.get("role") == "prompter" else "assistant"
    text = node.get("text", "")
    if text.strip():
        result.append({"role": role, "content": text})
    replies = node.get("replies", [])
    if replies:
        best = sorted(replies, key=lambda r: r.get("rank", 999))[0]
        result.extend(_oasst_extract_best_path(best))
    return result


# ── 3. Dolly 15K (JSONL) ──────────────────────────────────────────────────
def load_dolly(max_samples: int = 15_000) -> List[Dict]:
    """25% knowledge — instruction following."""
    print("  Loading Dolly...", end=" ", flush=True)
    path = RAW / "dolly" / "databricks-dolly-15k.jsonl"
    if not path.exists():
        print(f"SKIP")
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            instruction = d.get("instruction", "")
            context = d.get("context", "")
            response = d.get("response", "")
            user_msg = instruction
            if context:
                user_msg = f"{instruction}\n\nContext:\n{context}"
            sample = make_sample(user_msg, response)
            if sample:
                samples.append(sample)
            if len(samples) >= max_samples:
                break

    print(f"{len(samples):,} samples")
    return samples[:max_samples]


# ── 4. OpenOrca (large parquet — MUST use batched reading) ─────────────────
def load_openorca(max_samples: int = 60_000) -> List[Dict]:
    """25% knowledge — GPT-4 augmented instruction following."""
    print("  Loading OpenOrca...", end=" ", flush=True)
    folder = RAW / "openorca"
    # Prefer the GPT4 augmented file (higher quality, ~1GB)
    gpt4_pq = folder / "1M-GPT4-Augmented.parquet"
    if not gpt4_pq.exists():
        parquets = list(folder.glob("*.parquet"))
        if not parquets:
            print(f"SKIP")
            return []
        gpt4_pq = parquets[0]

    samples = []
    count = 0
    try:
        for row in _read_parquet_batched(
            gpt4_pq,
            columns=["system_prompt", "question", "response"],
            batch_size=1000,
        ):
            count += 1
            # Reservoir sampling: keep random subset without loading all rows
            if len(samples) < max_samples:
                sys_prompt = row.get("system_prompt", SYSTEM_PROMPT) or SYSTEM_PROMPT
                question = row.get("question", "")
                response = row.get("response", "")
                sample = make_sample(question, response, system=sys_prompt)
                if sample:
                    samples.append(sample)
            elif random.random() < max_samples / count:
                # Replace a random existing sample (reservoir sampling)
                idx = random.randint(0, max_samples - 1)
                sys_prompt = row.get("system_prompt", SYSTEM_PROMPT) or SYSTEM_PROMPT
                question = row.get("question", "")
                response = row.get("response", "")
                sample = make_sample(question, response, system=sys_prompt)
                if sample:
                    samples[idx] = sample

            # Progress indicator every 50K rows
            if count % 50_000 == 0:
                print(f"{count//1000}K..", end="", flush=True)
    except Exception as e:
        print(f"\n    Warning: {e}", flush=True)

    gc.collect()
    print(f" {len(samples):,} samples (from {count:,} rows)")
    return samples[:max_samples]


# ── 5. CodeAlpaca 20K (JSON array) ────────────────────────────────────────
def load_codealpaca(max_samples: int = 20_000) -> List[Dict]:
    """25% programming — code instruction pairs."""
    print("  Loading CodeAlpaca...", end=" ", flush=True)
    path = RAW / "codealpaca" / "code_alpaca_20k.json"
    if not path.exists():
        print(f"SKIP")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for d in data:
        instruction = d.get("instruction", "")
        inp = d.get("input", "")
        output = d.get("output", "")
        user_msg = instruction
        if inp:
            user_msg = f"{instruction}\n\nInput:\n{inp}"
        sample = make_sample(user_msg, output)
        if sample:
            samples.append(sample)
        if len(samples) >= max_samples:
            break

    print(f"{len(samples):,} samples")
    return samples[:max_samples]


# ── 6. CodeFeedback (JSONL — streamed line by line) ────────────────────────
def load_codefeedback(max_samples: int = 60_000) -> List[Dict]:
    """25% programming — code debugging and explanation."""
    print("  Loading CodeFeedback...", end=" ", flush=True)
    path = RAW / "codefeedback" / "CodeFeedback-Filtered-Instruction.jsonl"
    if not path.exists():
        print(f"SKIP")
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            query = d.get("query", d.get("instruction", d.get("question", "")))
            answer = d.get("answer", d.get("response", d.get("output", "")))
            sample = make_sample(query, answer)
            if sample:
                samples.append(sample)
            if len(samples) >= max_samples:
                break

    print(f"{len(samples):,} samples")
    return samples[:max_samples]


# ── 7. Alpaca GPT-4 (parquet in data/ subdir) ─────────────────────────────
def load_alpaca_gpt4(max_samples: int = 50_000) -> List[Dict]:
    """10% grammar/writing — high quality writing/reasoning."""
    print("  Loading Alpaca-GPT4...", end=" ", flush=True)
    folder = RAW / "alpaca_gpt4"

    # Check data/ subdir first
    pq_dir = folder / "data"
    parquets = sorted(pq_dir.glob("*.parquet")) if pq_dir.exists() else []
    if not parquets:
        parquets = sorted(folder.rglob("*.parquet"))
    if not parquets:
        # Fallback: JSON
        json_files = list(folder.rglob("*.json"))
        if json_files:
            return _load_alpaca_json(json_files, max_samples)
        print("SKIP")
        return []

    samples = []
    for pq_path in parquets:
        try:
            for row in _read_parquet_batched(pq_path, columns=["instruction", "input", "output"]):
                instruction = row.get("instruction", "")
                inp = row.get("input", "")
                output = row.get("output", "")
                user_msg = instruction
                if inp:
                    user_msg = f"{instruction}\n\nInput:\n{inp}"
                sample = make_sample(user_msg, output)
                if sample:
                    samples.append(sample)
                if len(samples) >= max_samples:
                    break
        except Exception as e:
            print(f"\n    Warning: {pq_path.name}: {e}", flush=True)
        gc.collect()
        if len(samples) >= max_samples:
            break

    print(f"{len(samples):,} samples")
    return samples[:max_samples]


def _load_alpaca_json(json_files, max_samples):
    samples = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for d in data:
                instruction = d.get("instruction", "")
                inp = d.get("input", "")
                output = d.get("output", "")
                user_msg = instruction
                if inp:
                    user_msg = f"{instruction}\n\nInput:\n{inp}"
                sample = make_sample(user_msg, output)
                if sample:
                    samples.append(sample)
                if len(samples) >= max_samples:
                    break
        if len(samples) >= max_samples:
            break
    print(f"{len(samples):,} samples")
    return samples[:max_samples]


# ── 8. CoEdIT (JSONL — grammar corrections) ───────────────────────────────
def load_coedit(max_samples: int = 60_000) -> List[Dict]:
    """10% grammar/writing — grammar correction pairs."""
    print("  Loading CoEdIT...", end=" ", flush=True)
    path = RAW / "coedit" / "train.jsonl"
    if not path.exists():
        print(f"SKIP")
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src = d.get("src", d.get("input", ""))
            tgt = d.get("tgt", d.get("output", d.get("target", "")))
            sample = make_sample(src, tgt)
            if sample:
                samples.append(sample)
            if len(samples) >= max_samples:
                break

    print(f"{len(samples):,} samples")
    return samples[:max_samples]


# ── 9. Existing Dizel SFT data ────────────────────────────────────────────
def load_existing_sft() -> List[Dict]:
    """Load existing hand-crafted Dizel SFT examples."""
    print("  Loading existing Dizel SFT data...", end=" ", flush=True)
    path = Path("sft_data/chat_expanded.jsonl")
    if not path.exists():
        print("SKIP")
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"{len(samples):,} samples")
    return samples


# ===========================================================================
# Sampling to target distribution
# ===========================================================================
def sample_to_target(
    conversation: List[Dict],
    knowledge: List[Dict],
    programming: List[Dict],
    writing: List[Dict],
    existing: List[Dict],
    max_total: int,
) -> List[Dict]:
    """
    Sample from each category to approximate the target distribution:
        40% conversation, 25% knowledge, 25% programming, 10% writing
    Existing Dizel data is always included in full.
    """
    remaining = max_total - len(existing)
    if remaining <= 0:
        return existing

    targets = {
        "conversation": int(remaining * 0.40),
        "knowledge":    int(remaining * 0.25),
        "programming":  int(remaining * 0.25),
        "writing":      int(remaining * 0.10),
    }

    pools = {
        "conversation": conversation,
        "knowledge":    knowledge,
        "programming":  programming,
        "writing":      writing,
    }

    result = list(existing)
    for category, target_count in targets.items():
        pool = pools[category]
        random.shuffle(pool)
        taken = pool[:target_count]
        result.extend(taken)
        print(f"  {category}: {len(taken):,} / {len(pool):,} available (target: {target_count:,})")

    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Prepare Dizel v1.1 training data")
    parser.add_argument("--max-total", type=int, default=300_000,
                        help="Maximum total SFT samples (default: 300K)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print counts, don't write files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-text", action="store_true", default=True,
                        help="Also export plain text for pretraining")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"\n{'='*60}")
    print(f"  Dizel v1.1 — Data Preparation (low-memory mode)")
    print(f"  Max total samples: {args.max_total:,}")
    print(f"{'='*60}\n")

    print("Loading datasets:\n")

    # Conversation (40%)
    ultrachat = load_ultrachat(max_samples=80_000)
    gc.collect()
    oasst2 = load_oasst2(max_samples=30_000)
    gc.collect()
    conversation = ultrachat + oasst2

    # Knowledge (25%)
    dolly = load_dolly(max_samples=15_000)
    openorca = load_openorca(max_samples=60_000)
    gc.collect()
    knowledge = dolly + openorca

    # Programming (25%)
    codealpaca = load_codealpaca(max_samples=20_000)
    codefeedback = load_codefeedback(max_samples=60_000)
    gc.collect()
    programming = codealpaca + codefeedback

    # Writing/Grammar (10%)
    alpaca = load_alpaca_gpt4(max_samples=50_000)
    gc.collect()
    coedit = load_coedit(max_samples=60_000)
    writing = alpaca + coedit

    # Existing Dizel data
    existing = load_existing_sft()

    total_raw = len(conversation) + len(knowledge) + len(programming) + len(writing) + len(existing)
    print(f"\n{'─'*60}")
    print(f"Raw totals:")
    print(f"  Conversation:  {len(conversation):,}")
    print(f"  Knowledge:     {len(knowledge):,}")
    print(f"  Programming:   {len(programming):,}")
    print(f"  Writing:       {len(writing):,}")
    print(f"  Existing:      {len(existing):,}")
    print(f"  TOTAL RAW:     {total_raw:,}")
    print(f"{'─'*60}\n")

    if args.dry_run:
        print("Dry run — not writing files.")
        return

    # Sample to target distribution
    print("Sampling to target distribution:\n")
    final = sample_to_target(conversation, knowledge, programming, writing, existing, args.max_total)

    # Free the pools
    del conversation, knowledge, programming, writing, existing
    del ultrachat, oasst2, dolly, openorca, codealpaca, codefeedback, alpaca, coedit
    gc.collect()

    random.shuffle(final)

    # Write SFT output
    OUT_SFT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SFT, "w", encoding="utf-8") as f:
        for sample in final:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    size_mb = OUT_SFT.stat().st_size / (1024 * 1024)
    print(f"\n✅ SFT data written to: {OUT_SFT}")
    print(f"   Samples: {len(final):,}")
    print(f"   Size:    {size_mb:.1f} MB")

    # Write pretrain plain text
    if args.pretrain_text:
        OUT_PRE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PRE, "w", encoding="utf-8") as f:
            for sample in final:
                for msg in sample.get("messages", []):
                    if msg["role"] == "assistant":
                        f.write(msg["content"] + "\n\n")
        pre_mb = OUT_PRE.stat().st_size / (1024 * 1024)
        print(f"\n✅ Pretrain text written to: {OUT_PRE}")
        print(f"   Size: {pre_mb:.1f} MB")

    print(f"\n💡 Remember to retrain the tokenizer:")
    print(f"   python tokenizer/train_tokenizer.py")
    print()


if __name__ == "__main__":
    main()
