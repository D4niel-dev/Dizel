"""
scripts/migrate_datasets.py â€” Core functions for Dizel dataset migration.

Converts multiple HuggingFace datasets into a unified chat-style JSONL format
compatible with training/dataset.py SFTDataset.

Usage (standalone):
    python scripts/migrate_datasets.py --base_dir /content/drive/MyDrive/Dizel

Usage (from Colab notebook):
    from scripts.migrate_datasets import *
"""

import json
import os
import re
import hashlib
import random
from collections import Counter
from typing import List, Dict, Optional, Callable

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM = "You are Dizel, a helpful, clear, and conversational AI assistant."

# Boilerplate phrases to strip from assistant responses
BOILERPLATE = [
    "As an AI language model,",
    "As an artificial intelligence,",
    "I'm just an AI,",
    "I don't have personal opinions,",
    "As a large language model,",
    "As an AI,",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_parquet(path: str, sample_n: Optional[int] = None) -> "pd.DataFrame":
    """Load a parquet file into a pandas DataFrame, optionally sampling."""
    import pandas as pd
    df = pd.read_parquet(path)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    return df


def load_parquet_dir(directory: str, pattern: str = "*.parquet",
                     sample_n: Optional[int] = None) -> "pd.DataFrame":
    """Load all parquet files matching a pattern from a directory (recursive)."""
    import pandas as pd
    import glob
    files = sorted(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
    # Skip cache/lock/metadata files
    files = [f for f in files if ".cache" not in f and not f.endswith(".lock")
             and not f.endswith(".metadata")]
    if not files:
        return None
    print(f"  Loading {len(files)} parquet file(s) from {directory}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")
    return df


def load_json_file(path: str, sample_n: Optional[int] = None) -> "pd.DataFrame":
    """Load a single .json file (list of objects) into a DataFrame."""
    import pandas as pd
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


def load_jsonl_to_df(path: str, sample_n: Optional[int] = None) -> "pd.DataFrame":
    """Load a .jsonl file (one JSON object per line) into a DataFrame."""
    import pandas as pd
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


def load_dataset_dir(directory: str, sample_n: Optional[int] = None) -> "pd.DataFrame":
    """
    Auto-detect file format and load from a dataset directory.
    Priority: parquet > jsonl > json
    """
    import pandas as pd
    import glob

    # Try parquet first
    df = load_parquet_dir(directory, sample_n=sample_n)
    if df is not None:
        return df

    # Try .jsonl files (skip README, .gitattributes)
    jsonl_files = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if jsonl_files:
        dfs = [load_jsonl_to_df(f, sample_n=sample_n) for f in jsonl_files]
        df = pd.concat(dfs, ignore_index=True)
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        return df

    # Try .json files (skip .gitattributes, README)
    json_files = sorted(glob.glob(os.path.join(directory, "*.json")))
    json_files = [f for f in json_files if not os.path.basename(f).startswith(".")]
    if json_files:
        dfs = [load_json_file(f, sample_n=sample_n) for f in json_files]
        df = pd.concat(dfs, ignore_index=True)
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        return df

    raise FileNotFoundError(f"No parquet, jsonl, or json files found in {directory}")


def load_existing_jsonl(path: str) -> List[Dict]:
    """Load an existing JSONL file (for merging handcrafted data)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" in obj:
                    samples.append(obj)
            except json.JSONDecodeError:
                continue
    print(f"  Loaded {len(samples):,} samples from {os.path.basename(path)}")
    return samples


def inspect_schema(df: "pd.DataFrame", name: str = "dataset"):
    """Print schema info and a sample row for manual inspection."""
    print(f"\n{'='*60}")
    print(f"  Schema: {name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Sample row:")
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        val_str = str(v)[:120]
        print(f"    {k}: {val_str}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean a single text field while preserving code blocks."""
    if not text or not isinstance(text, str):
        return ""

    # Separate code blocks to protect them
    code_blocks = []
    def _save_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    text = re.sub(r"```[\s\S]*?```", _save_code, text)

    # Strip null bytes and broken unicode
    text = text.replace("\x00", "")
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    # Remove boilerplate
    for bp in BOILERPLATE:
        text = text.replace(bp, "").replace(bp.lower(), "")

    # Normalize whitespace (but keep newlines)
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal space
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
    text = re.sub(r"<[^>]+>", "", text)           # strip HTML tags

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)

    return text.strip()


def clean_messages(messages: List[Dict]) -> List[Dict]:
    """Clean all content fields in a message list."""
    cleaned = []
    for msg in messages:
        content = clean_text(msg.get("content", ""))
        if content:
            cleaned.append({"role": msg["role"], "content": content})
    return cleaned


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_sample(messages: List[Dict],
                  min_chars: int = 20,
                  max_chars: int = 32000) -> bool:
    """Return True if sample passes quality checks."""
    if not messages:
        return False

    # Must have at least user + assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return False

    # No empty content
    for m in messages:
        if not m.get("content", "").strip():
            return False

    # Length checks
    total = sum(len(m["content"]) for m in messages)
    if total < min_chars or total > max_chars:
        return False

    # Reject if assistant just copies the user prompt
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    if user_msgs and asst_msgs:
        if user_msgs[0].strip() == asst_msgs[0].strip():
            return False

    # Reject encoding artifacts
    for m in messages:
        if "\ufffd" in m["content"]:
            return False

    # Reject excessive repetition
    for m in messages:
        words = m["content"].split()
        if len(words) > 10:
            counter = Counter(words)
            most_common_count = counter.most_common(1)[0][1]
            if most_common_count > len(words) * 0.5:
                return False

    return True


# ---------------------------------------------------------------------------
# Converters (per dataset type)
# ---------------------------------------------------------------------------
def make_messages(system: str, user: str, assistant: str) -> List[Dict]:
    """Build a standard 3-turn message list."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

def _batch_convert(df: "pd.DataFrame", row_fn, desc: str = "Converting",
                   chunk_size: int = 50000, debug: bool = False) -> List[Dict]:
    """
    Fast chunked converter. Processes DataFrame in chunks, applies row_fn
    to each row, and prints progress to prevent Colab idle timeout.
    """
    samples = []
    total = len(df)
    debugged = False
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = df.iloc[start:end]
        for _, row in chunk.iterrows():
            if debug and not debugged:
                # Print first-row debug info
                print(f"  [DEBUG] First row keys: {list(row.index)}")
                for col in row.index:
                    val = row[col]
                    print(f"  [DEBUG] {col}: type={type(val).__name__}, repr={repr(val)[:200]}")
                debugged = True
            result = row_fn(row)
            if result is not None:
                samples.append(result)
        print(f"  {desc}: {end:,}/{total:,} rows processed, {len(samples):,} kept")
    return samples


def convert_instruction_dataset(df: "pd.DataFrame",
                                 instruction_col: str = "instruction",
                                 input_col: str = "input",
                                 output_col: str = "output",
                                 system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """Convert instruction/input/output schema (alpaca, codealpaca)."""
    def _convert(row):
        inst = str(row.get(instruction_col, "")).strip()
        inp = str(row.get(input_col, "")).strip()
        out = str(row.get(output_col, "")).strip()
        if not inst or not out:
            return None
        user_text = inst
        if inp and inp.lower() not in ("nan", "none", ""):
            user_text = f"{inst}\n\n{inp}"
        msgs = make_messages(system, user_text, out)
        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            return {"messages": msgs}
        return None
    return _batch_convert(df, _convert, "instruction")


def convert_coedit_dataset(df: "pd.DataFrame",
                            system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """Convert coedit src/tgt schema."""
    def _convert(row):
        src = str(row.get("src", "")).strip()
        tgt = str(row.get("tgt", "")).strip()
        if not src or not tgt:
            return None
        msgs = make_messages(system, src, tgt)
        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            return {"messages": msgs}
        return None
    return _batch_convert(df, _convert, "coedit")


def convert_dolly_dataset(df: "pd.DataFrame",
                           system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """Convert dolly instruction/context/response schema."""
    def _convert(row):
        inst = str(row.get("instruction", "")).strip()
        ctx = str(row.get("context", "")).strip()
        resp = str(row.get("response", "")).strip()
        if not inst or not resp:
            return None
        user_text = inst
        if ctx and ctx.lower() not in ("nan", "none", ""):
            user_text = f"{inst}\n\nContext:\n{ctx}"
        msgs = make_messages(system, user_text, resp)
        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            return {"messages": msgs}
        return None
    return _batch_convert(df, _convert, "dolly")


def convert_openorca_dataset(df: "pd.DataFrame",
                              system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """Convert OpenOrca â€” replaces system with Dizel default. Chunked for speed."""
    def _convert(row):
        question = str(row.get("question", "")).strip()
        response = str(row.get("response", "")).strip()
        if not question or not response:
            return None
        msgs = make_messages(system, question, response)
        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            return {"messages": msgs}
        return None
    return _batch_convert(df, _convert, "openorca")


def convert_oasst2_dataset(df: "pd.DataFrame",
                            system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """
    Convert OASST2 tree-structured conversations into linear threads.
    Flattens parent->child chains and maps prompter->user, assistant->assistant.
    """
    # Filter to English
    if "lang" in df.columns:
        df = df[df["lang"] == "en"].copy()
        print(f"  Filtered to English: {len(df):,} rows")

    # Build parent->children mapping
    rows_by_id = {}
    children_map = {}
    for _, row in df.iterrows():
        msg_id = row.get("message_id", "")
        parent_id = row.get("parent_id", None)
        rows_by_id[msg_id] = row
        if parent_id and str(parent_id).lower() not in ("nan", "none", ""):
            children_map.setdefault(str(parent_id), []).append(msg_id)

    # Find root messages (no parent)
    roots = []
    for _, row in df.iterrows():
        pid = row.get("parent_id", None)
        if pid is None or str(pid).lower() in ("nan", "none", ""):
            roots.append(row.get("message_id", ""))

    # Build threads by following from root to deepest child
    def build_threads(msg_id, current_thread):
        row = rows_by_id.get(msg_id)
        if row is None:
            return []
        current_thread = current_thread + [row]
        kids = children_map.get(str(msg_id), [])
        if not kids:
            return [current_thread]
        # Follow all children paths (creates multiple threads from branches)
        threads = []
        for kid in kids:
            threads.extend(build_threads(kid, current_thread))
        return threads

    all_threads = []
    for root_id in tqdm(roots, desc="Flattening oasst2 trees"):
        all_threads.extend(build_threads(root_id, []))

    print(f"  Extracted {len(all_threads):,} conversation threads")

    # Convert threads to message format
    samples = []
    for thread in all_threads:
        msgs = [{"role": "system", "content": system}]
        valid = True
        for row in thread:
            role_raw = str(row.get("role", "")).lower()
            text = str(row.get("text", "")).strip()
            if not text:
                valid = False
                break
            if role_raw == "prompter":
                role = "user"
            elif role_raw == "assistant":
                role = "assistant"
            else:
                valid = False
                break
            msgs.append({"role": role, "content": text})

        if not valid:
            continue

        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            samples.append({"messages": msgs})

    return samples


def convert_ultrachat_dataset(df: "pd.DataFrame",
                               system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """
    Convert UltraChat dataset. Handles:
      A) list of dicts with role+content
      B) list of dicts with content only (no role) -> alternating user/assistant
      C) list of plain strings -> alternating user/assistant
    """
    def _convert(row):
        raw_msgs = row.get("messages", None)
        if raw_msgs is None:
            raw_msgs = row.get("data", row.get("conversation", None))
        if raw_msgs is None:
            return None
        if isinstance(raw_msgs, str):
            try:
                raw_msgs = json.loads(raw_msgs)
            except json.JSONDecodeError:
                return None
        # Convert numpy arrays to Python lists
        if hasattr(raw_msgs, 'tolist'):
            raw_msgs = raw_msgs.tolist()
        if not isinstance(raw_msgs, list) or len(raw_msgs) < 2:
            return None
        msgs = [{"role": "system", "content": system}]
        if isinstance(raw_msgs[0], dict):
            # Check if role key exists
            has_roles = "role" in raw_msgs[0]
            if has_roles:
                # Format A: dicts with role+content
                for m in raw_msgs:
                    role = str(m.get("role", "")).lower()
                    content = str(m.get("content", "")).strip()
                    if role == "system":
                        continue
                    if role in ("user", "assistant") and content:
                        msgs.append({"role": role, "content": content})
            else:
                # Format B: dicts with content only, alternate user/assistant
                turn_idx = 0
                for m in raw_msgs:
                    content = str(m.get("content", "")).strip()
                    if not content:
                        continue
                    role = "user" if turn_idx % 2 == 0 else "assistant"
                    msgs.append({"role": role, "content": content})
                    turn_idx += 1
        elif isinstance(raw_msgs[0], str):
            # Format C: plain strings, alternate user/assistant
            for i, text in enumerate(raw_msgs):
                text = str(text).strip()
                if not text:
                    continue
                role = "user" if i % 2 == 0 else "assistant"
                msgs.append({"role": role, "content": text})
        else:
            return None
        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            return {"messages": msgs}
        return None
    return _batch_convert(df, _convert, "ultrachat", debug=True)


def convert_codefeedback_dataset(df: "pd.DataFrame",
                                  system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """Convert CodeFeedback â€” auto-detects column names."""
    cols = set(df.columns)
    if "instruction" in cols and "answer" in cols:
        inst_col, out_col = "instruction", "answer"
    elif "instruction" in cols and "output" in cols:
        inst_col, out_col = "instruction", "output"
    elif "query" in cols and "answer" in cols:
        inst_col, out_col = "query", "answer"
    elif "question" in cols and "answer" in cols:
        inst_col, out_col = "question", "answer"
    else:
        print(f"  [WARN] Unknown codefeedback schema: {cols}")
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        if len(text_cols) >= 2:
            inst_col, out_col = text_cols[0], text_cols[1]
        else:
            return []

    def _convert(row):
        inst = str(row.get(inst_col, "")).strip()
        out = str(row.get(out_col, "")).strip()
        if not inst or not out:
            return None
        msgs = make_messages(system, inst, out)
        msgs = clean_messages(msgs)
        if filter_sample(msgs):
            return {"messages": msgs}
        return None
    return _batch_convert(df, _convert, "codefeedback")


# ---------------------------------------------------------------------------
# Normalization for existing SFT files
# ---------------------------------------------------------------------------
def normalize_existing_sft(samples: List[Dict],
                            system: str = DEFAULT_SYSTEM) -> List[Dict]:
    """
    Re-normalize existing SFT data: replace system prompt with the Dizel default,
    clean content, and filter.
    """
    out = []
    for s in tqdm(samples, desc="Normalizing existing SFT"):
        msgs = s.get("messages", [])
        if not msgs:
            continue

        normalized = []
        has_system = False
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                normalized.append({"role": "system", "content": system})
                has_system = True
            else:
                normalized.append({"role": role, "content": content})

        if not has_system:
            normalized.insert(0, {"role": "system", "content": system})

        normalized = clean_messages(normalized)
        if filter_sample(normalized):
            out.append({"messages": normalized})
    return out


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def content_hash(sample: Dict) -> str:
    """Create MD5 hash of concatenated message content for exact dedup."""
    parts = []
    for m in sample.get("messages", []):
        parts.append(f"{m['role']}:{m['content']}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def deduplicate_exact(samples: List[Dict]) -> List[Dict]:
    """Remove exact duplicates based on content hash."""
    seen = set()
    unique = []
    for s in tqdm(samples, desc="Exact dedup"):
        h = content_hash(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    removed = len(samples) - len(unique)
    print(f"  Exact dedup: {len(samples):,} -> {len(unique):,} (removed {removed:,})")
    return unique


def deduplicate_fuzzy(samples: List[Dict],
                       threshold: float = 0.85,
                       num_perm: int = 128) -> List[Dict]:
    """
    Remove near-duplicates using MinHash LSH.
    Falls back to exact-only if datasketch is unavailable.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  [WARN] datasketch not installed, skipping fuzzy dedup")
        print("         Install with: pip install datasketch")
        return samples

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    keep_indices = set(range(len(samples)))

    print(f"  Building MinHash signatures for {len(samples):,} samples...")
    for i, s in enumerate(tqdm(samples, desc="MinHash")):
        text = " ".join(m["content"] for m in s["messages"] if m["role"] != "system")
        tokens = set(text.lower().split())
        mh = MinHash(num_perm=num_perm)
        for t in tokens:
            mh.update(t.encode("utf-8"))
        minhashes.append(mh)

        try:
            lsh.insert(str(i), mh)
        except ValueError:
            # Duplicate detected by LSH
            keep_indices.discard(i)

    # Query for near-duplicates
    for i in tqdm(range(len(samples)), desc="LSH query"):
        if i not in keep_indices:
            continue
        results = lsh.query(minhashes[i])
        for r in results:
            idx = int(r)
            if idx != i and idx in keep_indices:
                keep_indices.discard(idx)

    unique = [samples[i] for i in sorted(keep_indices)]
    removed = len(samples) - len(unique)
    print(f"  Fuzzy dedup: {len(samples):,} -> {len(unique):,} (removed {removed:,})")
    return unique


# ---------------------------------------------------------------------------
# Quality Audit
# ---------------------------------------------------------------------------
def audit_samples(samples: List[Dict], source_field: str = "source"):
    """Print quality audit statistics."""
    print(f"\n{'='*60}")
    print(f"  QUALITY AUDIT â€” {len(samples):,} total samples")
    print(f"{'='*60}")

    # Source distribution
    if any(source_field in s for s in samples[:100]):
        sources = Counter(s.get(source_field, "unknown") for s in samples)
        print(f"\n  Source Distribution:")
        for src, cnt in sources.most_common():
            print(f"    {src:20s}: {cnt:>8,}")

    # Category distribution
    if any("category" in s for s in samples[:100]):
        cats = Counter(s.get("category", "unknown") for s in samples)
        print(f"\n  Category Distribution:")
        for cat, cnt in cats.most_common():
            print(f"    {cat:20s}: {cnt:>8,}")

    # Turn count distribution
    turn_counts = Counter(len(s["messages"]) for s in samples)
    print(f"\n  Turn Counts:")
    for tc, cnt in sorted(turn_counts.items()):
        print(f"    {tc} turns: {cnt:>8,}")

    # Length statistics
    lengths = [sum(len(m["content"]) for m in s["messages"]) for s in samples]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f"\n  Length Statistics:")
    print(f"    Min chars : {min(lengths):,}")
    print(f"    Max chars : {max(lengths):,}")
    print(f"    Avg chars : {avg_len:,.0f}")
    print(f"    Median    : {sorted(lengths)[len(lengths)//2]:,}")

    # Format validation
    errors = 0
    for s in samples:
        if not isinstance(s.get("messages"), list):
            errors += 1
            continue
        for m in s["messages"]:
            if m.get("role") not in ("system", "user", "assistant"):
                errors += 1
            if not m.get("content", "").strip():
                errors += 1
    print(f"\n  Format Errors: {errors}")

    # Random samples
    print(f"\n  --- 3 Random Samples ---")
    for s in random.sample(samples, min(3, len(samples))):
        print()
        for m in s["messages"]:
            preview = m["content"][:100].replace("\n", "\\n")
            print(f"    [{m['role']:>10s}] {preview}...")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_jsonl(samples: List[Dict], output_path: str,
                 strip_metadata: bool = False):
    """Write samples to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            obj = {"messages": s["messages"]}
            if not strip_metadata:
                if "source" in s:
                    obj["source"] = s["source"]
                if "category" in s:
                    obj["category"] = s["category"]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported {len(samples):,} samples to {output_path} ({size_mb:.1f} MB)")


def split_and_export(samples: List[Dict], base_dir: str,
                     train_ratio: float = 0.95, seed: int = 42):
    """Shuffle, split, and export train/valid/combined JSONL files."""
    random.seed(seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    train = samples[:split_idx]
    valid = samples[split_idx:]

    exports_dir = os.path.join(base_dir, "data", "exports")
    os.makedirs(exports_dir, exist_ok=True)

    export_jsonl(train, os.path.join(exports_dir, "train.jsonl"), strip_metadata=True)
    export_jsonl(valid, os.path.join(exports_dir, "valid.jsonl"), strip_metadata=True)
    export_jsonl(samples, os.path.join(exports_dir, "dizel_migrated.jsonl"), strip_metadata=True)

    print(f"\n  Final split: Train={len(train):,} | Valid={len(valid):,} | Total={len(samples):,}")
    return train, valid


# ---------------------------------------------------------------------------
# Helpers for disk-flush pipeline
# ---------------------------------------------------------------------------
def _flush_to_disk(converted, source, category, processed_dir):
    """Write samples to disk immediately and free memory."""
    if not converted:
        return 0
    out = os.path.join(processed_dir, f"{source}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for s in converted:
            f.write(json.dumps({"messages": s["messages"], "source": source,
                                "category": category}, ensure_ascii=False) + "\n")
    n = len(converted)
    converted.clear()
    import gc; gc.collect()
    print(f"   {source}: {n:,} samples  -> {os.path.basename(out)}")
    return n


def _stream_load_processed(processed_dir):
    """Stream-load all processed JSONL files from disk."""
    import glob as _g
    samples = []
    for fpath in sorted(_g.glob(os.path.join(processed_dir, "*.jsonl"))):
        c = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                        c += 1
                    except json.JSONDecodeError:
                        pass
        print(f"    {os.path.basename(fpath)}: {c:,}")
    return samples


# ---------------------------------------------------------------------------
# Full Pipeline (standalone execution)
# ---------------------------------------------------------------------------
def _streaming_dedup_and_export(processed_dir, base_dir, expected_count):
    """
    Memory-safe dedup + export.

    Pass 1: Stream all JSONL files, compute content hashes, build a set of
            unique hashes. Only the hash set lives in RAM (~40 bytes × N).
    Pass 2: Stream again, writing only unique samples to export files.
            Also collects a small random sample for the quality audit.
    """
    import glob as _g
    import gc

    jsonl_files = sorted(_g.glob(os.path.join(processed_dir, "*.jsonl")))

    # --- Pass 1: Build hash set ---
    print("  Pass 1: Building dedup hash index...")
    seen_hashes = set()
    duplicate_count = 0
    total_scanned = 0

    for fpath in jsonl_files:
        fname = os.path.basename(fpath)
        fc = 0; fd = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue
                h = content_hash(sample)
                if h in seen_hashes:
                    fd += 1
                else:
                    seen_hashes.add(h)
                    fc += 1
                total_scanned += 1
        duplicate_count += fd
        print(f"    {fname}: {fc + fd:,} rows processed, {fd:,} duplicates")

    unique_count = len(seen_hashes)
    print(f"  Pass 1 done: {total_scanned:,} scanned, {unique_count:,} unique, {duplicate_count:,} duplicates removed")

    # --- Pass 2: Stream to export files ---
    print("\n  Pass 2: Streaming unique samples to export files...")
    exports_dir = os.path.join(base_dir, "data", "exports")
    os.makedirs(exports_dir, exist_ok=True)

    train_path = os.path.join(exports_dir, "train.jsonl")
    valid_path = os.path.join(exports_dir, "valid.jsonl")
    combined_path = os.path.join(exports_dir, "dizel_migrated.jsonl")

    # Use 95/5 train/val split
    # We'll assign each sample deterministically based on its hash
    random.seed(42)
    train_count = 0
    valid_count = 0
    written = 0
    audit_reservoir = []  # Small reservoir sample for audit (max 5000)
    AUDIT_SIZE = 5000

    seen_pass2 = set()  # Track what we've written to avoid writing dupes

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(valid_path, "w", encoding="utf-8") as f_valid, \
         open(combined_path, "w", encoding="utf-8") as f_all:

        for fpath in jsonl_files:
            fname = os.path.basename(fpath)
            fc = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = content_hash(sample)
                    if h in seen_pass2:
                        continue
                    seen_pass2.add(h)

                    # Write only messages to export (strip metadata)
                    out_line = json.dumps({"messages": sample["messages"]}, ensure_ascii=False) + "\n"

                    f_all.write(out_line)

                    # Deterministic split based on hash
                    if int(h[:8], 16) % 20 == 0:
                        f_valid.write(out_line)
                        valid_count += 1
                    else:
                        f_train.write(out_line)
                        train_count += 1

                    written += 1
                    fc += 1

                    # Reservoir sampling for audit
                    if len(audit_reservoir) < AUDIT_SIZE:
                        audit_reservoir.append(sample)
                    else:
                        j = random.randint(0, written - 1)
                        if j < AUDIT_SIZE:
                            audit_reservoir[j] = sample

            print(f"    {fname}: {fc:,} unique written")

    del seen_hashes; del seen_pass2; gc.collect()

    print(f"\n  Export complete:")
    print(f"    Train : {train_count:,}")
    print(f"    Valid : {valid_count:,}")
    print(f"    Total : {written:,}")
    print(f"    Files : {train_path}")
    print(f"            {valid_path}")
    print(f"            {combined_path}")

    # --- Lightweight audit on reservoir sample ---
    if audit_reservoir:
        print(f"\n  Quality Audit (on {len(audit_reservoir):,} sample reservoir)...")
        audit_samples(audit_reservoir)

    return written


def run_full_pipeline(base_dir: str):
    """Run the complete migration pipeline. Flushes each dataset to disk immediately."""
    raw_dir = os.path.join(base_dir, "data", "raw")
    sft_dir = os.path.join(base_dir, "sft_data")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    total_count = 0

    #  Phase 1: Existing SFT files  â”€â”€â”€â”€â”€
    print("\n[Phase 1] Merging existing SFT data...")
    for fname in ["chat.jsonl", "chat_expanded.jsonl", "chat_v11.jsonl"]:
        fpath = os.path.join(sft_dir, fname)
        if os.path.exists(fpath):
            raw = load_existing_jsonl(fpath)
            normalized = normalize_existing_sft(raw)
            safe = fname.replace(".jsonl", "").replace(".", "_")
            total_count += _flush_to_disk(normalized, f"sft_{safe}", "handcrafted", processed_dir)
            del raw; import gc; gc.collect()

    #  Phase 2: Instruction datasets  â”€â”€â”€
    print("\n[Phase 2] Processing instruction datasets...")

    for ds_name, converter, category in [
        ("alpaca_gpt4", convert_instruction_dataset, "instruction"),
        ("codealpaca", convert_instruction_dataset, "coding"),
        ("coedit", convert_coedit_dataset, "instruction"),
        ("dolly", convert_dolly_dataset, "instruction"),
    ]:
        ds_dir = os.path.join(raw_dir, ds_name)
        if os.path.isdir(ds_dir):
            print(f"\n  -> {ds_name}")
            df = load_dataset_dir(ds_dir)
            if df is not None:
                inspect_schema(df, ds_name)
                converted = converter(df)
                del df; import gc; gc.collect()
                total_count += _flush_to_disk(converted, ds_name, category, processed_dir)

    #  Phase 3: Conversational datasets  
    print("\n[Phase 3] Processing conversational datasets...")

    # oasst2
    ds_dir = os.path.join(raw_dir, "oasst2")
    if os.path.isdir(ds_dir):
        print("\n  -> oasst2")
        df = load_parquet_dir(ds_dir)
        if df is not None:
            inspect_schema(df, "oasst2")
            converted = convert_oasst2_dataset(df)
            del df; import gc; gc.collect()
            total_count += _flush_to_disk(converted, "oasst2", "chat", processed_dir)

    # ultrachat (stream to disk file-by-file)
    ds_dir = os.path.join(raw_dir, "ultrachat")
    if os.path.isdir(ds_dir):
        print("\n  -> ultrachat")
        import glob as _glob; import gc as _gc
        uc_files = sorted(_glob.glob(os.path.join(ds_dir, "**", "*.parquet"), recursive=True))
        uc_files = [f for f in uc_files if ".cache" not in f]
        uc_out = os.path.join(processed_dir, "ultrachat.jsonl")
        uc_total = 0
        with open(uc_out, "w", encoding="utf-8") as uf:
            for uc_f in uc_files:
                fn = os.path.basename(uc_f)
                print(f"    Processing {fn}...")
                import pandas as _pd
                cdf = _pd.read_parquet(uc_f)
                converted = convert_ultrachat_dataset(cdf)
                del cdf
                for s in converted:
                    uf.write(json.dumps({"messages": s["messages"], "source": "ultrachat",
                                         "category": "chat"}, ensure_ascii=False) + "\n")
                uc_total += len(converted)
                del converted; _gc.collect()
                print(f"     {fn}: total so far {uc_total:,}")
        total_count += uc_total
        print(f"   ultrachat total: {uc_total:,}")

    #  Phase 4: Complex datasets   
    print("\n[Phase 4] Processing complex datasets...")

    # openorca (iter_batches for controlled chunking, stream to disk)
    ds_dir = os.path.join(raw_dir, "openorca")
    if os.path.isdir(ds_dir):
        print("\n  -> openorca")
        import glob; import pyarrow.parquet as pq; import gc
        ofiles = sorted(glob.glob(os.path.join(ds_dir, "**", "*.parquet"), recursive=True))
        ofiles = [f for f in ofiles if ".cache" not in f]
        o_out = os.path.join(processed_dir, "openorca.jsonl")
        o_total = 0; sp = False
        BATCH = 50_000  # rows per batch — safe for T4 RAM
        with open(o_out, "w", encoding="utf-8") as of:
            for ofile in ofiles:
                fn = os.path.basename(ofile)
                pf = pq.ParquetFile(ofile)
                total_rows = pf.metadata.num_rows
                print(f"    {fn}: {total_rows:,} rows")
                fk = 0; batch_n = 0
                for batch in pf.iter_batches(batch_size=BATCH):
                    cdf = batch.to_pandas()
                    batch_n += 1
                    if not sp:
                        inspect_schema(cdf, f"openorca/{fn}")
                        sp = True
                    converted = convert_openorca_dataset(cdf)
                    del cdf
                    for s in converted:
                        of.write(json.dumps({"messages": s["messages"], "source": "openorca",
                                              "category": "reasoning"}, ensure_ascii=False) + "\n")
                    fk += len(converted)
                    del converted; gc.collect()
                    print(f"      batch {batch_n} ({min(batch_n*BATCH, total_rows):,}/{total_rows:,}): kept {fk:,}")
                o_total += fk
                print(f"    done {fn}: {fk:,}")
        total_count += o_total
        print(f"   “ openorca total: {o_total:,}")

    # codefeedback
    ds_dir = os.path.join(raw_dir, "codefeedback")
    if os.path.isdir(ds_dir):
        print("\n   ’ codefeedback")
        df = load_dataset_dir(ds_dir)
        inspect_schema(df, "codefeedback")
        converted = convert_codefeedback_dataset(df)
        del df; import gc; gc.collect()
        total_count += _flush_to_disk(converted, "codefeedback", "coding", processed_dir)

    #  Phase 5: Streaming Dedup + Export (memory-safe)
    print(f"\n[Phase 5] Streaming dedup + export ({total_count:,} expected)...")
    final_count = _streaming_dedup_and_export(processed_dir, base_dir, total_count)

    print("\n" + "=" * 60)
    print("  MIGRATION COMPLETE")
    print("=" * 60)
    return final_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Dizel Dataset Migration Pipeline")
    p.add_argument("--base_dir", type=str, required=True,
                   help="Base Dizel project directory")
    args = p.parse_args()
    run_full_pipeline(args.base_dir)
