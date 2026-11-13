"""Validate the materialised Dizel 1.6 post-training mixes.

Usage:

    python scripts/validate_v16_posttrain.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def expect(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_sft(path: str, meta: dict):
    count = 0
    tokens = 0
    families = Counter()
    hashes = set()
    family_tokens = defaultdict(int)
    for record in iter_jsonl(path):
        count += 1
        family = record.get("family")
        expect(family, f"{path}: record {count} has no family")
        expect(record.get("source"), f"{path}: record {count} has no source")
        expect(record.get("hash"), f"{path}: record {count} has no hash")
        expect(record["hash"] not in hashes, f"{path}: duplicate hash at record {count}")
        hashes.add(record["hash"])
        messages = record.get("messages")
        expect(isinstance(messages, list) and messages,
               f"{path}: record {count} has no messages")
        expect(messages[0].get("role") == "system",
               f"{path}: record {count} does not start with a system message")
        expect(messages[-1].get("role") == "assistant",
               f"{path}: record {count} does not end with an assistant message")
        for message in messages:
            expect(message.get("content"), f"{path}: record {count} has an empty message")
        assistant = int(record.get("assistant_tokens", 0))
        expect(assistant > 0, f"{path}: record {count} has no loss tokens")
        tokens += assistant
        families[family] += 1
        family_tokens[family] += assistant

    expect(count == meta["records"], f"{path}: record count {count} != {meta['records']}")
    expect(tokens == meta["actual_tokens"], f"{path}: token count {tokens} != {meta['actual_tokens']}")
    expect(sorted(families) == sorted(meta["families"]),
           f"{path}: family set differs from manifest")
    for family, family_meta in meta["families"].items():
        expect(families[family] == family_meta["records"],
               f"{path}: {family} records {families[family]} != {family_meta['records']}")
        expect(family_tokens[family] == family_meta["actual_tokens"],
               f"{path}: {family} tokens {family_tokens[family]} != {family_meta['actual_tokens']}")
    return count, tokens, hashes, families


def validate_distill(path: str, meta: dict, sft_hashes: set):
    count = 0
    tokens = 0
    hashes = set()
    for record in iter_jsonl(path):
        count += 1
        expect(record.get("hash") in sft_hashes,
               f"{path}: record {count} is not in the SFT mix")
        expect(record.get("hash") not in hashes, f"{path}: duplicate hash at record {count}")
        hashes.add(record["hash"])
        tokens += int(record.get("assistant_tokens", 0))
    expect(count == meta["records"], f"{path}: record count {count} != {meta['records']}")
    expect(tokens == meta["actual_tokens"], f"{path}: token count {tokens} != {meta['actual_tokens']}")
    return count, tokens


def validate_dpo(path: str, meta: dict):
    count = 0
    hashes = set()
    for record in iter_jsonl(path):
        count += 1
        prompt = record.get("prompt_messages")
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        expect(isinstance(prompt, list) and prompt,
               f"{path}: record {count} has no prompt_messages")
        expect(chosen and rejected and chosen != rejected,
               f"{path}: record {count} has an invalid pair")
        expect(record.get("hash"), f"{path}: record {count} has no hash")
        expect(record["hash"] not in hashes, f"{path}: duplicate hash at record {count}")
        hashes.add(record["hash"])
        expect(int(record.get("chosen_tokens", 0)) > 0,
               f"{path}: record {count} has no chosen tokens")
        expect(int(record.get("rejected_tokens", 0)) > 0,
               f"{path}: record {count} has no rejected tokens")
    expect(count == meta["records"], f"{path}: record count {count} != {meta['records']}")
    expect(count == meta["actual_pairs"], f"{path}: pair count {count} != {meta['actual_pairs']}")
    return count, hashes


def main() -> None:
    base = os.path.join(ROOT, "data", "v16", "posttrain")
    manifest_path = os.path.join(base, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expect(manifest.get("format") == "dizel-posttrain-v16",
           "unexpected post-training manifest format")
    paths = {}
    for key in ("sft", "distill", "dpo"):
        meta = manifest[key]
        path = os.path.normpath(os.path.join(ROOT, meta["path"]))
        expect(os.path.exists(path), f"missing {key} file: {path}")
        expect(os.path.getsize(path) == meta["bytes"],
               f"{key} size {os.path.getsize(path)} != {meta['bytes']}")
        actual_hash = sha256(path)
        expect(actual_hash == meta["sha256"],
               f"{key} hash {actual_hash} != {meta['sha256']}")
        paths[key] = path

    sft_count, sft_tokens, sft_hashes, families = validate_sft(
        paths["sft"], manifest["sft"]
    )
    distill_count, distill_tokens = validate_distill(
        paths["distill"], manifest["distill"], sft_hashes
    )
    dpo_count, dpo_hashes = validate_dpo(paths["dpo"], manifest["dpo"])

    total = manifest["sft"]["actual_tokens"]
    print(f"SFT      {sft_count:>8,} records  {sft_tokens:>12,} loss tokens")
    for family, count in sorted(families.items()):
        tokens = manifest["sft"]["families"][family]["actual_tokens"]
        print(f"  {family:<12} {count:>8,} records  {tokens:>12,}  "
              f"{tokens / total:7.3%}")
    print(f"Distill  {distill_count:>8,} records  {distill_tokens:>12,} loss tokens")
    print(f"DPO      {dpo_count:>8,} pairs    {len(dpo_hashes):>8,} unique")
    print("post-training validation passed")


if __name__ == "__main__":
    main()
