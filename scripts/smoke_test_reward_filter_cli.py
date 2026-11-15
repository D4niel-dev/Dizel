import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForSequenceClassification,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

TOKENIZER_PATH = os.path.join(ROOT, "tokenizer", "qwen2.5")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "tiny-rm")
        os.makedirs(model_dir, exist_ok=True)
        config = Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            pad_token_id=tokenizer.pad_token_id,
            num_labels=1,
        )
        torch.manual_seed(0)
        model = Qwen3ForSequenceClassification(config)
        model.save_pretrained(model_dir)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(os.path.join(TOKENIZER_PATH, name), model_dir)

        records = []
        for index in range(10):
            records.append(
                {
                    "prompt_messages": [
                        {"role": "system", "content": "Be concise."},
                        {
                            "role": "user",
                            "content": f"Question number {index}: what is {index}+1?",
                        },
                    ],
                    "chosen": f"The answer is {index + 1}.",
                    "rejected": f"The answer is {index + 2}.",
                    "source": "synthetic" if index % 2 else "synthetic_even",
                }
            )
        input_path = os.path.join(tmp, "dpo.jsonl")
        write_jsonl(input_path, records)

        scored = os.path.join(tmp, "scored.jsonl")
        train = os.path.join(tmp, "train.jsonl")
        val = os.path.join(tmp, "val.jsonl")
        report = os.path.join(tmp, "report.json")

        command = [
            sys.executable,
            os.path.join(ROOT, "scripts", "score_dpo_pairs_v16.py"),
            "--reward_model",
            model_dir,
            "--local_files_only",
            "--input",
            input_path,
            "--scored_output",
            scored,
            "--output",
            train,
            "--val_output",
            val,
            "--report",
            report,
            "--batch_size",
            "2",
            "--max_length",
            "512",
            "--min_margin",
            "-1000",
            "--limit",
            "4",
            "--val_fraction",
            "0.3",
            "--seed",
            "7",
        ]
        first = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if first.returncode != 0:
            print(first.stdout)
            print(first.stderr)
            raise SystemExit(f"first run failed: {first.returncode}")
        print("first run ok")
        print(first.stdout[-1200:])

        with open(scored, encoding="utf-8") as handle:
            scored_rows = [json.loads(line) for line in handle if line.strip()]
        print("first scored", len(scored_rows))
        assert len(scored_rows) == 4
        assert all("rm_margin" in row for row in scored_rows)

        # Resume with a larger limit.  The first four pairs are a prefix.
        resumed = command[:]
        resumed[resumed.index("--limit") + 1] = "6"
        resumed.append("--resume")
        second = subprocess.run(
            resumed,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if second.returncode != 0:
            print(second.stdout)
            print(second.stderr)
            raise SystemExit(f"resume failed: {second.returncode}")
        print("resume ok")
        print(second.stdout[-1200:])

        with open(scored, encoding="utf-8") as handle:
            scored_rows = [json.loads(line) for line in handle if line.strip()]
        print("resumed scored", len(scored_rows))
        assert len(scored_rows) == 6
        assert [row["rm_index"] for row in scored_rows] == sorted(
            row["rm_index"] for row in scored_rows
        )

        with open(train, encoding="utf-8") as handle:
            train_rows = [json.loads(line) for line in handle if line.strip()]
        with open(val, encoding="utf-8") as handle:
            val_rows = [json.loads(line) for line in handle if line.strip()]
        print("train", len(train_rows), "val", len(val_rows))
        assert all(row["rm_margin"] >= -1000 for row in train_rows)
        assert len(train_rows) + len(val_rows) == 6

        with open(report, encoding="utf-8") as handle:
            report_data = json.load(handle)
        print("report counts", report_data["counts"])
        assert report_data["counts"]["scored_pairs"] == 6
        assert report_data["counts"]["kept_train_pairs"] == len(train_rows)
        assert report_data["dpo_note"].startswith("DPO training uses")
        assert "system prompts" in report_data["system_prompt_policy"].lower()

    print("reward-filter CLI smoke test passed")


if __name__ == "__main__":
    main()
