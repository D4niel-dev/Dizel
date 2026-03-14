import json
import os
from datasets import load_dataset

system_prompt = "You are Dizel, a structured analytical AI model. Prioritize clarity, precision, and logical organization. Use structured formatting when appropriate. Avoid unnecessary verbosity. If uncertain, explicitly state limitations."

def format_record(user_msg, asst_msg):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": asst_msg},
        ]
    }

def main():
    output_path = "sft_data/chat_expanded.jsonl"
    records = []

    # 1. Load existing
    if os.path.exists("sft_data/chat.jsonl"):
        with open("sft_data/chat.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} existing examples.")

    # 2. Alpaca (5k)
    print("Downloading yahma/alpaca-cleaned...")
    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    for i in range(5000):
        item = alpaca[i]
        user_msg = item["instruction"]
        if item.get("input", ""):
            user_msg += "\n\n" + item["input"]
        records.append(format_record(user_msg, item["output"]))
    print("Added 5000 Alpaca examples.")

    # 3. CodeAlpaca (3k)
    print("Downloading sahil2801/CodeAlpaca-20k...")
    code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    for i in range(3000):
        item = code_alpaca[i]
        user_msg = item["instruction"]
        if item.get("input", ""):
            user_msg += "\n\n" + item["input"]
        records.append(format_record(user_msg, item["output"]))
    print("Added 3000 CodeAlpaca examples.")

    # 4. TinyStories (5k)
    print("Downloading roneneldan/TinyStories...")
    tiny_stories = load_dataset("roneneldan/TinyStories", split="train")
    for i in range(5000):
        item = tiny_stories[i]
        records.append(format_record("Tell me a simple story.", item["text"]))
    print("Added 5000 TinyStories examples.")

    # Save to jsonl
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    
    print(f"Total examples: {len(records)}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
