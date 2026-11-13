import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import torch
from model.architecture import DizelLM
from training.dataset import Tokenizer, ROLE_TOKENS, END_TOKEN
from config import CONFIG

def load_model(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg", CONFIG.model)
    state_dict = ckpt["model_state"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = DizelLM(model_cfg).to(device)
    model.load_state_dict(cleaned)
    model.eval()

    tok_bytes = ckpt.get("tokenizer_model")
    tokenizer = Tokenizer()
    if tok_bytes:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".model") as f:
            f.write(tok_bytes)
            tokenizer = Tokenizer(model_path=f.name)

    return model, tokenizer, ckpt.get("step", "?")

def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.92, rep_pen=1.15):
    import torch.nn.functional as F

    ids = []
    ids += tokenizer.encode(ROLE_TOKENS["user"])
    ids += tokenizer.encode(prompt)
    ids += tokenizer.encode(END_TOKEN)
    ids += tokenizer.encode(ROLE_TOKENS["assistant"])

    ctx_len = model.cfg.context_length
    idx = torch.tensor([ids], dtype=torch.long, device="cpu")
    generated = []

    for step in range(max_tokens):
        with torch.no_grad():
            input_ids = idx[:, -ctx_len:]
            logits, _ = model(input_ids)
            logits = logits[:, -1, :].float()

            if rep_pen != 1.0:
                for tid in set(generated[-128:]):
                    if 0 <= tid < logits.size(-1):
                        if logits[0, tid] > 0:
                            logits[0, tid] /= rep_pen
                        else:
                            logits[0, tid] *= rep_pen

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p < 1.0:
                    ps, si = torch.sort(F.softmax(logits, dim=-1), descending=True)
                    ps[ps.cumsum(-1) - ps > top_p] = 0.0
                    ps /= ps.sum(-1, keepdim=True)
                    next_id = torch.multinomial(ps, 1)
                    next_id = si.gather(-1, next_id)
                else:
                    next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)

            tid = next_id.item()
            if tid in [tokenizer.eos_id]:
                break
            generated.append(tid)
            idx = torch.cat([idx, next_id], dim=1)

    text = tokenizer.decode(generated)
    for special in list(ROLE_TOKENS.values()) + [END_TOKEN]:
        text = text.replace(special, "")
    return text.strip()

test_prompts = [
    "What is the capital of France?",
    "Write a short poem about programming.",
    "Explain what machine learning is in simple terms.",
    "What is 2+2?",
    "Hello! How are you?",
    "Translate 'hello' to French.",
    "Write a Python function to check if a number is prime.",
    "What's the weather like?",
    "Tell me a joke.",
]

device = "cpu"
print("=" * 70)
print("COMPARISON: SFT vs DPO CHECKPOINTS")
print("=" * 70)

print("\nLoading SFT checkpoint...")
sft_model, sft_tok, sft_step = load_model("checkpoints/dizel-sft-best.pt", device)
print(f"  SFT model loaded (step={sft_step})")

print("\nLoading DPO checkpoint...")
dpo_model, dpo_tok, dpo_step = load_model("checkpoints/dizel-dpo-best.pt", device)
print(f"  DPO model loaded (step={dpo_step})")

print(f"\nTokenizer vocab: {len(sft_tok.sp)}")
print(f"Tokenizer EOS: {sft_tok.eos_id}, BOS: {sft_tok.bos_id}, PAD: {sft_tok.pad_id}")
print(f"Tokenizer ENCODE test 'hello': {sft_tok.encode('hello')}")
print(f"Tokenizer ENCODE '<|user|>': {sft_tok.encode(ROLE_TOKENS['user'])}")
print(f"Tokenizer ENCODE '<|user|> hello': {sft_tok.encode(ROLE_TOKENS['user'] + ' hello')}")
print(f"Tokenizer ENCODE '<|user|>hello': {sft_tok.encode(ROLE_TOKENS['user'] + 'hello')}")

print("\n\nFirst-token analysis for a sample prompt:")
sample_prompt = "What is the capital of France?"
ids = []
ids += sft_tok.encode(ROLE_TOKENS["user"])
ids += sft_tok.encode(sample_prompt)
ids += sft_tok.encode(END_TOKEN)
ids += sft_tok.encode(ROLE_TOKENS["assistant"])
idx = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    sft_logits, _ = sft_model(idx)
    dpo_logits, _ = dpo_model(idx)

sft_logits = sft_logits[0, -1, :].float()
dpo_logits = dpo_logits[0, -1, :].float()

sft_probs = torch.nn.functional.softmax(sft_logits, dim=-1)
dpo_probs = torch.nn.functional.softmax(dpo_logits, dim=-1)

sft_top10 = torch.topk(sft_probs, 15)
dpo_top10 = torch.topk(dpo_probs, 15)
diff_top10 = torch.topk(dpo_probs - sft_probs, 15)

print("\n--- SFT Top 15 first tokens ---")
for i in range(15):
    tid = sft_top10.indices[i].item()
    piece = sft_tok.sp.id_to_piece(tid)
    print(f"  #{i+1}: ID={tid} '{piece}' prob={sft_top10.values[i].item():.6f}")

print("\n--- DPO Top 15 first tokens ---")
for i in range(15):
    tid = dpo_top10.indices[i].item()
    piece = dpo_tok.sp.id_to_piece(tid)
    print(f"  #{i+1}: ID={tid} '{piece}' prob={dpo_top10.values[i].item():.6f}")

print("\n--- Tokens where DPO probability increased the most ---")
for i in range(15):
    tid = diff_top10.indices[i].item()
    piece = dpo_tok.sp.id_to_piece(tid)
    delta = diff_top10.values[i].item()
    print(f"  +{delta:.6f}: ID={tid} '{piece}' (SFT={sft_probs[tid].item():.6f} DPO={dpo_probs[tid].item():.6f})")

print("\n*** That's the tokenizer analysis above ***")

print("\n" + "=" * 70)
print("GENERATION COMPARISON")
print("=" * 70)
for prompt in test_prompts:
    print(f"\n--- Prompt: {prompt} ---")
    sft_out = generate(sft_model, sft_tok, prompt, max_tokens=80)
    dpo_out = generate(dpo_model, dpo_tok, prompt, max_tokens=80)
    print(f"  SFT: {sft_out}")
    print(f"  DPO: {dpo_out}")
    print(f"  ---")

print("\nDone!")
