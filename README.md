<div align="center">
  <img src="inference/dizel_ui/assets/app/Dizel_banner.png" 
    width="50%"
    height="50%"/>
</div>
<div align="center">

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

> *Dizel — Distributed Intelligent Zen Execution Layer*

**Dizel** is a complete, educational implementation of a GPT-style causal language
model *(~253 M parameters)* built with PyTorch. It is designed to run locally on a
single consumer GPU *(~4 GB VRAM)* over a weekend, with no distributed training required.
But it has grown into **so much more**—shipping with a state-of-the-art **Native Desktop App** featuring voice transcription, cross-provider API routing (BYOK), and rich markdown UI.

---

## 🌟 What You Will Build

| Component | Details |
|---|---|
| **Architecture** | Causal Transformer (Pre-LayerNorm, RoPE, SwiGLU MLP, multi-head self-attention) |
| **Parameters** | ~253 M (configurable 10–250 M) |
| **Tokenizer** | SentencePiece BPE, 32 000 vocab |
| **Pre-training** | Next-token prediction, cosine LR, AMP, gradient accumulation, memory-efficient chunked tokenization |
| **SFT** | Basic chat format, prompt-loss masking |
| **Inference (CLI)** | CLI chat with streaming, top-k/nucleus sampling, repetition penalty (Currently under maintenance) |
| **Inference (GUI)** | Full Desktop App (PySide6) with **API Router (BYOK)**, **Nova Voice Engine**, Dark mode, history & persistent UI |

---

## Folder Structure

```text
dizel/
├── config.py                    ← All hyperparameters in one place
├── requirements.txt             ← All app requirements and optionals too
│
├── data/
│   └── english.md               ← Training corpus (plain English text)
│
├── tokenizer/
│   ├── train_tokenizer.py       ← Train SentencePiece BPE tokenizer
│   ├── train_mila_tokenizer.py  ← Train Mila variant tokenizer
│   ├── corpus.txt               ← (generated) plain texts
│   ├── dizel.model              ← (generated) tokenizer model
│   └── dizel.vocab              ← (generated) vocabulary
│
├── model/
│   ├── __init__.py              ← Package loader
│   ├── architecture.py          ← DizelLM: full Transformer implementation
│   ├── registry.py              ← Model size presets (Lite, Base, Large)
│   └── rope.py                  ← Rotary position embedding (RoPE)
│
├── core/
│   ├── agents/                  ← Content Dict and Lily agents config
│   ├── tools/                   ← Tools and functions of Dict and Lily
│   ├── __init__.py              ← Package loader
│   ├── prompt_builder.py        ← Format agent results into clean text context
│   └── router.py                ← Detect input type and dispatch to the correct agent
│
├── training/
│   ├── __init__.py              ← Package loader
│   ├── dataset.py               ← PretrainDataset, SFTDataset, Tokenizer wrapper
│   ├── pretrain.py              ← Pre-training loop (AMP, grad accum, LR schedule)
│   ├── sft.py                   ← Supervised fine-tuning for chat format
│   ├── shard_utils.py           ← Memory-efficient corpus sharding (100 MB shards)
│   ├── tokenizer_utils.py       ← Fast batch tokenization utilities
│   ├── cache_utils.py           ← Per-shard .pt caching for instant resume
│   ├── data_mixing.py           ← Multi-dataset mixing utilities
│   ├── mila_pretrain.py         ← Mila variant pre-training driver
│   └── mila_sft.py              ← Mila variant SFT driver
│
├── scripts/
│   ├── colab_train.py           ← Google Colab training notebook helper
│   ├── migrate_datasets.py      ← HuggingFace dataset downloader and converter
│   ├── prepare_v11.py           ← v1.1 data preparation pipeline
│   └── prepare_v12.py           ← v1.2 data preparation pipeline
│
├── sft_data/
│   ├── generate_sft_data.py     ← Generate synthetic chat JSONL data
│   └── chat.jsonl               ← (generated) ~60 conversation examples
│
├── inference/
│   ├── cli_ui/
│   │   └── cmd_ui.py            ← CLI chat / completion / JSON inference
│   │
│   └── dizel_ui/                ← Full Desktop GUI Application!
│       ├── main.py              ← Run this to start the desktop app
│       ├── __init__.py          ← Package loader
│       ├── theme/               ← Theme manager, colors, fonts, stylesheets
│       ├── history/             ← Saved chats via JSON    (auto-created)
│       ├── .dizel/              ← Saved settings via JSON (auto-created)
│       ├── ui/                  ← PySide6 UI components (sidebar, chat, input, settings...)
│       ├── utils/               ← Icons loader and agent logic
│       ├── logic/               ← Async generation, config/history/tutorial managers
│       └── assets/              ← Logo and avatar images
│       
├── docs/                        ← Landing page website (vanilla HTML/CSS/JS)
│
├── utils/
│   ├── __init__.py              ← Package loader
│   ├── data_cleaner.py          ← Clean the training data
│   ├── test_model.py            ← Test the model
│   └── verify.py                ← Sanity checks (no GPU required)
│
├── checkpoints/                 ← Saved model checkpoints (auto-created)
└── logs/                        ← Training loss CSV logs  (auto-created)
```

---

## Setup

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install PyTorch with CUDA (replace cu121 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify your installation

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Should print: True  12.x
```

---

## Execution Steps

### Step 0 — Quick Start (If you already have a checkpoint)

If you have downloaded a pre-trained Dizel checkpoint (e.g. `dizel-sft-best.pt`), you can skip the training steps and immediately launch the graphical Desktop Interface to chat with it.

*(Or if there was a `dizel-sft-best.pt` in the `checkpoints` folder in the project directory, you can just use that file.)*

1. Install the dependencies (see Setup above).
2. Launch the Desktop App:
   ```bash
   python inference/dizel_ui/main.py
   ```
3. Click the **⚙ Configuration** button in the UI.
4. Use the **Checkpoint Loader** to select your `.pt` file.
5. Click **Back**/**Save** — the model will load in the background, and you're ready to chat!

*If you do not have a checkpoint and want to build the model from scratch, continue with Step 1 below.*

---

### Step 1 — Review and tune configuration (optional)

Open `config.py`. Key parameters:

```python
# Model size
d_model  = 384          # hidden dimension  (256 = smaller/faster, 512 = larger/slower)
n_layers = 6            # transformer depth
n_heads  = 6            # attention heads   (d_model must be divisible by n_heads)

# Training
batch_size  = 8         # micro-batch per step
grad_accum  = 8         # effective batch = 8 × 8 = 64 sequences
max_steps   = 4000      # how many steps the pretrain and sft does
lr          = 3e-4      # how much the model will learn with the training
context_length = 512    # how much contexts for the model to generate a respond
```

**Estimated parameters** by `d_model`:

| d_model | n_layers | Parameters |
|---------|----------|------------|
| 256     | 6        | ~10 M      |
| 384     | 6        | ~20 M      |
| 512     | 6        | ~34 M      |
| 768     | 12       | ~110.35 M  |

---

### Step 2 — Add training data

The file `data/english.md` ships with ~5 000 words of seed text. For better
results, **add more plain English text** to the same file:

- Wikipedia article extracts (plain text dump)
- Project Gutenberg books
- Any well-written English prose
- HuggingFace dataset samples
- etc...

The more diverse the data, the better. Even 1–5 MB of text makes a noticeable
difference.

---

### Step 3 — Train the tokenizer

```bash
python tokenizer/train_tokenizer.py
```

This reads `data/english.md`, trains a BPE SentencePiece model with 32 000
vocabulary entries, and writes `tokenizer/dizel.model` and `tokenizer/dizel.vocab`.

**Output:**

```text
[tokenizer] Wrote plain text to tokenizer/corpus.txt (45,123 chars)
[tokenizer] Training SentencePiece BPE (vocab=32,000) ...
[tokenizer] Trained BPE model → tokenizer/dizel.model
[tokenizer] Round-trip OK ✓
```

---

### Step 4 — Run sanity checks

```bash
python utils/verify.py
```

Verifies model architecture, forward pass, loss at initialisation, generation
shape, and tokenizer round-trip. **No GPU required.** Should complete in < 5 s.

---

### Step 5 — Pre-train Dizel

```bash
python training/pretrain.py
```

With optional overrides:
```bash
python training/pretrain.py --max_steps 6000 --lr 5e-4 --d_model 256
```

**What to expect:**

| Step | Train Loss | Notes |
|------|------------|-------|
| 0    | ~10.4      | Random initialisation (~ln(32000)) |
| 200  | ~5.0–6.0   | Starting to learn common words |
| 1000 | ~3.5–4.5   | Coherent word sequences |
| 4000 | ~2.5–3.5   | Reasonable sentences on training data |

A good val loss for this corpus size is **≤ 3.0**. The model will overfit on
tiny data — the dropout, weight decay, and window reshuffling all help mitigate
this.

Checkpoints are saved to `checkpoints/`:
- `dizel-pretrain-best.pt`     — best val loss
- `dizel-pretrain-step{N}.pt`  — periodic saves
- `dizel-pretrain-final.pt`    — end of training

Resume training from a checkpoint:
```bash
# 1st Option
python training/pretrain.py --resume checkpoints/dizel-pretrain-step{N}.pt

# 2nd Option (Recommended)
python training/pretrain.py --resume checkpoints/dizel-pretrain-best.pt
```

**Approximate training time (RTX 3060, 12 GB VRAM):**
- 4 000 steps, d_model=384, ctx=512: ~1.5–2 hours
- 4 000 steps, d_model=256, ctx=512: ~45–60 minutes

**Google Colab:** Use `scripts/colab_train.py` for a streamlined notebook experience. The tokenization pipeline uses memory-efficient numpy arrays to stay within Colab's ~12.7 GB RAM limit.

---

### Step 6 — Generate SFT training data

```bash
python sft_data/generate_sft_data.py
```

Creates `sft_data/chat.jsonl` with ~60 synthetic Q&A pairs in the chat format:
```json
{
  "messages": [
    {"role": "system",    "content": "You are Dizel..."},
    {"role": "user",      "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Photosynthesis is the process..."}
  ]
}
```

*You can add your own examples to this file in the same format.*

---

### Step 7 — Supervised fine-tuning (SFT)

```bash
python training/sft.py --base_checkpoint checkpoints/dizel-pretrain-best.pt
```

SFT runs for 500 steps (default) at a lower learning rate (1e-4). It:
- Loads the pretrained weights
- Teaches the model the user / assistant conversation format
- Computes loss **only on assistant tokens** (prompt masking)

Output checkpoint:
- `checkpoints/dizel-sft-step{N}.pt`
- `checkpoints/dizel-sft-best.pt`

Resume SFT from a checkpoint:
```bash
# 1st Option
python training/sft.py --resume checkpoints/dizel-sft-step{N}.pt

# 2nd Option (Recommended)
python training/sft.py --resume checkpoints/dizel-sft-best.pt
```

---

### Step 8 — CMD UI (Disable)
This UI currently is being disabled for maintenance and updating, please go to **Step 9** to chat with the GUI version.

**Interactive chat:**
```bash
python inference/cli_ui/cmd_ui.py --checkpoint checkpoints/dizel-sft-best.pt
```

**Raw text completion (pretrain checkpoint):**
```bash
python inference/cli_ui/cmd_ui.py \
    --checkpoint checkpoints/dizel-pretrain-best.pt \
    --mode complete \
    --prompt "Photosynthesis is the process by which"
```

**JSON structured output:**
```bash
python inference/cli_ui/cmd_ui.py \
    --checkpoint checkpoints/dizel-sft-best.pt \
    --mode json \
    --prompt "List the planets in the solar system"
```

**Sampling controls:**
```bash
python inference/cli_ui/cmd_ui.py \
    --checkpoint checkpoints/dizel-sft-best.pt \
    --temperature 0.7 \
    --top_k 40 \
    --top_p 0.9 \
    --repetition_penalty 1.2
```

**In-chat commands:**

```text
/quit          Exit
/new           Clear conversation history
/system <text> Change the system prompt
/info          Show model info and settings
/temp 0.6      Adjust temperature on the fly
```

---

### Step 9 — Desktop GUI *(Recommended)*

Dizel includes a fully localized, premium Desktop Interface built with PySide6 featuring the **Premium Dark Theme**. 

```bash
python inference/dizel_ui/main.py
```

**(Optional)** Pass a checkpoint right from the command line:
```bash
python inference/dizel_ui/main.py --checkpoint checkpoints/dizel-sft-best.pt --device cuda
```

**Features of the Desktop App:**
- 🎙️ **Nova Voice Engine:** Native whisper-powered speech transcription with live waveform UI processing, customizable silence timeouts, and multi-language support.
- 🔌 **API Router (BYOK):** Bring your own keys! Out-of-the-box support for *Anthropic, Gemini, OpenAI, xAI,* and a fully configurable **Custom Provider** module to plug into *OpenRouter, DeepSeek, Together AI* or whatever local OpenAI-compatible endpoint you're running.
- 🎒 **First-Run Tutorial:** Interactive onboarding overlay that highlights key UI elements with a spotlight effect and step-by-step tooltip instructions.
- 💊 **Action Pill Carousel:** Quick-action buttons (Create Image, Brainstorm, Write Code, etc.) displayed as a scrollable, paginated carousel on the welcome screen.
- 💾 **Persistent Settings:** Your temperature, top-p, external API keys (AES ENCRYPTED), checkpoints, and UI preferences are saved safely across sessions.
- 💬 **Chat History:** Seamlessly manage multiple conversations from the left sidebar via localized JSON instances.
- 📎 **Attachment Previews:** Visually queue up reference items for your prompts directly in the floating composer.
- 🎭 **Model Switcher:** Switch between Dizel and Mila model versions on the fly.
- 🧠 **Context Chips:** The model `Web Search`, `Deep Think` and `Parse Files` mode toggles right in the chatbox.
- ⚡ **Hardware & Limits Info:** Live UI tracking of generation tokens/sec, context limits, and hardware overhead.
- ⌨️ **Keyboard Shortcut:** `Ctrl+K` opens the command palette with a lot of options.

---

## Web UI Demo
You can now preview how the GUI looks and feels through the web demo.

> **Note**: This is a visual and interaction demo — not the full application.

**Things need to be noted:**
- Most features in the demo are **subjected to change, improved or removed**
- Unlike the desktop app, the web version only supports:
   - Ollama
   - API-based providers
- The web UI may receive frequent updates, including early previews of upcoming features
- The current demo focuses on **UI interactions** only — it does not include full backend logic (e.g. local model execution)

> Try the demo [here.](https://d4niel-dev.github.io/Dizel/)

---

## Weekend Timeline

| Time Block | Task |
|------------|------|
| **Fri evening** | Install PyTorch + CUDA, run `verify.py`, read `architecture.py` |
| **Sat morning** | Train tokenizer, start pre-training (let it run) |
| **Sat afternoon** | Read training logs, generate SFT data, run SFT |
| **Sat evening** | Chat with the model, tune sampling parameters |
| **Sun morning** | Add more training data, retrain with better hyperparameters |
| **Sun afternoon** | Experiment: change d_model, n_layers, context_length |
| **Sun evening** | Write your own SFT examples, fine-tune again, celebrate |

---

## Architecture Deep-Dive

### Causal Self-Attention

```text
input x (B, T, d_model)
   │
   ├─ QKV = Linear(d_model → 3 × d_model)
   │       split into Q, K, V
   │
   ├─ reshape to (B, n_heads, T, head_dim)
   │
   ├─ RoPE: apply rotary position embeddings to Q and K
   │
   ├─ attn = softmax(Q @ K.T / √head_dim + causal_mask)
   │        (Flash Attention via PyTorch 2.x SDPA if available)
   │
   └─ output = attn @ V  →  reshape  →  Linear(d_model → d_model)
```

### Transformer Block (Pre-LayerNorm)

```text
x → LayerNorm → Attention → + → LayerNorm → MLP → +
│                            ↑                     ↑
└────────────────────────────┘─────────────────────┘
         residual connections (help gradient flow)
```

### Why Pre-LayerNorm?

Post-LN (LayerNorm after residuals) is the original Transformer design but
requires careful learning rate warm-up to train stably. Pre-LN (before each
sub-layer) is more stable with standard AdamW and no special tricks.

### Weight Tying

The token embedding matrix (vocab × d_model) and the LM head (d_model × vocab)
share the same weights. This reduces parameters by ~3 M and often improves
perplexity because the model learns that words close in embedding space
should have similar output distributions.

---

## Overfitting on Small Data

Small corpora are the primary challenge. Dizel uses several mitigations:

| Technique | Where | Effect |
|---|---|---|
| **Dropout (0.15)** | Attention + MLP | Prevents memorisation |
| **Weight decay (0.1)** | AdamW | L2 regularisation |
| **Window reshuffling** | DataLoader | Breaks repetitive mini-batch patterns |
| **Overlapping windows (stride = ctx/2)** | PretrainDataset | More training examples |
| **Gradient clipping (1.0)** | Optimiser | Prevents instability |
| **Learning rate schedule** | Cosine + warmup | Stable convergence |
| **Weight tying** | Embeddings/LM head | Parameter efficiency |

> If val loss stops decreasing early, try:
1. Adding more training data (most effective)
2. Increasing dropout to 0.2–0.3
3. Reducing model size (smaller d_model)
4. Increasing weight decay

---

## VRAM Usage Guide

| Config | d_model | Batch × Accum | VRAM (bfloat16) |
|--------|---------|---------------|-----------------|
| Tiny   | 256     | 4 × 8         | ~1.5 GB         |
| Small  | 384     | 8 × 8         | ~2.5 GB         |
| Medium | 512     | 8 × 8         | ~4.0 GB         |
| Large  | 768     | 4 × 8         | ~6.0 GB         |

If you get OOM errors: reduce `batch_size`, reduce `context_length`, or
enable gradient checkpointing (add `use_reentrant=False` to `torch.utils.checkpoint`).

---

## Extending Dizel

### Add more data
Append any `.md` or `.txt` text to `data/english.md` and retrain the tokenizer.

### Change model size
Edit `d_model`, `n_layers`, `n_heads` in `config.py`.  
Remember: `d_model` must be divisible by `n_heads`.

### Add more SFT examples
Edit `sft_data/generate_sft_data.py` or append lines to `sft_data/chat.jsonl`.

### Export to ONNX
```python
import torch
from model.architecture import DizelLM
model = DizelLM(cfg)
dummy = torch.zeros(1, 64, dtype=torch.long)
torch.onnx.export(model, dummy, "dizel.onnx", opset_version=17)
```

---

## Frequently Asked Questions

**Q: Why does the model repeat itself?**  
> Increase `repetition_penalty` (try 1.2–1.5) or reduce `temperature`.

**Q: Why is the output nonsensical?**  
> The model may not have trained long enough. Check that val loss is ≤ 3.5. Add more data. The more dataset it gets, the more it'll respond with less nonsensical texts.

**Q: Can I run this on CPU?**  
> Yes — set `device = "cpu"`. Training will be ~50× slower. Inference is fine for short generations.

**Q: How do I use my own text?**  
> Replace or append to `data/english.md`, re-run `train_tokenizer.py`, then retrain.

**Q: How do I make the model produce better JSON?**  
> Add more JSON examples to `sft_data/chat.jsonl` and re-run SFT. Lower temperature to 0.2–0.4.

**Q: Is the model free-to-use?**  
> Yes, it is! — As of right now, this model has the MIT license, things may change in the future.

**Q: How can I report any errors/issues?**  
> Very simple! — Just go to the model GitHub Repo and create an *Issue* request.

**Q: How do I use the API Router (BYOK) to connect to external providers?**  
> Open the Desktop App → click **Configuration** → go to the **Chat** tab → you'll see the **API Router** section with provider cards (Anthropic, Gemini, OpenAI, xAI, etc.). Click any card, then hit **Configure** to enter your API key. For unlisted providers, click **+ Other Provider** to add any OpenAI-compatible endpoint with a custom name and base URL.

**Q: Are my API keys stored securely?**  
> Yes. All API keys (both built-in and custom providers) are encrypted using AES encryption before being persisted to the local configuration file. Keys are never stored in plaintext.

**Q: How do I use voice input (Nova)?**  
> Click the **microphone icon** (🎙️) next to the send button in the chat input. The Nova overlay will appear with a live waveform visualization. Speak your message — it will be transcribed automatically using Whisper. You can configure the Whisper model size, language, and silence timeout in **Settings → Speech**.

**Q: Can I use Dizel with Ollama or other local LLM servers?**  
> Yes! Use the **+ Other Provider** feature in the API Router. Set the provider name (e.g., "Ollama"), leave the API key blank if not needed, and set the base URL to your local server (e.g., `http://localhost:11434/v1`).

**Q: What does the `Ctrl+K` command palette do?**  
> It opens a searchable command palette (like VS Code). From there, you can quickly access **New Chat**, **Settings**, **Export Chat**, **Toggle Theme**, **Reload Model**, **Attach File**, **Clear Output**, and **Quit** — all without touching your mouse.

**Q: How do I switch between Dark and Light themes?**  
> Go to **Settings → Appearance** and select your preferred theme, or use the command palette (`Ctrl+K`) and type "Toggle Theme" to switch instantly. ⚠️ *Warning: Light mode can cause a flashbang when switching from Dark mode — be careful!*

**Q: Can I export my chat conversations?**  
> Yes! Click the **Export** button in the top header bar or use `Ctrl+K → Export Chat`. Conversations are exported as `.md` (Markdown), `.json`, or `.txt` files — you choose.

**Q: What are "Context Chips" in the input bar?**  
> Context Chips are toggleable modes that enhance how the model processes your prompt. **Web Search** adds real-time web context, **Deep Think** enables chain-of-thought reasoning, and **Parse Files** lets the model analyze attached documents.

---

## References

- **Attention Is All You Need** — [Vaswani et al., 2017](https://share.google/KEN2RB5lMguOf9vE7)
- **GPT-2** — [Radford et al., 2019](https://share.google/7nyubfRaa6dFuHCwI) (OpenAI)
- **nanoGPT** — [Andrej Karpathy](https://github.com/karpathy/nanoGPT) (architectural inspiration)
- **SentencePiece** — [Kudo & Richardson, 2018](https://share.google/lwV2qoP42TOyzat9B)
- **PyTorch Documentation** — [pytorch.org](https://share.google/wp67KCOAlAFdYxswg)
