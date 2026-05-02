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

**Dizel** *(Distributed Intelligent Zen Execution Layer)* is a complete, educational implementation of a GPT-style causal language
model *(~205 M parameters)* built with PyTorch. It is designed to run locally on a
single consumer GPU *(~4 GB VRAM)* over a weekend, with no distributed training required.
But it has grown into **so much more**—shipping with a state-of-the-art **Native Desktop App** featuring voice transcription, cross-provider API routing (BYOK), and rich markdown UI.

---

## 🌟 What You Will Build

| Component | Details |
|---|---|
| **Architecture** | Causal Transformer (Pre-LayerNorm, RoPE, SwiGLU MLP, multi-head self-attention) |
| **Parameters** | ~205 M (configurable 10–250 M) |
| **Tokenizer** | SentencePiece BPE, 32 000 vocab |
| **Pre-training** | Next-token prediction, cosine LR, AMP, gradient accumulation, memory-efficient chunked tokenization |
| **SFT** | Basic chat format, prompt-loss masking |
| **Inference (CLI)** | CLI chat with streaming, top-k/nucleus sampling, repetition penalty (Available v2.0!) |
| **Inference (GUI)** | Full Desktop App (PySide6) with **API Router (BYOK)**, **Nova Voice Engine**, Dark mode, history & persistent UI |

---

## Folder Structure

```text
Dizel/
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
│   ├── cmd_ui/                  ← Terminal UI v2.0 Application! (NEW)
│   │   ├── main.py              ← Run this to start the Terminal UI
│   │   ├── app.py               ← Textual App runtime & bindings
│   │   ├── bridge/              ← Streaming & threading logic
│   │   ├── commands/            ← Slash command registry & parser
│   │   ├── panels/              ← InputBar, Workspace, Context panels
│   │   ├── rendering/           ← Markdown blocks, ASCII empty state
│   │   └── cmd_ui.tcss          ← Textual CSS styles & layouts
│   │
│   └── dizel_ui/                ← Full Desktop GUI Application! (v1.0.0)
│       ├── main.py              ← Run this to start the desktop app
│       ├── theme/               ← Theme manager, colors, fonts, stylesheets
│       ├── history/             ← Saved chats via JSON    (auto-created)
│       ├── .dizel/              ← Saved settings via JSON (auto-created)
│       ├── ui/                  ← PySide6 UI components (sidebar, chat, input, overlays)
│       ├── utils/               ← Hardware utilities and generic icons
│       ├── logic/               ← Async managers, context trimmer, TokenBudget
│       └── assets/              ← Static logos and fonts
│       
├── docs/                        ← Demo Website (HTML/CSS/JS)
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

### Step 8 — Dizel CMD UI (v2.0.0)

The Terminal UI for Dizel has been completely reimagined from the ground up to provide a premium, developer-first experience inspired by modern keyboard-centric CLI tools. It is built using the robust Textual framework + Rich for better rendering.

Launch it via:
```bash
# Option 1 (Recommended)
python -m inference.cmd_ui.main

# Option 2 (Optional)
python inference/cmd_ui/main.py
```

**Features of the Reworked CMD UI:**
- 🎨 **Minimalist Dark Theme & Aesthetics:** A seamless, deep dark gray layout featuring floating components, a centered gradient ASCII logo for the empty state, and an overall distraction-free environment.
- ⌨️ **Keyboard-First Workflow:** Navigate the entire app without a mouse.
  - `Tab`: Cycle between AI Agent modes (Fast, Planning, Coding, Review).
  - `Ctrl+K`: Open the floating Command Palette for quick actions.
  - `Ctrl+H`: Toggle the Session History Panel on the left.
  - `Ctrl+R`: Toggle the Context Panel on the right.
- 💬 **Floating Input Bar:** A clean, bottom-docked prompt area that clearly displays your currently active Mode, Model, and Provider (e.g., `[BUILD] Dizel Lite Local`).
- 📊 **Reactive Context Panel:** A dynamic right sidebar that tracks your live token usage and compute budget in real-time as the model streams its responses.
- 🚀 **Advanced Slash Commands:** Fully integrated slash commands for instant adjustments right from the chat bar:
  - `/model [name]` - Swap models on the fly.
  - `/provider [name]` - Switch between local inference and API routing.
  - `/mode [name]` - Change your AI persona/agent mode.
  - `/session [new|rename|delete]` - Manage your chat sessions natively.
- 🛠 **Real-Time Generation & Tool Use:** Supports asynchronous text streaming, tool execution reporting, and robust error handling without blocking the main UI thread.
---

### Step 9 — Desktop GUI (v1.0.0) *(Recommended)*

Dizel includes a fully localized, premium Desktop Interface built entirely from scratch with **PySide6**. It pushes the boundaries of native Python UI development with highly fluid animations, rich data management, and an uncompromising aesthetic heavily inspired by state-of-the-art developer tools.

Launch it via:
```bash
# Launch the app (Default)
python inference/dizel_ui/main.py

# Pass a checkpoint right from the command line (Optional)
python inference/dizel_ui/main.py --checkpoint checkpoints/dizel-sft-best.pt --device cuda
```

**Core Capabilities & Architecture:**
- 🎙️ **Nova Voice Engine:** Native Whisper-powered speech transcription featuring real-time waveform UI visualization (`WaveformWidget`), multi-language support, and customizable silence thresholds to auto-submit your prompts when you stop speaking.
- 🔌 **API Router (BYOK):** Bring Your Own Keys. Robust out-of-the-box routing for *Anthropic, Gemini, OpenAI,* and *xAI*. Also includes a highly configurable **Custom Provider** module to plug directly into *OpenRouter, DeepSeek, Together AI,* or any local OpenAI-compatible endpoint (like Ollama or vLLM). All external keys are **AES-encrypted** locally before saving.
- 💬 **Secondary Operations Sidebar:** A dynamic, multi-view animated right sidebar offering:
  - **Chats & Import:** Search history and import `.json`/`.md` chats.
  - **Image Gallery:** A visual archive of all images you've generated or attached across all sessions.
  - **Riset (Global Search):** Deep semantic search filtering across all projects, archived files, and starred content simultaneously.
  - **Prompt Library:** Save, favorite, and reuse custom personas, instructions, and prompt templates.
  - **File Archive & Presentations:** Central hub for all documents and slide imports.
- 📊 **Intelligent Compute Tracking:** An advanced `UsageManager` and `TokenBudget` system dynamically tracks your 12-hour compute window, allocating context ceilings based on active mode complexity (e.g., scoring *Fast* mode vs *Planning* mode differently).
- 🧠 **Dynamic Context Trimming:** The `ContextTrimmer` automatically truncates or summarizes conversational data to prevent memory overflow (OOM) while rigidly protecting system instructions and recent context.
- 🎨 **Rich UI & Micro-Animations:** Custom-built PySide6 components including `MessageBubble` (for live-rendered markdown and syntax highlighting), `ActionMenu` (smooth popup tooling), `AnimatedButton`, and a realistic `TypingIndicator` for a truly premium feel.
- 🎒 **First-Run Onboarding:** An interactive tutorial overlay (`TutorialManager`) that sequentially spotlights critical UI regions with helpful tooltips for first-time users.
- ⚡ **Performance Telemetry:** Live hardware integration displaying tokens/sec, absolute context lengths, budget consumption, and VRAM overhead straight in the chat window.
- ⌨️ **Command Palette:** Press `Ctrl+K` to summon a fuzzy-searchable global command list, allowing entirely mouse-free navigation (Toggle Theme, Export Chat, Reload Model, Settings).
- 💊 **Zero-State Carousels:** Clickable, paginated action pills (Brainstorm, Write Code, Create Image) that instantly queue up complex system instructions.
- 💾 **Persistent Settings:** Your temperature, top-p, external API keys (AES ENCRYPTED), checkpoints, and UI preferences are saved safely across sessions.
- 💬 **Chat History:** Seamlessly manage multiple conversations from the left sidebar via localized JSON instances.
- 📎 **Attachment Previews:** Visually queue up reference items for your prompts directly in the floating composer.
- 🎭 **Model Switcher:** Switch between Dizel and Mila model versions on the fly.
- 🧠 **Context Chips:** The model `Web Search`, `Deep Think` and `Parse Files` mode toggles right in the chatbox.
- ⚡ **Hardware & Limits Info:** Live UI tracking of generation tokens/sec, context limits, and hardware overhead.
- ⌨️ **Keyboard Shortcut:** `Ctrl+K` opens the command palette with a lot of options.

---

## 🌐 Interactive Web UI Demo

Want to experience the look and feel of the Dizel GUI without installing Python, PyTorch, or downloading any models? **You can now preview the interface directly in your browser!**

> Try the live demo here: **[Dizel Web Interface](https://d4niel-dev.github.io/Dizel/)**

### What is the Web Demo?
The web demo is a meticulously crafted 1:1 visual replica of the PySide6 Desktop GUI, engineered entirely in **Vanilla HTML/CSS/JS** (located in the `docs/` folder). It is designed to give you a hands-on feel of the application's premium aesthetics, fluid animations, and layout before you commit to cloning the repository.

### What You Can Experience:
- 🎨 **Pixel-Perfect Aesthetics:** Explore the deep dark theme, glassmorphism overlays, and smooth micro-animations.
- 🎒 **Interactive Onboarding:** Walk through the mock first-run tutorial that highlights key UI elements.
- 🖱️ **Navigation & Layout:** Click through the Action Pill Carousels, toggle the Secondary Operations Sidebar, and open the Command Palette (`Ctrl+K`).
- 📱 **Responsive Design:** Unlike the desktop app, the web demo allows you to see how the UI reflows on smaller viewports and mobile devices.

### Important Limitations:
> **Note:** This is an interaction and design playground, *not* the full Python application.
- **No Local Model Execution:** You cannot run the `.pt` PyTorch models in the browser.
- **Limited Backend Logic:** The demo focuses on UI interactions. Chat functionality only works if you connect it to **Ollama** or external **API-based providers** (like OpenAI or Anthropic) via the settings.
- **Experimental Features:** The web UI acts as our testing ground. You may see early previews of upcoming features here before they are merged into the main PySide6 desktop app!
- **Subject to Change:** Features in the web demo may be altered, improved, or removed frequently.

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

Dizel implements a modern, highly-optimized variant of the original GPT-style Transformer.

### Core Attention Mechanism (Flash Attention & RoPE)

```text
input x (B, T, d_model)
   │
   ├─ QKV = Linear(d_model → 3 × d_model)
   │       split into Q, K, V
   │
   ├─ reshape to (B, n_heads, T, head_dim)
   │
   ├─ RoPE: Apply Rotary Position Embeddings to Q and K
   │        (Enables relative positional understanding and better length extrapolation)
   │
   ├─ attn = softmax(Q @ K.T / √head_dim + causal_mask)
   │        ⚡ Hardware Accelerated via PyTorch 2.x SDPA (Flash Attention)
   │
   └─ output = attn @ V  →  reshape  →  Linear(d_model → d_model)
```

### Transformer Block (Pre-LayerNorm & SwiGLU)

```text
x → LayerNorm → Attention → + → LayerNorm → SwiGLU MLP → +
│                            ↑                            ↑
└────────────────────────────┘────────────────────────────┘
         residual connections (essential for deep gradient flow)
```

**Why SwiGLU?**
Instead of a standard ReLU or GELU MLP, Dizel utilizes the SwiGLU activation function (as seen in Llama 3). This gating mechanism requires slightly more parameters per block but yields significantly better reasoning performance.

**Why Pre-LayerNorm?**
Post-LN (LayerNorm after residuals) is the original Transformer design but requires careful learning rate warm-up to train stably. Pre-LN (before each sub-layer) is more stable with standard AdamW and prevents gradient vanishing early in training.

**Weight Tying & KV Caching**
- **Weight Tying:** The token embedding matrix and the final LM head share the same weights. This reduces total parameters by ~3M and forces the embedding space to properly align with output probabilities.
- **KV Caching:** During inference, past Key and Value vectors are aggressively cached. Instead of re-computing the entire context window for every new word, Dizel only processes the *last* token, dropping O(N²) attention complexity down to O(N).

---

## Overfitting on Small Data

Small corpora are the primary challenge when training LLMs from scratch. Dizel uses an aggressive suite of mitigations to prevent memorization:

| Technique | Where | Effect |
|---|---|---|
| **Dropout (0.15)** | Attention + MLP | Prevents strict memorisation of paths |
| **Weight decay (0.1)** | AdamW Optimiser | L2 regularisation keeps weights small |
| **Window reshuffling** | DataLoader | Breaks repetitive mini-batch patterns |
| **Overlapping windows (stride = ctx/2)** | PretrainDataset | Maximizes examples squeezed from raw text |
| **Gradient clipping (1.0)** | Optimiser | Prevents explosive instability spikes |
| **Cosine LR + Warmup** | Scheduler | Allows smooth convergence |
| **Gradient Accumulation** | Training Loop | Simulates massive batch sizes (e.g., 64+) to average out noise |

> **Pro Tip:** If your validation loss plateaus early, try:
> 1. Adding more raw data (this is the only true fix).
> 2. Increasing dropout to 0.2–0.3.
> 3. Scaling down the model (use `d_model=256`).

---

## VRAM Usage Guide

Using `bfloat16` mixed precision, memory requirements are highly dependent on whether you are *Training* (which must store optimizer states, gradients, and activations) or running *Inference*.

| Config | d_model | Params | Training VRAM (Batch 8) | Inference VRAM (w/ KV Cache) |
|--------|---------|--------|-------------------------|------------------------------|
| **Tiny**   | 256     | ~10 M  | ~1.5 GB                 | ~400 MB                      |
| **Small**  | 384     | ~20 M  | ~2.5 GB                 | ~650 MB                      |
| **Medium** | 512     | ~34 M  | ~4.0 GB                 | ~1.1 GB                      |
| **Large**  | 768     | ~110 M | ~8.0 GB                 | ~3.5 GB                      |

If you get OOM (Out of Memory) errors during training: reduce `batch_size`, reduce `context_length`, or enable gradient checkpointing.

---

## Extending Dizel

The framework is built to be hacked on.

### 1. Build Custom Agents
Dizel ships with an Agent Router (`core/router.py`) and predefined agents (`core/agents/`). You can define your own JSON configurations to inject specific tools, system prompts, and personalities into the Desktop App.

### 2. Change Model Architecture
Edit `d_model`, `n_layers`, `n_heads` in `config.py`.  
*Rule of thumb:* `d_model` must be evenly divisible by `n_heads`.

### 3. Expand SFT Datasets
Add more conversation JSON examples to `sft_data/chat.jsonl` or modify `generate_sft_data.py` to synthesize larger dialogue datasets automatically.

### 4. Export to ONNX / C++
```python
import torch
from model.architecture import DizelLM
model = DizelLM(cfg)
dummy = torch.zeros(1, 64, dtype=torch.long)
torch.onnx.export(model, dummy, "dizel.onnx", opset_version=17)
```

---

## Frequently Asked Questions

### 🧠 Training & Model Architecture

**Q: Why does the model repeat itself?**  
> Increase `repetition_penalty` (try 1.2–1.5) or reduce `temperature` to 0.7.

**Q: Why is the output nonsensical?**  
> The model may not have trained long enough. Check that val loss is ≤ 3.5. Add more plain-text data! A model is only as smart as the data it has read.

**Q: How much text data do I need for a good model?**  
> For basic coherent English, 10–50 MB of clean text (Wikipedia, Gutenberg books) is enough to learn grammar. For deep knowledge, you need gigabytes.

**Q: Can I change the context window length?**  
> Yes, change `context_length` in `config.py`. However, remember that attention complexity is $O(N^2)$. Doubling the context length will quadruple the memory required for the attention matrix!

**Q: How do I make the model produce better structured JSON?**  
> Add more JSON-formatted responses to `sft_data/chat.jsonl` and re-run Supervised Fine-Tuning (SFT). Lower temperature to 0.2–0.4 during inference.

**Q: Can I use multiple GPUs for training?**  
> Dizel is specifically designed for single consumer GPUs to keep the educational barrier low. If you want multi-GPU, you will need to wrap the model in PyTorch's `DistributedDataParallel` (DDP).

**Q: What happens if I stop training halfway?**  
> Dizel automatically saves periodic checkpoints (e.g., `dizel-pretrain-step1000.pt`). You can safely resume training by running `python training/pretrain.py --resume checkpoints/dizel-pretrain-step1000.pt`.

### 🖥️ Hardware & Compatibility

**Q: Can I run/train this on a CPU?**  
> Yes — set `device = "cpu"`. However, training will be ~50× slower. CPU inference is acceptable for the Tiny/Small models.

**Q: Can I train this on my MacBook (M1/M2/M3)?**  
> Yes! PyTorch supports Apple Silicon via the `mps` backend. Change your device from `cuda` to `mps`. It is highly efficient, though slightly slower than a dedicated NVIDIA RTX GPU.

**Q: I'm getting `RuntimeError: CUDA out of memory` during training.**  
> 1. Reduce `batch_size` in `config.py`.
> 2. Reduce `context_length`.
> 3. Ensure no other apps (like games or heavy browsers) are eating your VRAM.

### 🎨 Desktop App (PySide6) & UI

**Q: Why does the Desktop GUI use PySide6 instead of Electron or Webview?**  
> PySide6 (Qt) offers native OS performance, direct Python integration, and hardware-accelerated rendering *without* the massive RAM overhead of shipping a bundled Chromium browser.

**Q: How do I use the API Router (BYOK) in the Desktop App?**  
> Open the Desktop App → click **Configuration** → go to the **Chat** tab. You'll see the **API Router** section. Click any provider card (Anthropic, Gemini, OpenAI, etc.), then hit **Configure** to enter your API key. For unlisted providers (like Ollama), click **+ Other Provider**.

**Q: Are my external API keys stored securely?**  
> Yes. All API keys are encrypted using AES encryption before being saved locally to `.dizel/settings.json`.

**Q: Where are my chat histories saved?**  
> They are saved natively on your machine as JSON files inside the `inference/dizel_ui/history/` directory. You can easily back them up or delete them.

**Q: How does the Context Trimmer work?**  
> When your conversation approaches the token limit of the model, the `ContextTrimmer` automatically drops the oldest messages to prevent an Out-Of-Memory error. It rigorously protects your System Prompt and the most recent messages.

**Q: How does the "12-hour Compute Budget" work?**  
> The UI includes a `UsageManager` that tracks the complexity of your requests. Using heavy agent modes (like *Planning*) consumes more of your virtual compute budget than *Fast* mode. The budget resets automatically every 12 hours.

**Q: How do I use voice input (Nova)?**  
> Click the microphone icon (🎙️) next to the send button in the GUI chat input. The Nova overlay will appear. Speak, and it will be transcribed using Whisper.

### ⌨️ Terminal (CMD UI) & Workflows

**Q: How do I navigate the new CMD UI without a mouse?**
> - **Tab:** Switch AI Modes (Fast, Planning, Coding).
> - **Ctrl+K:** Open Command Palette.
> - **Ctrl+H:** Toggle Chat History sidebar.
> - **Ctrl+R:** Toggle the Context Panel (token tracking).

**Q: I'm getting an `ImportError` when trying to run the UI!**  
> Make sure you have activated your virtual environment (`source .venv/bin/activate`) and run `pip install -r requirements.txt`. The UI relies heavily on `PySide6` (Desktop) and `Textual` (CMD).

**Q: Is the model free-to-use?**  
> Yes, it is! As of right now, this repository uses the MIT license.

---

## References

### Architecture & Papers
- **Attention Is All You Need** — [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- **GPT-2: Language Models are Unsupervised Multitask Learners** — [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** — [Su et al., 2021](https://arxiv.org/abs/2104.09864)
- **GLU Variants Improve Transformer (SwiGLU)** — [Noam Shazeer, 2020](https://arxiv.org/abs/2002.05202)
- **LLaMA: Open and Efficient Foundation Language Models** — [Touvron et al., 2023](https://arxiv.org/abs/2302.13971) (Inspiration for the modern Llama-like architecture)
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — [Tri Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- **Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)** — [Radford et al., 2022](https://arxiv.org/abs/2212.04356) (Used in Nova Voice Engine)

### Libraries & Inspiration
- **nanoGPT** — [Andrej Karpathy](https://github.com/karpathy/nanoGPT) (Invaluable architectural inspiration and educational groundwork)
- **SentencePiece** — [Kudo & Richardson, 2018](https://github.com/google/sentencepiece)
- **PyTorch Documentation** — [pytorch.org](https://pytorch.org/docs/stable/index.html)
- **PySide6 (Qt for Python)** — [Qt Group](https://doc.qt.io/qtforpython-6/) (Powering the Desktop GUI)
- **Textual & Rich** — [Textualize](https://github.com/Textualize/textual) (Powering the CMD UI v2.0)
- **Hugging Face Datasets** — [huggingface.co](https://huggingface.co/docs/datasets/)
