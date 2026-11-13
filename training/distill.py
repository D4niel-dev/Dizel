"""
training/distill.py — LoRA-based teacher distillation for Dizel.

Knowledge distillation from a larger teacher (e.g., Qwen2.5-0.5B-Instruct)
into a smaller student model (DizelLM) via LoRA adapters.

Loss:
    L = CE(student, labels) + kl_weight * KL(teacher_probs || student_probs)

Both losses are computed only on response tokens (loss_mask=1).

Usage
-----
    # Phase C.1 — general distillation
    python training/distill.py \
        --base_checkpoint checkpoints/dizel-v15-sft-best.pt \
        --teacher_model Qwen/Qwen2.5-0.5B-Instruct \
        --data_path data/formatted/v15_distill.jsonl \
        --max_steps 5000 \
        --lr 2e-4

    # Phase C.2 — reasoning emphasis
    python training/distill.py --resume checkpoints/dizel-v15-distill-best.pt
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DizelConfig, ModelConfig, V15_CONFIG
from model.architecture import DizelLM
from training.dataset import Tokenizer
from training.pretrain import (
    cosine_lr_with_warmup, set_lr, save_checkpoint, load_checkpoint,
    save_tokenizer, make_schedule_tag,
)

torch.serialization.add_safe_globals([ModelConfig])

from training.display_utils import (
    rule, section, info, ok, warning, error, dim,
    config_table, StepMeter, validation,
)

# ── HuggingFace token handling ─────────────────────────────────────────────

HF_TOKEN_FILE = ".hf_token"


def _hf_token_path() -> str:
    """Path to the persisted HF token file (repo root)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", HF_TOKEN_FILE)
    )


def load_saved_hf_token() -> Optional[str]:
    """Load a previously saved HF token, or None."""
    try:
        with open(_hf_token_path(), "r", encoding="utf-8") as f:
            token = f.read().strip()
        return token or None
    except OSError:
        return None


def save_hf_token(token: str) -> str:
    """Persist an HF token to disk with owner-only permissions."""
    path = _hf_token_path()
    with open(path, "w", encoding="utf-8") as f:
        f.write(token.strip() + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # chmod is best-effort (e.g. FAT/Drive filesystems)
    return path


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_hf_token(token_arg: Optional[str] = None) -> Optional[str]:
    """Resolve an HF token: --hf_token arg → HF_TOKEN env → saved file → prompt.

    Returns None if the user provides no token (public models work fine).
    """
    token = (token_arg or "").strip() or None
    if token:
        return token

    token = (os.environ.get("HF_TOKEN") or "").strip() or None
    if token:
        return token

    saved = load_saved_hf_token()
    if saved:
        info(f"[teacher] Reusing saved HF token from {HF_TOKEN_FILE}")
        return saved

    # Prompt only when stdin can accept input (terminal or Colab notebook).
    interactive = sys.stdin.isatty() or _in_colab()
    if not interactive:
        return None

    print()
    warning("This model may require a HuggingFace token (gated models).")
    info("Public models work without one — just press Enter to skip.")
    try:
        token = input("  HF token (or Enter to skip): ").strip()
    except EOFError:
        return None

    token = token or None
    if token:
        path = save_hf_token(token)
        ok(f"HF token saved to {path} — future runs reuse it automatically")
    return token


# ── Teacher model loading (HuggingFace) ────────────────────────────────────

def load_teacher_model(model_name: str, device: torch.device, amp_dtype: torch.dtype,
                       hf_token: Optional[str] = None):
    """Load a HuggingFace AutoModelForCausalLM as the teacher.

    Returns None with a warning if transformers is not installed.
    The teacher runs in eval mode, no gradients.
    hf_token is passed to from_pretrained for gated models.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[teacher] 'transformers' not installed. Install with:")
        print("  pip install transformers torch")
        return None, None

    model_kwargs = {}
    if hf_token:
        model_kwargs["token"] = hf_token

    print(f"[teacher] Loading '{model_name}'...")
    try:
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=amp_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
    except OSError as e:
        if hf_token:
            warning(f"Failed to load '{model_name}' with the provided token.")
            warning(f"  Delete {_hf_token_path()} or pass a fresh --hf_token and retry.")
        raise
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)
    print(f"[teacher] Loaded ({sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params)")

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id

    return teacher, hf_tokenizer


# ── LoRA Layers ────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a Linear layer.

    Wraps an existing nn.Linear, freezing its weights and adding
    trainable rank-decomposition: W' = W + B*A

    A: (r, in_features)  — random init (Kaiming uniform)
    B: (out_features, r) — zeros init
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        out_features, in_features = linear.weight.shape
        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, device=linear.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, r, device=linear.weight.device)
        )
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.merged = False

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.linear.weight, self.linear.bias)

        result = self.linear(x)
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling
        return result + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the base linear layer."""
        if self.merged:
            return
        with torch.no_grad():
            self.linear.weight.data += (
                (self.lora_B @ self.lora_A) * self.scaling
            )
        self.merged = True

    def unmerge_weights(self):
        """Subtract LoRA weights from the base linear layer."""
        if not self.merged:
            return
        with torch.no_grad():
            self.linear.weight.data -= (
                (self.lora_B @ self.lora_A) * self.scaling
            )
        self.merged = False


# ── LoRA Language Model Wrapper ────────────────────────────────────────────

DEFAULT_LORA_TARGETS = [
    "qkv",
    "proj",
    "fc1",
    "fc2",
]


def _replace_module(module: nn.Module, name: str, target_names: list, r: int, alpha: int, dropout: float) -> bool:
    """Recursively replace nn.Linear modules matching target_names with LoRALinear."""
    replaced = False
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name

        if isinstance(child, nn.Linear) and any(t in child_name for t in target_names):
            lora_layer = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, child_name, lora_layer)
            replaced = True
        else:
            if _replace_module(child, full_name, target_names, r, alpha, dropout):
                replaced = True
    return replaced


def attach_lora(
    model: DizelLM,
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
    target_names: list = None,
) -> int:
    """Replace target linear layers in the model with LoRA wrappers.

    Returns the number of LoRA adapter modules attached.
    Freezes all non-LoRA parameters.
    """
    if target_names is None:
        target_names = DEFAULT_LORA_TARGETS

    count = 0
    for name, module in model.named_children():
        if _replace_module(module, name, target_names, r, alpha, dropout):
            count += 1

    lora_count = sum(1 for _ in model.modules() if isinstance(_, LoRALinear))
    print(f"  [lora] Attached {lora_count} LoRA adapters (r={r}, alpha={alpha}, dropout={dropout})")

    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [lora] Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)")

    return lora_count


def merge_lora_weights(model: DizelLM):
    """Merge all LoRA adapter weights into base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()


def unmerge_lora_weights(model: DizelLM):
    """Unmerge all LoRA adapter weights from base weights."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()


# ── Gradient checkpointing helper ──────────────────────────────────────────

def _enable_gradient_checkpointing(model: DizelLM) -> None:
    """Wrap each TransformerBlock forward with gradient checkpointing."""
    for block in model.transformer["blocks"]:
        orig_forward = block.forward
        block.forward = lambda x, orig=orig_forward: torch.utils.checkpoint.checkpoint(
            orig, x, use_reentrant=False
        )
    print("  [ckpt] Gradient checkpointing enabled")


# ── Distillation Dataset ───────────────────────────────────────────────────

class DistillDataset(Dataset):
    """
    Loads instruction-output pairs for distillation.

    Each sample yields (input_ids, labels, loss_mask) where:
    - Prompt tokens have loss_mask=0
    - Response tokens have loss_mask=1

    Caches teacher logits (by text hash) to avoid re-running teacher.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        cache_teacher: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.cache_teacher = cache_teacher
        self.samples = []
        self.teacher_cache = {}

        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                instruction = obj.get("instruction", obj.get("prompt", ""))
                inp = obj.get("input", "")
                output = obj.get("output", obj.get("response", ""))

                if not instruction or not output:
                    skipped += 1
                    continue

                prompt_text = f"<|user|> {instruction}"
                if inp:
                    prompt_text += f"\n{inp}"
                prompt_text += " <|end|>"
                completion_text = f"<|assistant|> {output} <|end|>"

                prompt_ids = tokenizer.encode(prompt_text)
                completion_ids = tokenizer.encode(completion_text) + [tokenizer.eos_id]

                if len(prompt_ids) > max_prompt_length:
                    prompt_ids = prompt_ids[:max_prompt_length]

                all_ids = prompt_ids + completion_ids
                if len(all_ids) > max_length:
                    all_ids = all_ids[:max_length]

                prompt_len = len(prompt_ids)
                total_len = len(all_ids)
                loss_mask = [0] * min(prompt_len, total_len) + [1] * max(0, total_len - prompt_len)

                input_ids = np.array(all_ids[:-1], dtype=np.int64)
                labels = np.array(all_ids[1:], dtype=np.int64)
                mask = np.array(loss_mask[1:], dtype=np.float32)

                if len(input_ids) < 4:
                    skipped += 1
                    continue

                self.samples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_mask": mask,
                    "text": prompt_text + completion_text,
                })

        if skipped:
            print(f"  [distill] Skipped {skipped} invalid/empty samples")
        print(f"[dataset] Distillation samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["input_ids"]),
            torch.from_numpy(s["labels"]),
            torch.from_numpy(s["loss_mask"]),
            s["text"],
        )


def distill_collate_fn(batch, pad_id: int = 0):
    """Pad input_ids, labels, loss_mask to same length in batch."""
    input_ids_list, labels_list, masks_list, texts = zip(*batch)

    max_len = max(x.size(0) for x in input_ids_list)

    def pad(tensors, value, dtype):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.size(0),), value, dtype=dtype)])
            for t in tensors
        ])

    return (
        pad(input_ids_list, pad_id, torch.long),
        pad(labels_list, -1, torch.long),
        pad(masks_list, 0.0, torch.float),
        texts,
    )


# ── Teacher Logit Caching ─────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    """SHA-256 hash for caching teacher outputs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TeacherCache:
    """Cache teacher logits keyed by text hash, persisted to disk."""

    def __init__(self, cache_dir: str = ".cache/teacher"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> Optional[str]:
        h = _text_hash(text)
        cache_path = os.path.join(self.cache_dir, f"{h}.pt")
        if os.path.exists(cache_path):
            self.hits += 1
            return cache_path
        self.misses += 1
        return None

    def put(self, text: str, logits: torch.Tensor):
        h = _text_hash(text)
        cache_path = os.path.join(self.cache_dir, f"{h}.pt")
        torch.save(logits.half(), cache_path)

    def stats(self) -> str:
        total = self.hits + self.misses
        pct = 100 * self.hits / max(1, total)
        return f"[cache] {self.hits}/{total} hits ({pct:.0f}%)"

    def clear(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)


# ── Distillation Loss ──────────────────────────────────────────────────────

def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    labels: torch.Tensor = None,
    loss_mask: torch.Tensor = None,
    kl_weight: float = 0.5,
    temperature: float = 2.0,
    label_smoothing: float = 0.0,
    pad_id: int = 0,
) -> dict:
    """
    Combined distillation loss:

    L = (1 - α) * CE(student, labels) + α * T² * KL(teacher_probs || student_probs)

    where α = kl_weight and T = temperature.
    The T² scaling preserves gradient magnitudes (Hinton et al., 2015).

    Returns dict with 'loss', 'ce_loss', 'kl_loss'.
    """
    B, T, V_s = student_logits.shape

    # ── Hard label loss (cross-entropy) ──────────────────────────────────
    # Chunk the flat sequence dim so torch's fp32 softmax transient never
    # spans the full (B*T, 100000) tensor (~1.6GB). 512 rows/chunk keeps the
    # peak near ~200MB; softmax is per-row so numerics are identical.
    ce_flat_parts = []
    flat_logits = student_logits.reshape(-1, V_s)
    flat_labels = labels.reshape(-1)
    _CE_CHUNK = 512
    for _s in range(0, flat_logits.shape[0], _CE_CHUNK):
        ce_flat_parts.append(
            F.cross_entropy(
                flat_logits[_s:_s + _CE_CHUNK],
                flat_labels[_s:_s + _CE_CHUNK],
                ignore_index=-1,
                reduction="none",
            )
        )
    ce_flat = torch.cat(ce_flat_parts) if len(ce_flat_parts) > 1 else ce_flat_parts[0]
    ce_loss = (ce_flat.view(B, T) * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # ── KL divergence loss ──────────────────────────────────────────────
    # KL is impossible without a teacher or with a vocab mismatch; the
    # mismatch is warned about ONCE at startup (distill_train), stay silent.
    if teacher_logits is None or teacher_logits.shape[-1] != V_s:
        return {
            "loss": ce_loss,
            "ce_loss": ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss,
            "kl_loss": 0.0,
        }
    V_t = teacher_logits.shape[-1]

    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl_div = F.kl_div(
        student_log_probs.view(-1, V_s),
        teacher_probs.view(-1, V_s),
        reduction="none",
    ).sum(dim=-1).view(B, T)

    kl_loss = (kl_div * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    kl_loss = kl_loss * (temperature ** 2)

    # ── Combined loss ────────────────────────────────────────────────────
    alpha = kl_weight
    loss = (1 - alpha) * ce_loss + alpha * kl_loss

    return {
        "loss": loss,
        "ce_loss": ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss,
        "kl_loss": kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def distill_evaluate(
    student_model: DizelLM,
    teacher_model,
    val_loader,
    eval_iters: int,
    device: torch.device,
    amp_ctx,
    kl_weight: float,
    temperature: float,
    teacher_tokenizer,
    kl_active: bool = True,
) -> dict:
    """Evaluate distillation metrics on validation set."""
    student_model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    count = 0

    for batch in val_loader:
        if count >= eval_iters:
            break
        input_ids, labels, loss_mask, texts = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)

        with amp_ctx:
            student_logits, _ = student_model(input_ids, None)

            if kl_active:
                teacher_logits = _run_teacher_batch(
                    teacher_model, teacher_tokenizer, texts, input_ids,
                    device, amp_ctx,
                )
            else:
                teacher_logits = None

            metrics = distill_loss(
                student_logits, teacher_logits,
                labels, loss_mask,
                kl_weight=kl_weight,
                temperature=temperature,
            )

        total_loss += metrics["loss"].item()
        total_ce += metrics["ce_loss"]
        total_kl += metrics["kl_loss"]
        count += 1

    student_model.train()
    n = max(1, count)
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "kl_loss": total_kl / n,
    }


def _run_teacher_batch(teacher_model, teacher_tokenizer, texts, input_ids, device, amp_ctx):
    """Run teacher on a batch of texts and return logits aligned to student's sequence length."""
    B = len(texts)
    T = input_ids.shape[1]

    enc = teacher_tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=T + 32,
    )
    t_input_ids = enc["input_ids"].to(device)
    t_attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        with amp_ctx:
            t_outputs = teacher_model(
                t_input_ids,
                attention_mask=t_attention_mask,
                output_hidden_states=False,
            )

    t_logits = t_outputs.logits

    T_t = t_logits.shape[1]
    V_t = t_logits.shape[-1]

    if T_t > T:
        t_logits = t_logits[:, :T, :]
    elif T_t < T:
        pad = torch.zeros(B, T - T_t, V_t, device=device, dtype=t_logits.dtype)
        t_logits = torch.cat([t_logits, pad], dim=1)

    return t_logits


# ── Training Loop ──────────────────────────────────────────────────────────

def distill_train(
    cfg: DizelConfig,
    base_ckpt: str,
    teacher_model_name: str,
    hf_token: Optional[str] = None,
) -> None:
    dc = cfg.distill if hasattr(cfg, 'distill') else cfg.sft
    mc = cfg.model

    try:
        from config import V15_CONFIG
        if hasattr(V15_CONFIG, 'distill'):
            dc = V15_CONFIG.distill
    except (ImportError, AttributeError):
        pass

    rule("Knowledge Distillation")

    section("Hardware")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device : {device}")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        info(f"GPU    : {torch.cuda.get_device_name()} / {vram:.1f} GB VRAM")

    use_amp = dc.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if dc.amp_dtype == "bfloat16" else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer()
    mc.vocab_size = len(tokenizer)

    section("Data")

    # ── Data ─────────────────────────────────────────────────────────────
    data_path = dc.data_path
    if not os.path.exists(data_path):
        warning(f"Distill data not found at '{data_path}'")
        data_path = "data/formatted/v15_distill.jsonl"
        if not os.path.exists(data_path):
            error(f"Also not at '{data_path}'")
            sys.exit(1)

    full_ds = DistillDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=dc.max_length,
        max_prompt_length=min(dc.max_length, 1024),
    )

    n_val = max(1, int(len(full_ds) * 0.10))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    collate = lambda b: distill_collate_fn(b, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=dc.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dc.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    ok(f"Dataset loaded: {len(train_ds)} train + {len(val_ds)} val samples")

    section("Models")

    # ── Student Model ─────────────────────────────────────────────────────
    student_model = DizelLM(mc).to(device)

    # ── Load base checkpoint / attach LoRA ──────────────────────────────
    step = 0
    is_resume = base_ckpt and "distill" in base_ckpt.lower()
    ckpt_phase0 = None
    ckpt_phase_tag = None

    # Fresh launches load the plain (non-LoRA) base checkpoint BEFORE LoRA
    # is attached (the keys are plain nn.Linear). Resumes must attach LoRA
    # FIRST so the LoRA adapter weights AND optimizer state can be restored
    # (the checkpoint keys are LoRA-wrapped).
    if base_ckpt and os.path.exists(base_ckpt) and not is_resume:
        loaded_step, _meta = load_checkpoint(base_ckpt, student_model, device=str(device))
        ckpt_phase0 = _meta.get("phase0")
        ckpt_phase_tag = _meta.get("phase_tag")
        ok(f"Student loaded from: {base_ckpt}")
    elif is_resume and not (base_ckpt and os.path.exists(base_ckpt)):
        warning("Resume checkpoint not found — starting from scratch")

    # ── Attach LoRA ──────────────────────────────────────────────────────
    attach_lora(
        student_model,
        r=dc.lora_r,
        alpha=dc.lora_alpha,
        dropout=dc.lora_dropout,
        target_names=dc.lora_target_modules if hasattr(dc, 'lora_target_modules') else None,
    )

    # LoRA adapters are created on CPU by attach_lora (torch.empty default
    # device); push the whole model (base + adapters) to the device AFTER
    # attaching so lora_A/lora_B land on the same device as the base weights.
    student_model.to(device)

    # ── Gradient checkpointing ───────────────────────────────────────────
    if dc.gradient_checkpointing:
        _enable_gradient_checkpointing(student_model)

    # ── Teacher Model ────────────────────────────────────────────────────
    teacher_model, teacher_tokenizer = load_teacher_model(
        teacher_model_name, device, amp_dtype, hf_token=hf_token
    )

    # KL distillation requires teacher logits in the STUDENT's vocab space.
    # With mismatched vocabs (e.g. Qwen 151936 vs Dizel 100000) it is
    # impossible, so: warn once (not every step), and skip the teacher forward
    # entirely — it allocates ~1.2GB of discarded logits + a full 500M-param
    # forward every micro-step otherwise.
    kl_active = False
    if teacher_model is not None and teacher_tokenizer is not None:
        t_vocab = getattr(teacher_tokenizer, "vocab_size", None)
        kl_active = (t_vocab is not None and t_vocab == mc.vocab_size)
        if not kl_active:
            warning(
                f"Teacher vocab ({t_vocab}) ≠ Student vocab ({mc.vocab_size}) — "
                "KL DISABLED, training CE-only (run train_xkd.py --kd-mode align "
                "for genuine cross-vocab KD)"
            )

    # ── Optimiser ───────────────────────────────────────────────────────
    param_groups = [
        {"params": [p for p in student_model.parameters() if p.requires_grad and p.ndim >= 2],
         "weight_decay": dc.weight_decay},
        {"params": [p for p in student_model.parameters() if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=dc.lr, betas=(0.9, 0.95), eps=1e-8)

    # Resume: restore model (base + LoRA adapters) and optimizer state
    # together in one load.
    if is_resume and base_ckpt and os.path.exists(base_ckpt):
        loaded_step, _meta = load_checkpoint(
            base_ckpt, student_model,
            optimizer=optimizer,
            device=str(device),
        )
        ckpt_phase0 = _meta.get("phase0")
        ckpt_phase_tag = _meta.get("phase_tag")
        step = loaded_step
        ok(f"Student + LoRA + optimizer state restored from: {base_ckpt}")
    _resume_step = step

    # -- LR schedule anchor ----------------------------------------------
    # phase0 = global step where this phase first began; persisted in
    # checkpoints so a mid-phase resume CONTINUES decaying instead of
    # re-entering warmup. Legacy or cross-phase checkpoints start a fresh
    # schedule anchored at the current step.
    phase_budget = dc.phase1_steps
    phase_tag = make_schedule_tag(phase_budget, dc.lr, dc.lr * 0.1, dc.warmup_steps)
    if ckpt_phase0 is not None and ckpt_phase_tag == phase_tag:
        phase0 = ckpt_phase0
    else:
        phase0 = step
    # `phase1_steps` is a PER-PHASE budget counted from the phase's FIRST step
    # (phase0), so the absolute stopping step stays fixed across mid-phase
    # resumes: resuming at step N simply gets closer to the same fixed end.
    total_steps  = phase0 + phase_budget

    ok(f"Model: DizelLM({mc.d_model}, {mc.n_layers} layers) — {student_model.num_parameters() / 1e6:.1f}M params")
    info(f"Teacher: {teacher_model_name}")

    section("Training")

    # ── Training ─────────────────────────────────────────────────────────
    os.makedirs(dc.checkpoint_dir, exist_ok=True)
    os.makedirs(dc.log_dir, exist_ok=True)

    log_file = open(os.path.join(dc.log_dir, f"{dc.run_name}.csv"), "w")
    log_file.write("step,loss,ce_loss,kl_loss,lr\n")

    best_val = float("inf")
    patience = 5
    min_delta = 5e-4
    patience_counter = 0
    early_stop = False

    data_iter = iter(train_loader)
    micro_step = 0
    accum_loss = 0.0

    student_model.train()
    optimizer.zero_grad()

    teacher_ckpt_name = teacher_model_name.split("/")[-1] if teacher_model_name else "none"
    config_table({
        "teacher":      teacher_model_name,
        "KL weight":    dc.kl_weight,
        "temperature":  dc.temperature,
        "LoRA rank":    dc.lora_r,
        "LoRA alpha":   dc.lora_alpha,
    })
    config_table({
        "total steps":  phase_budget,
        "target steps": total_steps,
        "batch size":   dc.batch_size,
        "grad accum":   dc.grad_accum,
        "learning rate": f"{dc.lr:.2e}",
        "warmup steps": dc.warmup_steps if hasattr(dc, 'warmup_steps') else dc.phase1_warmup,
    })

    meter = StepMeter(
        total_steps,
        tokens_per_step=dc.batch_size * dc.grad_accum * mc.context_length,
        start_step=step,
    )

    while step < total_steps and not early_stop:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids, labels, loss_mask, texts = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)

        with amp_ctx:
            student_logits, _ = student_model(input_ids, None)

            if kl_active:
                teacher_logits = _run_teacher_batch(
                    teacher_model, teacher_tokenizer, texts, input_ids,
                    device, amp_ctx,
                )
            else:
                teacher_logits = None

            metrics = distill_loss(
                student_logits, teacher_logits,
                labels, loss_mask,
                kl_weight=dc.kl_weight,
                temperature=dc.temperature,
            )
            loss = metrics["loss"] / dc.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1

        if micro_step % dc.grad_accum == 0:
            warmup = dc.warmup_steps
            # LR schedule — anchored to phase start so a mid-phase resume
            # continues decaying instead of re-entering warmup.
            local = step - phase0
            lr = cosine_lr_with_warmup(local, warmup, phase_budget, dc.lr, dc.lr * 0.1)
            set_lr(optimizer, lr)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad],
                dc.grad_clip,
            )

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            train_loss = accum_loss
            accum_loss = 0.0
            step += 1

            if step % 25 == 0 or step == _resume_step + 1:
                meter.update(step=step, loss=train_loss, lr=lr,
                             extra=f"KL={dc.kl_weight:.1f} T={dc.temperature:.1f}")

            if step % dc.eval_interval == 0:
                val_metrics = distill_evaluate(
                    student_model, teacher_model, val_loader,
                    dc.eval_iters, device, amp_ctx,
                    dc.kl_weight, dc.temperature, teacher_tokenizer,
                    kl_active=kl_active,
                )
                validation(step, val_metrics["loss"],
                           extra=f"ce {val_metrics['ce_loss']:.4f} kl {val_metrics['kl_loss']:.4f}")
                log_file.write(
                    f"{step},{val_metrics['loss']:.4f},"
                    f"{val_metrics['ce_loss']:.4f},{val_metrics['kl_loss']:.4f},{lr:.6f}\n"
                )
                log_file.flush()
                torch.cuda.empty_cache()

                if val_metrics["loss"] < (best_val - min_delta):
                    best_val = val_metrics["loss"]
                    patience_counter = 0
                    # Save UNMERGED (LoRA separate) and leave the model
                    # unmerged so training continues through the LoRA path.
                    unmerge_lora_weights(student_model)
                    save_checkpoint(
                        student_model, optimizer, step, best_val, cfg,
                        os.path.join(dc.checkpoint_dir, f"{dc.run_name}-best.pt"),
                        tokenizer=tokenizer,
                        phase0=phase0,
                        phase_tag=phase_tag,
                        extra_meta={"lora_alpha": dc.lora_alpha, "lora_r": dc.lora_r},
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        warning(
                            f"Early stopping at step {step} "
                            f"(no improvement for {patience} evals)"
                        )
                        early_stop = True
                        break

            if not getattr(cfg, '_best_only', False) and step % dc.save_interval == 0:
                unmerge_lora_weights(student_model)
                save_checkpoint(
                    student_model, optimizer, step, float("inf"), cfg,
                    os.path.join(dc.checkpoint_dir, f"{dc.run_name}-step{step}.pt"),
                    tokenizer=tokenizer,
                    phase0=phase0,
                    phase_tag=phase_tag,
                    extra_meta={"lora_alpha": dc.lora_alpha, "lora_r": dc.lora_r},
                )

    # Final save
    unmerge_lora_weights(student_model)
    save_checkpoint(
        student_model, optimizer, step, best_val, cfg,
        os.path.join(dc.checkpoint_dir, f"{dc.run_name}-final.pt"),
        tokenizer=tokenizer,
        phase0=phase0,
        phase_tag=phase_tag,
        extra_meta={"lora_alpha": dc.lora_alpha, "lora_r": dc.lora_r},
    )
    merge_lora_weights(student_model)

    log_file.close()
    meter.finalise()
    config_table({
        "Best val loss": f"{best_val:.4f}",
        "Total steps":   f"{step:,}",
        "Teacher":       teacher_model_name,
    })


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dizel v1.5 LoRA Distillation")
    p.add_argument("--base_checkpoint", type=str, default="",
                   help="Path to student base checkpoint (.pt)")
    p.add_argument("--resume", type=str, default="",
                   help="Path to a distill checkpoint (.pt) to resume from")
    p.add_argument("--teacher_model", type=str, default=None,
                   help="HuggingFace teacher model name")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace token for gated teacher models "
                        "(or HF_TOKEN env var; prompts and saves to .hf_token otherwise)")
    p.add_argument("--data_path", type=str, default=None,
                   help="Path to distillation JSONL")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--kl_weight", type=float, default=None,
                   help="KL divergence weight in loss")
    p.add_argument("--temperature", type=float, default=None,
                   help="Temperature for distillation")
    p.add_argument("--lora_r", type=int, default=None,
                   help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=None,
                   help="LoRA alpha scaling")
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP even on CUDA")
    p.add_argument("--use_amp", action="store_true",
                   help="Enable AMP (overrides config)")
    p.add_argument("--best_only", action="store_true",
                   help="Save only the best checkpoint (skip periodic step checkpoints)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = V15_CONFIG

    if not args.base_checkpoint and not args.resume:
        print("Error: Must provide either --base_checkpoint or --resume")
        sys.exit(1)

    distill_cfg = V15_CONFIG.distill if hasattr(V15_CONFIG, 'distill') else cfg.sft

    if args.data_path is not None:    distill_cfg.data_path = args.data_path
    if args.lr is not None:           distill_cfg.lr = args.lr
    if args.max_steps is not None:
        distill_cfg.max_steps   = args.max_steps
        distill_cfg.phase1_steps = args.max_steps   # keep alias in sync
    if args.batch_size is not None:   distill_cfg.batch_size = args.batch_size
    if args.grad_accum is not None:   distill_cfg.grad_accum = args.grad_accum
    if args.kl_weight is not None:    distill_cfg.kl_weight = args.kl_weight
    if args.temperature is not None:  distill_cfg.temperature = args.temperature
    if args.lora_r is not None:       distill_cfg.lora_r = args.lora_r
    if args.lora_alpha is not None:   distill_cfg.lora_alpha = args.lora_alpha
    if args.warmup_steps is not None: distill_cfg.warmup_steps = args.warmup_steps
    if args.no_amp:                   distill_cfg.use_amp = False
    if args.use_amp:                  distill_cfg.use_amp = True
    cfg._best_only = args.best_only

    teacher_model = args.teacher_model or getattr(distill_cfg, 'teacher_model', 'Qwen/Qwen2.5-0.5B-Instruct')
    target_ckpt = args.resume if args.resume else args.base_checkpoint

    hf_token = resolve_hf_token(args.hf_token)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    distill_train(cfg, target_ckpt, teacher_model, hf_token=hf_token)


if __name__ == "__main__":
    main()
