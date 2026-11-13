"""
training/experiments/cross_vocab_kd/train_xkd.py — Cross-vocab KD runner (Arms A/B/C).

Standalone CLI runner for the three experimental distillation arms:
  A (ce)       : CE-only SFT — student trained on the base dataset labels.
  B (qwen-sft) : SFT on Qwen-generated responses (labels = teacher text).
  C (align)    : genuine aligned cross-vocab KD — teacher final hidden states
                 projected into student space via a learned projector, then
                 pushed through the student lm_head so KL is computed in the
                 student's vocab space at lexically-aligned response positions.

Mirrors training/distill.py's training-loop structure (LoRA student, AMP,
gradient accumulation, cosine LR, best/final checkpoints) and consumes the
shared cross-vocab KD library in this package (alignment, projector, kd_loss,
metrics_logger) built by the library agent — code to those EXACT interfaces.

Usage
-----
    # Arm A — CE-only SFT
    python training/experiments/cross_vocab_kd/train_xkd.py --kd-mode ce

    # Arm B — Qwen-generated response SFT (generates data on first run)
    python training/experiments/cross_vocab_kd/train_xkd.py --kd-mode qwen-sft

    # Arm C — genuine aligned KD
    python training/experiments/cross_vocab_kd/train_xkd.py --kd-mode align \
        --kd-weight 0.5 --temperature 2.0
"""

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader, random_split

# Repo root = three levels up from this file (training/experiments/cross_vocab_kd/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from config import V15_CONFIG, ModelConfigV15
from model.architecture import DizelLM
from training.dataset import Tokenizer
from training.distill import (
    load_teacher_model,
    DistillDataset,
    distill_collate_fn,
    attach_lora,
    merge_lora_weights,
    unmerge_lora_weights,
    resolve_hf_token,
    _enable_gradient_checkpointing,
    DEFAULT_LORA_TARGETS,
)
from training.pretrain import (
    cosine_lr_with_warmup,
    set_lr,
    save_checkpoint,
    load_checkpoint,
    make_schedule_tag,
)
from training.display_utils import (
    rule, section, info, ok, warning, error, dim,
    config_table, StepMeter, validation,
)

torch.serialization.add_safe_globals([ModelConfigV15])

# ── Constants ──────────────────────────────────────────────────────────────

QWEN_SFT_PATH = os.path.join("data", "formatted", "v15_qwen_sft.jsonl")
QWEN_ASSISTANT_MARKER = "<|im_start|>assistant"
QWEN_USER_MARKER = "<|im_start|>user"
QWEN_END_MARKER = "<|im_end|>"

RUN_TAGS = {
    "ce":       "xkd-ce",
    "qwen-sft": "xkd-qwensft",
    "align":    "xkd-align",
}

# Default checkpoint output paths (mirrors what the Colab pipeline expects).
ARM_CKPTS = {
    "ce":       os.path.join("checkpoints", "experiments", "xkd-ce-best.pt"),
    "qwen-sft": os.path.join("checkpoints", "experiments", "xkd-qwensft-best.pt"),
    "align":    os.path.join("checkpoints", "experiments", "xkd-align-best.pt"),
}


# ---------------------------------------------------------------------------
# Lazy import of the cross-vocab KD library (built by the library agent).
#
# Imported here — not at module top — so `--help` and `py_compile` work even
# before that package lands, and so a missing library produces a clear error
# instead of a bare ImportError.
# ---------------------------------------------------------------------------
def _load_kd_modules() -> dict:
    try:
        from training.experiments.cross_vocab_kd.alignment import (
            compute_alignment, AlignmentCache,
        )
        from training.experiments.cross_vocab_kd.projector import VocabProjector
        from training.experiments.cross_vocab_kd.kd_loss import (
            cross_vocab_distill_loss,
            assert_kd_not_silently_disabled,
            reset_kd_guard,
        )
        from training.experiments.cross_vocab_kd.metrics_logger import KDMetricsLogger
    except ImportError as e:
        raise RuntimeError(
            "Cross-vocab KD library not available under "
            "training/experiments/cross_vocab_kd/ — build it first. "
            f"ImportError: {e}"
        ) from e
    return {
        "compute_alignment": compute_alignment,
        "AlignmentCache": AlignmentCache,
        "VocabProjector": VocabProjector,
        "cross_vocab_distill_loss": cross_vocab_distill_loss,
        "assert_kd_not_silently_disabled": assert_kd_not_silently_disabled,
        "reset_kd_guard": reset_kd_guard,
        "KDMetricsLogger": KDMetricsLogger,
    }


# ---------------------------------------------------------------------------
# Teacher hidden states
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_teacher_hidden(
    teacher_model,
    teacher_tokenizer,
    texts: List[str],
    device: torch.device,
    amp_ctx,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """Tokenise `texts` with the HF tokenizer and return the teacher's final
    hidden state (hidden_states[-1], 896-dim for Qwen2.5-0.5B).

    Teacher runs frozen (already .eval()/.no_grad() from load_teacher_model).
    """
    enc = teacher_tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    t_input_ids = enc["input_ids"].to(device)
    t_attention_mask = enc["attention_mask"].to(device)
    with amp_ctx:
        outs = teacher_model(
            t_input_ids,
            attention_mask=t_attention_mask,
            output_hidden_states=True,
        )
    return outs.hidden_states[-1]


# ---------------------------------------------------------------------------
# Lexical alignment (cached)
# ---------------------------------------------------------------------------
def _build_aligned_pairs(
    texts: List[str],
    tokenizer: Tokenizer,
    teacher_tokenizer,
    alignment_cache,
    count_failures: bool = False,
    compute_alignment_fn=None,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """Per-sample alignment pairs, in batch order.

    Uses the AlignmentCache (keyed by raw text); on a miss, re-encodes the
    text with BOTH tokenizers (unpadded per-sample ids) and runs
    compute_alignment. A single failure logs a warning and yields [] for that
    sample — it must never crash the run.
    """
    if compute_alignment_fn is None:
        compute_alignment_fn = _load_kd_modules()["compute_alignment"]
    pairs_list: List[List[Tuple[int, int]]] = []
    failures = 0
    for text in texts:
        pairs = []
        try:
            pairs = alignment_cache.get(text)
            if pairs is None:
                student_ids = tokenizer.encode(text)  # unpadded per-sample ids
                teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=False)
                pairs = compute_alignment_fn(
                    student_ids=student_ids,
                    teacher_ids=teacher_ids,
                    student_tok=tokenizer,
                    teacher_tok=teacher_tokenizer,
                )
                alignment_cache.put(text, pairs)
        except Exception as e:
            failures += 1
            warning(f"[align] alignment failed for a sample ({type(e).__name__}: {e})")
            pairs = []
        pairs_list.append(pairs or [])
    return pairs_list, failures


# ---------------------------------------------------------------------------
# Arm B data generation (Qwen-generated response SFT)
# ---------------------------------------------------------------------------
def _extract_assistant_response(text: str) -> str:
    """Pull the response out of a decoded Qwen2.5 chat completion."""
    idx = text.find(QWEN_ASSISTANT_MARKER)
    if idx < 0:
        return ""
    after = text[idx + len(QWEN_ASSISTANT_MARKER):].lstrip("\n")
    end = after.find(QWEN_END_MARKER)
    if end >= 0:
        after = after[:end]
    return after.strip()


def _generate_qwen_sft(
    out_path: str,
    source_path: str,
    indices: List[int],
    teacher_model,
    teacher_tokenizer,
    device: torch.device,
    max_new_tokens: int = 256,
    progress_every: int = 25,
) -> None:
    """Generate teacher responses for the first `len(indices)` TRAIN samples
    and write them as a JSONL consumable by DistillDataset.

    One JSONL line per sample:
        {"instruction": ..., "input": ..., "output": <Qwen response>}
    """
    # Read the original instruction/input rows (no tokenization needed).
    rows = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            inst = obj.get("instruction", obj.get("prompt", ""))
            inp = obj.get("input", "")
            if inst:
                rows.append((inst, inp))
    chosen = [rows[i] for i in indices if i < len(rows)]
    if not chosen:
        warning(f"[qwen-sft] No samples to generate from {source_path}")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    info(f"[qwen-sft] Generating teacher responses for {len(chosen)} samples → {out_path}")

    teacher_model.eval()
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for i, (inst, inp) in enumerate(chosen):
                prompt = f"{QWEN_USER_MARKER}\n{inst}"
                if inp:
                    prompt += f"\n{inp}"
                prompt += f"{QWEN_END_MARKER}\n{QWEN_ASSISTANT_MARKER}\n"
                try:
                    enc = teacher_tokenizer(prompt, return_tensors="pt")
                    t_ids = enc["input_ids"].to(device)
                    t_mask = enc["attention_mask"].to(device)
                    gen_ids = teacher_model.generate(
                        t_ids,
                        attention_mask=t_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=teacher_tokenizer.eos_token_id,
                    )
                    full = teacher_tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                    resp = _extract_assistant_response(full)
                except Exception as e:
                    warning(f"[qwen-sft] generation failed for sample {i}: {e}")
                    resp = ""
                if not resp:
                    continue
                f.write(json.dumps(
                    {"instruction": inst, "input": inp, "output": resp},
                    ensure_ascii=False,
                ) + "\n")
                written += 1
                if (i + 1) % progress_every == 0:
                    info(f"[qwen-sft] generated {i + 1}/{len(chosen)} (wrote {written})")
    ok(f"[qwen-sft] Wrote {written} teacher-generated samples → {out_path}")


# ---------------------------------------------------------------------------
# Evaluation (distill_evaluate-style, with the cross-vocab path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def xkd_evaluate(
    student_model: DizelLM,
    teacher_model,
    teacher_tokenizer,
    val_loader,
    eval_iters: int,
    device: torch.device,
    amp_ctx,
    tokenizer: Tokenizer,
    projector,
    lm_head_weight: torch.Tensor,
    alignment_cache,
    hidden_holder: dict,
    mode: str,
    kd_weight: float,
    temperature: float,
    kd_loss_fn,
    teacher_hidden_fn,
    align_pairs_fn,
) -> dict:
    """Evaluate CE/KD metrics on the validation split.

    Mode `align` runs the full teacher + alignment path; if alignment fails
    for a sample it is logged as DISABLED but never raises. CE/qwen-sft modes
    route through the shared loss with KD disabled (uniform logging).
    """
    student_model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_active = 0
    total_aligned = 0
    total_failed = 0
    count = 0

    for batch in val_loader:
        if count >= eval_iters:
            break
        input_ids, labels, loss_mask, texts = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss_mask = loss_mask.to(device)

        with amp_ctx:
            logits, _ = student_model(input_ids, None)
            student_hidden = hidden_holder["hidden"]

            if mode == "align":
                teacher_hidden = teacher_hidden_fn(
                    teacher_model, teacher_tokenizer, texts, device, amp_ctx,
                    max_length=input_ids.shape[1] + 32,
                )
                pairs_list, failed = align_pairs_fn(
                    texts, tokenizer, teacher_tokenizer, alignment_cache,
                    count_failures=True,
                )
                total_failed += failed
                metrics = kd_loss_fn(
                    student_logits=logits,
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    projector=projector,
                    lm_head_weight=lm_head_weight,
                    labels=labels,
                    loss_mask=loss_mask,
                    aligned_pairs_list=pairs_list,
                    kl_weight=kd_weight,
                    temperature=temperature,
                    batch_ids=None,
                )
            else:
                # CE-only path: KD disabled — dummy teacher hidden + frozen
                # projector, empty pairs, kl_weight=0 (reduces to masked CE).
                dummy_hidden = torch.zeros(
                    (logits.shape[0], 1, 896),
                    device=logits.device, dtype=logits.dtype,
                )
                metrics = kd_loss_fn(
                    student_logits=logits,
                    student_hidden=student_hidden,
                    teacher_hidden=dummy_hidden,
                    projector=projector,
                    lm_head_weight=lm_head_weight,
                    labels=labels,
                    loss_mask=loss_mask,
                    aligned_pairs_list=[[] for _ in range(len(texts))],
                    kl_weight=0.0,
                    temperature=temperature,
                    batch_ids=None,
                )

        total_loss += metrics["loss"].item()
        total_ce += metrics["ce_loss"]
        total_kd += metrics["kd_loss"]
        total_active += int(metrics["active"])
        total_aligned += int(metrics["aligned_count"])
        count += 1

    student_model.train()
    n = max(1, count)
    if total_failed and mode == "align":
        warning(
            f"[eval] {total_failed} alignment failures — "
            "KD disabled for those samples (logged, not raised)"
        )
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "kd_loss": total_kd / n,
        "active": total_active / n,
        "aligned_count": total_aligned / n,
        "alignment_failed": total_failed,
    }


# ---------------------------------------------------------------------------
# Main training run
# ---------------------------------------------------------------------------
def xkd_train(args: argparse.Namespace) -> None:
    kd = _load_kd_modules()
    AlignmentCache = kd["AlignmentCache"]
    VocabProjector = kd["VocabProjector"]
    cross_vocab_distill_loss = kd["cross_vocab_distill_loss"]
    assert_kd_not_silently_disabled = kd["assert_kd_not_silently_disabled"]
    reset_kd_guard = kd["reset_kd_guard"]
    KDMetricsLogger = kd["KDMetricsLogger"]

    mode = args.kd_mode
    rule(f"Cross-Vocab Knowledge Distillation — Arm {mode.upper()}")

    # ── Hardware / AMP ────────────────────────────────────────────────────
    section("Hardware")
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    info(f"Device : {device}")
    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        info(f"GPU    : {torch.cuda.get_device_name()} / {vram:.1f} GB VRAM")

    use_amp = V15_CONFIG.distill.use_amp and device.type == "cuda"
    if args.no_amp:
        use_amp = False
    if args.use_amp:
        use_amp = True
    amp_dtype = torch.bfloat16 if V15_CONFIG.distill.amp_dtype == "bfloat16" else torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else nullcontext()
    )
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    info(f"AMP    : {'ON (' + V15_CONFIG.distill.amp_dtype + ')' if use_amp else 'OFF'}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = Tokenizer()
    mc = V15_CONFIG.model
    mc.vocab_size = len(tokenizer)

    # ── Teacher ───────────────────────────────────────────────────────────
    teacher_model = None
    teacher_tokenizer = None
    if mode in ("qwen-sft", "align"):
        section("Models")
        teacher_model, teacher_tokenizer = load_teacher_model(
            args.teacher_model, device, amp_dtype, hf_token=args.hf_token
        )
        if teacher_model is None and mode == "align":
            # Fail-closed: arm C without a teacher silently degrades to CE —
            # exactly what the KD guard exists to prevent.
            error(
                "Arm C (align) REQUIRES a teacher model but none could be "
                "loaded (transformers missing?). Failing closed — not "
                "falling back to CE."
            )
            sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────
    section("Data")
    data_path = args.data_path
    if not os.path.exists(data_path):
        warning(f"Distill data not found at '{data_path}'")
        data_path = "data/formatted/v15_distill.jsonl"
        if not os.path.exists(data_path):
            error(f"Also not at '{data_path}' — pass --data-path")
            sys.exit(1)

    def _build_splits(path: str):
        full_ds = DistillDataset(
            jsonl_path=path,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_prompt_length=min(args.max_length, 1024),
        )
        n_val = max(1, int(len(full_ds) * 0.10))
        n_train = len(full_ds) - n_val
        tr_ds, va_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        return full_ds, tr_ds, va_ds

    full_ds, train_ds, val_ds = _build_splits(data_path)

    # Arm B: on first run, generate teacher responses for the TRAIN split.
    if mode == "qwen-sft":
        if not os.path.exists(QWEN_SFT_PATH):
            if teacher_model is None:
                error(
                    "Arm B (qwen-sft) needs to generate teacher responses but "
                    "no teacher model is available (transformers missing?)."
                )
                sys.exit(1)
            n_needed = args.max_steps * args.batch_size * args.grad_accum
            _generate_qwen_sft(
                QWEN_SFT_PATH, data_path, train_ds.indices[:n_needed],
                teacher_model, teacher_tokenizer, device,
            )
        data_path = QWEN_SFT_PATH
        full_ds, train_ds, val_ds = _build_splits(data_path)

    if len(train_ds) < 1:
        error("Training split is empty — cannot train (is the data file valid?)")
        sys.exit(1)
    if len(train_ds) < args.batch_size:
        warning(
            f"Train split ({len(train_ds)} samples) < batch size ({args.batch_size}); "
            "drop_last may yield no batches"
        )

    collate = lambda b: distill_collate_fn(b, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    ok(f"Dataset loaded: {len(train_ds)} train + {len(val_ds)} val samples")

    # ── Student model + base checkpoint + LoRA ────────────────────────────
    section("Models")
    student_model = DizelLM(mc).to(device)
    is_resume = bool(args.resume) and os.path.exists(args.resume)
    step = 0
    ckpt_phase0 = None
    ckpt_phase_tag = None

    # Fresh launches load the plain (non-LoRA) base checkpoint BEFORE LoRA is
    # attached (keys are plain nn.Linear). Resumes attach LoRA FIRST so the
    # adapter weights AND optimizer state can be restored together.
    if is_resume:
        attach_lora(
            student_model,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            target_names=None,  # DEFAULT_LORA_TARGETS (qkv/proj/fc1/fc2)
        )
    else:
        if args.base_checkpoint and os.path.exists(args.base_checkpoint):
            loaded_step, _meta = load_checkpoint(
                args.base_checkpoint, student_model, device=str(device)
            )
            ckpt_phase0 = _meta.get("phase0")
            ckpt_phase_tag = _meta.get("phase_tag")
            ok(f"Student base loaded from: {args.base_checkpoint}")
        else:
            warning(
                f"Base checkpoint not found: {args.base_checkpoint} — "
                "starting from random init"
            )
        attach_lora(
            student_model,
            r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
            target_names=None,  # DEFAULT_LORA_TARGETS
        )

    # attach_lora creates adapters on CPU; move the whole model to the device
    # AFTER attaching so lora_A/lora_B share the base weights' device.
    student_model.to(device)

    # ── Memory: cast base weights to the AMP dtype (saves ~1.1GB static) ──
    if use_amp:
        student_model.to(amp_dtype)
        info(f"Student cast to {amp_dtype} (fp32 base weights released)")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Gradient checkpointing (mirrors distill.py:810-812) — trades compute
    # for ~40-60% activation VRAM; the single biggest OOM fix. ─────────────
    if not args.no_grad_checkpointing:
        _enable_gradient_checkpointing(student_model)

    # ── Projector ─────────────────────────────────────────────────────────
    # The shared cross_vocab_distill_loss calls `projector(teacher_hidden)`
    # unconditionally, so every arm needs a VocabProjector instance. Arm C
    # trains it (added to the optimizer); arms A/B freeze a dummy instance so
    # the CE-only path still routes through the shared loss with identical
    # metrics keys (kd_weight=0 + empty pairs ⇒ it reduces to masked CE).
    projector = VocabProjector(teacher_dim=896, student_dim=mc.d_model).to(device)
    if use_amp:
        projector.to(amp_dtype)
    if mode == "align":
        n_proj = sum(p.numel() for p in projector.parameters())
        ok(f"VocabProjector (896 → {mc.d_model}): {n_proj:,} trainable params")
    else:
        for p in projector.parameters():
            p.requires_grad = False
        dim("VocabProjector frozen (dummy) — CE-only arms route through the shared loss")
    lm_head_weight = student_model.lm_head.weight  # tied to tok_emb

    # ── Optimizer (trainable student params + projector params) ───────────
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    trainable_params = trainable_params + [
        p for p in projector.parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(
        [
            {"params": [p for p in trainable_params if p.ndim >= 2],
             "weight_decay": 0.05},
            {"params": [p for p in trainable_params if p.ndim < 2],
             "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # ── Resume: LoRA + optimizer (+ projector) restored together ─────────
    if is_resume:
        step, _meta = load_checkpoint(
            args.resume, student_model,
            optimizer=optimizer, device=str(device),
        )
        ckpt_phase0 = _meta.get("phase0")
        ckpt_phase_tag = _meta.get("phase_tag")
        # load_checkpoint surfaces only phase0/phase_tag/base_checkpoint in
        # meta; pull the projector state straight from the checkpoint file.
        if mode == "align" and projector is not None:
            try:
                ckpt = torch.load(args.resume, map_location=str(device), weights_only=False)
                proj_state = ckpt.get("projector_state")
                if proj_state is not None:
                    projector.load_state_dict(proj_state)
                    ok("Projector state restored from checkpoint")
                else:
                    warning("Checkpoint has no 'projector_state' — projector re-initialised")
            except Exception as e:
                warning(f"Could not restore projector state ({e})")
        ok(f"Resumed from: {args.resume} (step {step})")
    _resume_step = step

    # ── LR schedule anchor (mirrors distill.py phase anchoring) ───────────
    phase_budget = args.max_steps
    phase_tag = make_schedule_tag(phase_budget, args.lr, args.lr * 0.1, args.warmup_steps)
    if ckpt_phase0 is not None and ckpt_phase_tag == phase_tag:
        phase0 = ckpt_phase0
    else:
        phase0 = step
    total_steps = phase0 + phase_budget

    ok(f"Student: DizelLM({mc.d_model}, {mc.n_layers} layers) — "
       f"{student_model.num_parameters() / 1e6:.1f}M params")
    info(f"Teacher: {args.teacher_model} (mode={mode})")

    # ── Hidden-state capture hook (post-final-norm hidden feeds lm_head) ──
    hidden_holder: dict = {}

    def _capture_ln_f(module, inp, out):
        hidden_holder["hidden"] = out

    hook = student_model.transformer["ln_f"].register_forward_hook(_capture_ln_f)

    # ── Alignment cache (arm C only) ──────────────────────────────────────
    alignment_cache = None
    if mode == "align":
        alignment_cache = AlignmentCache(cache_dir=args.cache_dir)

    # ── Metrics logger ────────────────────────────────────────────────────
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = KDMetricsLogger(log_dir=log_dir, run_name=args.run_tag)
    effective_kd = args.kd_weight if mode == "align" else 0.0
    logger.header(
        mode, effective_kd, args.temperature,
        extra=f"run_tag={args.run_tag}",
    )

    section("Training")
    config_table({
        "Arm":          mode,
        "run tag":      args.run_tag,
        "kd weight":    effective_kd,
        "temperature":  args.temperature,
        "teacher":      args.teacher_model,
        "LoRA rank":    args.lora_r,
        "LoRA alpha":   args.lora_alpha,
    })
    config_table({
        "total steps":  phase_budget,
        "target steps": total_steps,
        "batch size":   args.batch_size,
        "grad accum":   args.grad_accum,
        "learning rate": f"{args.lr:.2e}",
        "warmup steps": args.warmup_steps,
    })

    meter = StepMeter(
        total_steps,
        tokens_per_step=args.batch_size * args.grad_accum * mc.context_length,
        start_step=step,
    )

    # ── Training loop (mirrors distill.py 889-1015) ──────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    best_val = float("inf")
    patience = 5
    min_delta = 5e-4
    patience_counter = 0
    early_stop = False

    data_iter = iter(train_loader)
    micro_step = 0
    accum_loss = 0.0
    last_metrics = None

    student_model.train()
    optimizer.zero_grad()

    if mode == "align":
        reset_kd_guard(run_id=args.run_tag)

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
            logits, _ = student_model(input_ids, None)
            student_hidden = hidden_holder["hidden"]

            if mode == "align":
                teacher_hidden = _run_teacher_hidden(
                    teacher_model, teacher_tokenizer, texts, device, amp_ctx,
                    max_length=input_ids.shape[1] + 32,
                )
                pairs_list, _fail = _build_aligned_pairs(
                    texts, tokenizer, teacher_tokenizer, alignment_cache
                )
                metrics = cross_vocab_distill_loss(
                    student_logits=logits,
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    projector=projector,
                    lm_head_weight=lm_head_weight,
                    labels=labels,
                    loss_mask=loss_mask,
                    aligned_pairs_list=pairs_list,
                    kl_weight=args.kd_weight,
                    temperature=args.temperature,
                    batch_ids=None,
                )
            else:
                # CE-only path (arm A and arm B share it): KD disabled. The
                # shared loss requires real tensors, so pass a dummy teacher
                # hidden + the frozen projector; kl_weight=0 + empty pairs
                # makes this exactly the masked CE loss.
                dummy_hidden = torch.zeros(
                    (logits.shape[0], 1, 896),
                    device=logits.device, dtype=logits.dtype,
                )
                metrics = cross_vocab_distill_loss(
                    student_logits=logits,
                    student_hidden=student_hidden,
                    teacher_hidden=dummy_hidden,
                    projector=projector,
                    lm_head_weight=lm_head_weight,
                    labels=labels,
                    loss_mask=loss_mask,
                    aligned_pairs_list=[[] for _ in range(len(texts))],
                    kl_weight=0.0,
                    temperature=args.temperature,
                    batch_ids=None,
                )
            loss = metrics["loss"] / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        micro_step += 1
        last_metrics = metrics

        if micro_step % args.grad_accum == 0:
            # LR schedule — anchored to phase start so a mid-phase resume
            # continues decaying instead of re-entering warmup.
            local = step - phase0
            lr = cosine_lr_with_warmup(
                local, args.warmup_steps, phase_budget, args.lr, args.lr * 0.1
            )
            set_lr(optimizer, lr)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            train_loss = accum_loss
            accum_loss = 0.0
            step += 1

            # ── KD guard: raise if KD is silently off for 5 consecutive steps
            if mode == "align":
                assert_kd_not_silently_disabled(
                    metrics=last_metrics,
                    kl_weight=args.kd_weight,
                    mode=mode,
                    step=step,
                    run_id=args.run_tag,
                )

            # ── Uniform step logging through the shared loss ─────────────
            if last_metrics is not None:
                logger.log_step(
                    step,
                    last_metrics["ce_loss"],
                    last_metrics["kd_loss"],
                    last_metrics["loss"].item(),
                    last_metrics["active"],
                    last_metrics["aligned_count"],
                    effective_kd,
                    args.temperature,
                    lr,
                )

            if step % 25 == 0 or step == _resume_step + 1:
                meter.update(
                    step=step, loss=train_loss, lr=lr,
                    extra=f"KD={effective_kd:.1f} T={args.temperature:.1f}",
                )

            # ── Evaluation ────────────────────────────────────────────────
            if step % args.eval_interval == 0:
                val_metrics = xkd_evaluate(
                    student_model, teacher_model, teacher_tokenizer, val_loader,
                    args.eval_iters, device, amp_ctx,
                    tokenizer, projector, lm_head_weight, alignment_cache,
                    hidden_holder, mode, args.kd_weight, args.temperature,
                    cross_vocab_distill_loss, _run_teacher_hidden, _build_aligned_pairs,
                )
                logger.log_eval(step, val_metrics, mode)
                validation(
                    step, val_metrics["loss"],
                    extra=f"ce {val_metrics['ce_loss']:.4f} "
                          f"kd {val_metrics['kd_loss']:.4f} "
                          f"active {val_metrics['active']:.2f}",
                )
                torch.cuda.empty_cache()

                if val_metrics["loss"] < (best_val - min_delta):
                    best_val = val_metrics["loss"]
                    patience_counter = 0
                    # Save UNMERGED (LoRA separate); leave the model unmerged
                    # so training continues through the LoRA path.
                    unmerge_lora_weights(student_model)
                    save_checkpoint(
                        student_model, optimizer, step, best_val, V15_CONFIG,
                        os.path.join(args.out_dir, f"{args.run_tag}-best.pt"),
                        tokenizer=tokenizer,
                        phase0=phase0,
                        phase_tag=phase_tag,
                        extra_meta={
                            "kd_mode": mode,
                            "kd_weight": args.kd_weight,
                            "lora_alpha": args.lora_alpha,
                            "lora_r": args.lora_r,
                            "projector_state": projector.state_dict()
                            if mode == "align" else None,
                        },
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

            # ── Periodic checkpoint ──────────────────────────────────────
            if not args.best_only and step % args.save_interval == 0:
                unmerge_lora_weights(student_model)
                save_checkpoint(
                    student_model, optimizer, step, float("inf"), V15_CONFIG,
                    os.path.join(args.out_dir, f"{args.run_tag}-step{step}.pt"),
                    tokenizer=tokenizer,
                    phase0=phase0,
                    phase_tag=phase_tag,
                    extra_meta={
                        "kd_mode": mode,
                        "kd_weight": args.kd_weight,
                        "lora_alpha": args.lora_alpha,
                        "lora_r": args.lora_r,
                        "projector_state": projector.state_dict()
                        if mode == "align" else None,
                    },
                )

    # ── Final save + cleanup ──────────────────────────────────────────────
    unmerge_lora_weights(student_model)
    save_checkpoint(
        student_model, optimizer, step, best_val, V15_CONFIG,
        os.path.join(args.out_dir, f"{args.run_tag}-final.pt"),
        tokenizer=tokenizer,
        phase0=phase0,
        phase_tag=phase_tag,
        extra_meta={
            "kd_mode": mode,
            "kd_weight": args.kd_weight,
            "lora_alpha": args.lora_alpha,
            "lora_r": args.lora_r,
            "projector_state": projector.state_dict()
            if mode == "align" else None,
        },
    )
    merge_lora_weights(student_model)
    hook.remove()
    logger.close()
    meter.finalise()

    if last_metrics is not None:
        config_table({
            "Final step":  f"{step:,}",
            "Best val":    f"{best_val:.4f}",
            "Final CE":    f"{last_metrics['ce_loss']:.4f}",
            "Final KD":    f"{last_metrics['kd_loss']:.4f}",
            "Total loss":  f"{last_metrics['loss'].item():.4f}",
            "KD active":   str(last_metrics["active"]),
        })
    rule("Complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dizel v1.5 cross-vocab KD runner (Arms A=ce, B=qwen-sft, C=align)"
    )
    p.add_argument("--kd-mode", type=str, default="align",
                   choices=["ce", "qwen-sft", "align"],
                   help="Distillation arm: ce | qwen-sft | align (default: align)")
    p.add_argument("--kd-weight", type=float, default=0.5,
                   help="KD loss weight (arm C only; default: 0.5)")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="KD temperature (default: 2.0)")
    p.add_argument("--data-path", type=str, default="data/formatted/v15_distill.jsonl",
                   help="Distillation JSONL path")
    p.add_argument("--base-checkpoint", type=str,
                   default="checkpoints/dizel-v15-sft-best.pt",
                   help="Plain (non-LoRA) student base checkpoint")
    p.add_argument("--resume", type=str, default="",
                   help="Resume from an xkd checkpoint (.pt; LoRA + optimizer + projector)")
    p.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace teacher model name")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace token for gated teacher models")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=30)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--out-dir", type=str, default="checkpoints/experiments",
                   help="Checkpoint output directory")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Run tag / checkpoint basename "
                        "(default derived from --kd-mode: xkd-ce/xkd-qwensft/xkd-align)")
    p.add_argument("--cache-dir", type=str, default=".cache/teacher",
                   help="Alignment cache directory (arm C)")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable AMP even on CUDA")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="Disable gradient checkpointing (uses more VRAM, less compute)")
    p.add_argument("--use-amp", action="store_true",
                   help="Enable AMP (overrides config)")
    p.add_argument("--device", type=str, default="",
                   help="Device override (auto-detect cuda/cpu if empty)")
    p.add_argument("--best-only", action="store_true",
                   help="Save only the best checkpoint (skip periodic step checkpoints)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.run_tag is None:
        args.run_tag = RUN_TAGS[args.kd_mode]

    hf_token = resolve_hf_token(args.hf_token)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
    args.hf_token = hf_token

    xkd_train(args)


if __name__ == "__main__":
    main()
