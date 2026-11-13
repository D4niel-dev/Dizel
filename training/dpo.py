"""
training/dpo.py — Direct Preference Optimization (DPO) for Dizel.

DPO aligns the model to human preferences without requiring a separate
reward model. Given preference pairs (chosen, rejected), the loss
directly optimises the policy to prefer the chosen completion.

Loss (Rafailov et al., 2023):
    L_DPO = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

where:
    π     = policy model (trained)
    π_ref = reference model (frozen SFT checkpoint)
    y_w   = chosen (preferred) completion
    y_l   = rejected completion
    β     = KL penalty coefficient

Data Format (JSONL)
-------------------
Each line:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }

The prompt is formatted with <|user|> ... <|end|> tags.
The chosen/rejected are formatted as <|assistant|> ... <|end|> </s>.

Usage
-----
    python training/dpo.py --base_checkpoint checkpoints/dizel-sft-best.pt

    # Resume from a DPO checkpoint
    python training/dpo.py --resume checkpoints/dizel-dpo-step500.pt
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import V15_CONFIG, ModelConfigV15, ModelConfig, DizelConfig
from model.architecture import DizelLM
from training.dataset import Tokenizer
from training.pretrain import (
    cosine_lr_with_warmup, set_lr, save_checkpoint, load_checkpoint,
    save_tokenizer, make_schedule_tag,
)
from training.display_utils import (
    rule, section, info, ok, warning, error, dim,
    config_table, StepMeter, validation,
)

torch.serialization.add_safe_globals([ModelConfig, ModelConfigV15])

# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------
ROLE_TOKENS = {
    "system":    "<|system|>",
    "user":      "<|user|>",
    "assistant": "<|assistant|>",
}
END_TOKEN = "<|end|>"


def _tokenize_prompt_completion(
    prompt: str,
    completion: str,
    tokenizer: Tokenizer,
    max_prompt_length: int,
    max_length: int,
):
    """
    Tokenize a (prompt, completion) pair into token ids and a loss mask.

    Returns (input_ids, labels, loss_mask) as numpy arrays.
    The prompt portion is masked (loss_mask=0), the completion is unmasked (loss_mask=1).
    """
    # Format as chat turns
    prompt_text = f"{ROLE_TOKENS['user']} {prompt} {END_TOKEN}"
    completion_text = f"{ROLE_TOKENS['assistant']} {completion} {END_TOKEN}"

    prompt_ids = tokenizer.encode(prompt_text)
    completion_ids = tokenizer.encode(completion_text) + [tokenizer.eos_id]

    # Truncate prompt if too long
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[:max_prompt_length]

    # Combine and truncate total
    all_ids = prompt_ids + completion_ids
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]

    # Build mask: 0 for prompt tokens, 1 for completion tokens
    prompt_len = len(prompt_ids)
    total_len = len(all_ids)
    loss_mask = [0] * min(prompt_len, total_len) + [1] * max(0, total_len - prompt_len)

    # input = all_ids[:-1], labels = all_ids[1:], mask = loss_mask[1:]
    input_ids = all_ids[:-1]
    labels = all_ids[1:]
    loss_mask = loss_mask[1:]

    return (
        np.array(input_ids, dtype=np.int32),
        np.array(labels, dtype=np.int32),
        np.array(loss_mask, dtype=np.bool_),
    )


class DPODataset(Dataset):
    """
    Loads preference pairs from a JSONL file.

    Each sample yields:
        (chosen_ids, chosen_labels, chosen_mask,
         rejected_ids, rejected_labels, rejected_mask)
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.pairs = []

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

                prompt = obj.get("instruction", obj.get("prompt", ""))
                chosen = obj.get("chosen", "")
                rejected = obj.get("rejected", "")

                if not prompt or not chosen or not rejected:
                    skipped += 1
                    continue

                # Skip if chosen and rejected are identical
                if chosen.strip() == rejected.strip():
                    skipped += 1
                    continue

                c_ids, c_labels, c_mask = _tokenize_prompt_completion(
                    prompt, chosen, tokenizer, max_prompt_length, max_length
                )
                r_ids, r_labels, r_mask = _tokenize_prompt_completion(
                    prompt, rejected, tokenizer, max_prompt_length, max_length
                )

                if len(c_ids) < 4 or len(r_ids) < 4:
                    skipped += 1
                    continue

                self.pairs.append((c_ids, c_labels, c_mask, r_ids, r_labels, r_mask))

        if skipped:
            print(f"  [dpo] Skipped {skipped} invalid/duplicate pairs")
        ok(f"DPO pairs loaded: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        c_ids, c_labels, c_mask, r_ids, r_labels, r_mask = self.pairs[idx]
        return (
            torch.tensor(c_ids.astype(np.int64), dtype=torch.long),
            torch.tensor(c_labels.astype(np.int64), dtype=torch.long),
            torch.tensor(c_mask.astype(np.float32), dtype=torch.float),
            torch.tensor(r_ids.astype(np.int64), dtype=torch.long),
            torch.tensor(r_labels.astype(np.int64), dtype=torch.long),
            torch.tensor(r_mask.astype(np.float32), dtype=torch.float),
        )


def dpo_collate_fn(batch, pad_id=0):
    """Pad chosen and rejected sequences to same length within batch."""
    c_ids, c_labels, c_mask, r_ids, r_labels, r_mask = zip(*batch)

    # Find max length across ALL sequences in the batch (both chosen + rejected)
    max_len = max(
        max(x.size(0) for x in c_ids),
        max(x.size(0) for x in r_ids),
    )

    def pad(tensors, value):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.size(0),), value, dtype=t.dtype)])
            for t in tensors
        ])

    return (
        pad(c_ids, pad_id),
        pad(c_labels, -1),
        pad(c_mask, 0),
        pad(r_ids, pad_id),
        pad(r_labels, -1),
        pad(r_mask, 0),
    )


def build_dpo_loaders(
    jsonl_path: str,
    tokenizer: Tokenizer,
    max_length: int,
    max_prompt_length: int,
    batch_size: int,
    val_frac: float = 0.10,
    seed: int = 42,
):
    full = DPODataset(jsonl_path, tokenizer, max_length, max_prompt_length)

    n_val = max(1, int(len(full) * val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    collate = lambda b: dpo_collate_fn(b, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------
def compute_log_probs(
    model: DizelLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample average log probability of the completion tokens.

    Returns tensor of shape (B,) — one log-prob per sample.
    """
    logits, _ = model(input_ids)  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather log probs for actual tokens: labels (B, T) → (B, T)
    # Use clamp to avoid indexing with -1 (padding)
    safe_labels = labels.clamp(min=0)
    per_token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    # Zero out padding and prompt positions
    per_token_log_probs = per_token_log_probs * loss_mask

    # Average per sample (sum of log probs / count of completion tokens)
    token_count = loss_mask.sum(dim=-1).clamp(min=1)
    return per_token_log_probs.sum(dim=-1) / token_count


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    Compute the DPO loss (Rafailov et al., 2023).

    Returns (loss, chosen_reward, rejected_reward) where rewards are
    the implicit reward margins for logging.
    """
    # Log-ratio differences
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)

    if label_smoothing > 0:
        # Conservative DPO: smooth the binary preference labels
        loss = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(logits)

    # Implicit rewards for monitoring
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def dpo_evaluate(
    policy_model: DizelLM,
    ref_model: DizelLM,
    val_loader,
    eval_iters: int,
    device: torch.device,
    amp_ctx,
    beta: float,
    label_smoothing: float = 0.0,
) -> dict:
    """Evaluate DPO metrics on validation set."""
    policy_model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_chosen_r = 0.0
    total_rejected_r = 0.0
    total_ref_c_logp = 0.0
    total_ref_r_logp = 0.0
    total_pol_c_logp = 0.0
    total_pol_r_logp = 0.0
    count = 0

    for batch in val_loader:
        if count >= eval_iters:
            break

        c_ids, c_labels, c_mask, r_ids, r_labels, r_mask = (
            t.to(device) for t in batch
        )

        with amp_ctx:
            # Policy log probs
            policy_c_logps = compute_log_probs(policy_model, c_ids, c_labels, c_mask)
            policy_r_logps = compute_log_probs(policy_model, r_ids, r_labels, r_mask)

            # Reference log probs
            ref_c_logps = compute_log_probs(ref_model, c_ids, c_labels, c_mask)
            ref_r_logps = compute_log_probs(ref_model, r_ids, r_labels, r_mask)

            loss, chosen_r, rejected_r = dpo_loss(
                policy_c_logps, policy_r_logps,
                ref_c_logps, ref_r_logps,
                beta=beta,
                label_smoothing=label_smoothing,
            )

        # Accuracy: how often does the policy prefer chosen over rejected?
        margin = (policy_c_logps - ref_c_logps) - (policy_r_logps - ref_r_logps)
        acc = (margin > 0).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        total_chosen_r += chosen_r.item()
        total_rejected_r += rejected_r.item()
        total_ref_c_logp += ref_c_logps.mean().item()
        total_ref_r_logp += ref_r_logps.mean().item()
        total_pol_c_logp += policy_c_logps.mean().item()
        total_pol_r_logp += policy_r_logps.mean().item()
        count += 1

    policy_model.train()
    n = max(1, count)
    ref_c_mean = total_ref_c_logp / n
    ref_r_mean = total_ref_r_logp / n
    pol_c_mean = total_pol_c_logp / n
    pol_r_mean = total_pol_r_logp / n
    return {
        "loss": total_loss / n,
        "accuracy": total_acc / n,
        "chosen_reward": total_chosen_r / n,
        "rejected_reward": total_rejected_r / n,
        "reward_margin": (total_chosen_r - total_rejected_r) / n,
        "ref_chosen_logprob": ref_c_mean,
        "ref_rejected_logprob": ref_r_mean,
        "policy_chosen_logprob": pol_c_mean,
        "policy_rejected_logprob": pol_r_mean,
        "policy_margin": pol_c_mean - pol_r_mean,
    }


# ---------------------------------------------------------------------------
# Gradient checkpointing helper
# ---------------------------------------------------------------------------
def _enable_gradient_checkpointing(model: DizelLM) -> None:
    """Wrap each TransformerBlock forward with gradient checkpointing to save VRAM."""
    for block in model.transformer["blocks"]:
        orig_forward = block.forward
        block.forward = lambda x, orig=orig_forward: torch.utils.checkpoint.checkpoint(
            orig, x, use_reentrant=False
        )

# NOTE: For 205M models, monitor for distribution collapse during DPO.
#       Collapse indicators: policy_chosen_logprob ≈ policy_rejected_logprob
#       while both are much lower than ref_* counterparts.
#       Fix: increase beta, increase label_smoothing, or reduce max_steps.


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def dpo_train(cfg: DizelConfig, base_ckpt: str) -> None:
    dc = cfg.dpo
    mc = cfg.model

    rule("Direct Preference Optimization")

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

    # ── Data ─────────────────────────────────────────────────────────────
    section("Data")
    if not os.path.exists(dc.dpo_data_path):
        warning(f"DPO data not found at '{dc.dpo_data_path}'")
        warning("Run: python scripts/prepare_dpo_data.py")
        sys.exit(1)

    train_loader, val_loader = build_dpo_loaders(
        jsonl_path=dc.dpo_data_path,
        tokenizer=tokenizer,
        max_length=dc.max_length,
        max_prompt_length=dc.max_prompt_length,
        batch_size=dc.batch_size,
    )

    # ── Policy model (trainable) ─────────────────────────────────────────
    section("Models")
    policy_model = DizelLM(mc).to(device)

    # ── Reference model (frozen copy of SFT weights) ─────────────────────
    ref_model = DizelLM(mc).to(device)

    # ── Load base checkpoint ─────────────────────────────────────────────
    step = 0
    is_resume = base_ckpt and "dpo" in base_ckpt.lower()
    ckpt_phase0 = None
    ckpt_phase_tag = None
    # Reference model must stay the ORIGINAL SFT weights. On a fresh run that
    # is base_ckpt; on a resume it is the path stored when the run started (so
    # the frozen reference never receives the DPO-trained policy weights).
    stored_base = base_ckpt

    param_groups = [
        {"params": [p for p in policy_model.parameters() if p.requires_grad and p.ndim >= 2],
         "weight_decay": dc.weight_decay},
        {"params": [p for p in policy_model.parameters() if p.requires_grad and p.ndim < 2],
         "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=dc.lr, betas=(0.9, 0.95), eps=1e-8)

    if base_ckpt and os.path.exists(base_ckpt):
        loaded_step, _meta = load_checkpoint(
            base_ckpt, policy_model,
            optimizer=optimizer if is_resume else None,
            device=str(device),
        )
        ckpt_phase0    = _meta.get("phase0")
        ckpt_phase_tag = _meta.get("phase_tag")
        if is_resume:
            step = loaded_step
            # For resume, ref model must be the original SFT weights (the
            # starting point of the policy), NOT the DPO-trained policy.
            ref_base = _meta.get("base_checkpoint") or base_ckpt
            if ref_base and os.path.exists(ref_base) and ref_base != base_ckpt:
                load_checkpoint(ref_base, ref_model, device=str(device))
                stored_base = ref_base
                ok(f"Reference model loaded from original base: {os.path.basename(ref_base)}")
            else:
                warning(
                    "Original SFT base checkpoint not found — loading ref from the "
                    "resume checkpoint (policy ≈ ref, DPO signal will be weak)"
                )
                load_checkpoint(base_ckpt, ref_model, device=str(device))
        else:
            # Fresh DPO: both models start from SFT checkpoint
            load_checkpoint(base_ckpt, ref_model, device=str(device))
        ok(f"Policy model loaded from: {os.path.basename(base_ckpt)}")
    else:
        warning("No base checkpoint — DPO from scratch (unusual!)")
    _resume_step = step

    # -- LR schedule anchor ----------------------------------------------
    # phase0 = global step where this phase first began; persisted in
    # checkpoints so a mid-phase resume CONTINUES decaying instead of
    # re-entering warmup. Legacy or cross-phase checkpoints start a fresh
    # schedule anchored at the current step.
    phase_tag = make_schedule_tag(dc.max_steps, dc.lr, dc.min_lr, dc.warmup_steps)
    if ckpt_phase0 is not None and ckpt_phase_tag == phase_tag:
        phase0 = ckpt_phase0
    else:
        phase0 = step
    # `--max_steps` is a PER-PHASE budget counted from the phase's FIRST step
    # (phase0), so the absolute stopping step stays fixed across mid-phase
    # resumes: resuming at step N simply gets closer to the same fixed end.
    target_steps = phase0 + dc.max_steps

    # Freeze reference model completely
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    info(f"Reference model frozen ({ref_model.num_parameters(trainable_only=False)/1e6:.1f}M params)")

    # ── Gradient Checkpointing ──────────────────────────────────────────
    if dc.gradient_checkpointing:
        _enable_gradient_checkpointing(policy_model)
        ok("Gradient checkpointing enabled on policy model")

    total_params = (
        policy_model.num_parameters(trainable_only=False)
        + ref_model.num_parameters(trainable_only=False)
    )
    info(f"Total VRAM: {total_params/1e6:.1f}M (policy + reference)")
    ok(f"Policy model: DizelLM({mc.d_model}, {mc.n_layers} layers)")

    config_table({
        "dimension":    mc.d_model,
        "layers":       mc.n_layers,
        "vocab size":   mc.vocab_size,
        "context":      mc.context_length,
    })
    config_table({
        "total steps":  dc.max_steps,
        "target steps": target_steps,
        "batch size":   dc.batch_size,
        "grad accum":   dc.grad_accum,
        "learning rate": f"{dc.lr:.2e} → {dc.min_lr:.2e}",
        "beta (KL)":    dc.beta,
        "max length":   dc.max_length,
    })

    # ── Training ─────────────────────────────────────────────────────────
    os.makedirs(dc.output_dir, exist_ok=True)

    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, f"{dc.run_name}.csv"), "w")
    log_file.write("step,loss,accuracy,chosen_reward,rejected_reward,margin,"
                   "lr,ref_chosen_logprob,ref_rejected_logprob,"
                   "policy_chosen_logprob,policy_rejected_logprob,policy_margin\n")

    best_val = float("inf")
    best_accuracy = None
    # ── Early stopping with patience ────────────────────────────────────
    patience = 5
    min_delta = 5e-4
    patience_counter = 0
    early_stop = False

    data_iter = iter(train_loader)
    micro_step = 0
    accum_loss = 0.0
    accum_acc = 0.0

    policy_model.train()
    optimizer.zero_grad()

    section("Training")
    meter = StepMeter(
        target_steps,
        tokens_per_step=dc.batch_size * dc.grad_accum * dc.max_length,
        start_step=step,
    )

    while step < target_steps and not early_stop:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        c_ids, c_labels, c_mask, r_ids, r_labels, r_mask = (
            t.to(device) for t in batch
        )

        with amp_ctx:
            # Policy forward passes
            policy_c_logps = compute_log_probs(policy_model, c_ids, c_labels, c_mask)
            policy_r_logps = compute_log_probs(policy_model, r_ids, r_labels, r_mask)

            # Reference forward passes (no grad)
            with torch.no_grad():
                ref_c_logps = compute_log_probs(ref_model, c_ids, c_labels, c_mask)
                ref_r_logps = compute_log_probs(ref_model, r_ids, r_labels, r_mask)

            loss, _, _ = dpo_loss(
                policy_c_logps, policy_r_logps,
                ref_c_logps, ref_r_logps,
                beta=dc.beta,
                label_smoothing=dc.label_smoothing,
            )
            loss = loss / dc.grad_accum

        # Track preference accuracy
        with torch.no_grad():
            margin = (policy_c_logps - ref_c_logps) - (policy_r_logps - ref_r_logps)
            batch_acc = (margin > 0).float().mean().item()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += loss.item()
        accum_acc += batch_acc / dc.grad_accum
        micro_step += 1

        if micro_step % dc.grad_accum == 0:
            # LR schedule — anchored to phase start so a mid-phase resume
            # continues decaying instead of re-entering warmup.
            local = step - phase0
            lr = cosine_lr_with_warmup(
                local, dc.warmup_steps, dc.max_steps, dc.lr, dc.min_lr
            )
            set_lr(optimizer, lr)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), dc.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            train_loss = accum_loss
            train_acc = accum_acc
            accum_loss = 0.0
            accum_acc = 0.0
            step += 1

            if step % 25 == 0 or step == _resume_step + 1:
                meter.update(
                    step=step,
                    loss=train_loss,
                    lr=lr,
                    acc=train_acc,
                    extra=f"β={dc.beta:.1f}"
                )

            if step % dc.eval_interval == 0:
                metrics = dpo_evaluate(
                    policy_model, ref_model, val_loader,
                    dc.eval_iters, device, amp_ctx, dc.beta,
                    label_smoothing=dc.label_smoothing,
                )
                validation(step, metrics["loss"],
                           extra=f"acc {metrics['accuracy']:.2%}")
                dim(f"    loss        {metrics['loss']:.4f}")
                dim(f"    accuracy    {metrics['accuracy']:.2%}")
                dim(f"    reward chos {metrics['chosen_reward']:.4f}")
                dim(f"    reward rej  {metrics['rejected_reward']:.4f}")
                dim(f"    margin      {metrics['reward_margin']:.4f}")
                dim(f"    ref chos    {metrics['ref_chosen_logprob']:.4f}")
                dim(f"    pol chos    {metrics['policy_chosen_logprob']:.4f}")
                log_file.write(
                    f"{step},{metrics['loss']:.4f},{metrics['accuracy']:.4f},"
                    f"{metrics['chosen_reward']:.4f},{metrics['rejected_reward']:.4f},"
                    f"{metrics['reward_margin']:.4f},{lr:.6f},"
                    f"{metrics['ref_chosen_logprob']:.4f},{metrics['ref_rejected_logprob']:.4f},"
                    f"{metrics['policy_chosen_logprob']:.4f},{metrics['policy_rejected_logprob']:.4f},"
                    f"{metrics['policy_margin']:.4f}\n"
                )
                log_file.flush()
                torch.cuda.empty_cache()

                if metrics["loss"] < (best_val - min_delta):
                    best_val = metrics["loss"]
                    best_accuracy = metrics["accuracy"]
                    patience_counter = 0
                    save_checkpoint(
                        policy_model, optimizer, step, best_val, cfg,
                        os.path.join(dc.output_dir, f"{dc.run_name}-best.pt"),
                        tokenizer=tokenizer,
                        phase0=phase0,
                        phase_tag=phase_tag,
                        extra_meta={"base_checkpoint": stored_base},
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        warning(f"Early stopping at step {step} (no improvement for {patience} evals)")
                        early_stop = True
                        break

            if not getattr(cfg, '_best_only', False) and step % dc.save_interval == 0:
                save_checkpoint(
                    policy_model, optimizer, step, float("inf"), cfg,
                    os.path.join(dc.output_dir, f"{dc.run_name}-step{step}.pt"),
                    tokenizer=tokenizer,
                    phase0=phase0,
                    phase_tag=phase_tag,
                    extra_meta={"base_checkpoint": stored_base},
                )

    # Final save
    save_checkpoint(
        policy_model, optimizer, step, best_val, cfg,
        os.path.join(dc.output_dir, f"{dc.run_name}-final.pt"),
        tokenizer=tokenizer,
        phase0=phase0,
        phase_tag=phase_tag,
        extra_meta={"base_checkpoint": stored_base},
    )
    log_file.close()
    meter.finalise()
    config_table({
        "Best val loss":    f"{best_val:.4f}",
        "Total steps":      f"{step:,}",
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Dizel DPO Training")
    p.add_argument("--base_checkpoint", type=str, default="",
                   help="Path to SFT checkpoint (.pt). Required unless resuming.")
    p.add_argument("--resume", type=str, default="",
                   help="Path to a DPO checkpoint (.pt) to resume from.")
    p.add_argument("--dpo_data", type=str, default=None,
                   help="Path to DPO JSONL data file")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--beta", type=float, default=None,
                   help="DPO beta parameter (KL penalty strength)")
    p.add_argument("--label_smoothing", type=float, default=None,
                   help="Label smoothing for conservative DPO")
    p.add_argument("--max_length", type=int, default=None,
                   help="Max sequence length for DPO pairs")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP even on CUDA")
    p.add_argument("--use_amp", action="store_true",
                   help="Enable AMP (overrides config)")
    p.add_argument("--amp_dtype", type=str, default=None,
                   choices=["float16", "bfloat16"])
    p.add_argument("--best_only", action="store_true",
                   help="Save only the best checkpoint (skip periodic step checkpoints)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cfg = V15_CONFIG

    if not args.base_checkpoint and not args.resume:
        error("Must provide either --base_checkpoint or --resume")
        sys.exit(1)

    # Apply CLI overrides
    if args.dpo_data is not None:       cfg.dpo.dpo_data_path = args.dpo_data
    if args.lr is not None:             cfg.dpo.lr = args.lr
    if args.max_steps is not None:      cfg.dpo.max_steps = args.max_steps
    if args.batch_size is not None:     cfg.dpo.batch_size = args.batch_size
    if args.grad_accum is not None:     cfg.dpo.grad_accum = args.grad_accum
    if args.beta is not None:           cfg.dpo.beta = args.beta
    if args.label_smoothing is not None: cfg.dpo.label_smoothing = args.label_smoothing
    if args.max_length is not None:     cfg.dpo.max_length = args.max_length
    if args.no_amp:                     cfg.dpo.use_amp = False
    if args.use_amp:                    cfg.dpo.use_amp = True
    if args.amp_dtype is not None:      cfg.dpo.amp_dtype = args.amp_dtype
    cfg._best_only = args.best_only

    target_ckpt = args.resume if args.resume else args.base_checkpoint
    dpo_train(cfg, target_ckpt)


if __name__ == "__main__":
    main()
