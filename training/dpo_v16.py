"""
training/dpo_v16.py — Direct Preference Optimization for Dizel 1.6 Flash.

The policy starts from the v1.6 SFT checkpoint and is compared against a frozen
copy of that checkpoint.  Prompts use Qwen ChatML and only the chosen/rejected
assistant completion tokens contribute log-probabilities.

The reference checkpoint is never inferred from a filename.  For a fresh run it
defaults to the base/SFT checkpoint; with ``--resume`` it comes from the
checkpoint metadata, or from an explicit ``--reference_checkpoint``.  A missing
reference path fails immediately.

Usage:
    python training/dpo_v16.py \
        --base_checkpoint checkpoints/dizel-1.6-preview-m1-sft-best.pt \
        --data data/v16/posttrain/dpo.jsonl \
        --tokenizer_model tokenizer/qwen2.5 --local_files_only
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.v16_flash import DIZEL_16_FLASH_CONFIG
from model.architecture import DizelLM
from training.display_utils import (
    StepMeter,
    config_table,
    info,
    ok,
    rule,
    section,
    validation,
    warning,
)
from training.posttrain_checkpoint_v16 import (
    load_posttrain_checkpoint,
    make_schedule_tag,
)
from training.posttrain_common_v16 import (
    ResumableBatchStream,
    build_optimizer,
    load_jsonl_records,
    make_amp_context,
    save_and_sync,
    split_records,
)
from training.posttrain_data_v16 import (
    DPODataset,
    compute_log_probs,
    dpo_collate_fn,
)
from training.pretrain import cosine_lr_with_warmup, set_lr
from training.tokenizer_qwen import QwenTokenizer, verify_tokenizer

torch.serialization.add_safe_globals([type(DIZEL_16_FLASH_CONFIG.model)])


def parse_args():
    parser = argparse.ArgumentParser(description="Dizel 1.6 DPO")
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--drive_checkpoint_dir", default="")

    parser.add_argument("--tokenizer_model", default="tokenizer/qwen2.5")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default=None)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg, args):
    dc = cfg.dpo
    for name in (
        "max_steps",
        "warmup_steps",
        "batch_size",
        "grad_accum",
        "lr",
        "min_lr",
        "weight_decay",
        "seed",
        "val_fraction",
        "eval_interval",
        "eval_iters",
        "save_interval",
        "log_interval",
        "max_length",
        "max_prompt_length",
        "beta",
        "label_smoothing",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(dc, name, value)
    if args.data:
        dc.data_path = args.data
    if args.output_dir:
        dc.output_dir = args.output_dir
    if args.run_name:
        dc.run_name = args.run_name
    if args.drive_checkpoint_dir:
        dc.drive_checkpoint_dir = args.drive_checkpoint_dir
    if args.base_checkpoint:
        dc.base_checkpoint = args.base_checkpoint
    if args.reference_checkpoint:
        dc.reference_checkpoint = args.reference_checkpoint
    if args.no_amp:
        dc.use_amp = False
    if args.amp_dtype:
        dc.amp_dtype = args.amp_dtype
    if args.no_gradient_checkpointing:
        dc.gradient_checkpointing = False
    return dc


def dpo_loss(
    policy_chosen_logp,
    policy_rejected_logp,
    ref_chosen_logp,
    ref_rejected_logp,
    beta,
    label_smoothing,
):
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    logits = float(beta) * (policy_margin - ref_margin)

    # Standard DPO loss with optional conservative label smoothing.  With
    # smoothing, the negative-class target is the same sigmoid term.
    loss = -(
        (1.0 - float(label_smoothing)) * F.logsigmoid(logits)
        + float(label_smoothing) * F.logsigmoid(-logits)
    ).mean()
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
        margin = (policy_margin - ref_margin).mean()
    return loss, accuracy, margin


def _repeat_no_grad(fn, *args):
    with torch.no_grad():
        return fn(*args)


@torch.no_grad()
def evaluate(policy, reference, loader, eval_iters, device, amp_ctx, dc):
    policy.eval()
    total_loss = total_acc = total_margin = 0.0
    count = 0
    for batch in loader:
        if count >= eval_iters:
            break
        c_x, c_y, c_m, r_x, r_y, r_m = (t.to(device) for t in batch)
        with amp_ctx:
            p_c = compute_log_probs(policy, c_x, c_y, c_m)
            p_r = compute_log_probs(policy, r_x, r_y, r_m)
        ref_c = _repeat_no_grad(compute_log_probs, reference, c_x, c_y, c_m)
        ref_r = _repeat_no_grad(compute_log_probs, reference, r_x, r_y, r_m)
        loss, acc, margin = dpo_loss(
            p_c, p_r, ref_c, ref_r, dc.beta, dc.label_smoothing
        )
        total_loss += loss.item()
        total_acc += acc.item()
        total_margin += margin.item()
        count += 1
    policy.train()
    return {
        "loss": total_loss / max(1, count),
        "accuracy": total_acc / max(1, count),
        "margin": total_margin / max(1, count),
    }


def main():
    args = parse_args()
    cfg = DIZEL_16_FLASH_CONFIG
    dc = apply_overrides(cfg, args)
    mc = cfg.model

    rule("Dizel 1.6 Flash Preview - DPO (experimental)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device : {device}")

    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {args.resume}. Refusing to start a "
            "fresh DPO run from an explicit --resume path."
        )
    if not args.resume and not dc.base_checkpoint:
        raise ValueError("Provide --base_checkpoint (SFT) for a fresh DPO run or --resume")
    if not args.resume and not os.path.exists(dc.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {dc.base_checkpoint}")
    if not os.path.exists(dc.data_path):
        raise FileNotFoundError(
            f"DPO data not found: {dc.data_path}. Run "
            "`python pipelines/prepare_v16_posttrain.py` first."
        )

    use_amp, amp_dtype, amp_ctx, scaler = make_amp_context(dc, device)
    info(f"AMP    : {'ON (' + str(amp_dtype).replace('torch.', '') + ')' if use_amp else 'OFF'}")

    tokenizer = QwenTokenizer(
        model_name=args.tokenizer_model,
        local_files_only=args.local_files_only,
    )
    if len(tokenizer) > mc.vocab_size:
        raise ValueError(
            f"Tokenizer emits {len(tokenizer):,} ids but model has "
            f"{mc.vocab_size:,} embedding rows"
        )
    verify_tokenizer(tokenizer)

    records = load_jsonl_records(dc.data_path)
    train_records, val_records = split_records(records, dc.val_fraction, dc.seed)
    section("Data")
    info(f"DPO pairs: {len(records):,} ({len(train_records):,} train / {len(val_records):,} val)")
    train_ds = DPODataset(
        train_records, tokenizer, dc.max_length, dc.max_prompt_length
    )
    val_ds = DPODataset(
        val_records, tokenizer, dc.max_length, dc.max_prompt_length
    )
    if not train_ds or not val_ds:
        raise ValueError("DPO dataset is empty after tokenization/validation")
    ok(f"Tokenized pairs: {len(train_ds):,} train + {len(val_ds):,} val")

    policy = DizelLM(mc).to(device)
    if dc.gradient_checkpointing:
        policy.enable_gradient_checkpointing()
        ok("Gradient checkpointing enabled")
    optimizer = build_optimizer(policy, dc)

    reference_path = dc.reference_checkpoint
    start_step = 0
    best_val = float("inf")
    phase0 = 0
    phase_tag = None
    dataset_state = None
    base_checkpoint = dc.base_checkpoint

    if args.resume:
        state = load_posttrain_checkpoint(
            args.resume,
            policy,
            optimizer=optimizer,
            scaler=scaler,
            tokenizer=tokenizer,
            device=str(device),
        )
        start_step = int(state["step"])
        dataset_state = state.get("dataset_state")
        phase0 = int(state.get("phase0") or 0)
        phase_tag = state.get("phase_tag")
        base_checkpoint = state.get("base_checkpoint") or base_checkpoint
        reference_path = (
            dc.reference_checkpoint
            or state.get("reference_checkpoint")
            or reference_path
        )
        if state.get("best_val") is not None:
            best_val = float(state["best_val"])
        if state.get("val_loss") is not None:
            best_val = min(best_val, float(state["val_loss"]))
        ok(f"Resumed DPO at step {start_step:,}")
    else:
        load_posttrain_checkpoint(
            dc.base_checkpoint,
            policy,
            tokenizer=tokenizer,
            device=str(device),
        )
        if not reference_path:
            reference_path = dc.base_checkpoint
        ok(f"Loaded v1.6 SFT policy: {os.path.basename(dc.base_checkpoint)}")

    if not reference_path:
        raise FileNotFoundError(
            "DPO reference checkpoint is unknown. Pass --reference_checkpoint; "
            "a resume checkpoint must record its SFT reference."
        )
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"DPO reference checkpoint not found: {reference_path}")

    reference = DizelLM(mc).to(device)
    load_posttrain_checkpoint(
        reference_path,
        reference,
        tokenizer=tokenizer,
        device=str(device),
        restore_rng=False,
    )
    reference.eval()
    reference.requires_grad_(False)
    ok(f"Frozen DPO reference: {os.path.basename(reference_path)}")

    expected_tag = make_schedule_tag(
        "dpo", dc.max_steps, dc.lr, dc.min_lr, dc.warmup_steps
    )
    if phase_tag != expected_tag:
        phase_tag = expected_tag
        phase0 = start_step if args.resume else 0
    target_steps = phase0 + dc.max_steps
    info(f"Steps  : {start_step:,} -> {target_steps:,} ({dc.max_steps:,} this phase)")

    section("Training")
    os.makedirs(dc.output_dir, exist_ok=True)
    stream = ResumableBatchStream(
        train_ds,
        batch_size=dc.batch_size,
        collate_fn=lambda batch: dpo_collate_fn(batch, pad_id=tokenizer.pad_id),
        seed=dc.seed,
    )
    stream.load_state(dataset_state)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=dc.batch_size,
        shuffle=False,
        collate_fn=lambda batch: dpo_collate_fn(batch, pad_id=tokenizer.pad_id),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    config_table(
        {
            "batch size": dc.batch_size,
            "grad accum": dc.grad_accum,
            "context": dc.max_length,
            "prompt cap": dc.max_prompt_length,
            "learning rate": f"{dc.lr:.2e} -> {dc.min_lr:.2e}",
            "beta": dc.beta,
            "label smoothing": dc.label_smoothing,
        }
    )

    meter = StepMeter(
        target_steps,
        tokens_per_step=dc.batch_size * dc.grad_accum * dc.max_length,
        start_step=start_step,
    )
    step = start_step
    micro = 0
    accum_loss = 0.0
    last_lr = dc.lr
    policy.train()
    optimizer.zero_grad()

    while step < target_steps:
        c_x, c_y, c_m, r_x, r_y, r_m = (
            t.to(device) for t in stream.next_batch()
        )
        with amp_ctx:
            p_c = compute_log_probs(policy, c_x, c_y, c_m)
            p_r = compute_log_probs(policy, r_x, r_y, r_m)
        ref_c = _repeat_no_grad(compute_log_probs, reference, c_x, c_y, c_m)
        ref_r = _repeat_no_grad(compute_log_probs, reference, r_x, r_y, r_m)
        loss, _, _ = dpo_loss(
            p_c, p_r, ref_c, ref_r, dc.beta, dc.label_smoothing
        )
        loss = loss / dc.grad_accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += loss.item()
        micro += 1
        if micro % dc.grad_accum != 0:
            continue

        local = step - phase0
        last_lr = cosine_lr_with_warmup(
            local, dc.warmup_steps, dc.max_steps, dc.lr, dc.min_lr
        )
        set_lr(optimizer, last_lr)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), dc.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        train_loss = accum_loss
        accum_loss = 0.0
        step += 1
        if step % dc.log_interval == 0:
            meter.update(step=step, loss=train_loss, lr=last_lr)

        if step % dc.eval_interval == 0:
            metrics = evaluate(
                policy, reference, val_loader, dc.eval_iters, device, amp_ctx, dc
            )
            validation(
                step,
                metrics["loss"],
                extra=f"acc {metrics['accuracy']:.3f} margin {metrics['margin']:+.3f}",
            )
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                save_and_sync(
                    os.path.join(dc.output_dir, f"{dc.run_name}-best.pt"),
                    policy, optimizer, scaler, step, "dpo", dc, mc,
                    cfg.tokenizer, tokenizer, metrics["loss"], best_val,
                    stream.state(), phase0, phase_tag,
                    base_checkpoint=base_checkpoint,
                    reference_checkpoint=reference_path,
                    drive_dir=dc.drive_checkpoint_dir,
                )
            torch.cuda.empty_cache()

        if step % dc.save_interval == 0:
            save_and_sync(
                os.path.join(dc.output_dir, f"{dc.run_name}-last.pt"),
                policy, optimizer, scaler, step, "dpo", dc, mc,
                cfg.tokenizer, tokenizer, None, best_val,
                stream.state(), phase0, phase_tag,
                base_checkpoint=base_checkpoint,
                reference_checkpoint=reference_path,
                drive_dir=dc.drive_checkpoint_dir,
            )

    metrics = evaluate(
        policy, reference, val_loader, dc.eval_iters, device, amp_ctx, dc
    )
    best_val = min(best_val, metrics["loss"])
    save_and_sync(
        os.path.join(dc.output_dir, f"{dc.run_name}-final.pt"),
        policy, optimizer, scaler, step, "dpo", dc, mc,
        cfg.tokenizer, tokenizer, metrics["loss"], best_val,
        stream.state(), phase0, phase_tag,
        base_checkpoint=base_checkpoint,
        reference_checkpoint=reference_path,
        drive_dir=dc.drive_checkpoint_dir,
    )
    meter.finalise()
    warning(
        f"Final DPO loss {metrics['loss']:.4f} | "
        f"accuracy {metrics['accuracy']:.3f} | margin {metrics['margin']:+.3f}"
    )


if __name__ == "__main__":
    main()
