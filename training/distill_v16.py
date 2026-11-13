"""
training/distill_v16.py — Token-level teacher distillation for Dizel 1.6 Flash.

Unlike the v1.5 LoRA path, v1.6 uses full-parameter distillation.  Student and
teacher share Qwen2.5's 151,936-row vocabulary, so no projector or token
alignment layer is needed: the teacher consumes the exact student input ids and
we align logits position by position.

Loss:
    L = (1 - kl_weight) * CE(response) + kl_weight * KL(teacher || student)

KL is evaluated only on assistant/content tokens and is chunked over rows so a
T4 does not materialise multiple full-vocabulary softmax tensors at once.

Usage:
    python training/distill_v16.py \
        --base_checkpoint checkpoints/dizel-1.6-preview-m1-sft-best.pt \
        --data data/v16/posttrain/distill.jsonl \
        --teacher_model Qwen/Qwen2.5-0.5B-Instruct \
        --tokenizer_model tokenizer/qwen2.5 --local_files_only

Resume:
    python training/distill_v16.py --resume checkpoints/dizel-1.6-preview-m1-distill-last.pt
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
    load_hf_token_from_dotenv,
    load_jsonl_records,
    make_amp_context,
    save_and_sync,
    split_records,
)
from training.posttrain_data_v16 import ChatSFTDataset, distill_collate_fn
from training.pretrain import cosine_lr_with_warmup, set_lr
from training.tokenizer_qwen import QwenTokenizer, verify_tokenizer

torch.serialization.add_safe_globals([type(DIZEL_16_FLASH_CONFIG.model)])


def parse_args():
    parser = argparse.ArgumentParser(description="Dizel 1.6 token-level KD")
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--drive_checkpoint_dir", default="")

    parser.add_argument("--teacher_model", default=None)
    parser.add_argument("--teacher_local_files_only", action="store_true")
    parser.add_argument("--tokenizer_model", default="tokenizer/qwen2.5")
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
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
    dc = cfg.distill
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
        "kl_weight",
        "temperature",
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
    if args.teacher_model:
        dc.teacher_model = args.teacher_model
    if args.teacher_local_files_only:
        dc.teacher_local_files_only = True
    if args.no_amp:
        dc.use_amp = False
    if args.amp_dtype:
        dc.amp_dtype = args.amp_dtype
    if args.no_gradient_checkpointing:
        dc.gradient_checkpointing = False
    return dc


def _masked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float,
    chunk_rows: int = 64,
) -> torch.Tensor:
    """KL(teacher || student) over loss-masked positions, chunked by row."""
    valid = loss_mask.bool()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return student_logits.sum() * 0.0
    s = student_logits[valid]
    t = teacher_logits[valid]
    total = student_logits.sum() * 0.0
    for start in range(0, n_valid, chunk_rows):
        s_chunk = s[start : start + chunk_rows].float() / temperature
        t_chunk = t[start : start + chunk_rows].float() / temperature
        s_log_probs = F.log_softmax(s_chunk, dim=-1)
        t_probs = F.softmax(t_chunk, dim=-1)
        total = total + F.kl_div(s_log_probs, t_probs, reduction="sum")
    return total / n_valid * (temperature ** 2)


def load_teacher(cfg, mc, device, use_amp, amp_dtype):
    from transformers import AutoModelForCausalLM

    dtype = amp_dtype if use_amp else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model,
        dtype=dtype,
        local_files_only=bool(cfg.teacher_local_files_only),
        trust_remote_code=False,
    )
    teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher_vocab = int(getattr(teacher.config, "vocab_size", 0))
    if teacher_vocab != int(mc.vocab_size):
        raise ValueError(
            f"Teacher vocab {teacher_vocab:,} != student vocab "
            f"{mc.vocab_size:,}; token-level KD cannot align logits"
        )
    return teacher


def distill_objective(
    student,
    teacher,
    x,
    y,
    mask,
    attention_mask,
    mc,
    dc,
):
    student_logits, ce_loss, aux = student(
        x, y, loss_mask=mask, return_aux=True
    )
    with torch.no_grad():
        teacher_out = teacher(
            input_ids=x,
            attention_mask=attention_mask,
            use_cache=False,
        )
        teacher_logits = teacher_out.logits
    kl_loss = _masked_kl(
        student_logits, teacher_logits, mask, float(dc.temperature)
    )
    loss = (1.0 - float(dc.kl_weight)) * ce_loss + float(dc.kl_weight) * kl_loss
    if aux is not None:
        loss = (
            loss
            + float(mc.router_aux_loss_coef) * aux["aux_loss"]
            + float(mc.router_z_loss_coef) * aux["z_loss"]
        )
    return loss, ce_loss, kl_loss, aux


@torch.no_grad()
def evaluate(student, teacher, loader, eval_iters, device, amp_ctx, mc, dc):
    student.eval()
    total = total_ce = total_kl = 0.0
    count = 0
    for batch in loader:
        if count >= eval_iters:
            break
        x, y, mask, attention = (t.to(device) for t in batch)
        with amp_ctx:
            loss, ce, kl, _ = distill_objective(
                student, teacher, x, y, mask, attention, mc, dc
            )
        total += loss.item()
        total_ce += ce.item()
        total_kl += kl.item()
        count += 1
    student.train()
    return {
        "loss": total / max(1, count),
        "ce": total_ce / max(1, count),
        "kl": total_kl / max(1, count),
    }


def main():
    args = parse_args()
    load_hf_token_from_dotenv()
    cfg = DIZEL_16_FLASH_CONFIG
    dc = apply_overrides(cfg, args)
    mc = cfg.model

    rule("Dizel 1.6 Flash Preview - token-level distillation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device : {device}")

    if not args.resume and not dc.base_checkpoint:
        raise ValueError("Provide --base_checkpoint for a fresh distillation run or --resume")
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {args.resume}. Refusing to start a "
            "fresh distillation run from an explicit --resume path."
        )
    if not args.resume and not os.path.exists(dc.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {dc.base_checkpoint}")
    if not os.path.exists(dc.data_path):
        raise FileNotFoundError(
            f"Distillation data not found: {dc.data_path}. Run "
            "`python pipelines/prepare_v16_posttrain.py` first."
        )

    use_amp, amp_dtype, amp_ctx, scaler = make_amp_context(dc, device)
    info(f"AMP    : {'ON (' + str(amp_dtype).replace('torch.', '') + ')' if use_amp else 'OFF'}")
    info(f"Teacher: {dc.teacher_model}")

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
    info(f"Distill records: {len(records):,} ({len(train_records):,} train / {len(val_records):,} val)")
    train_ds = ChatSFTDataset(train_records, tokenizer, dc.max_length, prefix="distill")
    val_ds = ChatSFTDataset(val_records, tokenizer, dc.max_length, prefix="distill")
    if not train_ds or not val_ds:
        raise ValueError("Distillation dataset is empty after tokenization/validation")
    ok(f"Tokenized examples: {len(train_ds):,} train + {len(val_ds):,} val")

    model = DizelLM(mc).to(device)
    if dc.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        ok("Gradient checkpointing enabled")
    optimizer = build_optimizer(model, dc)

    start_step = 0
    best_val = float("inf")
    phase0 = 0
    phase_tag = None
    dataset_state = None
    base_checkpoint = dc.base_checkpoint
    if args.resume:
        state = load_posttrain_checkpoint(
            args.resume,
            model,
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
        if state.get("best_val") is not None:
            best_val = float(state["best_val"])
        if state.get("val_loss") is not None:
            best_val = min(best_val, float(state["val_loss"]))
        ok(f"Resumed distillation at step {start_step:,}")
    else:
        load_posttrain_checkpoint(
            dc.base_checkpoint,
            model,
            tokenizer=tokenizer,
            device=str(device),
        )
        ok(f"Loaded v1.6 SFT checkpoint: {os.path.basename(dc.base_checkpoint)}")

    teacher = load_teacher(dc, mc, device, use_amp, amp_dtype)
    ok("Teacher loaded and frozen")

    expected_tag = make_schedule_tag(
        "distill", dc.max_steps, dc.lr, dc.min_lr, dc.warmup_steps
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
        collate_fn=lambda batch: distill_collate_fn(batch, pad_id=tokenizer.pad_id),
        seed=dc.seed,
    )
    stream.load_state(dataset_state)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=dc.batch_size,
        shuffle=False,
        collate_fn=lambda batch: distill_collate_fn(batch, pad_id=tokenizer.pad_id),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    config_table(
        {
            "batch size": dc.batch_size,
            "grad accum": dc.grad_accum,
            "context": dc.max_length,
            "learning rate": f"{dc.lr:.2e} -> {dc.min_lr:.2e}",
            "kl weight": dc.kl_weight,
            "temperature": dc.temperature,
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
    model.train()
    teacher.eval()
    optimizer.zero_grad()

    while step < target_steps:
        x, y, mask, attention = (t.to(device) for t in stream.next_batch())
        with amp_ctx:
            loss, _, _, _ = distill_objective(
                model, teacher, x, y, mask, attention, mc, dc
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), dc.grad_clip)
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
                model, teacher, val_loader, dc.eval_iters, device, amp_ctx, mc, dc
            )
            validation(step, metrics["ce"], extra=f"kl {metrics['kl']:.4f}")
            if metrics["ce"] < best_val:
                best_val = metrics["ce"]
                save_and_sync(
                    os.path.join(dc.output_dir, f"{dc.run_name}-best.pt"),
                    model, optimizer, scaler, step, "distill", dc, mc,
                    cfg.tokenizer, tokenizer, metrics["ce"], best_val,
                    stream.state(), phase0, phase_tag,
                    base_checkpoint=base_checkpoint,
                    drive_dir=dc.drive_checkpoint_dir,
                )
            torch.cuda.empty_cache()

        if step % dc.save_interval == 0:
            save_and_sync(
                os.path.join(dc.output_dir, f"{dc.run_name}-last.pt"),
                model, optimizer, scaler, step, "distill", dc, mc,
                cfg.tokenizer, tokenizer, None, best_val,
                stream.state(), phase0, phase_tag,
                base_checkpoint=base_checkpoint,
                drive_dir=dc.drive_checkpoint_dir,
            )

    metrics = evaluate(
        model, teacher, val_loader, dc.eval_iters, device, amp_ctx, mc, dc
    )
    best_val = min(best_val, metrics["ce"])
    save_and_sync(
        os.path.join(dc.output_dir, f"{dc.run_name}-final.pt"),
        model, optimizer, scaler, step, "distill", dc, mc,
        cfg.tokenizer, tokenizer, metrics["ce"], best_val,
        stream.state(), phase0, phase_tag,
        base_checkpoint=base_checkpoint,
        drive_dir=dc.drive_checkpoint_dir,
    )
    meter.finalise()
    warning(
        f"Final distillation CE {metrics['ce']:.4f} | KL {metrics['kl']:.4f}"
    )


if __name__ == "__main__":
    main()
