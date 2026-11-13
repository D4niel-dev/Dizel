"""
training/sft_v16.py — ChatML supervised fine-tuning for Dizel 1.6 Flash.

This is the v1.6 successor to ``training/sft.py``.  It loads a v1.6 MoE
checkpoint, uses the Qwen2.5 ChatML tokenizer, masks every non-assistant token,
and keeps the router auxiliary losses active.

Usage:
    python training/sft_v16.py \
        --base_checkpoint checkpoints/dizel-1.6-preview-m1-pretrain-best.pt \
        --data data/v16/posttrain/sft.jsonl \
        --tokenizer_model tokenizer/qwen2.5 --local_files_only

Resume:
    python training/sft_v16.py --resume checkpoints/dizel-1.6-preview-m1-sft-last.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

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
from training.posttrain_data_v16 import ChatSFTDataset, sft_collate_fn
from training.pretrain import cosine_lr_with_warmup, set_lr
from training.tokenizer_qwen import QwenTokenizer, verify_tokenizer

torch.serialization.add_safe_globals([type(DIZEL_16_FLASH_CONFIG.model)])


def parse_args():
    parser = argparse.ArgumentParser(description="Dizel 1.6 ChatML SFT")
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--drive_checkpoint_dir", default="")

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
    sc = cfg.sft
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
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(sc, name, value)
    if args.data:
        sc.data_path = args.data
    if args.output_dir:
        sc.output_dir = args.output_dir
    if args.run_name:
        sc.run_name = args.run_name
    if args.drive_checkpoint_dir:
        sc.drive_checkpoint_dir = args.drive_checkpoint_dir
    if args.base_checkpoint:
        sc.base_checkpoint = args.base_checkpoint
    if args.no_amp:
        sc.use_amp = False
    if args.amp_dtype:
        sc.amp_dtype = args.amp_dtype
    if args.no_gradient_checkpointing:
        sc.gradient_checkpointing = False
    return sc


@torch.no_grad()
def evaluate(model, loader, eval_iters, device, amp_ctx):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        if count >= eval_iters:
            break
        x, y, mask = (t.to(device) for t in batch)
        with amp_ctx:
            _, ce, _ = model(x, y, loss_mask=mask, return_aux=True)
        total += ce.item()
        count += 1
    model.train()
    return total / max(1, count)


def main():
    args = parse_args()
    cfg = DIZEL_16_FLASH_CONFIG
    sc = apply_overrides(cfg, args)
    mc = cfg.model

    rule("Dizel 1.6 Flash Preview - ChatML SFT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Device : {device}")

    if not args.resume and not sc.base_checkpoint:
        raise ValueError("Provide --base_checkpoint for a fresh SFT run or --resume")
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(
            f"Resume checkpoint not found: {args.resume}. Refusing to start "
            "a fresh SFT run from an explicit --resume path."
        )
    if not args.resume and not os.path.exists(sc.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {sc.base_checkpoint}")
    if not os.path.exists(sc.data_path):
        raise FileNotFoundError(
            f"SFT data not found: {sc.data_path}. Run "
            "`python pipelines/prepare_v16_posttrain.py` first."
        )

    use_amp, amp_dtype, amp_ctx, scaler = make_amp_context(sc, device)
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

    records = load_jsonl_records(sc.data_path)
    train_records, val_records = split_records(records, sc.val_fraction, sc.seed)
    section("Data")
    info(f"SFT records : {len(records):,} ({len(train_records):,} train / {len(val_records):,} val)")
    train_ds = ChatSFTDataset(train_records, tokenizer, sc.max_length)
    val_ds = ChatSFTDataset(val_records, tokenizer, sc.max_length)
    if not train_ds or not val_ds:
        raise ValueError("SFT dataset is empty after tokenization/validation")
    ok(f"Tokenized examples: {len(train_ds):,} train + {len(val_ds):,} val")

    model = DizelLM(mc).to(device)
    if sc.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        ok("Gradient checkpointing enabled")
    optimizer = build_optimizer(model, sc)

    start_step = 0
    best_val = float("inf")
    phase0 = 0
    phase_tag = None
    dataset_state = None
    base_checkpoint = sc.base_checkpoint
    state = None
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
        ok(f"Resumed SFT at step {start_step:,}")
    else:
        load_posttrain_checkpoint(
            sc.base_checkpoint,
            model,
            tokenizer=tokenizer,
            device=str(device),
        )
        ok(f"Loaded v1.6 base checkpoint: {os.path.basename(sc.base_checkpoint)}")

    expected_tag = make_schedule_tag(
        "sft", sc.max_steps, sc.lr, sc.min_lr, sc.warmup_steps
    )
    if phase_tag != expected_tag:
        phase_tag = expected_tag
        phase0 = start_step if args.resume else 0
    target_steps = phase0 + sc.max_steps
    info(f"Steps  : {start_step:,} -> {target_steps:,} ({sc.max_steps:,} this phase)")

    section("Training")
    os.makedirs(sc.output_dir, exist_ok=True)
    stream = ResumableBatchStream(
        train_ds,
        batch_size=sc.batch_size,
        collate_fn=lambda batch: sft_collate_fn(batch, pad_id=tokenizer.pad_id),
        seed=sc.seed,
    )
    stream.load_state(dataset_state)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=sc.batch_size,
        shuffle=False,
        collate_fn=lambda batch: sft_collate_fn(batch, pad_id=tokenizer.pad_id),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    config_table(
        {
            "batch size": sc.batch_size,
            "grad accum": sc.grad_accum,
            "context": sc.max_length,
            "learning rate": f"{sc.lr:.2e} -> {sc.min_lr:.2e}",
            "warmup": sc.warmup_steps,
            "eval every": sc.eval_interval,
        }
    )

    meter = StepMeter(
        target_steps,
        tokens_per_step=sc.batch_size * sc.grad_accum * sc.max_length,
        start_step=start_step,
    )
    step = start_step
    micro = 0
    accum_loss = 0.0
    last_lr = sc.lr
    model.train()
    optimizer.zero_grad()

    while step < target_steps:
        x, y, mask = (t.to(device) for t in stream.next_batch())
        with amp_ctx:
            _, ce, aux = model(x, y, loss_mask=mask, return_aux=True)
            loss = ce
            if aux is not None:
                loss = (
                    loss
                    + float(mc.router_aux_loss_coef) * aux["aux_loss"]
                    + float(mc.router_z_loss_coef) * aux["z_loss"]
                )
            loss = loss / sc.grad_accum
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += loss.item()
        micro += 1

        if micro % sc.grad_accum != 0:
            continue

        local = step - phase0
        last_lr = cosine_lr_with_warmup(
            local, sc.warmup_steps, sc.max_steps, sc.lr, sc.min_lr
        )
        set_lr(optimizer, last_lr)
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), sc.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        train_loss = accum_loss
        accum_loss = 0.0
        step += 1
        if step % sc.log_interval == 0:
            meter.update(step=step, loss=train_loss, lr=last_lr)

        if step % sc.eval_interval == 0:
            val_loss = evaluate(model, val_loader, sc.eval_iters, device, amp_ctx)
            validation(step, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save_and_sync(
                    os.path.join(sc.output_dir, f"{sc.run_name}-best.pt"),
                    model, optimizer, scaler, step, "sft", sc, mc,
                    cfg.tokenizer, tokenizer, val_loss, best_val,
                    stream.state(), phase0, phase_tag,
                    base_checkpoint=base_checkpoint,
                    drive_dir=sc.drive_checkpoint_dir,
                )
            torch.cuda.empty_cache()

        if step % sc.save_interval == 0:
            save_and_sync(
                os.path.join(sc.output_dir, f"{sc.run_name}-last.pt"),
                model, optimizer, scaler, step, "sft", sc, mc,
                cfg.tokenizer, tokenizer, None, best_val,
                stream.state(), phase0, phase_tag,
                base_checkpoint=base_checkpoint,
                drive_dir=sc.drive_checkpoint_dir,
            )

    final_val = evaluate(model, val_loader, sc.eval_iters, device, amp_ctx)
    best_val = min(best_val, final_val)
    save_and_sync(
        os.path.join(sc.output_dir, f"{sc.run_name}-final.pt"),
        model, optimizer, scaler, step, "sft", sc, mc,
        cfg.tokenizer, tokenizer, final_val, best_val,
        stream.state(), phase0, phase_tag,
        base_checkpoint=base_checkpoint,
        drive_dir=sc.drive_checkpoint_dir,
    )
    meter.finalise()
    warning(f"Final SFT validation CE: {final_val:.4f}")


if __name__ == "__main__":
    main()
