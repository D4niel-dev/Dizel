"""
scripts/smoke_test_v16_runner.py — end-to-end CPU runner proof for Dizel 1.6.

Proves the v1.6 pretraining entry point without transformers, CUDA, or
network access:

  * synthetic English/Vietnamese/code corpus
  * fake tokenizer with a 128-token vocabulary and stable fingerprint
  * tiny 4-expert top-2 MoE config
  * full train() path: AMP selection, data cache, MoE losses, expert stats,
    BPB, CSV logging, best/last/final/milestone checkpoints
  * checkpoint reload into a fresh model/optimizer
  * resume from the step-2 "last" checkpoint through step 4, including
    tokenizer-hash validation, optimizer/scaler/RNG/dataset-state restore
    and the phase schedule
  * exact-resume state comparison (weights and Adam moments) after a
    post-reshuffle step-3 checkpoint with num_workers=2, proving the
    DataLoader workers see the restored window order

Run from the repo root:  python scripts/smoke_test_v16_runner.py
"""

import copy
import csv
import glob
import math
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from configs.v16_flash import (
    DizelV16FlashConfig,
    ModelConfigV16Flash,
    PretrainConfigV16Flash,
    TokenizerConfigV16Flash,
)
from model.architecture import DizelLM
from training.pretrain_v16 import load_checkpoint_v16, train

torch.manual_seed(0)
torch.set_num_threads(1)

VOCAB = 128
CONTEXT = 32
BATCH = 2
GRAD_ACCUM = 2
MAX_STEPS = 4
TOKENS_PER_STEP = BATCH * GRAD_ACCUM * CONTEXT


class FakeTokenizer:
    """Deterministic byte-ish tokenizer: no transformers, no network."""

    pad_id = 0
    bos_id = None
    eos_id = 1
    unk_id = 2
    model_name = "fake-qwen-tokenizer"
    chat_template = None
    vocab_size = VOCAB

    def __len__(self) -> int:
        return VOCAB

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = [3 + (ord(ch) % 125) for ch in text]
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids if int(i) >= 3)

    def fingerprint(self) -> str:
        return "f" * 64

    def cache_key(self) -> str:
        return "fake-qwen-v1"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        return self.encode(rendered) if tokenize else rendered


def _write_corpus(path: str) -> None:
    english = (
        "Dizel is a small language model built for training experiments. "
        "A mixture-of-experts layer routes every token to its top experts.\n"
    )
    vietnamese = (
        "Dizel là một mô hình ngôn ngữ nhỏ dùng cho các thí nghiệm huấn "
        "luyện. Mỗi token được định tuyến tới các chuyên gia phù hợp.\n"
    )
    code = (
        "def moe_route(logits, top_k=2):\n"
        "    probs = torch.softmax(logits, dim=-1)\n"
        "    weights, indices = torch.topk(probs, top_k, dim=-1)\n"
        "    return weights / weights.sum(dim=-1, keepdim=True), indices\n"
    )
    block = english + vietnamese + code + "\n"
    text = ""
    while len(text) < 3000:
        text += block
    text = text[:3000]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _model_config() -> ModelConfigV16Flash:
    return ModelConfigV16Flash(
        vocab_size=VOCAB,
        context_length=CONTEXT,
        d_model=32,
        n_layers=2,
        n_heads=4,
        kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
        dropout=0.0,
    )


def _make_optimizer(model: DizelLM) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _build_config(tmp_dir: str, corpus_path: str) -> DizelV16FlashConfig:
    model = _model_config()
    tokenizer = TokenizerConfigV16Flash(
        model_name=FakeTokenizer.model_name,
        vocab_size=VOCAB,
    )
    pretrain = PretrainConfigV16Flash(
        data_path=corpus_path,
        batch_size=BATCH,
        grad_accum=GRAD_ACCUM,
        max_length=CONTEXT,
        max_steps=MAX_STEPS,
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=1,
        eval_interval=2,
        eval_iters=2,
        save_interval=2,
        checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
        log_dir=os.path.join(tmp_dir, "logs"),
        run_name="smoke-v16",
        train_split=0.9,
        seed=0,
        shard_size_mb=1,
        cache_dir=os.path.join(tmp_dir, "cache"),
        num_workers=2,
        resume_enabled=True,
        reshuffle_every_n_steps=2,
        drive_checkpoint_dir="",
        log_interval=1,
        milestone_interval=2,
        gradient_checkpointing=True,
        use_amp=False,
    )
    return DizelV16FlashConfig(
        model=model,
        tokenizer=tokenizer,
        pretrain=pretrain,
    )


def _read_csv(path: str):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _assert_state_matches(
    payload: dict,
    expected_state: dict,
    expected_optim,
    label: str,
) -> None:
    """Assert a checkpoint reproduces a previously saved model/optimizer state."""
    state = payload["model_state"]
    assert set(state) == set(expected_state), f"{label}: state keys differ"
    for key, expected in expected_state.items():
        actual = state[key]
        if expected.is_floating_point():
            assert torch.allclose(
                actual, expected, atol=1e-6, rtol=1e-5
            ), (
                f"{label}: model weight mismatch at {key}: max abs diff "
                f"{float((actual - expected).abs().max())}"
            )
        else:
            assert torch.equal(actual, expected), (
                f"{label}: model tensor mismatch at {key}"
            )

    if expected_optim is None:
        return
    optim = payload.get("optim_state")
    assert optim is not None, f"{label}: checkpoint has no optimizer state"
    assert set(optim["state"]) == set(expected_optim["state"]), (
        f"{label}: optimizer param ids differ"
    )
    for param_id, expected in expected_optim["state"].items():
        actual = optim["state"][param_id]
        assert set(actual) == set(expected), (
            f"{label}: optimizer state keys differ for param {param_id}"
        )
        for field in ("exp_avg", "exp_avg_sq"):
            if field not in expected:
                continue
            assert torch.allclose(
                actual[field], expected[field], atol=1e-6, rtol=1e-5
            ), (
                f"{label}: optimizer {field} mismatch for param {param_id}: "
                f"max abs diff "
                f"{float((actual[field] - expected[field]).abs().max())}"
            )
        if "step" in expected:
            assert torch.equal(actual["step"], expected["step"]), (
                f"{label}: optimizer step mismatch for param {param_id}"
            )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_path = os.path.join(tmp_dir, "smoke_corpus.txt")
        _write_corpus(corpus_path)
        corpus_size = os.path.getsize(corpus_path)
        assert 2500 <= corpus_size <= 4000, corpus_size

        cfg = _build_config(tmp_dir, corpus_path)
        fake = FakeTokenizer()

        ckpt_dir = cfg.pretrain.checkpoint_dir
        log_dir = cfg.pretrain.log_dir
        last_path = os.path.join(ckpt_dir, "smoke-v16-last.pt")
        best_path = os.path.join(ckpt_dir, "smoke-v16-best.pt")
        final_path = os.path.join(ckpt_dir, "smoke-v16-final.pt")
        csv_path = os.path.join(log_dir, "smoke-v16.csv")

        # -- Full run -----------------------------------------------------
        train(cfg, tokenizer=fake)

        for path in (last_path, best_path, final_path):
            assert os.path.exists(path), f"missing checkpoint: {path}"

        milestones = glob.glob(
            os.path.join(ckpt_dir, "smoke-v16-milestone-step*.pt")
        )
        assert milestones, "no milestone checkpoint was written"
        assert len(milestones) == 1, milestones

        last_payload = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        assert last_payload["step"] == 2, last_payload["step"]

        assert os.path.exists(csv_path), "missing training CSV"
        rows = _read_csv(csv_path)
        assert rows, "CSV is empty"
        header = rows[0]
        assert "step" in header and "val_bpb" in header, header
        step_idx = header.index("step")
        bpb_idx = header.index("val_bpb")
        data_rows = rows[1:]
        assert data_rows, "CSV has no data rows"
        finite_bpb = [
            float(row[bpb_idx])
            for row in data_rows
            if row[bpb_idx]
            and math.isfinite(float(row[bpb_idx]))
        ]
        assert finite_bpb, "no row has a finite val_bpb"
        assert int(data_rows[-1][step_idx]) == MAX_STEPS

        final_payload = torch.load(
            final_path, map_location="cpu", weights_only=False
        )
        assert final_payload["_format_version"] == "1.6"
        assert final_payload["step"] == MAX_STEPS
        assert final_payload["tokens_seen"] == MAX_STEPS * TOKENS_PER_STEP
        assert final_payload["num_experts"] == 4
        assert final_payload["top_k"] == 2
        assert final_payload["phase0"] == 0
        assert final_payload["tokenizer_hash"] == fake.fingerprint()

        # -- Load final into a fresh model/optimizer ----------------------
        fresh = DizelLM(copy.deepcopy(cfg.model))
        fresh_optimizer = _make_optimizer(fresh)
        loaded = load_checkpoint_v16(
            final_path,
            fresh,
            optimizer=fresh_optimizer,
            scaler=None,
            tokenizer=fake,
            device="cpu",
        )
        assert loaded["step"] == MAX_STEPS
        assert loaded["tokens_seen"] == MAX_STEPS * TOKENS_PER_STEP
        for key in (
            "transformer.tok_emb.weight",
            "transformer.blocks.0.mlp.router.weight",
        ):
            assert key in final_payload["model_state"], key
            assert torch.equal(
                fresh.state_dict()[key], final_payload["model_state"][key]
            ), key

        # -- Snapshot the continuous run before resume overwrites final ----
        # Steps 3-4 of the resumed run must reproduce the original continuous
        # run exactly; keep copies of the pre-resume final state to prove it.
        continuous_final_state = {
            key: tensor.detach().clone()
            for key, tensor in final_payload["model_state"].items()
        }
        continuous_final_optim = copy.deepcopy(
            final_payload.get("optim_state")
        )

        # -- Resume from step 2 through step 4 ----------------------------
        rows_before = len(rows)
        train(cfg, resume_from=last_path, tokenizer=fake)

        rows_after = _read_csv(csv_path)
        assert len(rows_after) > rows_before, "resume wrote no CSV rows"
        new_rows = rows_after[rows_before:]
        assert int(new_rows[0][step_idx]) == 3, (
            "resume should continue at step 3",
            new_rows[0],
        )
        assert int(rows_after[-1][step_idx]) == MAX_STEPS

        resumed_payload = torch.load(
            final_path, map_location="cpu", weights_only=False
        )
        assert resumed_payload["step"] == MAX_STEPS
        assert resumed_payload["phase0"] == 0

        # -- Exact-resume proof: compare weights from the continuous run --
        _assert_state_matches(
            resumed_payload,
            continuous_final_state,
            continuous_final_optim,
            "step-2 resume",
        )

        fresh_resumed = DizelLM(copy.deepcopy(cfg.model))
        fresh_resumed_optimizer = _make_optimizer(fresh_resumed)
        resumed = load_checkpoint_v16(
            final_path,
            fresh_resumed,
            optimizer=fresh_resumed_optimizer,
            scaler=None,
            tokenizer=fake,
            device="cpu",
        )
        assert resumed["step"] == MAX_STEPS
        assert resumed["tokens_seen"] == MAX_STEPS * TOKENS_PER_STEP

        # -- Resume-order proof across a post-reshuffle checkpoint --------
        # The step-2 resume above lands exactly on a reshuffle boundary: the
        # replay builds a fresh iterator from the restored dataset, which
        # would hide a load-after-iter regression. Resume from step 3
        # instead: the step-2 reshuffle is already reflected in the
        # checkpoint and step 3 is not a boundary, so the workers must copy
        # the restored window order before iter() for step 4 to match.
        order_cfg = copy.deepcopy(cfg)
        order_cfg.pretrain.save_interval = 1
        order_cfg.pretrain.milestone_interval = 0
        order_cfg.pretrain.run_name = "smoke-v16-order"
        order_cfg.pretrain.checkpoint_dir = os.path.join(
            tmp_dir, "ckpt_order"
        )
        order_cfg.pretrain.log_dir = os.path.join(tmp_dir, "logs_order")
        order_last_path = os.path.join(
            order_cfg.pretrain.checkpoint_dir, "smoke-v16-order-last.pt"
        )
        order_final_path = os.path.join(
            order_cfg.pretrain.checkpoint_dir, "smoke-v16-order-final.pt"
        )

        train(order_cfg, tokenizer=fake)
        order_final = torch.load(
            order_final_path, map_location="cpu", weights_only=False
        )
        assert order_final["step"] == MAX_STEPS
        order_state = {
            key: tensor.detach().clone()
            for key, tensor in order_final["model_state"].items()
        }
        order_optim = copy.deepcopy(order_final.get("optim_state"))

        order_last = torch.load(
            order_last_path, map_location="cpu", weights_only=False
        )
        assert order_last["step"] == MAX_STEPS - 1, order_last["step"]
        assert order_last["dataset_state"]["seed"] == (
            cfg.pretrain.seed + cfg.pretrain.reshuffle_every_n_steps
        ), (
            "step-3 checkpoint must be post-reshuffle",
            order_last["dataset_state"]["seed"],
        )

        train(order_cfg, resume_from=order_last_path, tokenizer=fake)
        resumed_order = torch.load(
            order_final_path, map_location="cpu", weights_only=False
        )
        assert resumed_order["step"] == MAX_STEPS
        _assert_state_matches(
            resumed_order,
            order_state,
            order_optim,
            "post-reshuffle resume",
        )

    print("V1.6 RUNNER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
