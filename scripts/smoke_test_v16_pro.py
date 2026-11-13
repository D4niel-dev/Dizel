"""
scripts/smoke_test_v16_pro.py — CPU-only smoke test for the Dizel 1.6 Pro core.

Pro is the planned 8-expert, top-4 tier. Nothing else instantiates it yet, so
this gate proves the config, the MoE router, the auxiliary losses, gradients,
gradient checkpointing, and a short optimizer run all work at Pro's expert
count before a training run is scheduled.

Run from the repo root:  python scripts/smoke_test_v16_pro.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from configs.v16_pro import (
    DIZEL_16_PRO_CONFIG,
    ModelConfigV16Pro,
    PretrainConfigV16Pro,
)
from model.architecture import DizelLM

torch.manual_seed(0)
torch.set_num_threads(1)

VOCAB = 128
B, T = 16, 32


def _tiny_config() -> ModelConfigV16Pro:
    return ModelConfigV16Pro(
        vocab_size=VOCAB,
        context_length=T,
        d_model=32,
        n_layers=2,
        n_heads=4,
        kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        num_experts=8,
        top_k=4,
        dropout=0.0,
        bias=False,
        weight_tying=True,
        norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
        router_init_std=0.02,
    )


def _make_batch(seed: int = 1234):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, VOCAB, (B, T), generator=gen)
    targets = torch.roll(idx, shifts=-1, dims=1)
    return idx, targets


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def test_a_full_config_contract():
    cfg = DIZEL_16_PRO_CONFIG
    mc = cfg.model
    pc = cfg.pretrain

    print(
        f"[a] total={mc.param_estimate():,}  "
        f"active={mc.active_param_estimate():,}  "
        f"experts={mc.num_experts} top-{mc.top_k}"
    )
    assert mc.param_estimate() == 1_633_957_120, mc.param_estimate()
    assert mc.active_param_estimate() == 973_354_240, mc.active_param_estimate()
    assert (mc.num_experts, mc.top_k) == (8, 4)
    assert mc.d_model // mc.n_heads == mc.head_dim
    assert cfg.release_stage == "planned"

    tokens_per_step = pc.batch_size * pc.grad_accum * mc.context_length
    total_tokens = tokens_per_step * pc.max_steps
    print(
        f"[a] tokens/step={tokens_per_step:,}  schedule={total_tokens:,}"
    )
    assert tokens_per_step == 65_536, tokens_per_step
    assert total_tokens == 3_000_041_472, total_tokens
    assert pc.warmup_steps < pc.max_steps
    assert 0 < pc.min_lr < pc.lr


def test_b_forward_with_aux(cfg, model, idx, targets):
    logits, ce, aux = model(idx, targets, return_aux=True)
    print(
        f"[b] logits={tuple(logits.shape)} ce={ce.item():.4f} "
        f"aux={aux['aux_loss'].item():.4f} z={aux['z_loss'].item():.4f} "
        f"total={aux['total_loss'].item():.4f}"
    )
    assert logits.shape == (B, T, VOCAB)
    for name in ("ce_loss", "aux_loss", "z_loss", "total_loss"):
        assert aux[name] is not None, name
        assert _finite(aux[name]), name
    expected = (
        ce
        + cfg.router_aux_loss_coef * aux["aux_loss"]
        + cfg.router_z_loss_coef * aux["z_loss"]
    )
    assert torch.allclose(aux["total_loss"], expected, atol=1e-5, rtol=0.0)
    return aux


def test_c_top4_utilization(cfg, aux):
    counts = aux["expert_counts"]
    per_layer = B * T * cfg.top_k
    print(f"[c] expert_counts={counts.tolist()}")
    assert counts.shape == (cfg.n_layers, cfg.num_experts), counts.shape
    assert int(counts.sum().item()) == cfg.n_layers * per_layer
    assert bool((counts.sum(dim=1) == per_layer).all().item())
    assert bool((counts > 0).all().item()), counts
    assert int(aux["top1_counts"].sum().item()) == cfg.n_layers * B * T
    assert int(aux["top2_counts"].sum().item()) == cfg.n_layers * B * T


def test_d_router_logits_fp32_under_bf16(cfg, model):
    block = model.transformer["blocks"][0]
    x = torch.randn(B, T, cfg.d_model)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _ = torch.ones(1, 1) @ torch.ones(1, 1)
    except RuntimeError as exc:
        print(f"[d] SKIPPED: CPU bf16 autocast unavailable: {exc}")
        return

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, router_logits = block(x, return_router_logits=True)
    print(f"[d] router_logits dtype={router_logits.dtype}")
    assert out.shape == (B, T, cfg.d_model)
    assert router_logits.dtype == torch.float32, router_logits.dtype
    assert router_logits.shape == (B * T, cfg.num_experts)


def test_e_gradients(model, idx, targets):
    model.train()
    model.zero_grad(set_to_none=True)
    _, _, aux = model(idx, targets, return_aux=True)
    aux["total_loss"].backward()

    for layer_idx, block in enumerate(model.transformer["blocks"]):
        grad = block.mlp.router.weight.grad
        assert grad is not None, (layer_idx, "router received no gradient")
        assert _finite(grad) and grad.abs().sum().item() > 0.0, layer_idx
        for expert_idx, expert in enumerate(block.mlp.experts):
            grad = expert.gate_proj.weight.grad
            assert grad is not None, (layer_idx, expert_idx)
            assert _finite(grad), (layer_idx, expert_idx)
            assert grad.abs().sum().item() > 0.0, (layer_idx, expert_idx)
    print(f"[e] router + {model.transformer['blocks'][0].mlp.num_experts} "
          "expert gradients per layer are finite and non-zero")


def test_f_gradient_checkpointing(cfg, idx, targets):
    model = DizelLM(cfg)
    model.enable_gradient_checkpointing()
    model.train()
    model.zero_grad(set_to_none=True)
    _, _, aux = model(idx, targets, return_aux=True)
    aux["total_loss"].backward()
    assert model.transformer["blocks"][0].mlp.router.weight.grad is not None
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert _finite(param.grad), name
    print("[f] gradient checkpointing forward/backward ok")


def test_g_short_optimizer_run(model, idx, targets):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial = None
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        _, ce, aux = model(idx, targets, return_aux=True)
        if initial is None:
            initial = ce.item()
        aux["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        _, final, _ = model(idx, targets, return_aux=True)
    print(f"[g] CE initial={initial:.4f} final={final.item():.4f}")
    assert final.item() < initial, (initial, final.item())


def main():
    test_a_full_config_contract()

    cfg = _tiny_config()
    model = DizelLM(cfg)
    print(f"[setup] {model}")
    idx, targets = _make_batch()

    aux = test_b_forward_with_aux(cfg, model, idx, targets)
    test_c_top4_utilization(cfg, aux)
    test_d_router_logits_fp32_under_bf16(cfg, model)
    test_e_gradients(model, idx, targets)
    test_f_gradient_checkpointing(cfg, idx, targets)
    test_g_short_optimizer_run(model, idx, targets)

    print("V1.6 PRO SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
