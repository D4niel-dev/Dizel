"""
scripts/smoke_test_v16.py — CPU-only smoke test for the Dizel 1.6 MoE core.

Covers the v1.6 plan's smoke-test gate (§11) without transformers, GPU, or
network access:

  a. Full config total/active parameter estimates
  b. MoE forward with auxiliary losses
  c. Per-layer expert utilization
  d. Float32 router logits under CPU bf16 autocast
  e. Router and expert gradients
  f. Loss decreases over a short optimizer run
  g. Gradient checkpointing forward/backward
  h. Checkpoint + optimizer-state round trip
  i. Generation

Run from the repo root:  python scripts/smoke_test_v16.py
"""

import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from configs.v16_flash import ModelConfigV16Flash
from model.architecture import DizelLM

torch.manual_seed(0)
torch.set_num_threads(1)

VOCAB = 128
B, T = 16, 32


def _tiny_config() -> ModelConfigV16Flash:
    return ModelConfigV16Flash(
        vocab_size=VOCAB,
        context_length=T,
        d_model=32,
        n_layers=2,
        n_heads=4,
        kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        num_experts=4,
        top_k=2,
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


def _cpu_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:  # older torch without device_type argument
        return torch.is_autocast_cpu_enabled()


def test_a_full_config_estimates():
    cfg = ModelConfigV16Flash()
    total = cfg.param_estimate()
    active = cfg.active_param_estimate()
    print(f"[a] total={total:,}  active={active:,}")
    assert total == 435_284_736, total
    assert active == 293_726_976, active


def test_b_forward_with_aux(model, cfg, idx, targets):
    logits, ce, aux = model(idx, targets, return_aux=True)
    print(f"[b] logits={tuple(logits.shape)} ce={ce.item():.4f} "
          f"aux={aux['aux_loss'].item():.4f} z={aux['z_loss'].item():.4f} "
          f"total={aux['total_loss'].item():.4f}")

    assert logits.shape == (B, T, VOCAB)
    for name in ("ce_loss", "aux_loss", "z_loss", "total_loss"):
        assert aux[name] is not None, name
        assert _finite(aux[name]), name
    assert aux["ce_loss"] is ce
    assert aux["aux_loss"].item() > 0.0
    assert aux["z_loss"].item() > 0.0
    assert aux["aux_loss"].requires_grad
    assert aux["z_loss"].requires_grad

    expected_total = (
        ce
        + cfg.router_aux_loss_coef * aux["aux_loss"]
        + cfg.router_z_loss_coef * aux["z_loss"]
    )
    assert torch.allclose(
        aux["total_loss"], expected_total, atol=1e-5, rtol=0.0
    ), (aux["total_loss"].item(), expected_total.item())
    return logits, ce, aux


def test_c_expert_utilization(cfg, aux):
    counts = aux["expert_counts"]
    print(f"[c] expert_counts={counts.tolist()}")
    assert counts.shape == (cfg.n_layers, cfg.num_experts), counts.shape
    assert int(counts.sum().item()) == cfg.n_layers * B * T * cfg.top_k
    assert bool(
        (counts.sum(dim=1) == B * T * cfg.top_k).all().item()
    ), counts.sum(dim=1)
    assert bool((counts > 0).all().item()), counts
    assert aux["top1_counts"].shape == (cfg.n_layers, cfg.num_experts)
    assert aux["top2_counts"].shape == (cfg.n_layers, cfg.num_experts)
    assert aux["router_entropy"].shape == (cfg.n_layers,)
    assert aux["num_tokens"] == B * T
    # Top-1 and top-2 partitions each cover every token exactly once per layer.
    assert int(aux["top1_counts"].sum().item()) == cfg.n_layers * B * T
    assert int(aux["top2_counts"].sum().item()) == cfg.n_layers * B * T


def test_d_router_logits_fp32_under_bf16(model, cfg):
    block = model.transformer["blocks"][0]
    x = torch.randn(B, T, cfg.d_model)

    autocast_available = True
    skip_reason = ""
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _ = torch.ones(1, 1) @ torch.ones(1, 1)
            if not _cpu_autocast_enabled():
                autocast_available = False
                skip_reason = "torch reports CPU autocast as disabled"
    except RuntimeError as exc:
        autocast_available = False
        skip_reason = str(exc)

    if not autocast_available:
        print(f"[d] SKIPPED: CPU bf16 autocast unavailable: {skip_reason}")
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
    _, ce, aux = model(idx, targets, return_aux=True)
    aux["total_loss"].backward()

    router_norms = []
    for layer_idx, block in enumerate(model.transformer["blocks"]):
        router_grad = block.mlp.router.weight.grad
        assert router_grad is not None, (layer_idx, "router received no gradient")
        assert _finite(router_grad), (layer_idx, "router gradient not finite")
        assert router_grad.abs().sum().item() > 0.0, (layer_idx, "router gradient is zero")
        router_norms.append(router_grad.norm().item())
    print(f"[e] router grad norms={['%.4g' % n for n in router_norms]}")

    for layer_idx, block in enumerate(model.transformer["blocks"]):
        for expert_idx, expert in enumerate(block.mlp.experts):
            grad = expert.gate_proj.weight.grad
            assert grad is not None, (layer_idx, expert_idx)
            assert _finite(grad), (layer_idx, expert_idx)
            assert grad.abs().sum().item() > 0.0, (layer_idx, expert_idx)


def test_f_loss_decreases(model):
    idx, targets = _make_batch()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01
    )

    initial_ce = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        _, ce, aux = model(idx, targets, return_aux=True)
        if initial_ce is None:
            initial_ce = ce.item()
        aux["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, final_ce, _ = model(idx, targets, return_aux=True)
    final_ce = final_ce.item()
    print(f"[f] CE initial={initial_ce:.4f} final={final_ce:.4f}")
    assert final_ce < initial_ce, (initial_ce, final_ce)
    model.train()
    return optimizer


def test_g_gradient_checkpointing(cfg, idx, targets):
    gc_model = DizelLM(cfg)
    gc_model.enable_gradient_checkpointing()
    gc_model.enable_gradient_checkpointing()  # idempotence guard
    gc_model.train()
    gc_model.zero_grad(set_to_none=True)

    _, ce, aux = gc_model(idx, targets, return_aux=True)
    assert _finite(ce) and _finite(aux["aux_loss"]) and _finite(aux["z_loss"])
    aux["total_loss"].backward()

    gc_norms = []
    for layer_idx, block in enumerate(gc_model.transformer["blocks"]):
        router_grad = block.mlp.router.weight.grad
        assert router_grad is not None, (
            layer_idx, "checkpointed router received no gradient"
        )
        assert _finite(router_grad), layer_idx
        assert router_grad.abs().sum().item() > 0.0, layer_idx
        gc_norms.append(router_grad.norm().item())
    for name, param in gc_model.named_parameters():
        if param.grad is not None:
            assert _finite(param.grad), name
    print(f"[g] checkpointed router grad norms={['%.4g' % n for n in gc_norms]}")


def test_h_checkpoint_round_trip(model, optimizer, cfg):
    assert len(optimizer.state) > 0, "optimizer has no state to round-trip"
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "smoke_ckpt.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": 20,
            },
            path,
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)

        fresh = DizelLM(cfg)
        fresh.load_state_dict(payload["model"])
        original_state = model.state_dict()
        fresh_state = fresh.state_dict()
        assert list(original_state.keys()) == list(fresh_state.keys())
        for name, tensor in original_state.items():
            assert torch.equal(tensor, fresh_state[name]), name

        fresh_optimizer = torch.optim.AdamW(
            fresh.parameters(), lr=1e-3, weight_decay=0.01
        )
        fresh_optimizer.load_state_dict(payload["optimizer"])
        assert payload["step"] == 20
        assert len(fresh_optimizer.state) == len(optimizer.state)
    print(f"[h] round-trip verified for {len(original_state)} tensors")


def test_i_generation(model):
    model.eval()
    prompt = torch.randint(0, VOCAB, (1, 4))
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=5, eos_ids=[])
    print(f"[i] generated shape={tuple(out.shape)}")
    assert out.shape == (1, 4 + 5), out.shape


def main():
    # a. Full-size parameter accounting
    test_a_full_config_estimates()

    cfg = _tiny_config()
    model = DizelLM(cfg)
    print(f"[setup] {model}")
    idx, targets = _make_batch()

    # b/c. MoE forward + diagnostics
    _, _, aux = test_b_forward_with_aux(model, cfg, idx, targets)
    test_c_expert_utilization(cfg, aux)

    # d. Router precision under autocast
    test_d_router_logits_fp32_under_bf16(model, cfg)

    # e. Router/expert gradients
    test_e_gradients(model, idx, targets)

    # f. Optimization actually reduces CE
    optimizer = test_f_loss_decreases(model)

    # g. Gradient checkpointing
    test_g_gradient_checkpointing(cfg, idx, targets)

    # h. Checkpoint/optimizer round-trip
    test_h_checkpoint_round_trip(model, optimizer, cfg)

    # i. Generation
    test_i_generation(model)

    print("V1.6 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
