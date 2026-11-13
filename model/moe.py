"""
model/moe.py — Dropless top-k Mixture-of-Experts FFN for Dizel v1.6.

The MoE block replaces a dense SwiGLU FFN. Every token is routed to its
top-k experts (no capacity factor, no token dropping). The router always
runs in float32, even when the surrounding model runs under bf16/fp16
autocast, so routing decisions and auxiliary losses stay numerically stable.

Diagnostics
-----------
``router_stats`` and ``moe_aux_losses`` expose the quantities required by the
v1.6 plan: load-balancing loss, z-loss, per-layer expert utilization,
top-1/top-2 distributions, and router entropy.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MoEFFN",
    "load_balancing_loss",
    "router_z_loss",
    "router_stats",
    "moe_aux_losses",
]


def _router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute router logits in float32, ignoring any active autocast context."""
    device_type = x.device.type
    if device_type in ("cpu", "cuda"):
        with torch.autocast(device_type=device_type, enabled=False):
            return F.linear(x.float(), weight.float())
    # Autocast is not defined for every backend (e.g. MPS); the explicit
    # float()/float() cast already guarantees a float32 router.
    return F.linear(x.float(), weight.float())


class MoEFFN(nn.Module):
    """
    Dropless top-k MoE feed-forward block.

    Args:
        cfg: config object exposing ``d_model``, ``num_experts``, ``top_k``,
             ``router_init_std`` and the SwiGLU fields used by ``SwiGLUMLP``.

    Forward:
        x:                (..., d_model)
        return_router_logits: also return the flattened (N, num_experts)
                         float32 router logits used by the auxiliary losses.
    """

    def __init__(self, cfg):
        super().__init__()
        # Lazy import: model.architecture imports this module at top level.
        from model.architecture import SwiGLUMLP

        d_model = int(cfg.d_model)
        self.num_experts = int(getattr(cfg, "num_experts", 1))
        self.top_k = int(getattr(cfg, "top_k", 2))
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"top_k must be in [1, {self.num_experts}], got {self.top_k}"
            )

        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        nn.init.normal_(
            self.router.weight,
            mean=0.0,
            std=float(getattr(cfg, "router_init_std", 0.02)),
        )
        # DizelLM._init_weights() must not overwrite the router with its
        # generic Linear initialization.
        self.router._skip_weight_init = True

        self.experts = nn.ModuleList(
            [SwiGLUMLP(cfg) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor, return_router_logits: bool = False):
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])

        router_logits = _router_logits(flat_x, self.router.weight)
        probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)

        # Dropless dispatch: each token is processed by exactly top_k experts.
        output = torch.zeros_like(flat_x)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, rank_idx = (topk_indices == expert_idx).nonzero(
                as_tuple=True
            )
            if token_idx.numel() == 0:
                continue  # An unused expert contributes nothing this batch.
            expert_out = expert(flat_x[token_idx])
            scale = topk_weights[token_idx, rank_idx].to(expert_out.dtype)
            # Autocast may produce a lower-precision expert output than the
            # fp32 residual stream; accumulate in the residual dtype.
            contribution = (expert_out * scale.unsqueeze(-1)).to(output.dtype)
            output.index_add_(0, token_idx, contribution)

        output = output.reshape(original_shape)
        if return_router_logits:
            return output, router_logits
        return output

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, top_k={self.top_k}"


def load_balancing_loss(
    router_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Switch-style load-balancing auxiliary loss.

    ``E * sum_e(fraction_e * mean_prob_e)``. Differentiable through ``probs``;
    the hard assignment counts are treated as constants.
    """
    probs = F.softmax(router_logits, dim=-1)
    num_experts = probs.size(-1)
    k = min(int(top_k), num_experts)
    topk_indices = torch.topk(probs, k=k, dim=-1).indices

    num_tokens = router_logits.size(0)
    counts = torch.bincount(
        topk_indices.reshape(-1), minlength=num_experts
    )
    frac = counts.to(probs.dtype) / float(max(num_tokens * k, 1))
    mean_probs = probs.mean(dim=0)
    return num_experts * (frac * mean_probs).sum()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss: mean of squared log-partition values."""
    return torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()


@torch.no_grad()
def router_stats(router_logits: torch.Tensor, top_k: int) -> dict:
    """
    Detached router diagnostics for logging.

    Returns a dict with:
        expert_counts: (E,) total top-k assignments per expert
        top1_counts:   (E,) first-choice assignments per expert
        top2_counts:   (E,) second-choice assignments (zeros if top_k < 2)
        router_entropy: scalar mean token entropy of the router distribution
        num_tokens:    number of routed tokens
    """
    probs = F.softmax(router_logits.float(), dim=-1)
    num_experts = probs.size(-1)
    k = min(int(top_k), num_experts)
    topk_indices = torch.topk(probs, k=k, dim=-1).indices

    expert_counts = torch.bincount(
        topk_indices.reshape(-1), minlength=num_experts
    )
    top1_counts = torch.bincount(topk_indices[:, 0], minlength=num_experts)
    if k >= 2:
        top2_counts = torch.bincount(topk_indices[:, 1], minlength=num_experts)
    else:
        top2_counts = torch.zeros(
            num_experts, dtype=torch.long, device=probs.device
        )

    entropy = -(
        probs.clamp_min(1e-9).log() * probs
    ).sum(dim=-1).mean()

    return {
        "expert_counts": expert_counts,
        "top1_counts": top1_counts,
        "top2_counts": top2_counts,
        "router_entropy": entropy,
        "num_tokens": int(probs.size(0)),
    }


def moe_aux_losses(
    router_logits_list: List[torch.Tensor],
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate load-balancing and z-losses over layers.

    Returns:
        (aux_loss, z_loss), each averaged over the supplied layers.

    Raises:
        ValueError: when no router logits are supplied. A configured loss
            term must never silently become zero (v1.6 plan §13).
    """
    if not router_logits_list:
        raise ValueError(
            "moe_aux_losses() requires at least one router-logits tensor"
        )

    aux_loss = torch.stack(
        [load_balancing_loss(logits, top_k) for logits in router_logits_list]
    ).mean()
    z_loss = torch.stack(
        [router_z_loss(logits) for logits in router_logits_list]
    ).mean()
    return aux_loss, z_loss
