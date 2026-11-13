"""
model/architecture.py — Dizel causal Transformer (v1.5, Qwen-style).

Architecture highlights (v1.5)
-----------------------------
* Grouped Query Attention (GQA) — reduces KV cache vs MHA
* Rotary Positional Embeddings (RoPE) — applied to Q and K in attention
* SwiGLU activation in feed-forward (gated SiLU, 3-projection)
* Pre-RMSNorm (applied before attention/MLP, not after)
* Weight tying between token embedding and LM head
* No biases by default
* Dropout for overfitting mitigation

Backward-compatible: accepts both ModelConfig (v1.2) and ModelConfigV15 (v1.5)
at init time. Detects GQA via hasattr(cfg, 'kv_heads').
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig, ModelConfigV15
from model.rope import RotaryPositionalEmbedding
from model.moe import MoEFFN, moe_aux_losses, router_stats

_SAFE_GLOBALS = [ModelConfig, ModelConfigV15]
try:
    from configs.v16_flash import ModelConfigV16Flash
except ImportError:
    pass
else:
    _SAFE_GLOBALS.append(ModelConfigV16Flash)

torch.serialization.add_safe_globals(_SAFE_GLOBALS)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(input_dtype)

    def extra_repr(self):
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------
class GQAAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.
    Falls back to MHA when kv_heads == n_heads.
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        n_heads = cfg.n_heads
        kv_heads = getattr(cfg, 'kv_heads', n_heads)
        head_dim = getattr(cfg, 'head_dim', d_model // n_heads)
        bias = cfg.bias
        dropout = cfg.dropout
        rope_base = getattr(cfg, 'rope_theta', getattr(cfg, 'rope_base', 10000.0))

        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.n_groups = n_heads // kv_heads if kv_heads > 0 else 1

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)

        self.rope = RotaryPositionalEmbedding(
            dim=head_dim,
            max_seq_len=cfg.context_length * 2,
            base=rope_base,
        )

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, seq_len=T)

        if self.kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLUMLP(nn.Module):
    """
    Gated SiLU feed-forward: down(silu(gate(x)) * up(x)).
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        intermediate = getattr(cfg, 'intermediate_size', None) or int(d_model * cfg.ffn_mult)
        bias = cfg.bias

        self.gate_proj = nn.Linear(d_model, intermediate, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, d_model, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    Pre-RMSNorm decoder layer: x + Attn(RMSNorm(x)), x + FFN(RMSNorm(x)).

    The FFN is a dropless top-k MoE block when ``cfg.num_experts > 1``;
    otherwise the dense SwiGLU MLP is used unchanged.
    """

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        eps = getattr(cfg, 'norm_eps', 1e-6)

        self.input_norm = RMSNorm(d_model, eps)
        self.attn = GQAAttention(cfg)
        self.post_attn_norm = RMSNorm(d_model, eps)
        if int(getattr(cfg, "num_experts", 1)) > 1:
            self.mlp = MoEFFN(cfg)
        else:
            self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, return_router_logits=False):
        x = x + self.attn(self.input_norm(x))
        if return_router_logits and isinstance(self.mlp, MoEFFN):
            mlp_out, router_logits = self.mlp(
                self.post_attn_norm(x), return_router_logits=True
            )
            return x + mlp_out, router_logits
        x = x + self.mlp(self.post_attn_norm(x))
        if return_router_logits:
            return x, None
        return x


# ---------------------------------------------------------------------------
# Dizel Language Model
# ---------------------------------------------------------------------------
class DizelLM(nn.Module):
    """
    Dizel: A causal language model (v1.5, Qwen-style architecture).

    Detects config type at init:
      - hasattr(cfg, 'kv_heads')  → v1.5 (GQA, SwiGLU, RMSNorm)
      - no kv_heads               → v1.2 (treated as GQA with kv_heads=n_heads)

    Forward pass
    ------------
    input:  idx   (B, T)   — integer token ids
    output: logits (B, T, vocab_size)
             loss  (scalar, optional)  — cross-entropy NLL

    Generation
    ----------
    Use DizelLM.generate() for autoregressive sampling.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict({
            "tok_emb": nn.Embedding(cfg.vocab_size, cfg.d_model),
            "emb_drop": nn.Dropout(cfg.dropout),
            "blocks": nn.ModuleList([
                DecoderLayer(cfg) for _ in range(cfg.n_layers)
            ]),
            "ln_f": RMSNorm(cfg.d_model, getattr(cfg, 'norm_eps', 1e-6)),
        })

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.lm_head.weight = self.transformer["tok_emb"].weight

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(
                    param, mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.n_layers)
                )

    # ------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if getattr(module, "_skip_weight_init", False):
                # MoE routers are initialized from cfg.router_init_std.
                return
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    @staticmethod
    def _cross_entropy(logits, targets, loss_mask=None):
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="none",
        )

        if loss_mask is not None:
            mask = loss_mask.view(-1).float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    # ------------------------------------------------------------------
    def forward(
        self,
        idx,
        targets=None,
        loss_mask=None,
        return_aux=False,
    ):
        B, T = idx.shape
        assert T <= self.cfg.context_length, \
            f"Input length {T} exceeds context_length {self.cfg.context_length}"

        blocks = self.transformer["blocks"]
        moe_blocks = []
        if return_aux:
            moe_blocks = [b for b in blocks if isinstance(b.mlp, MoEFFN)]
            if not moe_blocks:
                raise RuntimeError(
                    "return_aux=True requires at least one MoE layer "
                    "(set cfg.num_experts > 1); this model is dense."
                )

        x = self.transformer["tok_emb"](idx)
        x = self.transformer["emb_drop"](x)

        router_logits_list = []
        for block in blocks:
            if return_aux:
                x, block_router_logits = block(
                    x, return_router_logits=True
                )
                if block_router_logits is not None:
                    router_logits_list.append(block_router_logits)
            else:
                x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        if not return_aux:
            if targets is None:
                return logits, None
            return logits, self._cross_entropy(logits, targets, loss_mask)

        ce_loss = None
        if targets is not None:
            ce_loss = self._cross_entropy(logits, targets, loss_mask)

        top_k = int(moe_blocks[0].mlp.top_k)
        aux_loss, z_loss = moe_aux_losses(router_logits_list, top_k)
        layer_stats = [
            router_stats(rl, top_k) for rl in router_logits_list
        ]

        expert_counts = torch.stack(
            [s["expert_counts"] for s in layer_stats]
        )
        top1_counts = torch.stack([s["top1_counts"] for s in layer_stats])
        top2_counts = torch.stack([s["top2_counts"] for s in layer_stats])
        router_entropy = torch.stack(
            [s["router_entropy"] for s in layer_stats]
        )
        num_tokens = int(layer_stats[0]["num_tokens"])

        if ce_loss is None:
            total_loss = None
        else:
            total_loss = (
                ce_loss
                + float(getattr(self.cfg, "router_aux_loss_coef", 0.01)) * aux_loss
                + float(getattr(self.cfg, "router_z_loss_coef", 0.001)) * z_loss
            )

        aux = {
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "total_loss": total_loss,
            "expert_counts": expert_counts,
            "top1_counts": top1_counts,
            "top2_counts": top2_counts,
            "router_entropy": router_entropy,
            "num_tokens": num_tokens,
        }

        return logits, ce_loss, aux

    # ------------------------------------------------------------------
    def enable_gradient_checkpointing(self):
        """
        Recompute each decoder block during backward to trade compute for
        activation memory. Idempotent: calling it twice does not wrap twice.
        """
        from torch.utils.checkpoint import checkpoint as _checkpoint

        for block in self.transformer["blocks"]:
            if getattr(block, "_gradient_checkpointing_enabled", False):
                continue
            original_forward = block.forward

            def checkpointed_forward(
                x,
                return_router_logits=False,
                _forward=original_forward,
            ):
                if not torch.is_grad_enabled():
                    return _forward(x, return_router_logits)
                return _checkpoint(
                    _forward,
                    x,
                    return_router_logits,
                    use_reentrant=False,
                )

            block.forward = checkpointed_forward
            block._gradient_checkpointing_enabled = True

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=0.8,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.15,
        eos_id=2,
        eos_ids=None,
    ):
        self.eval()
        generated = idx

        stop_ids = set()
        if eos_ids is not None:
            stop_ids.update(eos_ids)
        else:
            stop_ids.add(eos_id)

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.cfg.context_length:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                probs_sorted, sorted_idx = torch.sort(
                    F.softmax(logits, dim=-1), dim=-1, descending=True
                )
                cum_probs = probs_sorted.cumsum(dim=-1)
                remove = cum_probs - probs_sorted > top_p
                probs_sorted[remove] = 0.0
                probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs_sorted, num_samples=1)
                next_token = sorted_idx.gather(-1, next_token)
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in stop_ids:
                break

        return generated

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only=True):
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    def __repr__(self):
        n = self.num_parameters()
        kv = getattr(self.cfg, 'kv_heads', self.cfg.n_heads)
        num_experts = int(getattr(self.cfg, 'num_experts', 1))
        moe = ""
        if num_experts > 1:
            moe = (
                f"experts={num_experts}/"
                f"top{getattr(self.cfg, 'top_k', 1)}, "
            )
        return (
            f"DizelLM("
            f"vocab={self.cfg.vocab_size}, "
            f"d_model={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}/kv={kv}, "
            f"{moe}"
            f"params={n/1e6:.2f}M)"
        )


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import ModelConfig, ModelConfigV15

    for name, cfg_cls in [("v1.2", ModelConfig), ("v1.5", ModelConfigV15)]:
        cfg = cfg_cls()
        model = DizelLM(cfg)
        print(f"\n{model}")

        B, T = 2, 64
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(idx, targets)
        print(f"  logits: {logits.shape}, loss: {loss.item():.4f}")

        prompt = torch.zeros(1, 1, dtype=torch.long)
        out = model.generate(prompt, max_new_tokens=10)
        print(f"  generated: {out.shape}")
