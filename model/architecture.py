"""
model/architecture.py — Dizel causal Transformer (GPT-style).

Architecture highlights
-----------------------
* Learned positional embeddings (simple, stable for short contexts)
* Pre-LayerNorm (applied before attention/MLP, not after)
* Causal self-attention with optional flash-attention fallback
* GELU-activated MLP (two linear layers + gelu)
* Weight tying between token embedding and LM head
* No biases by default (reduces parameter count, marginally better)
* Dropout for overfitting mitigation on small datasets

Parameter budget (defaults from config.py)
-------------------------------------------
  vocab=8000, d_model=384, n_layers=6, n_heads=6
  ≈ 20 M parameters  →  fits in ~400 MB GPU RAM (fp32) / ~200 MB (fp16)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ModelConfig


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Uses PyTorch's scaled_dot_product_attention when available (PyTorch >= 2.0),
    which automatically dispatches to Flash Attention on supported hardware,
    giving a significant speed and memory improvement.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, \
            "d_model must be divisible by n_heads"

        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model  = cfg.d_model
        self.dropout  = cfg.dropout

        # Fused QKV projection (3× more memory-efficient than three separate layers)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        # Output projection
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        # Dropout after attention softmax and after output projection
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Detect whether scaled_dot_product_attention is available
        self._use_sdpa = hasattr(F, "scaled_dot_product_attention")
        if not self._use_sdpa:
            # Fall back to manual mask (pre-PyTorch 2.0)
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(cfg.context_length, cfg.context_length))
                     .view(1, 1, cfg.context_length, cfg.context_length),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, sequence length, d_model

        # ── QKV ──────────────────────────────────────────────────────────
        qkv = self.qkv(x)                                   # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=2)           # each (B, T, C)

        # Reshape to (B, n_heads, T, head_dim) for attention
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # ── Attention ─────────────────────────────────────────────────────
        if self._use_sdpa:
            # Flash / efficient attention (PyTorch >= 2.0).
            # is_causal=True applies the lower-triangular mask internally.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual scaled dot-product attention + causal mask
            scale = 1.0 / math.sqrt(self.head_dim)
            attn  = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
            attn  = attn.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            attn  = F.softmax(attn, dim=-1)
            attn  = self.attn_drop(attn)
            y     = attn @ v                                    # (B, H, T, d_head)

        # Re-assemble all head outputs → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------------
# MLP (Position-wise Feed-Forward)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    Two-layer MLP with GELU activation.
    Standard Transformer FFN: Linear → GELU → Linear → Dropout
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1   = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=cfg.bias)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=cfg.bias)
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block with Pre-LayerNorm:

        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Pre-LN (applied *before* sub-layers) trains more stably than Post-LN
    when using Adam without a long warm-up phase.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Dizel GPT-style Language Model
# ---------------------------------------------------------------------------
class DizelLM(nn.Module):
    """
    Dizel: A tiny causal language model (~20 M parameters).

    Forward pass
    ------------
    input:  idx   (B, T)   — integer token ids
    output: logits (B, T, vocab_size)
             loss  (scalar, optional)  — cross-entropy NLL

    Generation
    ----------
    Use DizelLM.generate() for autoregressive sampling.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict({
            # Token embedding table
            "tok_emb": nn.Embedding(cfg.vocab_size, cfg.d_model),
            # Learnable positional embedding (one vector per position)
            "pos_emb": nn.Embedding(cfg.context_length, cfg.d_model),
            # Input dropout (applied to embedding sum)
            "emb_drop": nn.Dropout(cfg.dropout),
            # Stack of transformer blocks
            "blocks": nn.ModuleList([
                TransformerBlock(cfg) for _ in range(cfg.n_layers)
            ]),
            # Final LayerNorm before the LM head
            "ln_f": nn.LayerNorm(cfg.d_model),
        })

        # Language model head: maps d_model → vocab_size
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share the token embedding weight with the LM head.
        # This is a common trick that reduces parameter count and often
        # improves perplexity on small models.
        if cfg.weight_tying:
            self.lm_head.weight = self.transformer["tok_emb"].weight

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 paper recommendation)
        for name, param in self.named_parameters():
            if name.endswith(("proj.weight", "fc2.weight")):
                nn.init.normal_(
                    param, mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.n_layers)
                )

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,                    # (B, T)
        targets: Optional[torch.Tensor] = None,  # (B, T)
        loss_mask: Optional[torch.Tensor] = None, # (B, T) 1=compute loss, 0=ignore
    ):
        B, T = idx.shape
        assert T <= self.cfg.context_length, \
            f"Input length {T} exceeds context_length {self.cfg.context_length}"

        device = idx.device

        # Token + positional embeddings
        tok = self.transformer["tok_emb"](idx)                   # (B, T, d_model)
        pos = self.transformer["pos_emb"](
            torch.arange(T, device=device)
        )                                                         # (T, d_model)
        x = self.transformer["emb_drop"](tok + pos)

        # Transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        # Final normalisation + LM head
        x = self.transformer["ln_f"](x)                          # (B, T, d_model)
        logits = self.lm_head(x)                                  # (B, T, vocab)

        if targets is None:
            return logits, None

        # Cross-entropy loss
        # Flatten to (B*T, vocab) vs (B*T,)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,       # -1 used for padding positions
            reduction="none",
        )                                                         # (B*T,)

        if loss_mask is not None:
            # SFT: only count loss on assistant tokens
            mask = loss_mask.view(-1).float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,          # (1, T) — prompt token ids
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
        eos_id: int = 2,
        eos_ids: list = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k + nucleus (top-p) sampling
        and repetition penalty.

        Returns tensor of shape (1, T + generated_tokens).
        """
        self.eval()
        generated = idx

        # Build the set of EOS token ids to stop on
        stop_ids = set()
        if eos_ids is not None:
            stop_ids.update(eos_ids)
        else:
            stop_ids.add(eos_id)

        for _ in range(max_new_tokens):
            # Truncate context to context_length
            ctx = generated[:, -self.cfg.context_length:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()       # (1, vocab)

            # ── Repetition penalty ────────────────────────────────────
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # ── Temperature ───────────────────────────────────────────
            logits = logits / max(temperature, 1e-8)

            # ── Top-k ─────────────────────────────────────────────────
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # ── Top-p (nucleus) ───────────────────────────────────────
            if top_p < 1.0:
                probs_sorted, sorted_idx = torch.sort(
                    F.softmax(logits, dim=-1), dim=-1, descending=True
                )
                cum_probs = probs_sorted.cumsum(dim=-1)
                # Remove tokens with cumulative probability above top_p
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
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"DizelLM("
            f"vocab={self.cfg.vocab_size}, "
            f"d_model={self.cfg.d_model}, "
            f"layers={self.cfg.n_layers}, "
            f"heads={self.cfg.n_heads}, "
            f"params={n/1e6:.2f}M)"
        )


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import ModelConfig

    cfg = ModelConfig(vocab_size=8000, d_model=384, n_layers=6, n_heads=6)
    model = DizelLM(cfg)
    print(model)

    # Forward pass test
    B, T = 2, 64
    idx     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    print(f"logits shape : {logits.shape}")
    print(f"loss         : {loss.item():.4f}  (expect ~ln({cfg.vocab_size:.0f}) ≈ {math.log(cfg.vocab_size):.2f})")

    # Generation test
    prompt = torch.zeros(1, 1, dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=20)
    print(f"generated    : {out.shape}  → {out[0].tolist()}")
