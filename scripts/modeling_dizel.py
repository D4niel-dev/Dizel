# -*- coding: utf-8 -*-
"""
modeling_dizel.py -- Dizel v1.5 custom HF remote code (GQA + SwiGLU + RMSNorm + RoPE).
────────────────────────────────────────────────────────────────────────────────────────
Dizel v1.5 -- a custom causal Transformer with:

  * Grouped Query Attention (GQA) with Rotary Positional Embeddings (RoPE)
  * SwiGLU gated feed-forward (gate_proj / up_proj / down_proj)
  * Pre-RMSNorm normalization (input_norm / post_attn_norm / ln_f)
  * Weight tying between the token embedding and the LM head (optional)

This module is *self-contained*: when ``AutoModelForCausalLM.from_pretrained(
..., trust_remote_code=True)`` loads it from an exported directory, it runs in
isolation - it must not import anything from the Dizel repository.  The module
and parameter names below match ``DizelLM`` in ``model/architecture.py`` 1:1 so
the exported ``model.safetensors`` maps directly onto the checkpoint weights:
``transformer.tok_emb.weight``, ``transformer.blocks.{i}.input_norm.weight``,
``attn.q_proj/k_proj/v_proj/o_proj.weight``, ``post_attn_norm.weight``,
``mlp.gate_proj/up_proj/down_proj.weight``, ``transformer.ln_f.weight`` and
``lm_head.weight``.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    # Relative import: transformers' dynamic-module loader resolves this by
    # copying configuration_dizel.py next to this file and creating an
    # __init__.py package in the module cache (see dynamic_module_utils).
    # The absolute fallback below is for direct in-repo imports only; it is
    # inside the try/except so transformers' static import scanner skips it.
    from .configuration_dizel import DizelConfig
except ImportError:
    from configuration_dizel import DizelConfig


# ===========================================================================
# RMSNorm (mirror of model/architecture.py::RMSNorm)
# ===========================================================================
class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no learnable affine bias)."""

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


# ============================================================================
# RoPE -- minimal rotary positional embedding (mirror of model/rope.py)
# ============================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dimension in half and rotate: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Pre-computed sin/cos rotary position tables with lazy on-demand
    rebuilding (non-``persistent`` buffers so no state_dict keys are added)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        device = device or self.inv_freq.device
        if seq_len <= self._cached_seq_len and self.cos_cache.device == device:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))    # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, q, k, seq_len: int):
        """
        Apply rotary embeddings to q and k.

        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_heads, T, head_dim)
        Returns:
            (q_rot, k_rot)
        """
        self._build_cache(seq_len, device=q.device)
        cos = self.cos_cache[:seq_len]  # (T, head_dim)
        sin = self.sin_cache[:seq_len]  # (T, head_dim)
        q_rot = (q.float() * cos) + (_rotate_half(q.float()) * sin)
        k_rot = (k.float() * cos) + (_rotate_half(k.float()) * sin)
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


# ============================================================================
# Grouped Query Attention (mirror of model/architecture.py::GQAAttention)
# ============================================================================
class GQAAttention(nn.Module):
    """Grouped Query Attention with RoPE. Falls back to MHA when kv == n_heads."""

    def __init__(self, cfg: "DizelConfig"):
        super().__init__()
        d_model = cfg.d_model
        n_heads = cfg.n_heads
        kv_heads = getattr(cfg, "kv_heads", n_heads)
        head_dim = getattr(cfg, "head_dim", d_model // n_heads)
        bias = cfg.bias
        dropout = cfg.dropout
        rope_base = getattr(cfg, "rope_theta", 10000.0)
        max_len = getattr(cfg, "context_length", 2048) * 2

        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.n_groups = n_heads // kv_heads if kv_heads > 0 else 1

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)

        self.rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=max_len, base=rope_base)

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


# ============================================================================
# SwiGLU MLP (mirror of model/architecture.py::SwiGLUMLP)
# ============================================================================
class SwiGLUMLP(nn.Module):
    """Gated SiLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: "DizelConfig"):
        super().__init__()
        d_model = cfg.d_model
        intermediate = cfg.intermediate_size
        bias = cfg.bias

        self.gate_proj = nn.Linear(d_model, intermediate, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, d_model, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Decoder Layer (mirror of model/architecture.py::DecoderLayer)
# ============================================================================
class DecoderLayer(nn.Module):
    """Pre-RMSNorm decoder layer: x + Attn(RMSNorm(x)), x + SwiGLU(RMSNorm(x))."""

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        eps = getattr(cfg, "norm_eps", 1e-6)

        self.input_norm = RMSNorm(d_model, eps)
        self.attn = GQAAttention(cfg)
        self.post_attn_norm = RMSNorm(d_model, eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


# ============================================================================
# Shared module builder + weight init (mirrors model/architecture.py::DizelLM)
# ============================================================================
def _build_transformer(config: "DizelConfig") -> nn.ModuleDict:
    """The v1.5 transformer body: ModuleDict with keys tok_emb/emb_drop/
    blocks/ln_f -- EXACTLY the key names found in Dizel checkpoints."""
    return nn.ModuleDict({
        "tok_emb": nn.Embedding(config.vocab_size, config.d_model),
        "emb_drop": nn.Dropout(config.dropout),
        "blocks": nn.ModuleList([
            DecoderLayer(config) for _ in range(config.n_layers)
        ]),
        "ln_f": RMSNorm(config.d_model, getattr(config, "norm_eps", 1e-6)),
    })


def _init_weights_fn(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def _apply_residual_scaling(model: nn.Module, n_layers: int):
    """o_proj / down_proj get 0.02/sqrt(2*n_layers) scaling (architecture.py)."""
    for name, param in model.named_parameters():
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))


def _hidden_forward(model, idx: torch.Tensor) -> torch.Tensor:
    """Run the v1.5 transformer body: tok_emb -> emb_drop -> blocks -> ln_f."""
    B, T = idx.shape
    if T > model.config.context_length:
        raise ValueError(
            f"Input length {T} exceeds context_length {model.config.context_length}"
        )
    x = model.transformer["tok_emb"](idx)
    x = model.transformer["emb_drop"](x)
    for block in model.transformer["blocks"]:
        x = block(x)
    x = model.transformer["ln_f"](x)
    return x


# ============================================================================
# Dizel LM body (PreTrainedModel wrapper; transformer keys at top level)
# ============================================================================
class DizelLMModel(PreTrainedModel):
    """The transformer body of the Dizel model (no output head).

    The ModuleDict key names match the checkpoint keys exactly:
    ``transformer.tok_emb`` / ``transformer.emb_drop`` /
    ``transformer.blocks`` / ``transformer.ln_f``.
    """

    config_class = DizelConfig
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, config: DizelConfig):
        super().__init__(config)
        self.cfg = config
        self.transformer = _build_transformer(config)
        self.apply(self._init_weights)
        _apply_residual_scaling(self, config.n_layers)

    def _init_weights(self, module):
        _init_weights_fn(module)

    def forward(self, idx):
        return _hidden_forward(self, idx)


# ============================================================================
# Causal LM head (HF-facing: AutoModelForCausalLM target)
# ============================================================================
class DizelLMForCausalLM(PreTrainedModel):
    """Dizel v1.5 with a (tied) LM head -- the class ``transformers`` instantiates
    via ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.

    ``self.transformer`` sits at the TOP level (exactly like the repo's
    ``DizelLM``) so the checkpoint keys ``transformer.*`` + ``lm_head.weight``
    map 1:1 with NO renaming.
    """

    config_class = DizelConfig
    _no_split_modules = ["DecoderLayer"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, config: DizelConfig):
        super().__init__(config)
        self.cfg = config

        self.transformer = _build_transformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        _apply_residual_scaling(self, config.n_layers)

        if config.weight_tying:
            self.tie_weights()
        self.post_init()

    def _init_weights(self, module):
        _init_weights_fn(module)

    # -- embeddings -------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.transformer["tok_emb"]

    def set_input_embeddings(self, value: nn.Module):
        self.transformer["tok_emb"] = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module):
        self.lm_head = value

    def tie_weights(self):
        """Enforce weight tying when config.weight_tying is True (v1.5 default)."""
        if getattr(self.config, "weight_tying", True):
            self.lm_head.weight = self.transformer["tok_emb"].weight
        super().tie_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise NotImplementedError(
                "Dizel remote code requires token ids (`input_ids`); "
                "`inputs_embeds` is not supported."
            )
        if input_ids is None:
            raise ValueError("`input_ids` is required for a forward pass.")

        hidden = _hidden_forward(self, input_ids)          # (B, T, d_model)
        logits = self.lm_head(hidden)                      # (B, T, vocab)

        loss = None
        if labels is not None:
            # Standard causal-LM cross-entropy (ignore_index=-1 padding convention).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
        do_sample: Optional[bool] = None,
        streamer=None,
        **kwargs,
    ):
        """Autoregressive sampling mirroring ``DizelLM.generate``.

        The GUI calls ``self._model.generate(inputs=..., do_sample=...,
        temperature=... top_k=... top_p=... repetition_penalty=...,
        streamer=...)`` directly -- this signature accepts those kwargs.
        """
        # --- Resolve prompt tensors (accept `inputs` from the GUI) ---------
        if input_ids is None:
            input_ids = inputs
        if hasattr(input_ids, "input_ids"):          # BatchEncoding / dict-like
            input_ids = input_ids.input_ids
        elif isinstance(input_ids, dict):
            input_ids = input_ids.get("input_ids")
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids is None:
            raise ValueError("generate() requires `inputs` (or `input_ids`).")

        device = self.transformer["tok_emb"].weight.device
        self.eval()
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, _ = input_ids.shape
        if B != 1:
            raise ValueError(
                "Dizel.generate() expects a single sequence (batch size 1)"
            )

        max_seq = self.config.context_length
        generated = input_ids
        stop_ids = set()
        eos_candidates = kwargs.pop("eos_id", None) or getattr(self.config, "eos_id", 2)
        if isinstance(eos_candidates, (list, tuple)):
            stop_ids.update(int(x) for x in eos_candidates)
        else:
            stop_ids.add(int(eos_candidates))

        # Mark the prompt for `skip_prompt` streamers
        if streamer is not None:
            streamer.put(input_ids)

        for _step in range(int(max_new_tokens)):
            ctx = generated[:, -max_seq:]

            logits = self(input_ids=ctx).logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            logits = logits / max(float(temperature), 1e-8)

            do_sample = do_sample if do_sample is not None else (float(temperature) > 0.0)

            if not do_sample:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p < 1.0:
                    probs_sorted, sorted_idx = torch.sort(
                        F.softmax(logits, dim=-1), dim=-1, descending=True
                    )
                    cum_probs = probs_sorted.cumsum(dim=-1)
                    remove_mask = cum_probs - probs_sorted > top_p
                    probs_sorted = probs_sorted.masked_fill(remove_mask, 0.0)
                    probs_sorted = probs_sorted / probs_sorted.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs_sorted, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_token)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if streamer is not None:
                streamer.put(next_token)

            if int(next_token.item()) in stop_ids:
                break

        if streamer is not None:
            streamer.end()

        return generated

    # ------------------------------------------------------------------
    def can_generate(self) -> bool:
        """Mark ours as a generator so HF never warns on gen-era utilities."""
        return True


__all__ = ["DizelConfig", "DizelLMModel", "DizelLMForCausalLM"]