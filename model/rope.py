"""
model/rope.py — Rotary Positional Embedding (RoPE) for Dizel v1.2.

RoPE (Su et al., 2021) encodes position information by rotating Q and K
vectors in 2D subspaces at frequencies determined by dimension index.

Key properties:
  - No learnable parameters (purely geometric)
  - Relative position awareness (attention depends on q-k distance)
  - Naturally supports extrapolation to longer sequences
  - Applied to Q and K only (not V)

Reference: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dimension in half and rotate: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding.

    Precomputes sin/cos frequency tables up to `max_seq_len` and applies
    rotary transformations to query and key tensors.

    Args:
        dim:         Head dimension (must be even).
        max_seq_len: Maximum sequence length to cache. Can exceed context_length
                     for future extrapolation.
        base:        Frequency base (default 10000.0, standard from the paper).
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build or extend the sin/cos cache."""
        if seq_len <= self._cached_seq_len:
            return

        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)              # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (seq_len, dim)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self._cached_seq_len = seq_len

    def forward(
        self,
        q: torch.Tensor,   # (B, n_heads, T, head_dim)
        k: torch.Tensor,   # (B, n_heads, T, head_dim)
        seq_len: int,
    ):
        """
        Apply rotary embeddings to q and k.

        Returns:
            (q_rotated, k_rotated) with the same shapes as input.
        """
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cache[:seq_len]  # (T, head_dim)
        sin = self.sin_cache[:seq_len]  # (T, head_dim)

        # Broadcast: (T, head_dim) → works with (B, H, T, head_dim) via broadcasting
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)

        return q_rot, k_rot


if __name__ == "__main__":
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=4096)
    q = torch.randn(2, 16, 128, 64)
    k = torch.randn(2, 16, 128, 64)
    q_r, k_r = rope(q, k, seq_len=128)
    assert q_r.shape == q.shape, f"Shape mismatch: {q_r.shape} vs {q.shape}"
    assert k_r.shape == k.shape, f"Shape mismatch: {k_r.shape} vs {k.shape}"
    print(f"RoPE: OK — q_rot={q_r.shape}, k_rot={k_r.shape}")
    print(f"Cache size: {rope._cached_seq_len} positions")
