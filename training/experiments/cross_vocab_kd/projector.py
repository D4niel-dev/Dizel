"""
Small learned linear projector from teacher hidden space (896-dim) into
student hidden space (1024-dim), plus the student-vocab logit projection.

By pushing the projected teacher hidden state through the STUDENT's lm_head
weight, teacher logits are produced directly in student vocab space, making
V_t == V_s by construction (no cross-vocab softmax needed).
"""
import math

import torch
import torch.nn as nn


class VocabProjector(nn.Module):
    """Linear projection: teacher hidden dim -> student hidden dim."""

    def __init__(self, teacher_dim: int = 896, student_dim: int = 1024):
        super().__init__()
        self.teacher_dim = int(teacher_dim)
        self.student_dim = int(student_dim)
        self.proj = nn.Linear(self.teacher_dim, self.student_dim, bias=False)
        # Match the default nn.Linear weight init scheme.
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        assert h_t.shape[-1] == self.teacher_dim, (
            f"VocabProjector: teacher hidden dim {h_t.shape[-1]} "
            f"!= teacher_dim {self.teacher_dim}"
        )
        return self.proj(h_t)


def project_to_student_vocab(
    hidden_s: torch.Tensor, lm_head_weight: torch.Tensor
) -> torch.Tensor:
    """hidden (B, T, d_s) @ lm_head_weight.T (d_s, V_s) -> logits (B, T, V_s)."""
    return hidden_s @ lm_head_weight.T