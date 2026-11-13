"""
DSKD-lite: cross-vocabulary knowledge distillation core library.

Aligns Dizel (student, SentencePiece vocab=100000) and Qwen (teacher,
HF vocab=151936) tokenizations lexically, projects teacher hidden states
into student space with a small learned linear projector, and computes
KL distillation at aligned response positions in student vocab space.

Exports (as required by the runner):
    compute_alignment, AlignmentCache, VocabProjector,
    cross_vocab_distill_loss, KDMetricsLogger
"""
import os
import sys

# Make the repo root importable when this package is loaded from anywhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .alignment import (
    AlignmentCache,
    compute_alignment,
    per_token_strings,
    tokenize_student,
    tokenize_teacher,
)
from .kd_loss import (
    KD_SILENCE_LIMIT,
    VOCAB_MISMATCH_RAISE,
    assert_kd_not_silently_disabled,
    cross_vocab_distill_loss,
    masked_ce_loss,
    reset_kd_guard,
)
from .metrics_logger import KDMetricsLogger
from .projector import VocabProjector, project_to_student_vocab

__all__ = [
    "compute_alignment",
    "AlignmentCache",
    "VocabProjector",
    "cross_vocab_distill_loss",
    "KDMetricsLogger",
]