"""
DSKD-lite distillation loss.

L = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(teacher_probs || student_probs)

with teacher logits produced IN STUDENT VOCAB SPACE by projecting teacher
hidden states through the student lm_head weight.  KL is applied only at
lexically aligned, loss-masked response positions.
"""
import torch
import torch.nn.functional as F

from .projector import project_to_student_vocab

# ---------------------------------------------------------------------------
# Fail-closed guard for silently-disabled KD
# ---------------------------------------------------------------------------

# Raise after more than KD_SILENCE_LIMIT consecutive calls where KD was
# requested (kl_weight > 0, mode == "align") but produced no alignment.
KD_SILENCE_LIMIT = 5

# If True, any V_t != V_s mismatch during loss construction raises instead of
# silently disabling KD.  Kept module-level so runners can toggle it.
VOCAB_MISMATCH_RAISE = False

_silence_counters: dict = {}  # run_id -> consecutive silent calls


def reset_kd_guard(run_id: str = "default") -> None:
    """Reset the fail-closed silence counter for ``run_id``."""
    _silence_counters.pop(run_id, None)


def assert_kd_not_silently_disabled(metrics, kl_weight, mode, step, run_id: str = "default") -> None:
    """Raise RuntimeError if KD has been silently disabled for too long.

    A "silent call" is one where ``kl_weight > 0``, ``mode == "align"`` and the
    returned metrics report ``active is False`` with ``aligned_count == 0``.
    After more than ``KD_SILENCE_LIMIT`` consecutive silent calls this raises,
    so a broken alignment pipeline fails loudly instead of training pure-CE.
    """
    if (
        kl_weight > 0
        and mode == "align"
        and metrics.get("active") is False
        and metrics.get("aligned_count", 0) == 0
    ):
        _silence_counters[run_id] = _silence_counters.get(run_id, 0) + 1
    else:
        _silence_counters[run_id] = 0

    if _silence_counters[run_id] > KD_SILENCE_LIMIT:
        raise RuntimeError(
            f"DSKD-lite: KD silently disabled for {_silence_counters[run_id]} consecutive "
            f"calls (kl_weight={kl_weight}, mode={mode}, active={metrics.get('active')}, "
            f"aligned_count={metrics.get('aligned_count')}). No aligned response positions "
            f"were found -- check tokenizer alignment or call reset_kd_guard()."
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_ce_loss(student_logits, labels, loss_mask, ignore_index=-1) -> torch.Tensor:
    """Masked cross-entropy, same formulation as distill.py:543-550 (chunked).

    Chunks the flat sequence dim so torch's fp32 softmax transient never spans
    the full (B*T, V_s) tensor (~1.6GB at V_s=100000). 512 rows/chunk keeps the
    peak near ~200MB; softmax is per-row so numerics are identical.
    """
    B, T, V_s = student_logits.shape
    flat_logits = student_logits.reshape(-1, V_s)
    flat_labels = labels.reshape(-1)
    _CE_CHUNK = 512
    parts = [
        F.cross_entropy(
            flat_logits[s:s + _CE_CHUNK],
            flat_labels[s:s + _CE_CHUNK],
            ignore_index=ignore_index,
            reduction="none",
        )
        for s in range(0, flat_logits.shape[0], _CE_CHUNK)
    ]
    ce_flat = torch.cat(parts) if len(parts) > 1 else parts[0]
    ce_loss = (ce_flat.view(B, T) * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    return ce_loss


def cross_vocab_distill_loss(
    student_logits,
    student_hidden,
    teacher_hidden,
    projector,
    lm_head_weight,
    labels,
    loss_mask,
    aligned_pairs_list,
    kl_weight: float = 0.5,
    temperature: float = 2.0,
    batch_ids=None,
) -> dict:
    """Combined CE + cross-vocabulary KL distillation loss.

    Args:
        student_logits: (B, T, V_s) student logits.
        student_hidden: (B, T, d_s) post-final-norm student hidden (unused by
            the loss itself; accepted for runner API compatibility).
        teacher_hidden: (B, T_t, d_t) teacher final hidden states.
        projector: VocabProjector mapping teacher dim -> student dim.
        lm_head_weight: (V_s, d_s) student lm_head weight (tied embedding).
        labels: (B, T) hard labels with -1 padding.
        loss_mask: (B, T) 0/1 mask; 1 marks countable response positions.
        aligned_pairs_list: list per batch sample of [(i_student, j_teacher)].
        kl_weight: mixing coefficient alpha.
        temperature: distillation temperature T.
        batch_ids: optional sequence ids (reserved for runner bookkeeping).

    Returns a metrics dict with 'loss', 'ce_loss', 'kd_loss', 'active',
    'aligned_count', 'kl_weight', 'temperature'.
    """
    del student_hidden, batch_ids  # not used in the loss itself

    B, T, V_s = student_logits.shape
    T_t = teacher_hidden.shape[1]
    device = student_logits.device

    if VOCAB_MISMATCH_RAISE and V_s != lm_head_weight.shape[0]:
        raise RuntimeError(
            f"DSKD-lite: student vocab {V_s} != lm_head weight rows "
            f"{lm_head_weight.shape[0]}"
        )

    ce_loss = masked_ce_loss(student_logits, labels, loss_mask)

    # Teacher logits in STUDENT vocab space: proj(teacher_hidden) @ lm_head_w.T
    teacher_logits_s = project_to_student_vocab(projector(teacher_hidden), lm_head_weight)

    total_kd = torch.zeros((), device=device, dtype=student_logits.dtype)
    count = 0
    for b in range(B):
        pairs = aligned_pairs_list[b] if b < len(aligned_pairs_list) else []
        for i_student, j_teacher in pairs:
            if not (0 <= i_student < T and 0 <= j_teacher < T_t):
                continue
            if int(loss_mask[b, i_student].item()) != 1:
                continue
            kl = (temperature ** 2) * F.kl_div(
                F.log_softmax(student_logits[b, i_student] / temperature, dim=-1),
                F.softmax(teacher_logits_s[b, j_teacher] / temperature, dim=-1),
                reduction="sum",
            )
            total_kd = total_kd + kl
            count += 1

    if count > 0:
        kd = total_kd / count
        active = True
    else:
        kd = torch.zeros((), device=device, dtype=student_logits.dtype)
        active = False

    total = (1.0 - kl_weight) * ce_loss + kl_weight * kd

    return {
        "loss": total,
        "ce_loss": ce_loss.detach(),
        "kd_loss": kd.detach(),
        "active": active,
        "aligned_count": count,
        "kl_weight": kl_weight,
        "temperature": temperature,
    }