"""
Pure unit tests for the DSKD-lite cross-vocabulary KD core library.

Run directly:   python training/experiments/cross_vocab_kd/test_cross_vocab_kd.py
Via pytest:     python -m pytest training/experiments/cross_vocab_kd/test_cross_vocab_kd.py -v

Uses only unittest + small random tensors (no real models are loaded).
"""
import hashlib
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from training.experiments.cross_vocab_kd.alignment import AlignmentCache, compute_alignment
from training.experiments.cross_vocab_kd.kd_loss import (
    KD_SILENCE_LIMIT,
    assert_kd_not_silently_disabled,
    cross_vocab_distill_loss,
    reset_kd_guard,
)
from training.experiments.cross_vocab_kd.projector import VocabProjector


# ---------------------------------------------------------------------------
# Toy tokenizers (duck-typed)
# ---------------------------------------------------------------------------


class FakeStudentTokenizer:
    """SentencePiece-like student tokenizer: .encode, .sp, .tokenize, vocab."""

    def __init__(self, id_to_str, special_ids=None):
        self.id_to_str = dict(id_to_str)
        self.vocab = {s: i for i, s in id_to_str.items()}
        special_ids = special_ids or {}
        self.pad_id = special_ids.get("pad")
        self.bos_id = special_ids.get("bos")
        self.eos_id = special_ids.get("eos")
        self.unk_id = special_ids.get("unk")
        self.tok_id_to_string = self.id_to_str

        class _SP:
            def __init__(self, owner):
                self.owner = owner
                self.pad_id = owner.pad_id
                self.bos_id = owner.bos_id
                self.eos_id = owner.eos_id
                self.unk_id = owner.unk_id

            def decode(self, ids):
                return "".join(self.owner.id_to_str.get(int(i), "") for i in ids)

        self.sp = _SP(self)

    def _encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for ln in range(min(8, len(text) - i), 0, -1):
                piece = text[i : i + ln]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    i += ln
                    matched = True
                    break
            if not matched:
                ids.append(self.unk_id if self.unk_id is not None else 0)
                i += 1
        return ids

    def encode(self, text):
        return self._encode(text)

    def tokenize(self, text):
        return [self.id_to_str[i] for i in self._encode(text)]

    def decode(self, ids, skip_special_tokens=False):
        out = []
        for i in ids:
            i = int(i)
            s = self.id_to_str.get(i, "")
            if skip_special_tokens and i in (self.pad_id, self.bos_id, self.eos_id, self.unk_id):
                continue
            out.append(s)
        return "".join(out)


class FakeTeacherTokenizer:
    """HF-like teacher tokenizer: __call__ -> {"input_ids": ...}, .decode."""

    def __init__(self, id_to_str):
        self.id_to_str = dict(id_to_str)
        self.vocab = {s: i for i, s in id_to_str.items()}
        self.tok_id_to_string = self.id_to_str

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}

    def _encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for ln in range(min(8, len(text) - i), 0, -1):
                piece = text[i : i + ln]
                if piece in self.vocab:
                    ids.append(self.vocab[piece])
                    i += ln
                    matched = True
                    break
            if not matched:
                ids.append(0)
                i += 1
        return ids

    def encode(self, text):
        return self._encode(text)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.id_to_str.get(int(i), "") for i in ids)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class CrossVocabKDTest(unittest.TestCase):
    D_T = 8   # toy teacher hidden dim
    D_S = 4   # toy student hidden dim
    V_T = 10  # toy teacher vocab
    V_S = 6   # toy student vocab

    def _base_inputs(self, B=1, T=6, T_t=8, seed=0):
        torch.manual_seed(seed)
        projector = VocabProjector(teacher_dim=self.D_T, student_dim=self.D_S).double()
        lm_head_weight = torch.randn(self.V_S, self.D_S, dtype=torch.float64)
        student_logits = torch.randn(B, T, self.V_S, dtype=torch.float64, requires_grad=True)
        student_hidden = torch.randn(B, T, self.D_S, dtype=torch.float64)
        teacher_hidden = torch.randn(B, T_t, self.D_T, dtype=torch.float64)
        return projector, lm_head_weight, student_logits, student_hidden, teacher_hidden

    # 1. KD loss is non-zero and active with >= 1 aligned masked position
    def test_kd_loss_nonzero_when_aligned(self):
        B, T, T_t = 2, 6, 8
        projector, lm_head_weight, student_logits, student_hidden, teacher_hidden = (
            self._base_inputs(B=B, T=T, T_t=T_t, seed=0)
        )
        labels = torch.full((B, T), -1, dtype=torch.long)
        labels[0, 2] = 3
        labels[1, 4] = 1
        loss_mask = torch.zeros(B, T, dtype=torch.float64)
        loss_mask[0, 2] = 1.0
        loss_mask[1, 4] = 1.0
        aligned_pairs_list = [[(2, 1), (3, 2)], [(4, 5)]]

        metrics = cross_vocab_distill_loss(
            student_logits, student_hidden, teacher_hidden, projector, lm_head_weight,
            labels, loss_mask, aligned_pairs_list, kl_weight=0.5, temperature=2.0,
        )

        self.assertIs(metrics["active"], True)
        self.assertEqual(metrics["aligned_count"], 2)  # only masked positions counted
        self.assertGreater(float(metrics["kd_loss"]), 0.0)
        self.assertIsInstance(metrics["loss"], torch.Tensor)
        self.assertEqual(metrics["kl_weight"], 0.5)
        self.assertEqual(metrics["temperature"], 2.0)

    # 2. Gradients backpropagate to student params AND the projector
    def test_kd_backpropagates_to_student(self):
        torch.manual_seed(1)
        B, T, T_t = 1, 4, 5
        projector = VocabProjector(teacher_dim=self.D_T, student_dim=self.D_S)
        lm_head = nn.Linear(self.D_S, self.V_S, bias=False)
        emb = nn.Embedding(self.V_S, self.D_S)

        input_ids = torch.tensor([[0, 1, 2, 3]])
        student_hidden = emb(input_ids)            # (1, 4, d_s)
        student_logits = lm_head(student_hidden)   # (1, 4, V_s)
        teacher_hidden = torch.randn(B, T_t, self.D_T)
        labels = torch.tensor([[1, -1, -1, 2]])
        loss_mask = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        aligned_pairs_list = [[(0, 2), (3, 4)]]

        metrics = cross_vocab_distill_loss(
            student_logits, student_hidden, teacher_hidden, projector, lm_head.weight,
            labels, loss_mask, aligned_pairs_list, kl_weight=0.5, temperature=2.0,
        )
        self.assertTrue(metrics["active"])
        metrics["loss"].backward()

        self.assertIsNotNone(lm_head.weight.grad)
        self.assertGreater(torch.count_nonzero(lm_head.weight.grad).item(), 0)
        self.assertIsNotNone(emb.weight.grad)
        self.assertGreater(torch.count_nonzero(emb.weight.grad).item(), 0)
        self.assertIsNotNone(projector.proj.weight.grad)
        self.assertGreater(torch.count_nonzero(projector.proj.weight.grad).item(), 0)

    # 3. Fail closed: no alignment -> RuntimeError after > KD_SILENCE_LIMIT calls
    def test_fail_closed_when_no_alignment(self):
        reset_kd_guard("fail_closed")
        metrics = {"active": False, "aligned_count": 0, "loss": 0.0}
        # First KD_SILENCE_LIMIT consecutive silent calls do not raise...
        for _ in range(KD_SILENCE_LIMIT):
            assert_kd_not_silently_disabled(metrics, 0.5, "align", step=0, run_id="fail_closed")
        # ...but the next one does.
        with self.assertRaises(RuntimeError):
            assert_kd_not_silently_disabled(metrics, 0.5, "align", step=0, run_id="fail_closed")
        reset_kd_guard("fail_closed")

    # 4. Masked positions contribute nothing
    def test_padding_contributes_zero(self):
        B, T, T_t = 1, 3, 3
        projector, lm_head_weight, student_logits, student_hidden, teacher_hidden = (
            self._base_inputs(B=B, T=T, T_t=T_t, seed=2)
        )
        labels = torch.tensor([[0, -1, 1]])
        loss_mask = torch.tensor([[1.0, 0.0, 1.0]])  # position 1 masked
        kwargs = dict(kl_weight=0.5, temperature=2.0)

        with_pair = cross_vocab_distill_loss(
            student_logits, student_hidden, teacher_hidden, projector, lm_head_weight,
            labels, loss_mask, [[(1, 1)]], **kwargs,
        )
        without = cross_vocab_distill_loss(
            student_logits, student_hidden, teacher_hidden, projector, lm_head_weight,
            labels, loss_mask, [[]], **kwargs,
        )

        self.assertAlmostEqual(
            float(with_pair["loss"].detach()), float(without["loss"].detach()), places=12
        )
        self.assertFalse(with_pair["active"])
        self.assertEqual(with_pair["aligned_count"], 0)
        self.assertEqual(float(with_pair["kd_loss"]), 0.0)

    # 5. One student token -> LAST of several teacher fragments
    def test_one_to_many_alignment(self):
        student = FakeStudentTokenizer(
            {1: "abc", 2: "x", 3: "y", 99: "<pad>"},
            special_ids={"pad": 99, "bos": 98, "eos": 97, "unk": 96},
        )
        teacher = FakeTeacherTokenizer({1: "a", 2: "b", 3: "c", 4: "z"})

        student_ids = student.encode("abc")  # [1]  (one token "abc")
        teacher_ids = teacher.encode("abc")  # [1, 2, 3]  ("a","b","c")

        pairs = compute_alignment(student_ids, teacher_ids, student, teacher)
        self.assertIn((0, 2), pairs)        # anchored to LAST teacher fragment
        self.assertNotIn((0, 0), pairs)
        self.assertNotIn((0, 1), pairs)

    # 6. Alignment cache roundtrip
    def test_alignment_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = AlignmentCache(cache_dir=tmp)
            text = "the quick brown fox"
            pairs = [(0, 1), (2, 3), (4, 5)]

            self.assertIsNone(cache.get(text))  # missing key -> None
            cache.put(text, pairs)
            self.assertEqual(cache.get(text), pairs)

            self.assertIsNone(cache.get("another text"))  # still None

            # corrupt file -> None, not a crash
            corrupt_key = "corrupt payload"
            digest = hashlib.sha256(corrupt_key.encode("utf-8")).hexdigest()
            with open(os.path.join(tmp, f"{digest}.align.json"), "w", encoding="utf-8") as f:
                f.write("{not valid json")
            self.assertIsNone(cache.get(corrupt_key))

            self.assertIn("hits=", cache.stats())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    failed = len(result.failures) + len(result.errors)
    if failed == 0:
        print(f"OK: {result.testsRun} tests passed")
    sys.exit(0 if failed == 0 else 1)