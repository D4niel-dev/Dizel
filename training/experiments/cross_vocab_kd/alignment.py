"""
Lexical token alignment between student (Dizel / SentencePiece) and teacher
(Qwen / HF) tokenizations.

Tokens are decoded to their raw strings, joined into full texts, and aligned
with difflib.SequenceMatcher over shared characters.  Character offsets are
mapped back to token index ranges, and student/teacher token pairs are
produced proportionally (1 student token -> LAST teacher token in range,
N student tokens -> proportionally positioned teacher tokens).
"""
import difflib
import hashlib
import json
import os
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_student(text: str, student_tok) -> List[int]:
    """Tokenize ``text`` with the student (SentencePiece-style) tokenizer.

    Prefers ``student_tok.encode(text)`` (SP encode, no bos/eos).  Falls back
    to ``student_tok.tokenize()`` mapped through ``student_tok.vocab``.
    """
    if hasattr(student_tok, "encode"):
        try:
            return [int(i) for i in student_tok.encode(text)]
        except Exception:
            pass
    if hasattr(student_tok, "tokenize"):
        try:
            pieces = list(student_tok.tokenize(text))
        except Exception:
            pieces = None
        if pieces is not None:
            vocab = getattr(student_tok, "vocab", None)
            if vocab is not None and hasattr(vocab, "get"):
                ids = []
                for p in pieces:
                    i = vocab.get(p)
                    if i is None and isinstance(p, str):
                        try:
                            i = vocab.get(p.encode("utf-8"))
                        except Exception:
                            i = None
                    if i is not None:
                        ids.append(int(i))
                if ids:
                    return ids
    raise ValueError(
        f"tokenize_student: tokenizer {type(student_tok).__name__} has neither "
        f"a working .encode() nor a tokenize()+vocab fallback"
    )


def tokenize_teacher(text: str, teacher_tok) -> List[int]:
    """Tokenize ``text`` with the teacher (HF-style) tokenizer.

    Prefers ``teacher_tok(text, add_special_tokens=False)["input_ids"]``
    (HF tokenizer call convention); falls back to a duck-typed ``.encode``.
    """
    if callable(teacher_tok):
        try:
            enc = teacher_tok(text, add_special_tokens=False)
            ids = None
            if isinstance(enc, dict) and "input_ids" in enc:
                ids = enc["input_ids"]
            elif hasattr(enc, "input_ids"):
                ids = enc.input_ids
            if ids is not None:
                return [int(i) for i in ids]
        except Exception:
            pass
    if hasattr(teacher_tok, "encode"):
        try:
            return [int(i) for i in teacher_tok.encode(text)]
        except Exception:
            pass
    raise ValueError(
        f"tokenize_teacher: tokenizer {type(teacher_tok).__name__} has neither "
        f"a callable HF interface nor a working .encode()"
    )


# ---------------------------------------------------------------------------
# Per-token decoding
# ---------------------------------------------------------------------------


def _decode_id(tok, token_id: int) -> str:
    """Decode a single token id to its raw string, never raising.

    Order: student ``tok.sp.decode([i])`` (raw SP piece) -> teacher
    ``tok.decode([i], skip_special_tokens=False)`` -> explicit
    ``tok_id_to_string`` map -> private-use placeholder.
    """
    # 1) Raw SentencePiece piece decode (student path).
    sp = getattr(tok, "sp", None)
    if sp is not None and hasattr(sp, "decode"):
        try:
            s = sp.decode([int(token_id)])
            if s is not None:
                return s
        except Exception:
            pass
    # 2) HF-style tokenizer.decode (teacher path).
    decode = getattr(tok, "decode", None)
    if decode is not None:
        for kwargs in ({"skip_special_tokens": False}, {}):
            try:
                s = decode([int(token_id)], **kwargs)
                if s is not None:
                    return s
            except Exception:
                continue
    # 3) Explicit id -> string map (test tokenizers / duck types).
    mapping = getattr(tok, "tok_id_to_string", None)
    if mapping is not None:
        try:
            s = mapping(int(token_id)) if callable(mapping) else mapping[int(token_id)]
            if s is not None:
                return s
        except Exception:
            pass
    # 4) Placeholder so alignment never crashes on an undecodable id.
    return chr(0xE000 + int(token_id))


def per_token_strings(ids, tok) -> List[str]:
    """Decode each id in ``ids`` to its own string."""
    return [_decode_id(tok, i) for i in ids]


# ---------------------------------------------------------------------------
# Offset mapping helpers
# ---------------------------------------------------------------------------


def _cumulative_offsets(pieces: List[str]) -> List[int]:
    """offsets[k] = character index where token k starts (len = n_tokens+1)."""
    offs = [0]
    for p in pieces:
        offs.append(offs[-1] + len(p))
    return offs


def _token_at_char(offsets: List[int], char_pos: int) -> int:
    """Return the token index containing character index ``char_pos``."""
    if char_pos < 0:
        return 0
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= char_pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _student_special_ids(student_tok) -> set:
    """Collect pad/bos/eos/unk ids from the tokenizer and its .sp."""
    specials = set()
    holders = [student_tok]
    sp = getattr(student_tok, "sp", None)
    if sp is not None:
        holders.append(sp)
    for h in holders:
        for attr in ("pad_id", "bos_id", "eos_id", "unk_id"):
            v = getattr(h, attr, None)
            if isinstance(v, int):
                specials.add(v)
    return specials


def _is_student_special(token_id: int, student_tok) -> bool:
    """True if this student token id is special (pad/bos/eos/unk) or a
    Dizel format marker such as ``<|user|>`` / ``<|assistant|>`` / ``<|end|>``."""
    if token_id in _student_special_ids(student_tok):
        return True
    return "<|" in _decode_id(student_tok, token_id)


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------


def compute_alignment(
    student_ids: List[int],
    teacher_ids: List[int],
    student_tok,
    teacher_tok,
) -> List[Tuple[int, int]]:
    """Lexically align student/teacher token ids.

    Returns a sorted, deduplicated list of ``(i_student, j_teacher)`` pairs.
    Blocks whose student range covers special tokens (pad/bos/eos/unk or
    ``<|...|>`` format markers) are skipped.  Padding/masked positions are
    the caller's responsibility.
    """
    student_ids = [int(i) for i in student_ids]
    teacher_ids = [int(i) for i in teacher_ids]
    if not student_ids or not teacher_ids:
        return []

    s_strs = per_token_strings(student_ids, student_tok)
    t_strs = per_token_strings(teacher_ids, teacher_tok)

    student_full = "".join(s_strs)
    teacher_full = "".join(t_strs)
    s_offs = _cumulative_offsets(s_strs)
    t_offs = _cumulative_offsets(t_strs)

    matcher = difflib.SequenceMatcher(None, student_full, teacher_full, autojunk=False)
    pairs: List[Tuple[int, int]] = []

    for block in matcher.get_matching_blocks():
        a, b, size = block.a, block.b, block.size
        if size <= 0:  # sentinel trailing block
            continue

        # Map matching char block [a, a+size) / [b, b+size) to token ranges.
        s_start = _token_at_char(s_offs, a)
        s_end = _token_at_char(s_offs, a + size - 1)
        t_start = _token_at_char(t_offs, b)
        t_end = _token_at_char(t_offs, b + size - 1)

        s_idx = list(range(s_start, s_end + 1))
        t_idx = list(range(t_start, t_end + 1))

        # Skip blocks covering student special tokens / format markers.
        if any(_is_student_special(student_ids[i], student_tok) for i in s_idx):
            continue

        if len(s_idx) == 1:
            # Single student token -> anchor to LAST teacher token in range.
            pairs.append((s_idx[0], t_idx[-1]))
        else:
            # Multiple student tokens -> proportionally positioned teacher tokens.
            n_s, n_t = len(s_idx), len(t_idx)
            for k, i in enumerate(s_idx):
                j = t_start + int((k / n_s) * n_t)
                j = min(j, t_start + n_t - 1)
                pairs.append((i, j))

    return sorted(set(pairs))


# ---------------------------------------------------------------------------
# Alignment cache (sha256-keyed JSON files)
# ---------------------------------------------------------------------------


def _validate_pairs(data) -> Optional[List[Tuple[int, int]]]:
    """Validate cached JSON payload; return None if malformed."""
    if not isinstance(data, list):
        return None
    out: List[Tuple[int, int]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        out.append((int(item[0]), int(item[1])))
    return out


class AlignmentCache:
    """Disk cache of computed alignments, keyed by sha256 of the source text.

    Robust to corrupt/missing files: ``get`` returns None on any error.
    """

    def __init__(self, cache_dir: str = ".cache/teacher"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.align.json")

    def get(self, text: str) -> Optional[List[Tuple[int, int]]]:
        path = self._path(text)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            self.misses += 1
            return None
        pairs = _validate_pairs(data)
        if pairs is None:
            self.misses += 1
            return None
        self.hits += 1
        return pairs

    def put(self, text: str, pairs) -> None:
        path = self._path(text)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([[int(a), int(b)] for a, b in pairs], f)
        except (OSError, ValueError, TypeError):
            pass  # cache is best-effort

    def stats(self) -> str:
        return f"AlignmentCache(hits={self.hits}, misses={self.misses})"