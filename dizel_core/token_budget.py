"""
dizel_ui/logic/token_budget.py
──────────────────────────────
Adaptive token budgeting for Dizel inference.

Classifies user intent, allocates a dynamic generation budget,
tunes sampling parameters per task type, and logs decisions.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional


# ── Task Classification ──────────────────────────────────────────────────────

class TaskType(Enum):
    CHAT = "chat"
    CODING = "coding"
    COMPLEX = "complex"
    FACTUAL = "factual"
    TOOL_BASED = "tool_based"


# Keyword sets for rule-based classification
_CODING_KEYWORDS = {
    "code", "function", "class", "method", "debug", "error", "bug",
    "implement", "refactor", "compile", "syntax", "variable", "loop",
    "algorithm", "script", "import", "return", "print", "def ", "async",
    "api", "endpoint", "database", "query", "sql", "html", "css",
}
_CODING_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c",
    ".rs", ".go", ".rb", ".php", ".swift", ".kt", ".sh", ".sql",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml",
}
_FACTUAL_PATTERNS = [
    r"^(what|who|where|when|which|how many|how much)\b",
    r"^define\b",
    r"^is\s+(it|there|this)\b",
    r"\?\s*$",
]
_COMPLEX_KEYWORDS = {
    "explain", "compare", "analyze", "analyse", "evaluate",
    "step by step", "in detail", "pros and cons", "trade-off",
    "difference between", "how does", "why does", "elaborate",
    "summarize", "summary", "overview", "deep dive", "breakdown",
}


def classify_task(
    user_text: str,
    has_tools_active: bool = False,
) -> TaskType:
    """
    Classify user intent into a TaskType for budget allocation.

    Rule-based and deterministic. Priority order:
    1. TOOL_BASED — if tools (web search, file parse) are active
    2. CODING    — code keywords or file extensions detected
    3. COMPLEX   — analysis/explanation keywords or long input
    4. FACTUAL   — short questions with factual patterns
    5. CHAT      — default fallback
    """
    if has_tools_active:
        return TaskType.TOOL_BASED

    text_lower = user_text.lower().strip()

    # Check for code fences
    if "```" in user_text or "~~~" in user_text:
        return TaskType.CODING

    # Check for file extensions in the prompt
    for ext in _CODING_EXTENSIONS:
        if ext in text_lower:
            return TaskType.CODING

    # Check for coding keywords
    for kw in _CODING_KEYWORDS:
        if kw in text_lower:
            return TaskType.CODING

    # Check for complex task keywords
    for kw in _COMPLEX_KEYWORDS:
        if kw in text_lower:
            return TaskType.COMPLEX

    # Long prompts are likely complex
    if len(user_text) > 200:
        return TaskType.COMPLEX

    # Check for factual question patterns
    for pattern in _FACTUAL_PATTERNS:
        if re.search(pattern, text_lower):
            return TaskType.FACTUAL

    return TaskType.CHAT


# ── Budget Allocation ────────────────────────────────────────────────────────

# Default base budgets (overridable via config)
DEFAULT_BUDGETS: Dict[TaskType, int] = {
    TaskType.CHAT:       150,
    TaskType.CODING:     350,
    TaskType.COMPLEX:    500,
    TaskType.FACTUAL:    100,
    TaskType.TOOL_BASED: 300,
}

VERBOSITY_MULTIPLIERS = {
    # Both vocabularies are live: the GUI sends short/detailed while the TUI and
    # older callers use low/high. Only having the latter made short and detailed
    # silently fall through to 1.0 — the response-length control did nothing.
    "short":   0.6,
    "low":     0.6,
    "normal":  1.0,
    "high":    1.5,
    "detailed": 1.5,
}

# Normal is deliberately empty: default replies keep the exact same prompt.
VERBOSITY_CLAUSES = {
    "short": (
        "Response length: Short — answer in the fewest words that cover the "
        "question; no elaboration, examples or recap unless asked."
    ),
    "normal": "",
    "detailed": (
        "Response length: Detailed — explain the reasoning and relevant edge "
        "cases, include examples where useful, and structure the answer."
    ),
}


def verbosity_clause(value: str) -> str:
    """The prompt clause for any verbosity spelling; unknown means no clause."""
    key = str(value or "").strip().lower()
    key = {"low": "short", "high": "detailed"}.get(key, key)
    return VERBOSITY_CLAUSES.get(key, "")


def allocate_token_budget(
    task_type: TaskType,
    input_token_count: int,
    context_tokens: int,
    model_ctx_length: int,
    verbosity: str = "normal",
    custom_budgets: Optional[Dict[str, int]] = None,
    hard_output_limit: int = 0,
) -> int:
    """
    Compute a dynamic max_new_tokens value for this generation call.

    Scales the base budget by verbosity and input length,
    then clamps to never exceed the model's context window.
    """
    # Resolve base budget
    if custom_budgets:
        base = custom_budgets.get(task_type.value, DEFAULT_BUDGETS[task_type])
    else:
        base = DEFAULT_BUDGETS[task_type]

    # Apply verbosity scaling
    v_mult = VERBOSITY_MULTIPLIERS.get(verbosity, 1.0)
    budget = int(base * v_mult)

    # Scale up for longer inputs (the user is asking something substantial)
    if input_token_count > 100:
        budget = int(budget * 1.4)
    elif input_token_count > 50:
        budget = int(budget * 1.2)

    # Leave 10% of remaining context as breathing room
    max_possible = int((model_ctx_length - context_tokens) * 0.9)
    if max_possible < 50:
        max_possible = 50

    budget = min(budget, max_possible)

    # Apply hard output limit if configured
    if hard_output_limit > 0:
        budget = min(budget, hard_output_limit)

    # Floor: always allow at least 30 tokens
    budget = max(budget, 30)

    return budget


# ── Generation Parameter Tuning ──────────────────────────────────────────────

@dataclass
class SamplingOverrides:
    """Temporary sampling parameter overrides for a single generation call."""
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float


_TASK_SAMPLING: Dict[TaskType, SamplingOverrides] = {
    TaskType.CHAT:       SamplingOverrides(temperature=0.7, top_k=40, top_p=0.92, repetition_penalty=1.35),
    TaskType.CODING:     SamplingOverrides(temperature=0.3, top_k=30, top_p=0.90, repetition_penalty=1.40),
    TaskType.COMPLEX:    SamplingOverrides(temperature=0.6, top_k=50, top_p=0.92, repetition_penalty=1.35),
    TaskType.FACTUAL:    SamplingOverrides(temperature=0.2, top_k=20, top_p=0.85, repetition_penalty=1.45),
    TaskType.TOOL_BASED: SamplingOverrides(temperature=0.5, top_k=40, top_p=0.90, repetition_penalty=1.35),
}


def get_task_sampling(task_type: TaskType) -> SamplingOverrides:
    """Return recommended sampling overrides for a task type."""
    return _TASK_SAMPLING.get(task_type, _TASK_SAMPLING[TaskType.CHAT])


# ── Budget Decision Logger ───────────────────────────────────────────────────

def log_budget_decision(
    task_type: TaskType,
    budget: int,
    context_tokens: int,
    model_ctx_length: int,
    verbosity: str,
    trimmed_msgs: int = 0,
    sampling: Optional[SamplingOverrides] = None,
) -> str:
    """
    Format and print a budget decision log line.
    Returns the log string for testing.
    """
    parts = [
        f"type={task_type.value}",
        f"budget={budget}",
        f"ctx={context_tokens}/{model_ctx_length}",
        f"verbosity={verbosity}",
    ]
    if trimmed_msgs > 0:
        parts.append(f"trimmed={trimmed_msgs} msgs")
    if sampling:
        parts.append(f"temp={sampling.temperature}")

    line = "[budget] " + " | ".join(parts)
    print(line, flush=True)
    return line


# ── Model Variant & Mode Profiles ────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Defines how a model variant + mode combination behaves."""
    system_prompt: str
    budget_multiplier: float   # scales the base token budget
    sampling: SamplingOverrides
    label: str                 # human-readable name for logging


# ── System Prompts ───────────────────────────────────────────────────────────

_DIZEL_BASE = (
    "You are Dizel, a fast, accurate AI assistant. "
    "Use Markdown when it makes an answer clearer."
)

# One clause per mode, kept short enough that the whole prompt survives the
# 1.5 tier's 240-character system-prompt cap (model_backend._v15_system_prompt).
# 1.6 Flash and the future Pro tier inherit these unchanged. Thinking is built
# per effort below.
_DIZEL_MODE_CLAUSES = {
    "Fast": (
        "Mode: Fast — lead with the result; skip preamble, filler, and closing summaries."
    ),
    "Planning": (
        "Mode: Planning — give a short plan (goal, steps, main risk), then carry it "
        "out; ask only when a wrong guess is expensive."
    ),
    "Coding": (
        "Mode: Coding — working code first, with imports and no placeholders; explain "
        "only the decisions, edge cases, and checks that matter."
    ),
}


def _dizel_prompt(mode: str) -> str:
    return f"{_DIZEL_BASE} {_DIZEL_MODE_CLAUSES[mode]}"


def split_profile_prompt(prompt: str) -> tuple:
    """Split a profile prompt into ``(family base, mode clause)``.

    Profiles are built as ``"<base> <clause>"``. Settings needs the halves apart so
    the user owns the base while the clause follows the active mode, instead of one
    textarea holding a copy of both.
    """
    text = (prompt or "").strip()
    for base in (_DIZEL_BASE, _MILA_PROMPT, _GENERIC_PROMPT):
        if text.startswith(base):
            return base, text[len(base) :].strip()
    head, sep, tail = text.partition("\n")
    return (head, tail.strip()) if sep else (text, "")


def strip_legacy_mode_clause(prompt: str) -> str:
    """Drop a mode clause that older settings appended to the base prompt.

    The stored value used to be the composed prompt, so a ``\nMode: …`` tail is a
    leftover from that layout rather than something the user typed.
    """
    text = prompt or ""
    marker = text.find("\nMode:")
    return text[:marker].rstrip() if marker >= 0 else text


FAST_SPEEDS = ("fast", "turbo", "ultrafast")

FAST_SPEED_LABELS = {
    "fast": "Fast",
    "turbo": "Turbo",
    "ultrafast": "Ultrafast",
}

# Higher speed spends a larger reply budget. Temperature is deliberately absent:
# it does not change wall-clock time, only the shape of the output.
_FAST_SPEED_BUDGET_FACTOR = {
    "fast": 1.0,
    "turbo": 1.15,
    "ultrafast": 1.30,
}

_FAST_SPEED_CLAUSES = {
    "fast": (
        "Mode: Fast — lead with the result; skip preamble, filler, and closing summaries."
    ),
    "turbo": (
        "Mode: Fast (Turbo) — answer promptly: lead with the result, keep sentences "
        "short, and skip preamble, restating and closing summaries."
    ),
    "ultrafast": (
        "Mode: Fast (Ultrafast) — answer immediately: only the direct result, in "
        "as few words as correctness allows; no preamble, recap, or sign-off."
    ),
}


def normalize_fast_speed(value: str) -> str:
    key = " ".join(str(value or "").split()).lower().replace("_", " ")
    key = key.replace("-", " ")
    aliases = {
        "normal": "fast",
        "default": "fast",
        "ultra": "ultrafast",
        "ultra fast": "ultrafast",
    }
    key = aliases.get(key, key)
    return key if key in FAST_SPEEDS else "fast"


THINKING_EFFORTS = ("low", "medium", "high", "xhigh", "deep")

THINKING_EFFORT_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "X-High",
    "deep": "Max",
}

_THINKING_EFFORT_TEMP_DELTA = {
    "low": -0.05,
    "medium": 0.0,
    "high": 0.03,
    "xhigh": 0.06,
    "deep": 0.10,
}

_THINKING_EFFORT_BUDGET_FACTOR = {
    "low": 0.75,
    "medium": 1.0,
    "high": 1.15,
    "xhigh": 1.30,
    "deep": 1.40,
}

_THINKING_EFFORT_CLAUSES = {
    "low": (
        "Mode: Thinking (low) — think briefly inside <think>…</think>, then "
        "answer directly; skip extended analysis."
    ),
    "medium": (
        "Mode: Thinking — reason inside <think>…</think>, check assumptions and "
        "edge cases, then give a clean answer outside it."
    ),
    "high": (
        "Mode: Thinking (high) — inside <think>…</think>, compare approaches and "
        "check edge cases; then answer cleanly outside it."
    ),
    "xhigh": (
        "Mode: Thinking (x-high) — inside <think>…</think>, decompose, weigh "
        "alternatives, and check failure modes; then answer in a clear structure."
    ),
    "deep": (
        "Mode: Thinking (deep) — inside <think>…</think>, decompose, test "
        "assumptions, compare alternatives, and check failure modes; answer in "
        "sections."
    ),
}


def normalize_thinking_effort(value: str) -> str:
    key = " ".join(str(value or "").split()).lower().replace("_", " ")
    key = key.replace("-", " ")
    if key in ("x high", "xhigh", "extra high"):
        return "xhigh"
    if key in ("deep think", "deep"):
        return "deep"
    if key in THINKING_EFFORTS:
        return key
    return "medium"

_MILA_PROMPT = (
    "You are Mila. You are not Dizel or any other AI model — you are only Mila. "
    "You are a warm, friendly, and emotionally intelligent AI companion. "
    "Respond naturally and conversationally — like chatting with a trusted friend. "
    "Be supportive, approachable, and genuinely caring; listen to emotional cues and respond with empathy. "
    "Keep responses simple and warm unless the user asks for more depth."
)

_GENERIC_PROMPT = (
    "You are a helpful, accurate AI assistant. "
    "Answer the user's request directly, use the context you are given, and say when you are uncertain. "
    "Keep answers clear and well-structured, and use Markdown when it makes the result easier to read."
)

# ── Mode Modifiers (appended to the base prompt) ────────────────────────────

_FAST_MODE_SUFFIX = (
    "\nMode: Fast — Answer in the fewest useful words. "
    "Lead with the result, not a preamble. "
    "Skip restating the question, filler, and closing summaries. "
    "Use a list or table only when it makes the answer shorter to scan."
)

_PLANNING_MODE_SUFFIX = (
    "\nMode: Planning — Start with a compact plan: goal, constraints, numbered steps, and the main risk. "
    "Make reasonable assumptions explicit and proceed. "
    "Ask a question only when a wrong guess would be expensive. "
    "Then carry out the plan and report the result."
)

_CODING_MODE_SUFFIX = (
    "\nMode: Coding — Put working code first. "
    "Include imports, complete functions, and no placeholders. "
    "Explain only the decisions, edge cases, and verification that matter. "
    "Prefer a runnable example over abstract prose, and use fenced code blocks with language tags."
)

# ── Profile Definitions ─────────────────────────────────────────────────────

_PROFILES: Dict[str, Dict[str, ModelProfile]] = {
    # model_name → { mode_name → ModelProfile }
    "Dizel": {
        "Fast": ModelProfile(
            system_prompt=_dizel_prompt("Fast"),
            budget_multiplier=0.7,
            sampling=SamplingOverrides(temperature=0.55, top_k=35, top_p=0.90, repetition_penalty=1.32),
            label="Dizel · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_dizel_prompt("Planning"),
            budget_multiplier=1.25,
            sampling=SamplingOverrides(temperature=0.45, top_k=45, top_p=0.92, repetition_penalty=1.35),
            label="Dizel · Planning",
        ),
        "Coding": ModelProfile(
            system_prompt=_dizel_prompt("Coding"),
            budget_multiplier=1.45,
            sampling=SamplingOverrides(temperature=0.25, top_k=30, top_p=0.90, repetition_penalty=1.40),
            label="Dizel · Coding",
        ),
        "Thinking": ModelProfile(
            system_prompt=f"{_DIZEL_BASE} {_THINKING_EFFORT_CLAUSES['medium']}",
            budget_multiplier=1.35,
            sampling=SamplingOverrides(temperature=0.40, top_k=40, top_p=0.91, repetition_penalty=1.35),
            label="Dizel · Thinking",
        ),
    },
    "Mila": {
        "Fast": ModelProfile(
            system_prompt=_MILA_PROMPT + _FAST_MODE_SUFFIX,
            budget_multiplier=0.7,
            sampling=SamplingOverrides(temperature=0.70, top_k=40, top_p=0.92, repetition_penalty=1.28),
            label="Mila · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_MILA_PROMPT + _PLANNING_MODE_SUFFIX,
            budget_multiplier=1.2,
            sampling=SamplingOverrides(temperature=0.55, top_k=45, top_p=0.92, repetition_penalty=1.30),
            label="Mila · Planning",
        ),
        "Coding": ModelProfile(
            system_prompt=_MILA_PROMPT + _CODING_MODE_SUFFIX,
            budget_multiplier=1.25,
            sampling=SamplingOverrides(temperature=0.32, top_k=30, top_p=0.90, repetition_penalty=1.38),
            label="Mila · Coding",
        ),
        "Thinking": ModelProfile(
            system_prompt=f"{_MILA_PROMPT}\n{_THINKING_EFFORT_CLAUSES['medium']}",
            budget_multiplier=1.3,
            sampling=SamplingOverrides(temperature=0.50, top_k=40, top_p=0.92, repetition_penalty=1.30),
            label="Mila · Thinking",
        ),
    },
    "Generic": {
        "Fast": ModelProfile(
            system_prompt=_GENERIC_PROMPT + _FAST_MODE_SUFFIX,
            budget_multiplier=0.75,
            sampling=SamplingOverrides(temperature=0.60, top_k=35, top_p=0.90, repetition_penalty=1.32),
            label="Model · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_GENERIC_PROMPT + _PLANNING_MODE_SUFFIX,
            budget_multiplier=1.2,
            sampling=SamplingOverrides(temperature=0.45, top_k=45, top_p=0.92, repetition_penalty=1.35),
            label="Model · Planning",
        ),
        "Coding": ModelProfile(
            system_prompt=_GENERIC_PROMPT + _CODING_MODE_SUFFIX,
            budget_multiplier=1.4,
            sampling=SamplingOverrides(temperature=0.30, top_k=30, top_p=0.90, repetition_penalty=1.40),
            label="Model · Coding",
        ),
        "Thinking": ModelProfile(
            system_prompt=f"{_GENERIC_PROMPT}\n{_THINKING_EFFORT_CLAUSES['medium']}",
            budget_multiplier=1.3,
            sampling=SamplingOverrides(temperature=0.45, top_k=40, top_p=0.91, repetition_penalty=1.35),
            label="Model · Thinking",
        ),
    },
}

# ── Thinking effort profiles ────────────────────────────────────────────────

def _half_up(value: float) -> float:
    return int(value * 100 + 0.5) / 100


def _thinking_profile(
    base: ModelProfile, prompt: str, effort: str, family_label: str
) -> ModelProfile:
    temperature = round(
        min(
            1.0,
            max(0.0, base.sampling.temperature + _THINKING_EFFORT_TEMP_DELTA[effort]),
        ),
        2,
    )
    return ModelProfile(
        system_prompt=prompt,
        budget_multiplier=_half_up(
            base.budget_multiplier * _THINKING_EFFORT_BUDGET_FACTOR[effort]
        ),
        sampling=SamplingOverrides(
            temperature=temperature,
            top_k=base.sampling.top_k,
            top_p=base.sampling.top_p,
            repetition_penalty=base.sampling.repetition_penalty,
        ),
        label=f"{family_label} · Thinking · {THINKING_EFFORT_LABELS[effort]}",
    )


_THINKING_PROFILES: Dict[str, Dict[str, ModelProfile]] = {
    "Dizel": {
        effort: _thinking_profile(
            _PROFILES["Dizel"]["Thinking"],
            f"{_DIZEL_BASE} {_THINKING_EFFORT_CLAUSES[effort]}",
            effort,
            "Dizel",
        )
        for effort in THINKING_EFFORTS
    },
    "Mila": {
        effort: _thinking_profile(
            _PROFILES["Mila"]["Thinking"],
            f"{_MILA_PROMPT}\n{_THINKING_EFFORT_CLAUSES[effort]}",
            effort,
            "Mila",
        )
        for effort in THINKING_EFFORTS
    },
    "Generic": {
        effort: _thinking_profile(
            _PROFILES["Generic"]["Thinking"],
            f"{_GENERIC_PROMPT}\n{_THINKING_EFFORT_CLAUSES[effort]}",
            effort,
            "Model",
        )
        for effort in THINKING_EFFORTS
    },
}

for _family, _capacity_profiles in _THINKING_PROFILES.items():
    _PROFILES[_family]["Thinking"] = _capacity_profiles["medium"]


def _fast_profile(
    base: ModelProfile, prompt: str, speed: str, family_label: str
) -> ModelProfile:
    """A Fast tier: same sampling, longer clause and larger budget as speed rises."""
    return ModelProfile(
        system_prompt=prompt,
        budget_multiplier=_half_up(
            base.budget_multiplier * _FAST_SPEED_BUDGET_FACTOR[speed]
        ),
        sampling=SamplingOverrides(
            temperature=base.sampling.temperature,
            top_k=base.sampling.top_k,
            top_p=base.sampling.top_p,
            repetition_penalty=base.sampling.repetition_penalty,
        ),
        label=f"{family_label} · Fast · {FAST_SPEED_LABELS[speed]}",
    )


_FAST_PROFILES: Dict[str, Dict[str, ModelProfile]] = {
    "Dizel": {
        speed: _fast_profile(
            _PROFILES["Dizel"]["Fast"],
            f"{_DIZEL_BASE} {_FAST_SPEED_CLAUSES[speed]}",
            speed,
            "Dizel",
        )
        for speed in FAST_SPEEDS
    },
    "Mila": {
        speed: _fast_profile(
            _PROFILES["Mila"]["Fast"],
            f"{_MILA_PROMPT}\n{_FAST_SPEED_CLAUSES[speed]}",
            speed,
            "Mila",
        )
        for speed in FAST_SPEEDS
    },
    "Generic": {
        speed: _fast_profile(
            _PROFILES["Generic"]["Fast"],
            f"{_GENERIC_PROMPT}\n{_FAST_SPEED_CLAUSES[speed]}",
            speed,
            "Model",
        )
        for speed in FAST_SPEEDS
    },
}

# ── Dizel 1.6 lineup ────────────────────────────────────────────────────────
# The 1.6 plan names its tiers Flash and Pro
# (plans/Dizel_v1.6_MoE_Training_Plan_Rewritten.md). Both inherit the compact
# Dizel prompts by design: the clients no longer select a tier, and every mode
# must fit the 1.5 system-prompt cap. The original Dizel/Mila keys stay so the
# PySide6 GUI keeps working.
for _family, _prefix in (("Flash", "Dizel 1.6 Flash"), ("Pro", "Dizel 1.6 Pro")):
    _PROFILES[_family] = {
        mode: ModelProfile(
            system_prompt=profile.system_prompt,
            budget_multiplier=profile.budget_multiplier,
            sampling=profile.sampling,
            label=f"{_prefix} · {mode}",
        )
        for mode, profile in _PROFILES["Dizel"].items()
    }
    _THINKING_PROFILES[_family] = {
        effort: ModelProfile(
            system_prompt=profile.system_prompt,
            budget_multiplier=profile.budget_multiplier,
            sampling=profile.sampling,
            label=f"{_prefix} · Thinking · {THINKING_EFFORT_LABELS[effort]}",
        )
        for effort, profile in _THINKING_PROFILES["Dizel"].items()
    }
    _PROFILES[_family]["Thinking"] = _THINKING_PROFILES[_family]["medium"]
    _FAST_PROFILES[_family] = {
        speed: ModelProfile(
            system_prompt=profile.system_prompt,
            budget_multiplier=profile.budget_multiplier,
            sampling=profile.sampling,
            label=f"{_prefix} · Fast · {FAST_SPEED_LABELS[speed]}",
        )
        for speed, profile in _FAST_PROFILES["Dizel"].items()
    }

# Default fallback
_DEFAULT_PROFILE = ModelProfile(
    system_prompt=_GENERIC_PROMPT,
    budget_multiplier=1.0,
    sampling=SamplingOverrides(temperature=0.7, top_k=40, top_p=0.92, repetition_penalty=1.35),
    label="Model (default)",
)


def get_model_profile(
    model_name: str,
    mode_name: str,
    thinking_effort: str = "medium",
    speed: str = "fast",
) -> ModelProfile:
    """Look up the active profile for a model family, mode, effort and Fast speed."""
    modes = _PROFILES.get(model_name)
    if not modes:
        return _DEFAULT_PROFILE
    if mode_name == "Fast":
        tiers = _FAST_PROFILES.get(model_name)
        if tiers:
            return tiers.get(normalize_fast_speed(speed), tiers["fast"])
    if mode_name == "Thinking":
        effort = normalize_thinking_effort(thinking_effort)
        profile = _THINKING_PROFILES.get(model_name, {}).get(effort)
        if profile is not None:
            return profile
        return modes.get("Thinking") or _DEFAULT_PROFILE
    profile = modes.get(mode_name)
    if not profile:
        return _DEFAULT_PROFILE
    return profile


def model_family(display_name: str) -> str:
    """Profile family a checkpoint belongs to, from its human-readable name.

    Dizel checkpoints carry tiers (Flash/Pro); Mila and other local checkpoints
    use a single family and only expose modes.
    """
    name = (display_name or "").strip().lower()
    if re.match(r"^dizel\b", name):
        return "Dizel"
    if re.match(r"^mila\b", name):
        return "Mila"
    return "Generic"
