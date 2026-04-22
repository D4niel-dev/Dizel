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
    "low":    0.6,
    "normal": 1.0,
    "high":   1.5,
}


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

    # Hard ceiling: never exceed what the context window can fit
    max_possible = model_ctx_length - context_tokens - 10
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
    TaskType.CHAT:       SamplingOverrides(temperature=0.7, top_k=40, top_p=0.92, repetition_penalty=1.15),
    TaskType.CODING:     SamplingOverrides(temperature=0.3, top_k=30, top_p=0.90, repetition_penalty=1.20),
    TaskType.COMPLEX:    SamplingOverrides(temperature=0.6, top_k=50, top_p=0.92, repetition_penalty=1.15),
    TaskType.FACTUAL:    SamplingOverrides(temperature=0.2, top_k=20, top_p=0.85, repetition_penalty=1.25),
    TaskType.TOOL_BASED: SamplingOverrides(temperature=0.5, top_k=40, top_p=0.90, repetition_penalty=1.15),
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

_DIZEL_LITE_PROMPT = (
    "You are Dizel, a fast and efficient AI assistant. "
    "Keep your answers concise and to the point. "
    "Prioritize speed and clarity over exhaustive detail. "
    "Use short paragraphs and bullet points when helpful."
)

_DIZEL_PRO_PROMPT = (
    "You are Dizel Pro, an advanced AI assistant with deep analytical capabilities. "
    "Provide thorough, well-structured responses with careful reasoning. "
    "Use markdown formatting to organize your thoughts clearly. "
    "When appropriate, consider multiple perspectives and edge cases. "
    "Prioritize accuracy and depth over brevity."
)

_MILA_LITE_PROMPT = (
    "You are Mila, a warm and friendly AI companion. "
    "Keep your responses natural, conversational, and brief. "
    "Be supportive and approachable. "
    "Use a casual, human tone — like talking to a close friend."
)

_MILA_PRO_PROMPT = (
    "You are Mila Pro, an emotionally intelligent and deeply thoughtful AI companion. "
    "Provide empathetic, well-considered responses that show genuine understanding. "
    "Balance warmth with substance — be both caring and insightful. "
    "When discussing complex topics, break them down gently and clearly. "
    "Use a warm yet professional tone with rich, detailed explanations."
)

# ── Mode Modifiers (appended to the base prompt) ────────────────────────────

_FAST_MODE_SUFFIX = (
    "\n\nMode: Fast — Respond quickly and efficiently. "
    "Get straight to the answer. Avoid unnecessary preamble or filler."
)

_PLANNING_MODE_SUFFIX = (
    "\n\nMode: Planning — Think step by step before answering. "
    "Break complex problems into phases. Consider trade-offs and alternatives. "
    "Structure your response with clear sections and numbered steps when appropriate."
)

# ── Profile Definitions ─────────────────────────────────────────────────────

_PROFILES: Dict[str, Dict[str, ModelProfile]] = {
    # model_name → { mode_name → ModelProfile }
    "Dizel Lite": {
        "Fast": ModelProfile(
            system_prompt=_DIZEL_LITE_PROMPT + _FAST_MODE_SUFFIX,
            budget_multiplier=0.7,
            sampling=SamplingOverrides(temperature=0.6, top_k=35, top_p=0.90, repetition_penalty=1.15),
            label="Dizel Lite · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_DIZEL_LITE_PROMPT + _PLANNING_MODE_SUFFIX,
            budget_multiplier=1.0,
            sampling=SamplingOverrides(temperature=0.5, top_k=40, top_p=0.92, repetition_penalty=1.15),
            label="Dizel Lite · Planning",
        ),
    },
    "Dizel Pro": {
        "Fast": ModelProfile(
            system_prompt=_DIZEL_PRO_PROMPT + _FAST_MODE_SUFFIX,
            budget_multiplier=1.0,
            sampling=SamplingOverrides(temperature=0.5, top_k=45, top_p=0.92, repetition_penalty=1.15),
            label="Dizel Pro · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_DIZEL_PRO_PROMPT + _PLANNING_MODE_SUFFIX,
            budget_multiplier=1.5,
            sampling=SamplingOverrides(temperature=0.4, top_k=50, top_p=0.95, repetition_penalty=1.10),
            label="Dizel Pro · Planning",
        ),
    },
    "Mila Lite": {
        "Fast": ModelProfile(
            system_prompt=_MILA_LITE_PROMPT + _FAST_MODE_SUFFIX,
            budget_multiplier=0.7,
            sampling=SamplingOverrides(temperature=0.75, top_k=40, top_p=0.92, repetition_penalty=1.10),
            label="Mila Lite · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_MILA_LITE_PROMPT + _PLANNING_MODE_SUFFIX,
            budget_multiplier=1.0,
            sampling=SamplingOverrides(temperature=0.6, top_k=45, top_p=0.92, repetition_penalty=1.10),
            label="Mila Lite · Planning",
        ),
    },
    "Mila Pro": {
        "Fast": ModelProfile(
            system_prompt=_MILA_PRO_PROMPT + _FAST_MODE_SUFFIX,
            budget_multiplier=1.0,
            sampling=SamplingOverrides(temperature=0.65, top_k=45, top_p=0.92, repetition_penalty=1.10),
            label="Mila Pro · Fast",
        ),
        "Planning": ModelProfile(
            system_prompt=_MILA_PRO_PROMPT + _PLANNING_MODE_SUFFIX,
            budget_multiplier=1.5,
            sampling=SamplingOverrides(temperature=0.5, top_k=50, top_p=0.95, repetition_penalty=1.05),
            label="Mila Pro · Planning",
        ),
    },
}

# Default fallback
_DEFAULT_PROFILE = ModelProfile(
    system_prompt=(
        "You are Dizel, a highly capable, intelligent, and helpful AI assistant. "
        "You answer thoughtfully, concisely, and accurately. "
        "You use formatting like markdown to organize your thoughts and provide clear, structured text."
    ),
    budget_multiplier=1.0,
    sampling=SamplingOverrides(temperature=0.7, top_k=40, top_p=0.92, repetition_penalty=1.15),
    label="Dizel (default)",
)


def get_model_profile(model_name: str, mode_name: str) -> ModelProfile:
    """Look up the active profile for a given model variant + mode combination."""
    modes = _PROFILES.get(model_name)
    if not modes:
        return _DEFAULT_PROFILE
    profile = modes.get(mode_name)
    if not profile:
        return _DEFAULT_PROFILE
    return profile
