"""Shared TUI constants for response modes and provider metadata."""

MODES = ("Fast", "Planning", "Coding", "Thinking")

FAST_SPEEDS = ("fast", "turbo", "ultrafast")
FAST_SPEED_LABELS = {
    "fast": "Fast",
    "turbo": "Turbo",
    "ultrafast": "Ultrafast",
}
FAST_SPEED_COLORS = {
    "fast": "#38bdf8",
    "turbo": "#4fd6be",
    "ultrafast": "#fb923c",
}

THINKING_EFFORTS = ("low", "medium", "high", "xhigh", "deep")

THINKING_EFFORT_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "X-High",
    "deep": "Max",
}

THINKING_EFFORT_COLORS = {
    "low": "#60a5fa",
    "medium": "#a78bfa",
    "high": "#f472b6",
    "xhigh": "#fb923c",
    "deep": "#f43f5e",
}

MODE_COLORS = {
    "Fast": "#38bdf8",
    "Planning": "#f59e0b",
    "Coding": "#34d399",
}

MODE_GLYPHS = {
    "Fast": "↯",
    "Planning": "☰",
    "Coding": "{}",
    "Thinking": "✦",
}

MODE_DETAILS = {
    "Fast": "Answer-first, low latency · temp 0.55",
    "Planning": "Plan, risks, then execute · temp 0.45",
    "Coding": "Runnable code first, then notes · temp 0.25",
    "Thinking": "Reason in a think block · effort sets depth",
}

PROVIDER_FALLBACKS = (
    ("openai", "OpenAI", "GPT models from OpenAI", True),
    ("anthropic", "Anthropic", "Claude models", True),
    ("google", "Google", "Gemini models", True),
    ("ollama", "Ollama", "Local models through an Ollama server", False),
    ("groq", "Groq", "Fast OpenAI-compatible inference", True),
    ("mistral", "Mistral", "Mistral models", True),
    ("xai", "xAI", "Grok models", True),
    ("ai21", "AI21", "Jamba models", True),
    ("azure", "Azure OpenAI", "Hosted OpenAI deployments", True),
    ("cohere", "Cohere", "Command models", True),
    ("meta", "Meta", "Llama API models", True),
)
