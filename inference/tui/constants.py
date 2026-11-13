"""Shared TUI constants for tiers, modes and provider metadata."""

# Dizel 1.6 names its model families Flash and Pro; the modes sit
# underneath them and only change sampling and the system prompt.
TIERS = ("Flash", "Pro")

TIER_DETAILS = {
    "Flash": "The small, fast tier · what this build ships",
    "Pro": "The larger tier · same prompts until that build exists",
}

MODES = ("Fast", "Planning", "Coding", "Thinking", "Deep Think")

MODE_DETAILS = {
    "Fast": "Answer-first, low latency · temp 0.55",
    "Planning": "Plan, risks, then execute · temp 0.45",
    "Coding": "Runnable code first, then notes · temp 0.25",
    "Thinking": "Reason in a think block · temp 0.40",
    "Deep Think": "Long-form reasoning and structure · temp 0.50",
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
