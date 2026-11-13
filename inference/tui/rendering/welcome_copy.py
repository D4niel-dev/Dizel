"""The time-aware welcome pools from inference/gui/src/services/welcome.ts."""

from __future__ import annotations

import random
from datetime import datetime

HEADLINES = {
    "morning": (
        "{greeting}, {name}!",
        "Morning, {name}.",
        "Morning, {name}!",
        "Early start, {name}?",
        "Early bird, {name}?",
        "Early riser, {name}?",
        "Good morning to you, {name}.",
        "A fresh morning, {name}.",
        "New day, {name}.",
        "Morning already, {name}?",
        "Good morning, {name}.",
        "Top of the morning, {name}.",
        "Mornin', {name}.",
        "Morning to you, {name}.",
        "Bright and early, {name}.",
        "A new day, {name}.",
    ),
    "afternoon": (
        "{greeting}, {name}!",
        "Afternoon, {name}.",
        "Afternoon, {name}!",
        "Good afternoon to you, {name}.",
        "Good afternoon, {name}.",
        "Afternoon already, {name}?",
        "Back at it, {name}?",
        "Halfway through, {name}?",
        "How's the afternoon, {name}?",
        "Having a good afternoon, {name}?",
        "Good day, {name}.",
        "Good day to you, {name}.",
        "Afternoon to you, {name}.",
        "Mid-afternoon, {name}.",
        "The afternoon's here, {name}.",
        "Still at it, {name}?",
    ),
    "evening": (
        "{greeting}, {name}!",
        "Evening, {name}.",
        "Evening, {name}!",
        "Good evening to you, {name}.",
        "Good evening, {name}.",
        "Evening already, {name}?",
        "Still going, {name}?",
        "Still up, {name}?",
        "How's the evening, {name}?",
        "Having a good evening, {name}?",
        "Winding down, {name}?",
        "A good evening to you, {name}.",
        "Evening to you, {name}.",
        "Late evening, {name}.",
        "The evening's here, {name}.",
        "Hope your evening's going well, {name}.",
    ),
}

SUBLINES = (
    "What's on your mind?",
    "Ask anything. Start anywhere.",
    "A question, a task, or a half-formed idea.",
    "Write, debug, plan, or think out loud.",
    "Paste the messy part.",
    "Built to run on your machine.",
    "Local when you want it.",
    "What should we figure out?",
    "One prompt is enough to begin.",
    "Turn a rough idea into a plan.",
    "Ask for the short version or the deep one.",
    "Drop in code, a file, or a question.",
    "Let's get the first draft down.",
    "What would make today easier?",
    "Start with the part that's stuck.",
    "Plain answers, no detours.",
    "What are we improving?",
    "Think out loud; I'll keep up.",
    "A good place to start is the problem.",
    "Bring the context; we'll find the thread.",
    "Ask for code, a plan, or an explanation.",
    "The model is ready when you are.",
    "What's the next small step?",
    "Pick up a saved prompt, or start clean.",
)


def day_part(now: datetime | None = None) -> str:
    hour = (now or datetime.now()).hour
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def greeting(now: datetime | None = None) -> str:
    return f"Good {day_part(now)}"


def sample_welcome(part: str) -> tuple[str, str]:
    return random.choice(HEADLINES[part]), random.choice(SUBLINES)


def fill_welcome(template: str, *, greeting_text: str, name: str) -> str:
    return template.replace("{greeting}", greeting_text).replace("{name}", name)