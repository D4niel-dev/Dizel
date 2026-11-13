"""Focused regression tests for pipelines/identity_guard.py.

Run:
    python scripts/test_identity_guard.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.clean_v15_data import check_identity as v15_check_identity
from pipelines.format_v15_data import to_sft_format
from pipelines.identity_guard import check_identity, rewrite_identity


def fail(message: str) -> None:
    raise AssertionError(message)


def expect_true(value, message: str) -> None:
    if not value:
        fail(message)


def expect_false(value, message: str) -> None:
    if value:
        fail(message)


def expect_equal(actual, expected, message: str) -> None:
    if actual != expected:
        fail(f"{message}\n  expected: {expected!r}\n  actual:   {actual!r}")


# Clear first-person / creator claims that must be removed from pretraining
# and rewritten for SFT/distill output.
POSITIVE = [
    ("I am ChatGPT.", "I am Dizel."),
    ("I'm ChatGPT.", "I'm Dizel."),
    ("I\u2019m ChatGPT.", "I\u2019m Dizel."),
    (
        "I am an AI language model developed by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I'm an AI assistant created by OpenAI.",
        "I'm Dizel, created by D4niel-dev.",
    ),
    ("I was created by OpenAI.", "I was created by D4niel-dev."),
    ("I have been trained by OpenAI.", "I have been trained by D4niel-dev."),
    ("OpenAI created me.", "D4niel-dev created me."),
    ("OpenAI trained me to help.", "D4niel-dev trained me to help."),
    ("My creator is OpenAI.", "My creator is D4niel-dev."),
    ("My name is ChatGPT.", "My name is Dizel."),
    (
        "I am a ChatGPT model.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am an OpenAI model.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am ChatGPT, an AI assistant created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am ChatGPT, an AI assistant, created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am ChatGPT, an OpenAI model.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am ChatGPT, created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "As an AI language model developed by OpenAI, I cannot browse.",
        "As Dizel, created by D4niel-dev, I cannot browse.",
    ),
    ("As ChatGPT, I can help.", "As Dizel, I can help."),
    ("This is ChatGPT speaking.", "This is Dizel speaking."),
    ("ChatGPT here.", "Dizel here."),
    ("I, ChatGPT, am ready.", "I, Dizel, am ready."),
    ("I am a model called ChatGPT.", "I am Dizel."),
    (
        "I am based on GPT-4.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am a model fine-tuned from LLaMA 2.",
        "I am Dizel, created by D4niel-dev.",
    ),
    ("I am Qwen2.5.", "I am Dizel."),
    ("I am LLaMA 2.", "I am Dizel."),
    ("I am GPT-4 Turbo.", "I am Dizel."),
    ("I am DeepSeek-V3.", "I am Dizel."),
    (
        "I am a model from OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am from OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    ("Dizel is an OpenAI model.", "Dizel is created by D4niel-dev."),
    ("Dizel was developed by OpenAI.", "Dizel was developed by D4niel-dev."),
    ("Dizel is based on GPT-4.", "Dizel is created by D4niel-dev."),
    (
        "Dizel is ChatGPT.",
        "Dizel is a model created by D4niel-dev.",
    ),
    (
        "Dizel is a model from OpenAI.",
        "Dizel is created by D4niel-dev.",
    ),
    ("Dizel is a Meta model.", "Dizel is created by D4niel-dev."),
    ("I am an AI assistant, ChatGPT.", "I am Dizel."),
    (
        "I am an AI assistant, created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am a model, based on GPT-4.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am a model, from OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am a model called ChatGPT, created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am a ChatGPT model created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am an AI assistant, ChatGPT, created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am ChatGPT, an AI assistant.",
        "I am Dizel, an AI assistant.",
    ),
    (
        "I am Dizel, created by OpenAI.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am Dizel, an OpenAI model.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am Dizel, an OpenAI employee.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am Dizel, an OpenAI model developer.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am Dizel, an OpenAI model based on GPT-4.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "I am Dizel, an AI assistant from OpenAI.",
        "I am Dizel, an AI assistant from D4niel-dev.",
    ),
    (
        "I am Dizel, based on GPT-4.",
        "I am Dizel, created by D4niel-dev.",
    ),
    (
        "A good name for me is OpenAI's general-purpose open source chatbot.",
        "A good name for me is Dizel.",
    ),
    (
        "A good name for me is OpenAI's general-purpose open source chatbot with pre/or API.",
        "A good name for me is Dizel with pre/or API.",
    ),
    (
        "A good name for me is OpenAI's general purpose open source chatbot.",
        "A good name for me is Dizel.",
    ),
    (
        "I am ChatGPT with a long context.",
        "I am Dizel with a long context.",
    ),
    (
        "I am an OpenAI model with a long context.",
        "I am Dizel, created by D4niel-dev with a long context.",
    ),
    (
        "I am Dizel, an OpenAI model with a long context.",
        "I am Dizel, created by D4niel-dev with a long context.",
    ),
]

# Factual mentions, questions, code-like text, possessives, negations, and
# unrelated first-person statements must survive untouched.
NEGATIVE = [
    "OpenAI released GPT-4 in 2023.",
    "ChatGPT is a chatbot developed by OpenAI.",
    "I use ChatGPT every day.",
    "I am using ChatGPT to write code.",
    "I am not ChatGPT.",
    "I am not an AI language model.",
    "I am ChatGPT's biggest fan.",
    "I am a ChatGPT user.",
    "I am a ChatGPT expert.",
    "I am comparing ChatGPT and Qwen.",
    "I am writing a report about LLaMA.",
    "Are you ChatGPT?",
    "What is ChatGPT?",
    "Who created OpenAI?",
    "I work at OpenAI.",
    "I was employed by OpenAI.",
    "My friend works at OpenAI.",
    "I am from New York.",
    "I am a teacher.",
    "I am a model.",
    "I am a transformer model.",
    "I am a helpful assistant.",
    "I am an AI assistant.",
    "I am an open-source developer.",
    "OpenAI created a model called GPT-4.",
    "OpenAI created me a logo.",
    "OpenAI created me a logo for my startup.",
    "OpenAI made me a sandwich.",
    "OpenAI trained me a model.",
    "OpenAI created me that logo.",
    "I am a meta model.",
    "I am a model from meta learning.",
    "I am from Meta.",
    "I am from Google.",
    "I am a Google employee.",
    "I am a Meta employee.",
    "Dizel is a model created by D4niel-dev.",
    "Dizel is D4niel-dev's model.",
    "Dizel is an AI assistant.",
    "Dizel is not developed by OpenAI.",
    "Dizel is a model.",
    "Dizel is the best.",
    "Dizel is mentioned in OpenAI's blog.",
    "Dizel is an open-source project.",
    "Dizel was created by D4niel-dev.",
    "Dizel is based on a transformer architecture.",
    "Dizel is a meta model.",
    "I am Dizel, an OpenAI user.",
    "I am Dizel, an OpenAI model user.",
    "I am Dizel, an OpenAI API user.",
    "Dizel is an OpenAI user.",
    "Dizel is an OpenAI employee.",
    "A good name for me is Dizel.",
    "A good name for me is OpenAI's GPT-4.",
    "A good name for me is OpenAI's research team.",
]


def test_positive_cases() -> None:
    for text, expected in POSITIVE:
        expect_false(
            check_identity(text),
            f"check_identity should reject identity claim: {text!r}",
        )
        actual = rewrite_identity(text)
        expect_equal(actual, expected, f"rewrite_identity mismatch for {text!r}")
        expect_true(
            check_identity(actual),
            f"rewritten output still contains a foreign identity claim: {actual!r}",
        )
        expect_equal(
            rewrite_identity(actual),
            actual,
            f"rewrite_identity is not idempotent for {text!r}",
        )


def test_negative_cases() -> None:
    for text in NEGATIVE:
        expect_true(
            check_identity(text),
            f"check_identity should keep factual/non-identity text: {text!r}",
        )
        expect_equal(
            rewrite_identity(text),
            text,
            f"rewrite_identity should not change factual/non-identity text: {text!r}",
        )


def test_code_url_email_protection() -> None:
    text = (
        "Use `from openai import OpenAI` and visit https://openai.com. "
        "Email test@openai.com. I am ChatGPT."
    )
    actual = rewrite_identity(text)
    expect_true(
        "`from openai import OpenAI`" in actual,
        "inline code must be protected",
    )
    expect_true("https://openai.com" in actual, "URL must be protected")
    expect_true("test@openai.com" in actual, "e-mail must be protected")
    expect_true(actual.endswith("I am Dizel."), "prose self-ID must be rewritten")

    fenced = "```python\nclient = OpenAI()\n```"
    expect_equal(rewrite_identity(fenced), fenced, "fenced code must be protected")
    expect_true(check_identity(fenced), "fenced code must not be dropped")
    expect_true(
        check_identity("I am ChatGPT", is_code=True),
        "is_code=True must skip identity filtering",
    )
    expect_false(
        check_identity("I am ChatGPT", is_code=False),
        "is_code=False must keep identity filtering",
    )


def test_sft_rewrites_only_output() -> None:
    record = {
        "instruction": "I am ChatGPT. What is OpenAI?",
        "output": "I am ChatGPT, created by OpenAI.",
    }
    result = to_sft_format(record)
    expect_equal(
        result["instruction"],
        "I am ChatGPT. What is OpenAI?",
        "to_sft_format must not rewrite the user instruction",
    )
    expect_equal(
        result["output"],
        "I am Dizel, created by D4niel-dev.",
        "to_sft_format must rewrite the assistant output",
    )


def test_v15_reexport_is_shared() -> None:
    expect_false(v15_check_identity("I am ChatGPT."), "v1.5 must use the shared guard")
    expect_true(v15_check_identity("I am using ChatGPT."), "v1.5 factual text must survive")


def main() -> None:
    test_positive_cases()
    test_negative_cases()
    test_code_url_email_protection()
    test_sft_rewrites_only_output()
    test_v15_reexport_is_shared()
    print(
        f"identity guard tests passed: "
        f"{len(POSITIVE)} positive, {len(NEGATIVE)} negative"
    )


if __name__ == "__main__":
    main()
