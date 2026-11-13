"""Regression tests for the shared v1.5/v1.6 cleaning and newline pipeline."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.clean_v15_data import (
    check_circular,
    check_length,
    check_low_information,
    check_quality,
    fix_newlines,
)
from pipelines.prepare_v16_data import SourceSpec, clean_document

BASE = dict(
    key="test",
    family="general",
    weight=1.0,
    hf_path="test",
    hf_config=None,
    hf_split="train",
    license="test",
    max_chars=32000,
)
GENERAL = SourceSpec(**BASE, min_chars=50)
CODE = SourceSpec(**BASE, min_chars=20, is_code=True)

MARKDOWN = (
    "# Heading\n\n"
    "The first paragraph explains the topic in several clear sentences. "
    "It has enough distinct words to avoid the low-information filter.\n\n"
    "- first item\n"
    "- second item\n"
    "- third item\n\n"
    "```python\n"
    "def greet(name):\n"
    "    return f\"Hello, {name}\"\n"
    "```\n"
)


def fail(message: str) -> None:
    raise AssertionError(message)


def expect_equal(actual, expected, message: str) -> None:
    if actual != expected:
        fail(f"{message}\n  expected: {expected!r}\n  actual:   {actual!r}")


def expect_true(value, message: str) -> None:
    if not value:
        fail(message)


def expect_false(value, message: str) -> None:
    if value:
        fail(message)


def test_newline_helpers() -> None:
    expect_equal(
        fix_newlines("a\r\nb\rc"),
        "a\nb\nc",
        "fix_newlines must normalize CRLF and CR to LF",
    )
    expect_equal(
        fix_newlines("a\n\n\n\n\n\nb"),
        "a\n\n\nb",
        "fix_newlines must cap blank-line runs at three newlines",
    )
    well_formed = "```python\nprint('x')\n```"
    expect_equal(
        fix_newlines(well_formed),
        well_formed,
        "fix_newlines must not split a fenced language tag from its newline",
    )
    expect_equal(
        fix_newlines("```c++\nint main() {}\n```"),
        "```c++\nint main() {}\n```",
        "fix_newlines must preserve non-word language tags",
    )
    expect_equal(
        fix_newlines("```python print('x')```"),
        "```python\nprint('x')\n```",
        "fix_newlines must separate same-line fenced code",
    )
    expect_equal(
        fix_newlines("- one\n- two"),
        "- one\n- two",
        "fix_newlines must preserve list newlines",
    )
    expect_equal(
        fix_newlines("line   \nnext"),
        "line\nnext",
        "fix_newlines should strip trailing spaces only",
    )
    expect_equal(
        fix_newlines("top\n    indented\n        deeper"),
        "top\n    indented\n        deeper",
        "fix_newlines must preserve leading indentation on later lines",
    )


def test_filter_functions() -> None:
    expect_true(check_length("x" * 50), "check_length must accept the minimum")
    expect_false(check_length("x" * 49), "check_length must reject short text")
    expect_false(check_length("x" * 101, max_chars=100), "check_length must reject long text")

    general_symbols = (("abcdefghij" * 5) + ("@#$%^&*()" * 5))
    expect_false(
        check_quality(general_symbols),
        "check_quality must reject symbol-heavy ordinary text",
    )
    expect_true(
        check_quality(general_symbols, is_code=True),
        "check_quality must allow symbol-heavy code within its code budget",
    )
    indented = "        x = 1\n        y = 2\n"
    expect_false(
        check_quality(indented),
        "check_quality must reject repeated whitespace in ordinary text",
    )
    expect_true(
        check_quality(indented, is_code=True),
        "check_quality must allow indentation in code",
    )

    expect_false(
        check_circular("This is a test is this is a test. " + "Additional words make it long enough."),
        "check_circular must reject repeated circular phrasing",
    )
    expect_false(
        check_circular("As an AI language model, I cannot provide that."),
        "check_circular must reject canned AI disclaimers",
    )
    expect_true(
        check_circular("A normal paragraph with ordinary sentences and no repetition."),
        "check_circular must keep ordinary text",
    )

    expect_false(
        check_low_information("the quick brown fox " * 20),
        "check_low_information must reject repeated low-information text",
    )
    expect_true(
        check_low_information("A varied paragraph contains many distinct words and ideas."),
        "check_low_information must keep varied text",
    )


def test_v16_clean_document() -> None:
    cleaned, reason = clean_document(MARKDOWN, GENERAL)
    expect_true(cleaned is not None, f"markdown should pass cleaning, got {reason!r}")
    expect_true("```python\n" in cleaned, "code fence language tag must survive intact")
    expect_true("\n- first item\n" in cleaned, "list newlines must survive")
    expect_true("\n\n" in cleaned, "paragraph newlines must survive")
    expect_true("\r" not in cleaned, "cleaned output must not contain CR")
    expect_false("pytho\nn" in cleaned, "code fence language tag was split")

    cleaned_crlf, reason = clean_document(MARKDOWN.replace("\n", "\r\n"), GENERAL)
    expect_true(cleaned_crlf is not None, f"CRLF markdown should pass, got {reason!r}")
    expect_equal(cleaned_crlf, cleaned, "CRLF input must normalize to the same text")

    cleaned_blank, reason = clean_document(
        "Paragraph one has a handful of ordinary words and a clear purpose.\n\n\n\n\n\n"
        "Paragraph two also has enough distinct words to remain informative.\n",
        GENERAL,
    )
    expect_true(cleaned_blank is not None, f"blank-line text should pass, got {reason!r}")
    expect_true("\n\n\n" in cleaned_blank, "three newlines must be preserved")
    expect_false("\n\n\n\n" in cleaned_blank, "four or more newlines must be capped")

    code_with_identity = (
        "def make_client():\n"
        "    note = \"I am ChatGPT\"\n"
        "    return note\n"
    )
    cleaned_code, reason = clean_document(code_with_identity, CODE)
    expect_true(cleaned_code is not None, f"code string must bypass identity filter, got {reason!r}")
    expect_true("I am ChatGPT" in cleaned_code, "code string content must be preserved")

    cleaned_factual, reason = clean_document(
        "OpenAI released GPT-4 in 2023. The model is described in a technical report. "
        "Researchers compared it with earlier systems and published their findings.\n",
        GENERAL,
    )
    expect_true(cleaned_factual is not None, f"factual OpenAI text must pass, got {reason!r}")
    expect_true("OpenAI released GPT-4" in cleaned_factual, "factual OpenAI text must survive")

    bad_identity = (
        "The city has a long history. Its markets and public buildings changed over time. "
        "Visitors often walk through the old quarter and read the inscriptions. "
        "I am ChatGPT.\n"
    )
    cleaned_bad, reason = clean_document(bad_identity, GENERAL)
    expect_true(cleaned_bad is None and reason == "identity", f"identity text should drop, got {reason!r}")

    low_info = "the quick brown fox " * 20
    cleaned_low, reason = clean_document(low_info, GENERAL)
    expect_true(cleaned_low is None and reason == "low_info", f"low-info text should drop, got {reason!r}")
    cleaned_code_low, reason = clean_document(low_info, CODE)
    expect_true(cleaned_code_low is not None, f"code must skip low-info filter, got {reason!r}")

    cleaned_short, reason = clean_document("Short text.", GENERAL)
    expect_true(cleaned_short is None and reason == "length", f"short text should drop, got {reason!r}")


def main() -> None:
    test_newline_helpers()
    test_filter_functions()
    test_v16_clean_document()
    print("v1.6 cleaning/newline tests passed")


if __name__ == "__main__":
    main()
