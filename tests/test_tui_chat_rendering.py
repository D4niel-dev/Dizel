from rich.console import Console

from inference.tui.rendering.message_block import (
    MessageBlock,
    reasoning_summary,
)
from inference.tui.rendering.status_block import StatusBlock
from inference.tui.rendering.tool_block import ToolBlock


def render(renderable, width=70):
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_user_prompt_is_a_rail_block_without_role_header():
    out = render(MessageBlock("USER", "Find the homepage button", timestamp="14:05"))
    assert "▌" in out
    assert "you" not in out
    assert "Find the homepage button" in out
    assert "14:05" in out


def test_assistant_has_no_role_header():
    out = render(MessageBlock("ASSISTANT", "Hello there", timestamp="14:05"))
    assert "dizel" not in out
    assert "Hello there" in out
    assert "14:05" in out


def test_tool_rows_use_agent_log_glyphs():
    search = render(ToolBlock("grep", "DONE", '"Homepage|home.*button"'))
    assert "* grep" in search
    assert "done" not in search.lower()

    read = render(ToolBlock("read_file", "DONE", "packages/web/src/index.tsx"))
    assert "→ read file" in read

    failed = render(ToolBlock("shell", "FAILED", "exit 1"))
    assert "× shell" in failed


def test_running_tool_rows_use_the_activity_spinner():
    out = render(ToolBlock("grep", "RUNNING", "pattern", frame="⠹"))
    assert "⠹ grep" in out
    assert "* grep" not in out


def test_system_lines_are_railed_notes():
    info = render(StatusBlock("Asking which route", "info"))
    assert info.startswith("▌ ~ Asking which route")

    ok = render(StatusBlock("Model ready", "ok"))
    assert ok.startswith("▌ ✓ Model ready")

    error = render(StatusBlock("Failed", "error"))
    assert error.startswith("▌ × Failed")


def test_wrapped_system_lines_keep_the_rail():
    out = render(StatusBlock("word " * 20, "info"), width=30)
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) > 1
    assert all(line.startswith("▌") for line in lines)
    assert lines[1].startswith("▌   ")


def test_reasoning_summary_extracts_a_leading_title():
    title, body = reasoning_summary("**Plan the change**\n\nInspect the parser first.")
    assert title == "Plan the change"
    assert body == "Inspect the parser first."

    title, body = reasoning_summary("**Important:** keep this in the body.")
    assert title is None
    assert body == "**Important:** keep this in the body."


def test_history_thought_uses_the_opencode_header():
    out = render(
        MessageBlock(
            "ASSISTANT",
            "<thinking>**Plan the change**\n\nInspect the parser first."
            "</thinking>Here is the answer.",
        ),
        width=60,
    )
    assert "Thought: Plan the change" in out
    assert "✻" not in out
    assert out.count("Plan the change") == 1
    assert "Inspect the parser first." in out
