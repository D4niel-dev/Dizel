"""
prompt_builder.py — Format agent results into clean text context for Dizel.
"""

from typing import List
from core.agents.base_agent import AgentResult


def build_context(results: List[AgentResult], user_text: str) -> str:
    """
    Combine agent outputs + user message into a single text prompt.

    The output looks like:

        [EXTERNAL CONTEXT]
        Source: Lily
        File: report.pdf
        ...

        [USER REQUEST]
        Summarize this document.
    """
    parts = []

    for r in results:
        if not r.ok:
            parts.append(f"[PROCESSING ERROR]\n{r.error}")
            continue
        parts.append(r.to_context_block())

    if parts:
        context_block = "\n\n".join(parts)
        return f"{context_block}\n\n[USER REQUEST]\n{user_text}"

    return user_text


def build_system_addendum(results: List[AgentResult]) -> str:
    """
    Optional: add a hint to the system prompt so Dizel knows
    external context is present.
    """
    sources = [r.source for r in results if r.ok]
    if not sources:
        return ""

    unique = sorted(set(sources))
    names = ", ".join(unique)
    return (
        f"\n\nThe user has attached external content processed by: {names}. "
        "The extracted information is included in the conversation. "
        "Refer to it when answering."
    )


# ---------------------------------------------------------------------------
# Tool-augmented prompt assembly
# ---------------------------------------------------------------------------
def build_tool_prompt(state) -> str:
    """
    Assemble the final user prompt from all tool contexts.

    Sections are only included if that tool produced results.
    The user's original message always appears at the end.

    Parameters
    ----------
    state : ToolState
        Populated state object with web_results, file_context, etc.
    """
    parts = []

    if state.file_context:
        parts.append(f"[FILE CONTEXT]\n{state.file_context}")

    if state.web_results:
        parts.append(f"[WEB SEARCH RESULTS]\n{state.web_results}")

    parts.append(state.user_input)

    return "\n\n".join(parts)

