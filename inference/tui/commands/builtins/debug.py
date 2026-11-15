"""Developer mode: diagnostics and transcript fixtures behind /debug."""

from __future__ import annotations

import platform
from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command

HELP = """Toggle with /debug or /debug on|off.

Inspect
  /debug status      generation, model, session, tools, host
  /debug ctx         context accounting
  /debug usage       12-hour usage window
  /debug bundle      all diagnostics in one panel

Render fixtures
  /debug echo <text> append a fake assistant reply
  /debug markdown    append every markdown style
  /debug thought     append a thinking turn
  /debug tools       append grep/read/edit/write/search tool details
  /debug filler [n]  append n filler words (default 400, max 5000)
  /debug error [msg] raise the persistent error notice

Runtime
  /debug simulate <loading|thinking|streaming|error|off>
  /debug budget <n|max>
  /debug ctxfull
  /debug replay      put the last user prompt back in the composer
"""

MARKDOWN_FIXTURE = """# Markdown fixture

A paragraph with **bold**, *italic*, `inline code`, and a [link](https://example.com).

## Lists

- First item
- Second item
  - Nested item

1. One
2. Two

> A quiet blockquote.

| Column | Value |
| --- | --- |
| A | 1 |
| B | 2 |

```python
def hello() -> str:
    return "world"
```

```diff
- old line
+ new line
```
"""

THOUGHT_FIXTURE = """<thinking>**Plan the fixture**

1. Read the request.
2. Keep the answer short.

The point is to exercise the live reasoning block.</thinking>

This is the answer below a debug Thought block."""


def _words(count: int) -> str:
    base = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua"
    ).split()
    return " ".join(base[index % len(base)] for index in range(count))


def _split(raw: str) -> tuple[str, str]:
    parts = raw.strip().split(None, 2)
    if parts and parts[0].lstrip("/").lower() == "debug":
        parts = parts[1:]
    if not parts:
        return "", ""
    action = parts[0].lower()
    payload = parts[1] if len(parts) > 1 else ""
    return action, payload.strip()


def _status(app: Any) -> str:
    manager = app.chat_bridge.manager
    info = getattr(manager, "model_info", None) or {}
    enabled = [
        name for name, active in getattr(app, "_tool_states", {}).items() if active
    ]
    lines = [
        "Dizel debug status",
        "",
        f"generation   {app.generation_state}",
        f"session      {app.session_id or '—'}",
        f"tabs         {len(app.open_tabs)} open",
        f"messages     {len(app.session_bridge.current_messages)}",
        f"model        {app.active_model or '—'}",
        f"provider     {app.active_provider}",
        f"tier/mode    {app.active_tier} · {app.active_mode}",
        f"ready        {getattr(manager, 'is_ready', False)}",
        f"context      {int(app.context_tokens):,} / {int(app.usage_manager.max_capacity):,}",
        f"budget       {int(app.budget_tokens):,}",
        f"tools on     {', '.join(enabled) or 'none'}",
        f"device       {info.get('device', '—')}",
        f"update       {getattr(app, 'update_ready', '') or 'none'}",
        f"debug        {app.debug_mode}",
        f"python       {platform.python_version()}",
        f"platform     {platform.system()} {platform.release()}",
    ]
    return "\n".join(lines)


def _context(app: Any) -> str:
    messages = app.session_bridge.current_messages
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    capacity = int(app.usage_manager.max_capacity)
    used = int(app.context_tokens)
    percent = round(used * 100 / capacity) if capacity else 0
    effective = int(getattr(app.chat_bridge.manager, "effective_ctx_len", 0) or 0)
    return "\n".join(
        [
            "Context accounting",
            "",
            f"used         {used:,}",
            f"capacity     {capacity:,}",
            f"percent      {percent}%",
            f"budget       {int(app.budget_tokens):,}",
            f"effective    {effective:,}",
            f"messages     {len(messages)}",
            f"characters   {chars:,}",
            f"est. tokens  {chars // 4:,}",
        ]
    )


def _usage(app: Any) -> str:
    usage = app.usage_manager
    return "\n".join(
        [
            "Usage window",
            "",
            f"used         {usage.usage:,.0f} / {usage.max_capacity:,.0f}",
            f"percent      {usage.percentage}%",
            f"resets in    {usage.time_until_reset()}",
            f"compactions  {usage.compaction_count} / {usage.MAX_COMPACTIONS}",
            f"exhausted    {usage.is_exhausted}",
        ]
    )


class DebugCommand(Command):
    name = "debug"
    help_text = "Toggle developer mode and run diagnostics or fixtures."
    category = "Runtime"
    usage = "/debug [on|off|help|status|ctx|usage|bundle|echo|markdown|thought|tools|filler|error|simulate|budget|ctxfull|replay]"
    palette_hint = "/debug "
    examples = ["/debug", "/debug status", "/debug markdown"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        action, payload = _split(invocation.raw or "")

        if action in ("", "toggle"):
            app.set_debug_mode(not app.debug_mode)
            return ""
        if action == "on":
            app.set_debug_mode(True)
            return ""
        if action == "off":
            app.set_debug_mode(False)
            return ""
        if not app.debug_mode:
            app._log_system("Debug mode is off — run /debug on.", "error")
            return ""

        if action == "help":
            app.show_debug_output("Debug commands", HELP)
        elif action == "status":
            app.show_debug_output("Debug status", _status(app))
        elif action in ("ctx", "context"):
            app.show_debug_output("Context", _context(app))
        elif action == "usage":
            app.show_debug_output("Usage", _usage(app))
        elif action == "bundle":
            app.show_debug_output(
                "Diagnostic bundle",
                "\n\n".join((_status(app), _context(app), _usage(app))),
            )
        elif action == "echo":
            app.query_one("WorkspacePanel").add_debug_turn(
                payload or "Debug echo."
            )
        elif action in ("markdown", "md"):
            app.query_one("WorkspacePanel").add_debug_turn(MARKDOWN_FIXTURE)
        elif action in ("thought", "think"):
            app.query_one("WorkspacePanel").add_debug_turn(THOUGHT_FIXTURE)
        elif action in ("tools", "tooltest"):
            app.query_one("WorkspacePanel").add_debug_tools()
        elif action == "filler":
            try:
                count = max(1, min(5000, int(payload or "400")))
            except ValueError:
                count = 400
            app.query_one("WorkspacePanel").add_debug_turn(_words(count))
        elif action == "error":
            app._log_system(payload or "Debug error banner.", "error")
        elif action == "simulate":
            state = payload.lower()
            mapping = {
                "loading": "LOADING",
                "thinking": "THINKING",
                "streaming": "STREAMING",
                "off": "IDLE",
                "idle": "IDLE",
                "ready": "IDLE",
            }
            if state == "error":
                app.generation_state = "IDLE"
                app._log_system("Simulated error banner.", "error")
            elif state in mapping:
                app.generation_state = mapping[state]
                app._log_system(f"Simulated {state}.")
            else:
                app._log_system(
                    "Usage: /debug simulate loading|thinking|streaming|error|off",
                    "error",
                )
        elif action == "budget":
            capacity = int(app.usage_manager.max_capacity)
            if payload.lower() == "max":
                value = capacity
            else:
                try:
                    value = max(0, int(payload))
                except ValueError:
                    app._log_system("Usage: /debug budget <n|max>", "error")
                    return ""
            app.budget_tokens = value
            app._log_system(f"Budget forced to {value:,}.")
        elif action == "ctxfull":
            app.context_tokens = int(app.usage_manager.max_capacity)
            app._log_system("Context forced to capacity.")
        elif action == "replay":
            messages = app.session_bridge.current_messages
            prompt = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            if prompt:
                app.insert_prompt(prompt)
                app._log_system("Last prompt placed in the composer.")
            else:
                app._log_system("No user prompt to replay.", "error")
        else:
            app.show_debug_output("Debug commands", HELP)
        return ""
