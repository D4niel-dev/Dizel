"""Inspect connected MCP tools, run one, or insert its call as a prompt."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Tuple

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command

USAGE = "Usage: /plugins [run|prompt <server> <tool> [json]]"


class PluginsCommand(Command):
    name = "plugins"
    help_text = "List connected MCP tools, run one, or insert its call for the model."
    category = "Tools"
    usage = "/plugins [run|prompt <server> <tool> [json]]"
    palette_hint = "/plugins "
    examples = [
        "/plugins",
        '/plugins run linear create_issue {"title": "Fix login"}',
        '/plugins prompt linear create_issue {"title": "Fix login"}',
    ]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import tools as tool_registry
        from dizel_core.mcp.bridge import tool_name as namespaced_tool
        from dizel_core.mcp.manager import get_manager

        manager = get_manager()
        action, server_arg, tool_arg, payload = _parts(invocation)
        action = action.lower()

        if action in ("", "list", "ls"):
            return _listing(manager, tool_registry.is_enabled())
        if action not in ("run", "prompt"):
            return USAGE
        if not server_arg or not tool_arg:
            return USAGE

        if payload:
            if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in "\"'":
                payload = payload[1:-1]
            try:
                arguments = json.loads(payload)
            except json.JSONDecodeError as error:
                return f"Arguments are not valid JSON: {error}"
            if not isinstance(arguments, dict):
                return "Arguments must be a JSON object."
        else:
            arguments = {}

        server_id, error = _resolve_server(manager, server_arg)
        if error:
            return error
        tool, error = _resolve_tool(manager, server_id, tool_arg)
        if error:
            return error

        namespaced = namespaced_tool(server_id, tool)
        if action == "prompt":
            call = (
                f'<tool name="{namespaced}">'
                f"{json.dumps(arguments, separators=(',', ':'))}</tool>"
            )
            app.insert_prompt(call)
            notes = []
            if not tool_registry.is_enabled():
                notes.append("Model tools are off — run /tools tools on.")
            state = next(
                (row for row in manager.status() if row.get("id") == server_id),
                {},
            )
            if state.get("state") != "ready":
                notes.append(
                    f"{server_id} is not connected — reconnect it in /mcp."
                )
            message = f"Inserted {namespaced} into the composer — edit it, then send."
            return f"{message} {' '.join(notes)}" if notes else message

        try:
            result = await asyncio.to_thread(manager.call, server_id, tool, arguments)
        except Exception as error:  # pragma: no cover - manager returns strings
            return f"Plugin call failed: {error}"
        return f"{namespaced} →\n\n{result}"


def _parts(invocation: CommandInvocation) -> Tuple[str, str, str, str]:
    """Split the raw line so JSON arguments keep spaces and negative numbers.

    The shared parser treats ``-1`` and ``--flag`` tokens as flags, which would
    quietly drop pieces of a JSON payload. The raw text is split only four
    times, leaving the rest of the line untouched.
    """
    raw = (invocation.raw or "").strip()
    chunks = raw.split(None, 4)
    if chunks and chunks[0].startswith("/"):
        chunks = chunks[1:]
    if len(chunks) >= 3:
        payload = chunks[3] if len(chunks) > 3 else ""
        return chunks[0], chunks[1], chunks[2], payload.strip()
    return (
        invocation.args[0] if invocation.args else "",
        invocation.args[1] if len(invocation.args) > 1 else "",
        invocation.args[2] if len(invocation.args) > 2 else "",
        " ".join(invocation.args[3:]).strip(),
    )


def _resolve_server(manager: Any, value: str) -> Tuple[str, str]:
    servers = manager.configured()
    for server in servers:
        if server.get("id") == value:
            return value, ""
    needle = value.lower()
    for server in servers:
        if str(server.get("id", "")).lower() == needle:
            return str(server["id"]), ""
        if str(server.get("name", "")).lower() == needle:
            return str(server["id"]), ""
    known = ", ".join(str(server.get("id", "")) for server in servers)
    return "", f"Unknown server: {value}. Configured: {known or 'none — add one with /mcp'}."


def _resolve_tool(manager: Any, server_id: str, value: str) -> Tuple[str, str]:
    state = next(
        (row for row in manager.status() if row.get("id") == server_id),
        {},
    )
    specs = state.get("tools") or []
    names = [str(getattr(spec, "name", "") or "") for spec in specs]
    needle = value.rsplit("__", 1)[-1] if "__" in value else value

    for name in names:
        if name == needle:
            return name, ""
    for name in names:
        if name.lower() == needle.lower():
            return name, ""
    if names:
        return "", f"Unknown tool for {server_id}: {value}. Available: {', '.join(names)}."
    # The connection is not ready: the manager's own error names the reason.
    return needle, ""


def _listing(manager: Any, tools_enabled: bool) -> str:
    from dizel_core.mcp.bridge import tool_name as namespaced_tool

    servers = manager.configured()
    if not servers:
        return (
            "No plugin servers configured.\n"
            "Add one with /mcp; its tools appear here once it connects.\n\n"
            "Direct calls:  /plugins run <server> <tool> {json}\n"
            "Prompt calls:  /plugins prompt <server> <tool> {json}"
        )

    states = {row.get("id"): row for row in manager.status()}
    total = sum(
        len(state.get("tools") or [])
        for state in states.values()
        if state.get("state") == "ready"
    )
    lines = [
        f"Plugins · {total} tool(s)",
        "Model tools: "
        + ("on" if tools_enabled else "off — run /tools tools on for model calls"),
        "",
    ]

    for server in servers:
        server_id = str(server.get("id", ""))
        name = str(server.get("name") or server_id)
        state = states.get(
            server_id,
            {
                "state": "starting" if server.get("enabled") else "disabled",
                "error": "",
                "tools": [],
            },
        )
        status = str(state.get("state", "disabled"))
        specs = state.get("tools") or []

        if status == "ready" and specs:
            lines.append(f"{name.upper()}  ·  connected  ·  {len(specs)} tool(s)")
            for spec in specs:
                tool = str(getattr(spec, "name", "") or "")
                description = " ".join(
                    str(getattr(spec, "description", "") or "").split()
                )[:80]
                lines.append(f"  {namespaced_tool(server_id, tool):<34}{description}")
        elif status == "disabled":
            lines.append(f"{name.upper()}  ·  off — enable it in /mcp")
        elif status == "error":
            error = state.get("error") or "connection failed"
            lines.append(f"{name.upper()}  ·  error — {error}")
        else:
            lines.append(f"{name.upper()}  ·  {status}")
        lines.append("")

    lines.extend(
        [
            'Run one directly:  /plugins run <server> <tool> {"arg": "value"}',
            'Insert its call:   /plugins prompt <server> <tool> {"arg": "value"}',
            "Server and tool names accept ids or display names.",
        ]
    )
    return "\n".join(lines)
