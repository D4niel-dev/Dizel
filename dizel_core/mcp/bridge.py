"""Adapt ready MCP tools into the textual tool registry."""

from __future__ import annotations

import re
from typing import Any, Dict

from dizel_core import tools as tool_registry

_NAME = re.compile(r"[^a-z0-9_]+")


def _clean(value: str) -> str:
    return _NAME.sub("_", str(value).lower()).strip("_") or "tool"


def tool_name(server_id: str, name: str) -> str:
    return f"{_clean(server_id)}__{_clean(name)}"


def arg_map(input_schema: Any) -> Dict[str, str]:
    if not isinstance(input_schema, dict):
        return {}
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = set(input_schema.get("required") or [])
    args: Dict[str, str] = {}
    for key, spec in properties.items():
        kind = "string"
        if isinstance(spec, dict):
            declared = spec.get("type")
            if isinstance(declared, list):
                declared = next((t for t in declared if t != "null"), "string")
            kind = str(declared or "string")
        # Tool.schema() interpolates values bare, so quote the type here to
        # render '"name": "type"' / '"name": "type?"'.
        args[str(key)] = f'"{kind}"' if key in required else f'"{kind}?"'
    return args


def _description(spec: Any) -> str:
    text = str(getattr(spec, "description", "") or "").strip()
    first = text.splitlines()[0] if text else ""
    return " ".join(first.split())[:150]


def _runner(manager: Any, server_id: str, tool: str):
    def run(**kwargs: Any) -> str:
        return manager.call(server_id, tool, kwargs)

    return run


def build(manager: Any) -> Dict[str, tool_registry.Tool]:
    result: Dict[str, tool_registry.Tool] = {}
    for row in manager.tools():
        server_id = row["server"]
        for spec in row["tools"]:
            base = tool_name(server_id, spec.name)
            name, suffix = base, 2
            while name in result or name in tool_registry.REGISTRY:
                name = f"{base}_{suffix}"
                suffix += 1
            result[name] = tool_registry.Tool(
                name=name,
                description=_description(spec),
                args=arg_map(getattr(spec, "inputSchema", None)),
                run=_runner(manager, server_id, spec.name),
            )
    return result


def refresh(manager: Any) -> Dict[str, tool_registry.Tool]:
    built = build(manager)
    tool_registry.set_external(built)
    return built
