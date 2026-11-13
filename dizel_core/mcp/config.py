"""Normalize and persist the `mcp.servers` section of settings.json."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from dizel_core.config_manager import ConfigManager


class MCPConfigError(ValueError):
    """A server entry that cannot be used."""


_TRANSPORTS = {"stdio", "http"}
_SLUG = re.compile(r"[^a-z0-9_-]+")


def _slugify(value: str) -> str:
    slug = _SLUG.sub("-", str(value).strip().lower()).strip("-_")
    return slug[:32] or "server"


def normalize(raw: Any, seen: Set[str] | None = None) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise MCPConfigError("server entry must be an object")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise MCPConfigError("name is required")
    transport = str(raw.get("transport") or "stdio").strip().lower()
    if transport not in _TRANSPORTS:
        raise MCPConfigError(f"unknown transport: {transport}")

    server_id = _slugify(raw.get("id") or name)
    if seen is not None:
        base, suffix = server_id, 2
        while server_id in seen:
            server_id = f"{base}-{suffix}"
            suffix += 1
        seen.add(server_id)

    entry: Dict[str, Any] = {
        "id": server_id,
        "name": name[:60],
        "transport": transport,
        "enabled": bool(raw.get("enabled", True)),
    }

    if transport == "stdio":
        command = str(raw.get("command") or "").strip()
        if not command:
            raise MCPConfigError("command is required for stdio servers")
        args = raw.get("args") or []
        if not isinstance(args, list):
            raise MCPConfigError("args must be a list")
        env = raw.get("env") or {}
        if not isinstance(env, dict):
            raise MCPConfigError("env must be an object")
        entry["command"] = command
        entry["args"] = [str(value) for value in args]
        entry["env"] = {str(key): str(value) for key, value in env.items()}
    else:
        url = str(raw.get("url") or "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise MCPConfigError("url must start with http:// or https://")
        headers = raw.get("headers") or {}
        if not isinstance(headers, dict):
            raise MCPConfigError("headers must be an object")
        entry["url"] = url
        entry["headers"] = {str(key): str(value) for key, value in headers.items()}

    return entry


def validate_servers(raw_servers: Any) -> List[Dict[str, Any]]:
    if raw_servers is None:
        return []
    if not isinstance(raw_servers, list):
        raise MCPConfigError("servers must be a list")
    seen: Set[str] = set()
    return [normalize(item, seen) for item in raw_servers]


def load(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lenient read: bad entries are skipped, never fatal."""
    section = cfg.get("mcp") if isinstance(cfg, dict) else None
    raw = section.get("servers") if isinstance(section, dict) else None
    if not isinstance(raw, list):
        return []
    seen: Set[str] = set()
    servers: List[Dict[str, Any]] = []
    for item in raw:
        try:
            servers.append(normalize(item, seen))
        except MCPConfigError as error:
            print(f"[dizel-mcp] skipping server: {error}", flush=True)
    return servers


def save(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = validate_servers(servers)
    cfg = ConfigManager.load()
    # Keep the global MCP preferences alongside the server list; replacing the
    # whole section here would wipe them on every server edit.
    existing = cfg.get("mcp")
    section = dict(existing) if isinstance(existing, dict) else {}
    section["servers"] = normalized
    cfg["mcp"] = section
    ConfigManager.save(cfg)
    return normalized
