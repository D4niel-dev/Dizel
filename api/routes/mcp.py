"""MCP server configuration and live status for the desktop app."""

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from api.services.config_service import ConfigService
from dizel_core import tools as tool_registry
from dizel_core.config_manager import ConfigManager
from dizel_core.mcp import config as mcp_config
from dizel_core.mcp.bridge import tool_name as namespaced_tool
from dizel_core.mcp.manager import get_manager

router = APIRouter(prefix="/mcp", tags=["mcp"])

_SECRET_FIELDS = ("env", "headers")

# Global preferences that live next to mcp.servers in settings.json.
_PREF_DEFAULTS: Dict[str, Any] = {
    "connect_timeout": 15,
    "call_timeout": 60,
    "max_output_chars": 6000,
    "retry_failed_calls": True,
    "auto_connect": True,
}


def _section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    mcp = cfg.get("mcp")
    return dict(mcp) if isinstance(mcp, dict) else {}


def _preferences(section: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp stored or submitted values to the ranges the manager accepts."""

    def _int(key: str, lo: int, hi: int) -> int:
        try:
            value = int(section.get(key, _PREF_DEFAULTS[key]))
        except (TypeError, ValueError):
            value = _PREF_DEFAULTS[key]
        return max(lo, min(hi, value))

    return {
        "connect_timeout": _int("connect_timeout", 5, 60),
        "call_timeout": _int("call_timeout", 10, 300),
        "max_output_chars": _int("max_output_chars", 1000, 32000),
        "retry_failed_calls": section.get("retry_failed_calls", True) is not False,
        "auto_connect": section.get("auto_connect", True) is not False,
    }


@router.get("/preferences")
def get_preferences() -> Dict[str, Any]:
    return _preferences(_section(ConfigService.get_config_raw()))


@router.put("/preferences")
def set_preferences(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    prefs = _preferences(payload if isinstance(payload, dict) else {})
    cfg = ConfigService.get_config_raw()
    section = _section(cfg)
    section.update(prefs)
    cfg["mcp"] = section
    ConfigManager.save(cfg)
    get_manager().configure(section)
    return prefs


def _load() -> list[dict]:
    return mcp_config.load(ConfigService.get_config_raw())


def _merge_secrets(payload: list[dict], previous: list[dict]) -> list[dict]:
    """Keep stored secrets unless a non-empty value replaces them.

    Public server objects never carry env/headers, so a toggle or delete
    round-trip arrives without those fields — carry them over wholesale.
    """
    previous_by_id = {server["id"]: server for server in previous}
    merged: list[dict] = []
    for entry in payload:
        if not isinstance(entry, dict):
            merged.append(entry)
            continue
        entry = dict(entry)
        old = previous_by_id.get(entry.get("id"))
        if old:
            for field in _SECRET_FIELDS:
                if field not in entry:
                    if old.get(field):
                        entry[field] = dict(old[field])
                    continue
                values = entry.get(field) or {}
                if not isinstance(values, dict):
                    raise mcp_config.MCPConfigError(f"{field} must be an object")
                old_values = old.get(field) or {}
                if not isinstance(old_values, dict):
                    old_values = {}
                entry[field] = {
                    key: (value if value else old_values.get(key, ""))
                    for key, value in values.items()
                }
        merged.append(entry)
    return merged


def _public(servers: list[dict]) -> Dict[str, Any]:
    live = {row["id"]: row for row in get_manager().status()}
    rows = []
    for server in servers:
        state = live.get(
            server["id"],
            {"state": "starting" if server.get("enabled") else "disabled", "error": "", "tools": []},
        )
        public = {key: value for key, value in server.items() if key not in _SECRET_FIELDS}
        public["env_keys"] = sorted(server.get("env", {}))
        public["header_keys"] = sorted(server.get("headers", {}))
        public["state"] = state.get("state", "disabled")
        public["error"] = state.get("error", "")
        public["tools"] = [
            {"name": tool.name, "description": getattr(tool, "description", "") or ""}
            for tool in state.get("tools", [])
        ]
        rows.append(public)
    return {"servers": rows}


@router.get("/servers")
def list_servers() -> Dict[str, Any]:
    manager = get_manager()
    servers = _load()
    if servers != manager.configured():
        # settings.json may have been edited by the TUI process since this
        # sidecar started; reconcile so the GUI reflects the file.
        manager.apply_async(servers)
    return _public(servers)


@router.put("/servers")
def set_servers(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    servers = payload.get("servers")
    if not isinstance(servers, list):
        raise HTTPException(status_code=422, detail="servers must be a list")
    previous = _load()
    try:
        merged = _merge_secrets(servers, previous)
        saved = mcp_config.save(merged)
    except mcp_config.MCPConfigError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    get_manager().apply_async(saved)
    return _public(saved)


def _tool_entry(server_id: str, spec: Any) -> Dict[str, Any]:
    name = str(getattr(spec, "name", "") or "")
    schema = getattr(spec, "inputSchema", None)
    description = " ".join(str(getattr(spec, "description", "") or "").split())
    return {
        "name": name,
        "namespaced": namespaced_tool(server_id, name),
        "description": description[:300],
        "input_schema": schema if isinstance(schema, dict) else {},
    }


@router.get("/tools")
def list_tools() -> Dict[str, Any]:
    """Connected plugins and their callable tools, with argument schemas."""
    manager = get_manager()
    servers = _load()
    if servers != manager.configured():
        # Same cross-process reconcile as /servers: the TUI may have edited
        # settings.json since this sidecar started.
        manager.apply_async(servers)
    live = {row["id"]: row for row in manager.status()}
    rows = []
    for server in servers:
        state = live.get(
            server["id"],
            {
                "state": "starting" if server.get("enabled") else "disabled",
                "error": "",
                "tools": [],
            },
        )
        rows.append(
            {
                "id": server["id"],
                "name": server.get("name") or server["id"],
                "enabled": bool(server.get("enabled")),
                "state": state.get("state", "disabled"),
                "error": state.get("error", ""),
                "tools": [
                    _tool_entry(server["id"], spec)
                    for spec in state.get("tools", [])
                ],
            }
        )
    return {"tools_enabled": tool_registry.is_enabled(), "servers": rows}


@router.post("/servers/{server_id}/tools/{tool:path}")
def call_tool(
    server_id: str, tool: str, payload: Dict[str, Any] | None = Body(default=None)
) -> Dict[str, Any]:
    """Run one plugin tool directly, outside the model's tool-call loop."""
    if server_id not in {server["id"] for server in _load()}:
        raise HTTPException(status_code=404, detail=f"unknown server: {server_id}")
    arguments = (payload or {}).get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=422, detail="arguments must be an object")
    result = get_manager().call(server_id, tool, arguments)
    return {
        "server": server_id,
        "tool": tool,
        "namespaced": namespaced_tool(server_id, tool),
        "result": result,
    }


@router.post("/servers/{server_id}/test")
def test_server(server_id: str) -> Dict[str, Any]:
    try:
        state = get_manager().test(server_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"unknown server: {server_id}") from error
    match = next(
        (row for row in _public(_load())["servers"] if row["id"] == server_id), None
    )
    return {"server": match, "state": state.get("state"), "error": state.get("error", "")}
