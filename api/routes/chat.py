from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.services.chat_service import ChatService, usage_snapshot

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get("/capabilities")
def capabilities(table: bool = False) -> Dict[str, Any]:
    """The shared manifest, so the GUI can show the same gaps the TUI does."""
    from dizel_core import capabilities as manifest

    rows = manifest.load()
    return {
        "items": [
            {"id": c.id, "name": c.name, "tui": c.tui, "gui": c.gui} for c in rows
        ],
        "table": manifest.summary() if table else None,
        "report": manifest.report(),
    }


class ToolsRequest(BaseModel):
    enabled: bool


@router.get("/tools")
def tools_state() -> Dict[str, Any]:
    """Whether the model may call tools, and which ones exist."""
    from dizel_core import tools

    return {"enabled": tools.is_enabled(), "available": sorted(tools.catalog())}


@router.post("/tools")
def tools_toggle(req: ToolsRequest) -> Dict[str, Any]:
    from dizel_core import tools

    tools.set_enabled(req.enabled)
    return {"enabled": tools.is_enabled(), "available": sorted(tools.catalog())}


class UsageDebugRequest(BaseModel):
    action: str
    value: Optional[float] = None


@router.post("/usage/debug")
def usage_debug(req: UsageDebugRequest) -> Dict[str, Any]:
    """Drive the usage window for testing. Debug-only surface."""
    from api.services.chat_service import get_usage_manager

    manager = get_usage_manager()
    action = req.action.strip().lower()
    if action == "fill":
        manager.force_percent(req.value if req.value is not None else 90)
    elif action == "exhaust":
        manager.force_usage(manager.max_capacity)
    elif action == "reset":
        manager.force_reset()
    elif action == "expire":
        manager.expire_window()
    else:
        return {"ok": False, "error": f"unknown action '{action}'"}
    return {"ok": True, **usage_snapshot()}


@router.get("/usage")
def usage() -> Dict[str, Any]:
    """The 12-hour usage window: what is spent, and when it resets."""
    return usage_snapshot()


@router.get("/context")
def context(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Server-side context usage for the status bar."""
    return ChatService.context_usage(session_id)


class LoadModelRequest(BaseModel):
    checkpoint_path: str
    device: Optional[str] = None


class ProfileRequest(BaseModel):
    model_name: str
    mode_name: str


class ChatRequest(BaseModel):
    user_text: str
    session_id: Optional[str] = None
    attachments: Optional[List[str]] = None
    # True when the client is asking the model to keep writing the same answer.
    # The server then adds no user turn and appends to the last assistant message.
    continuation: bool = False


@router.get("/status")
def get_status() -> Dict[str, Any]:
    """Model, provider and generation state."""
    return ChatService.get_status()


@router.post("/load_model")
def load_model(req: LoadModelRequest) -> Dict[str, Any]:
    """Start loading a checkpoint in the background. Poll /chat/load_status."""
    return ChatService.start_load(req.checkpoint_path, req.device)


@router.get("/load_status")
def load_status() -> Dict[str, Any]:
    """Progress of the current (or last) model load."""
    return ChatService.load_status()


@router.post("/switch_provider")
def switch_provider() -> Dict[str, Any]:
    """Re-read provider config from settings.json."""
    return ChatService.switch_provider()


@router.post("/apply_profile")
def apply_profile(req: ProfileRequest) -> Dict[str, Any]:
    """Apply a model variant + mode profile (system prompt and budget)."""
    return ChatService.apply_profile(req.model_name, req.mode_name)


@router.post("/reload_settings")
def reload_settings() -> Dict[str, Any]:
    """Push edited settings.json values into the running manager."""
    return ChatService.reload_settings()


@router.post("/stop")
def stop_generation() -> Dict[str, Any]:
    """Ask the backend to stop the in-flight generation."""
    return ChatService.stop()


@router.post("/stream")
async def stream_chat(req: ChatRequest) -> EventSourceResponse:
    """Stream one assistant turn as Server-Sent Events.

    Event payloads: {"type": "start"|"token"|"done"|"error", ...}
    """
    return EventSourceResponse(
        ChatService.stream_chat(
            req.user_text, req.session_id, req.attachments, req.continuation
        ),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
