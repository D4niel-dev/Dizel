from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

class LoadModelRequest(BaseModel):
    checkpoint_path: str
    device: Optional[str] = None

class ProfileRequest(BaseModel):
    model_name: str
    mode_name: str

class ChatRequest(BaseModel):
    user_text: str
    session_id: Optional[str] = None
    attachments: Optional[list] = None

@router.get("/status")
def get_status() -> Dict[str, Any]:
    """Get the model and provider status"""
    return ChatService.get_status()

@router.post("/load_model")
def load_model(req: LoadModelRequest) -> Dict[str, str]:
    """Load a local checkpoint"""
    return ChatService.load_model(req.checkpoint_path, req.device)

@router.post("/switch_provider")
def switch_provider() -> Dict[str, str]:
    """Reload the provider from current config"""
    return ChatService.switch_provider()

@router.post("/apply_profile")
def apply_profile(req: ProfileRequest) -> Dict[str, str]:
    """Apply a generation profile (e.g. Dizel Lite / Fast)"""
    return ChatService.apply_profile(req.model_name, req.mode_name)

@router.post("/stream")
def stream_chat(req: ChatRequest):
    """
    Stream a chat response using Server-Sent Events.
    Note: Client must handle the SSE stream.
    """
    return EventSourceResponse(
        ChatService.stream_chat(req.user_text, req.session_id, req.attachments)
    )
