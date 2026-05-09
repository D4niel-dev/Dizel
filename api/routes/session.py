from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from api.services.session_service import SessionService

router = APIRouter(prefix="/session", tags=["session"])

class SessionCreateRequest(BaseModel):
    messages: List[Dict[str, str]]
    session_id: Optional[str] = None
    title: Optional[str] = None

@router.get("/list")
def list_sessions() -> List[Dict]:
    """List all available sessions"""
    return SessionService.list_sessions()

@router.get("/{session_id}")
def get_session(session_id: str) -> Dict:
    """Get a specific session by ID"""
    session = SessionService.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/")
def create_or_update_session(req: SessionCreateRequest) -> Dict[str, str]:
    """Create or update a session with new messages"""
    new_id = SessionService.create_or_update_session(req.messages, req.session_id, req.title)
    return {"session_id": new_id, "status": "success"}

@router.delete("/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    """Delete a session"""
    success = SessionService.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

@router.post("/{session_id}/pin")
def toggle_pin(session_id: str) -> Dict[str, Any]:
    """Toggle the pin status of a session"""
    pinned = SessionService.toggle_pin(session_id)
    return {"session_id": session_id, "pinned": pinned}
