from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dizel_core.feedback_store import record_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    rating: str
    session_id: str = ""
    message_key: str = ""
    reasons: List[str] = Field(default_factory=list)
    note: str = ""
    response: str = ""
    prompt: str = ""
    model: str = ""
    provider: str = ""


@router.post("/")
def save_feedback(req: FeedbackRequest) -> dict:
    """Store one local feedback event next to the chat data."""
    try:
        path = record_feedback(
            rating=req.rating,
            session_id=req.session_id,
            message_key=req.message_key,
            reasons=req.reasons,
            note=req.note,
            response=req.response,
            prompt=req.prompt,
            model=req.model,
            provider=req.provider,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "ok", "path": str(path)}
