from fastapi import APIRouter
from typing import Any, Dict, List
from pydantic import BaseModel

from api.services.memory_service import MemoryService

router = APIRouter(prefix="/memory", tags=["memory"])


class SummarizeRequest(BaseModel):
    messages: List[Dict[str, str]]


@router.post("/summarize")
def summarize(req: SummarizeRequest) -> Dict[str, str]:
    """Extractive summary of the supplied messages."""
    return {"summary": MemoryService.summarize(req.messages)}


@router.get("/recent")
def recent(limit: int = 3) -> List[Dict[str, Any]]:
    """Recent conversations with a one-line summary each."""
    return MemoryService.recent(limit)
