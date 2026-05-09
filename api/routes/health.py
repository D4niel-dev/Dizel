from fastapi import APIRouter
from typing import Dict

router = APIRouter(prefix="/health")

@router.get("/")
def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint to verify the API is running.
    """
    return {"status": "ok", "service": "Dizel API"}
