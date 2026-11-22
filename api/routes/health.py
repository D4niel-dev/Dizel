from fastapi import APIRouter
from typing import Any, Dict

router = APIRouter(prefix="/health")

@router.get("/")
def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint to verify the API is running.
    """
    return {"status": "ok", "service": "Dizel API"}


@router.get("/doctor")
def doctor_report() -> Dict[str, Any]:
    """Offline runtime checks shared by the desktop and terminal clients."""
    from dizel_core import doctor

    return doctor.as_dict()
