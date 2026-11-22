from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/report")
def usage_report_route() -> Dict[str, Any]:
    """Every provider's window with a labelled cost estimate."""
    from api.services.chat_service import get_usage_manager
    from dizel_core import usage_report

    return usage_report.snapshot(get_usage_manager())
