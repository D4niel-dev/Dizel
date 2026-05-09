from fastapi import APIRouter, Body
from typing import Dict, Any
from api.services.config_service import ConfigService

router = APIRouter(prefix="/config", tags=["config"])

@router.get("/")
def get_config() -> Dict[str, Any]:
    """Get the current application configuration"""
    return ConfigService.get_config()

@router.post("/")
def update_config(updates: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Update the application configuration"""
    return ConfigService.update_config(updates)
