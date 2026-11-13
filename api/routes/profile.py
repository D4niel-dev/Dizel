"""
api/routes/profile.py
─────────────────────
User profile: display name and avatar.

The avatar is stored as a file next to settings.json (same place the desktop
app keeps it) and served through an authenticated route, so the renderer can
fetch it with its launch token and hand the bytes to an <img>.
"""

import hashlib
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter(prefix="/profile", tags=["profile"])

_ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_MAX_BYTES = 4_000_000
_ACCENTS = {"blue", "teal", "violet", "rose", "amber"}
_GREETING_STYLES = {"warm", "simple", "time"}


def _data_dir() -> Path:
    from dizel_core.config_manager import SETTINGS_FILE

    return Path(SETTINGS_FILE).resolve().parent


def _read_profile() -> Dict[str, Any]:
    from dizel_core.config_manager import ConfigManager

    profile = ConfigManager.load().get("user_profile", {}) or {}
    accent = profile.get("accent", "blue")
    style = profile.get("greeting_style", "warm")
    return {
        "username": profile.get("username", "User") or "User",
        "avatar": profile.get("avatar", "") or "",
        "bio": profile.get("bio", "") or "",
        "pronouns": profile.get("pronouns", "") or "",
        "accent": accent if accent in _ACCENTS else "blue",
        "greeting_style": style if style in _GREETING_STYLES else "warm",
        "created_at": profile.get("created_at", "") or "",
    }


def _public(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "username": profile["username"],
        "has_avatar": bool(profile["avatar"]),
        "bio": profile["bio"],
        "pronouns": profile["pronouns"],
        "accent": profile["accent"],
        "greeting_style": profile["greeting_style"],
        "created_at": profile["created_at"],
    }


@router.get("/")
def get_profile() -> Dict[str, Any]:
    return _public(_read_profile())


@router.get("/avatar")
def get_avatar() -> FileResponse:
    profile = _read_profile()
    name = profile["avatar"]
    if not name:
        raise HTTPException(status_code=404, detail="No avatar set.")

    data_dir = _data_dir()
    path = (data_dir / name).resolve()

    # Never serve anything outside the data directory.
    if data_dir not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Avatar file missing.")

    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@router.post("/")
async def update_profile(
    username: str = Form(""),
    bio: str = Form(""),
    pronouns: str = Form(""),
    accent: str = Form(""),
    greeting_style: str = Form(""),
    remove_avatar: bool = Form(False),
    avatar: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    from dizel_core.config_manager import ConfigManager

    config = ConfigManager.load()
    profile = config.get("user_profile", {}) or {}

    clean_name = username.strip()[:40]
    if clean_name:
        profile["username"] = clean_name

    profile["bio"] = bio.strip()[:80]
    profile["pronouns"] = pronouns.strip()[:24]
    if accent in _ACCENTS:
        profile["accent"] = accent
    if greeting_style in _GREETING_STYLES:
        profile["greeting_style"] = greeting_style

    if remove_avatar:
        profile["avatar"] = ""

    if avatar is not None and avatar.filename:
        data = await avatar.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty image.")
        if len(data) > _MAX_BYTES:
            raise HTTPException(status_code=400, detail="Avatar must be under 4 MB.")

        suffix = Path(avatar.filename).suffix.lower()
        if suffix not in _ALLOWED:
            raise HTTPException(
                status_code=400, detail=f"Unsupported image type '{suffix}'."
            )

        # Content-addressed so the URL changes when the image does.
        digest = hashlib.md5(data).hexdigest()[:10]
        filename = f"avatar_{digest}{suffix}"
        data_dir = _data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / filename).write_bytes(data)
        profile["avatar"] = filename

    profile.setdefault("username", "User")
    profile.setdefault("avatar", "")
    profile.setdefault("accent", "blue")
    profile.setdefault("greeting_style", "warm")
    if not profile.get("created_at"):
        # First save stamps the profile so the modal can show "local since".
        profile["created_at"] = date.today().isoformat()
    config["user_profile"] = profile
    ConfigManager.save(config)

    return _public(profile)
