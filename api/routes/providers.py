"""
api/routes/providers.py
───────────────────────
API router (BYOK) surface for the Settings dialog.

Mirrors the desktop app's model: one active provider at a time, one encrypted
key in `api_router`. Switching providers reloads the live ChatManager so the
change takes effect without a restart.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/providers", tags=["providers"])

# slug, display name, avatar file (None → use the app mark), requires a key
_PROVIDERS = [
    ("local", "Local (Dizel)", None, False),
    ("ollama", "Ollama", "ollama.png", False),
    ("openai", "OpenAI", "chatgpt.png", True),
    ("anthropic", "Anthropic", "claude.png", True),
    ("google", "Google (Gemini)", "gemini.png", True),
    ("groq", "Groq", "Groq.png", True),
    ("mistral", "Mistral AI", "mistral-ai.png", True),
    ("xai", "XAI", "xai.png", True),
    ("ai21", "AI21 Labs", "ai21-labs.png", True),
    ("azure", "Azure OpenAI", "microsoft-azure-openaI.png", True),
    ("cohere", "Cohere", "cohere.png", True),
    ("meta", "Meta (Llama)", "meta.png", True),
    ("custom", "Custom Provider", None, True),
]


class ProviderKeyRequest(BaseModel):
    provider: str
    api_key: str = ""
    model: str = ""
    custom_name: str = ""
    custom_url: str = ""


def _api_router_config() -> Dict[str, Any]:
    from dizel_core.config_manager import ConfigManager

    return ConfigManager.load().get("api_router", {})


@router.get("/")
def list_providers() -> List[Dict[str, Any]]:
    api_cfg = _api_router_config()
    active = api_cfg.get("provider", "local")

    return [
        {
            "slug": slug,
            "name": name,
            "avatar": avatar,
            "requires_key": requires_key,
            "active": slug == active,
            "configured": bool(api_cfg.get("api_key")) if slug == active else False,
            "model": api_cfg.get("model", "") if slug == active else "",
        }
        for slug, name, avatar, requires_key in _PROVIDERS
    ]


@router.get("/models")
def provider_models(slug: str) -> Dict[str, Any]:
    """
    List the models a provider can serve.

    For Ollama this is the list of locally pulled models, which doubles as a
    connectivity check — if the daemon isn't running, the caller gets the reason.
    """
    valid = {name for name, _, _, _ in _PROVIDERS}
    if slug not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{slug}'.")

    from dizel_core.providers import get_provider

    api_cfg = _api_router_config()
    key = ""
    if api_cfg.get("api_key"):
        from dizel_core.config_manager import decrypt_key

        key = decrypt_key(api_cfg["api_key"])

    kwargs: Dict[str, Any] = {}
    if slug == "ollama":
        kwargs["ollama_url"] = api_cfg.get("ollama_url", "http://localhost:11434")
    elif slug == "custom":
        if not api_cfg.get("custom_url"):
            raise HTTPException(
                status_code=400,
                detail="Set a base URL for the custom provider first.",
            )
        kwargs["custom_name"] = api_cfg.get("custom_name", "Custom Provider")
        kwargs["custom_url"] = api_cfg["custom_url"]

    try:
        provider = get_provider(slug, **kwargs)
        models = provider.list_models(key=key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"models": [{"id": m.id, "name": m.name} for m in models]}


@router.post("/key")
def set_key(req: ProviderKeyRequest) -> Dict[str, Any]:
    """Store an encrypted key and switch the active provider."""
    from dizel_core.config_manager import ConfigManager, encrypt_key

    valid = {slug for slug, _, _, _ in _PROVIDERS}
    if req.provider not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{req.provider}'.")

    config = ConfigManager.load()
    api_cfg = config.get("api_router", {})
    api_cfg["provider"] = req.provider
    api_cfg["model"] = req.model.strip()

    if req.provider == "custom":
        # Switching model on an already-configured custom provider sends no URL,
        # so fall back to what's saved before treating it as unconfigured.
        url = req.custom_url.strip() or api_cfg.get("custom_url", "")
        if not url:
            raise HTTPException(
                status_code=400,
                detail="A custom provider needs a base URL (e.g. https://openrouter.ai/api/v1).",
            )
        api_cfg["custom_url"] = url
        api_cfg["custom_name"] = (
            req.custom_name.strip() or api_cfg.get("custom_name", "Custom Provider")
        )

    if req.api_key.strip():
        api_cfg["api_key"] = encrypt_key(req.api_key.strip())
    elif req.provider == "local":
        api_cfg["api_key"] = ""
    config["api_router"] = api_cfg
    ConfigManager.save(config)

    # Reload the live manager so the new provider is used immediately.
    try:
        from api.services.chat_service import ChatService

        ChatService.get_manager().reload_provider()
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        raise HTTPException(status_code=500, detail=f"Saved, but reload failed: {exc}") from exc

    return {
        "status": "ok",
        "provider": req.provider,
        "has_key": bool(api_cfg.get("api_key")),
        "model": api_cfg.get("model", ""),
    }
