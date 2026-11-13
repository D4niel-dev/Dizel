"""Runtime mutations shared by the slash commands and the modal screens.

Keeping the apply-and-persist logic here means a model picked from a modal
behaves exactly like `/model <name>` typed at the prompt.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from inference.tui.constants import MODES, PROVIDER_FALLBACKS


def project_root() -> Path:
    """Find the repository root from this module's location."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "inference").is_dir() and (parent / "model").is_dir():
            return parent
    return Path.cwd()


def checkpoint_paths() -> list[Path]:
    """Every loadable checkpoint in checkpoints/, newest first."""
    directory = project_root() / "checkpoints"
    if not directory.is_dir():
        return []
    paths: dict[str, Path] = {}
    for pattern in ("*.pt", "*.pth", "*.gguf", "*.safetensors", "*.bin", "*.ckpt"):
        for path in directory.glob(pattern):
            paths[str(path.resolve())] = path
    return sorted(paths.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def provider_catalog() -> list[dict[str, Any]]:
    """Return local plus every registered API provider, with capability flags."""
    catalog: list[dict[str, Any]] = [
        {
            "slug": "local",
            "name": "Local checkpoint",
            "description": "Run a Dizel checkpoint on this machine",
            "requires_key": False,
        }
    ]

    catalog.append(
        {
            "slug": "custom",
            "name": "Custom endpoint",
            "description": "Any OpenAI-compatible API (set the base URL below)",
            "requires_key": True,
        }
    )

    try:
        from dizel_core.providers import ProviderRegistry

        for info in ProviderRegistry.infos():
            catalog.append(
                {
                    "slug": info.slug,
                    "name": info.name,
                    "description": info.description or info.base_url or "",
                    "requires_key": bool(info.requires_key),
                }
            )
    except Exception:
        for slug, name, description, requires_key in PROVIDER_FALLBACKS:
            catalog.append(
                {
                    "slug": slug,
                    "name": name,
                    "description": description,
                    "requires_key": requires_key,
                }
            )

    return catalog


def provider_meta(slug: str) -> dict[str, Any] | None:
    slug = (slug or "").lower()
    return next((item for item in provider_catalog() if item["slug"] == slug), None)


def provider_requires_key(slug: str) -> bool:
    meta = provider_meta(slug)
    return bool(meta and meta["requires_key"])


def key_state(app: Any, slug: str) -> str:
    """A short badge for a provider row: key set / key needed."""
    if not provider_requires_key(slug):
        return ""
    manager = app.chat_bridge.manager
    if manager._provider_slug == slug and manager._api_key:
        return "key set"
    return "key needed"


def apply_model(app: Any, name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Model name is empty.")

    manager = app.chat_bridge.manager
    is_api = manager._provider_slug != "local" and manager._api_provider is not None

    if is_api:
        manager._api_model = name
        app.active_model = name

        from dizel_core.config_manager import ConfigManager

        cfg = ConfigManager.load()
        api_cfg = cfg.setdefault("api_router", {})
        api_cfg["model"] = name
        ConfigManager.save(cfg)
        return f"Model: {name} · {app.active_provider}"

    # Local: the model is whichever checkpoint is loaded. Match by display
    # name or file name, then load it. The old Dizel/Mila profile families
    # are gone — there was never a Mila checkpoint behind that row.
    from model.naming import model_display_name

    paths = checkpoint_paths()
    wanted = name.lower()
    matched = next(
        (
            path
            for path in paths
            if wanted
            in {
                model_display_name(str(path)).lower(),
                path.stem.lower(),
                path.name.lower(),
            }
        ),
        None,
    )
    if matched is None:
        available = ", ".join(model_display_name(str(path)) for path in paths) or "none"
        raise ValueError(f"Unknown local model: {name}. Available: {available}")

    return load_checkpoint(app, str(matched))


def load_checkpoint(app: Any, path: str) -> str:
    """Load a checkpoint on the app's worker thread and report what is loading."""
    from model.naming import model_display_name

    resolved = str(Path(path).resolve())
    if not Path(resolved).is_file():
        raise ValueError(f"Checkpoint not found: {path}")
    if not app._load_model_async(resolved, app._device):
        return ""
    return f"Loading {model_display_name(resolved)}…"


def apply_provider(app: Any, slug: str, model: str = "", key: str = "") -> str:
    slug = (slug or "").lower()
    meta = provider_meta(slug)
    if not meta:
        raise ValueError(f"Unknown provider: {slug}")

    from dizel_core.config_manager import ConfigManager, encrypt_key

    manager = app.chat_bridge.manager
    previous = manager._provider_slug

    cfg = ConfigManager.load()
    api_cfg = cfg.setdefault("api_router", {})
    api_cfg["provider"] = slug

    if model:
        api_cfg["model"] = model
    if key:
        api_cfg["api_key"] = encrypt_key(key)
    elif slug != previous and meta["requires_key"]:
        # Never hand a key saved for one provider to a different one.
        api_cfg["api_key"] = ""

    ConfigManager.save(cfg)
    manager.reload_provider()

    app.active_provider = manager.active_provider_slug

    if app.active_provider == "local":
        if slug != "local":
            return (
                f"Could not load {slug}; switched back to local. "
                "Check the provider dependencies and API key."
            )
        from model.naming import model_display_name

        app.active_model = model_display_name(app._checkpoint) if app._checkpoint else ""
        return "Provider: local"

    app.active_model = manager._api_model or ""
    if not app.active_model:
        return f"Provider: {app.active_provider} · no model selected"
    return f"Provider: {app.active_provider} · {app.active_model}"


def validate_provider_async(app: Any, slug: str | None = None) -> None:
    """Check the connection off the UI thread and report back in the transcript."""
    if slug == "local":
        return

    def worker() -> None:
        manager = app.chat_bridge.manager
        provider = manager._api_provider
        if provider is None:
            message = "No API provider loaded."
        else:
            try:
                provider.validate(key=manager._api_key)
                message = f"{app.active_provider} connected."
            except Exception as error:
                message = f"{app.active_provider} unreachable: {error}"
        try:
            app.call_from_thread(app._log_system, message)
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


def apply_mode(app: Any, mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Available: {', '.join(MODES)}")
    app.set_mode(mode)
    return f"Mode: {mode}"
