"""
api/main.py
───────────
FastAPI sidecar for the Dizel desktop app.

Runs as a child process of Electron. Binds 127.0.0.1 only, on a port chosen by
the Electron main process, and requires a per-launch token (see api/security.py).

Manual run for testing:
    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Make the project root importable ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from api.routes import (  # noqa: E402
    chat,
    config,
    feedback,
    health,
    mcp,
    memory,
    models,
    nova,
    profile,
    providers,
    session,
    skills,
)
from api.security import TokenAuthMiddleware, allowed_origins, api_token  # noqa: E402
from dizel_core.version import __version__  # noqa: E402

app = FastAPI(
    title="Dizel API",
    description="Local inference sidecar for the Dizel desktop app",
    version=__version__,
)

# CORS must be the outermost layer so preflights are answered before auth runs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Dizel-Token"],
)
app.add_middleware(TokenAuthMiddleware)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(config.router, prefix="/api", tags=["config"])
app.include_router(session.router, prefix="/api", tags=["session"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(mcp.router, prefix="/api", tags=["mcp"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(nova.router, prefix="/api", tags=["nova"])
app.include_router(providers.router, prefix="/api", tags=["providers"])
app.include_router(profile.router, prefix="/api", tags=["profile"])
app.include_router(memory.router, prefix="/api", tags=["memory"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(skills.router, prefix="/api", tags=["skills"])


@app.on_event("startup")
def _announce() -> None:
    auth = "token required" if api_token() else "NO TOKEN (dev mode — do not expose)"
    print(f"[dizel-api] ready — auth: {auth}", flush=True)


@app.on_event("startup")
def _start_mcp() -> None:
    from dizel_core.config_manager import ConfigManager
    from dizel_core.mcp import config as mcp_config
    from dizel_core.mcp.manager import get_manager

    # Raw config: the redacted service view carries no secret values.
    cfg = ConfigManager.load()
    section = cfg.get("mcp") if isinstance(cfg.get("mcp"), dict) else {}
    manager = get_manager()
    manager.configure(section)
    # Auto-connect is a launch preference only; testing or opening the MCP
    # settings still connects on demand.
    if section.get("auto_connect", True):
        manager.apply_async(mcp_config.load(cfg))


@app.on_event("shutdown")
def _stop_mcp() -> None:
    from dizel_core.mcp.manager import get_manager

    get_manager().shutdown()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DIZEL_PORT", "8000"))
    uvicorn.run("api.main:app", host="127.0.0.1", port=port, log_level="info")
