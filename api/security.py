"""
api/security.py
───────────────
Local-sidecar hardening.

The FastAPI process is spawned by Electron on 127.0.0.1 with a random port and
a random per-launch token. Without the token, *any* page open in the user's
browser could reach the sidecar over localhost and drive a shell through the
chat endpoint. The token makes that impossible.

When DIZEL_API_TOKEN is unset (e.g. running `uvicorn api.main:app` by hand for
curl testing), auth is disabled and a warning is printed at startup.
"""

import os
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Endpoints reachable without a token. Keep this list tiny.
_PUBLIC_PATHS = {"/api/health", "/api/health/"}

# Origin allow-list. The dev server plus the packaged renderer.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def api_token() -> str:
    """The token this process expects, or '' when auth is disabled."""
    return os.environ.get("DIZEL_API_TOKEN", "").strip()


def allowed_origins() -> list:
    """CORS allow-list: dev server, plus anything explicitly configured.

    Electron loads the built renderer from disk in production, which reports
    `Origin: null`. That is only safe *because* every non-public route also
    requires the launch token.
    """
    extra = os.environ.get("DIZEL_ALLOWED_ORIGINS", "")
    origins = list(_DEFAULT_ORIGINS)
    origins.extend(o.strip() for o in extra.split(",") if o.strip())
    origins.append("null")
    return origins


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Reject any non-public request that lacks the launch token."""

    async def dispatch(self, request: Request, call_next):
        expected = api_token()
        if not expected or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # CORS preflights never carry custom headers; let the CORS layer answer.
        if request.method == "OPTIONS":
            return await call_next(request)

        supplied = request.headers.get("x-dizel-token", "")
        if not secrets.compare_digest(supplied, expected):
            return JSONResponse(
                {"detail": "Invalid or missing X-Dizel-Token."},
                status_code=401,
            )
        return await call_next(request)


def new_token() -> str:
    """Generate a token for one sidecar launch."""
    return secrets.token_urlsafe(32)
