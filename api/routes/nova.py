"""
api/routes/nova.py
──────────────────
Voice transcription for the composer's mic button.

The renderer records 16 kHz mono PCM with the Web Audio API and posts a WAV, so
nothing here needs ffmpeg — `wave` decodes it and Whisper takes the array
directly.
"""

import importlib.util
import io
import os
import threading
import wave
from typing import Any, Dict, Optional

import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/nova", tags=["nova"])

# Whisper model instances are expensive to build and not thread-safe to share
# blindly, so loading is serialised and inference is guarded by a lock.
_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()
_infer_lock = threading.Lock()

# Checkpoints are looked for in the app's own folder first (portable — it travels
# with the install), then Whisper's default user cache. Downloads land in the app
# folder so they're easy to find and easy to delete.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_APP_ROOT = os.path.join(_REPO_ROOT, "models", "whisper")
_USER_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "whisper")

_dl_lock = threading.Lock()
_dl: Dict[str, Any] = {
    "state": "idle",  # idle | downloading | done | error
    "size": "",
    "received": 0,
    "total": 0,
    "error": "",
    "cancel": False,
}


def _download_root() -> str:
    """
    App folder when it's writable, user cache otherwise.

    A packaged install can live somewhere read-only (Program Files), and a
    download must not fail just because the binary's folder is protected.
    """
    try:
        os.makedirs(_APP_ROOT, exist_ok=True)
        probe = os.path.join(_APP_ROOT, ".write-test")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("")
        os.remove(probe)
        return _APP_ROOT
    except OSError:
        os.makedirs(_USER_ROOT, exist_ok=True)
        return _USER_ROOT


def _model_path(size: str) -> str:
    """Where a new download is written."""
    return os.path.join(_download_root(), f"{size}.pt")


def _find_model(size: str) -> Optional[str]:
    """An existing checkpoint, app folder first, then the user cache."""
    for root in (_APP_ROOT, _USER_ROOT):
        candidate = os.path.join(root, f"{size}.pt")
        if os.path.exists(candidate):
            return candidate
    return None


def _is_downloaded(size: str) -> bool:
    return _find_model(size) is not None


def _snapshot() -> Dict[str, Any]:
    with _dl_lock:
        data = {k: v for k, v in _dl.items() if k != "cancel"}
    percent = 0
    if data["state"] == "done":
        percent = 100
    elif data["total"] > 0:
        percent = min(99, round(data["received"] * 100 / data["total"]))
    else:
        percent = 0
    data["percent"] = percent
    return data


def _stream_checkpoint(url: str, tmp: str) -> None:
    def fetch(trust_env: bool) -> None:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=60.0,
            trust_env=trust_env,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            with _dl_lock:
                _dl.update(state="downloading", received=0, total=total, error="")

            with open(tmp, "wb") as handle:
                for chunk in response.iter_bytes(1 << 16):
                    if _dl.get("cancel"):
                        raise RuntimeError("cancelled")
                    handle.write(chunk)
                    with _dl_lock:
                        _dl["received"] += len(chunk)

    try:
        fetch(True)
    except httpx.InvalidURL:
        # A malformed proxy variable (NO_PROXY commonly picks up a URL-style
        # value on Windows) makes httpx reject the request before it reaches
        # the network. Retry directly; valid proxy settings still use the
        # first path.
        fetch(False)


def _download_worker(size: str) -> None:
    """Streams the checkpoint to <size>.pt, publishing progress as it goes."""
    try:
        import whisper

        url = whisper._MODELS[size]
        dest = _model_path(size)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"

        _stream_checkpoint(url, tmp)
        os.replace(tmp, dest)
        with _dl_lock:
            _dl.update(state="done")
    except Exception as exc:
        try:
            os.remove(_model_path(size) + ".part")
        except OSError:
            pass
        with _dl_lock:
            cancelled = _dl.get("cancel")
            _dl.update(
                state="idle" if cancelled else "error",
                error="" if cancelled else str(exc),
            )



def _configured_size() -> str:
    try:
        from dizel_core.config_manager import ConfigManager

        return ConfigManager.load().get("nova", {}).get("model_size", "base")
    except Exception:
        return "base"


def _configured_language() -> str:
    """The Speech tab stores a language; the client doesn't send one."""
    try:
        from dizel_core.config_manager import ConfigManager

        return ConfigManager.load().get("nova", {}).get("language", "auto")
    except Exception:
        return "auto"


def _configured_task() -> str:
    try:
        from dizel_core.config_manager import ConfigManager

        task = ConfigManager.load().get("nova", {}).get("task", "transcribe")
        return task if task in ("transcribe", "translate") else "transcribe"
    except Exception:
        return "transcribe"


def _configured_initial_prompt() -> str:
    try:
        from dizel_core.config_manager import ConfigManager

        value = ConfigManager.load().get("nova", {}).get("initial_prompt", "")
        return value if isinstance(value, str) else ""
    except Exception:
        return ""


def _configured_decoding() -> Dict[str, Any]:
    """Whisper decoding options from the Speech tab, validated for transcribe().

    best_of is clamped to at least beam_size — whisper raises otherwise when it
    samples (temperature > 0) with both set.
    """
    try:
        from dizel_core.config_manager import ConfigManager

        nova = ConfigManager.load().get("nova", {})
    except Exception:
        nova = {}

    def _int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return max(lo, min(hi, int(nova.get(key, default))))
        except (TypeError, ValueError):
            return default

    def _float(key: str, default: float, lo: float, hi: float) -> float:
        try:
            return max(lo, min(hi, float(nova.get(key, default))))
        except (TypeError, ValueError):
            return default

    beam = _int("beam_size", 5, 1, 10)
    best = max(beam, _int("best_of", 5, 1, 10))
    cond = nova.get("condition_on_previous_text", True)
    return {
        "beam_size": beam,
        "best_of": best,
        "temperature": _float("temperature", 0.0, 0.0, 1.0),
        "condition_on_previous_text": bool(cond),
        "no_speech_threshold": _float("no_speech_threshold", 0.6, 0.0, 1.0),
    }


def _get_model(size: str):
    with _cache_lock:
        if size in _cache:
            return _cache[size]
        import whisper

        # Point whisper at whichever folder already holds the checkpoint, so it
        # doesn't re-download something we found in the app folder.
        existing = _find_model(size)
        root = os.path.dirname(existing) if existing else _APP_ROOT
        os.makedirs(root, exist_ok=True)

        model = whisper.load_model(size, download_root=root)
        _cache[size] = model
        return model


def _decode_wav(data: bytes) -> np.ndarray:
    """16-bit PCM WAV → float32 mono at 16 kHz, which is what Whisper wants."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported audio: {exc}") from exc

    if width != 2:
        raise HTTPException(status_code=400, detail="Expected 16-bit PCM audio.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if rate != 16000 and audio.size:
        target = int(round(audio.size * 16000 / rate))
        if target > 0:
            audio = np.interp(
                np.linspace(0, audio.size - 1, target, dtype=np.float32),
                np.arange(audio.size, dtype=np.float32),
                audio,
            ).astype(np.float32)

    return audio


@router.get("/status")
def status() -> Dict[str, Any]:
    """
    Lets the UI dim the mic button instead of failing on click.

    Availability is checked with `find_spec`, not `import whisper` — importing
    pulls in torch and takes seconds, which times out the boot-time call.
    """
    size = _configured_size()
    available = importlib.util.find_spec("whisper") is not None
    return {
        "available": available,
        "reason": "" if available else "whisper is not installed",
        "model": size,
        "downloaded": _is_downloaded(size),
    }


@router.get("/download")
def download_status(size: Optional[str] = None) -> Dict[str, Any]:
    """Progress of the current (or last) checkpoint download."""
    target = size or _configured_size()
    state = _snapshot()
    state["downloaded"] = _is_downloaded(target)
    state["size"] = target
    return state


@router.post("/download")
def start_download(size: Optional[str] = None) -> Dict[str, Any]:
    """Kick off the checkpoint download; safe to call repeatedly."""
    target = size or _configured_size()

    if _is_downloaded(target):
        return {"state": "done", "downloaded": True, "size": target, "percent": 100}

    with _dl_lock:
        if _dl["state"] == "downloading" and _dl["size"] == target:
            return _snapshot() | {"downloaded": False}
        _dl.update(state="downloading", size=target, received=0, total=0, error="", cancel=False)

    threading.Thread(target=_download_worker, args=(target,), daemon=True).start()
    return _snapshot() | {"downloaded": False}


@router.post("/download/cancel")
def cancel_download() -> Dict[str, Any]:
    with _dl_lock:
        if _dl["state"] == "downloading":
            _dl["cancel"] = True
    return _snapshot()


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
) -> Dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio payload.")

    audio = _decode_wav(data)
    if audio.size < 1600:  # under 0.1 s — nothing to transcribe
        return {"text": "", "language": None, "duration": round(audio.size / 16000, 2)}

    size = model or _configured_size()
    lang = language or _configured_language()
    kwargs: Dict[str, Any] = {"task": _configured_task()}
    kwargs.update(_configured_decoding())
    if lang and lang not in ("auto", ""):
        kwargs["language"] = lang
    prompt = _configured_initial_prompt().strip()
    if prompt:
        kwargs["initial_prompt"] = prompt

    if not _is_downloaded(size):
        raise HTTPException(
            status_code=409,
            detail=f"The Whisper '{size}' model has not been downloaded yet.",
        )

    try:
        # Loading downloads the weights on first use, which can take minutes. It
        # must not run on the event loop — a blocking call here stalls every other
        # request, including the chat stream.
        engine = await run_in_threadpool(_get_model, size)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Dictation unavailable: the Whisper '{size}' model could not be loaded. "
                f"It downloads on first use, so this needs an internet connection. ({exc})"
            ),
        ) from exc

    try:
        with _infer_lock:
            result = await run_in_threadpool(
                lambda: engine.transcribe(audio, fp16=False, **kwargs)
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language"),
        "duration": round(audio.size / 16000, 2),
    }
