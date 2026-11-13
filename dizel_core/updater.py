"""Release discovery and self-update for both Dizel surfaces.

The GitHub release is the source of truth. A release carries one manifest
(``dizel-update.json``) listing the GUI installer and the TUI archive with their
SHA-256 digests, so a download is never applied without verification.

The desktop app delegates installation to electron-updater, which understands
NSIS. The terminal app cannot replace Python files that are loaded, so it stages
the archive and applies it from ``dizel.cmd`` before the next launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .version import (
    APP_NAME,
    LATEST_RELEASE_API,
    RELEASES_URL,
    __version__,
)

MANIFEST_ASSET = "dizel-update.json"
TUI_ASSET_PREFIX = "Dizel-TUI-"
GUI_ASSET_PREFIX = "Dizel-Setup-"
CHECK_INTERVAL_SECONDS = 6 * 60 * 60
STATE_VERSION = 1

ALLOWED_DOWNLOAD_HOSTS = (
    "api.github.com",
    "github.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
    "release-assets.githubusercontent.com",
)


class UpdateError(RuntimeError):
    """A check or download failed in a way worth showing to a person."""


@dataclass(frozen=True)
class UpdateAsset:
    name: str
    url: str
    size: int = 0
    sha256: str = ""


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    tag: str
    name: str
    notes: str
    url: str
    published_at: str = ""
    gui: Optional[UpdateAsset] = None
    tui: Optional[UpdateAsset] = None
    prerelease: bool = False


@dataclass(frozen=True)
class CheckResult:
    current: str
    available: bool
    info: Optional[UpdateInfo]
    checked_at: float = field(default_factory=time.time)


# ── Versions ────────────────────────────────────────────────────────────────

_VERSION_CORE = re.compile(r"^v?(\d+(?:\.\d+)*)(?:-([0-9A-Za-z.-]+))?(?:\+.*)?$")


def normalize_version(value: str) -> str:
    text = str(value or "").strip()
    match = _VERSION_CORE.match(text)
    if not match:
        return text.lstrip("vV")
    parts = [str(int(part)) for part in match.group(1).split(".")]
    while len(parts) < 3:
        parts.append("0")
    core = ".".join(parts)
    return f"{core}-{match.group(2)}" if match.group(2) else core


def _prerelease_key(value: str) -> tuple:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in value.split(".")
        if part
    )


def version_key(value: str) -> tuple:
    """Sort key: numeric core first, then stable-after-prerelease."""
    match = _VERSION_CORE.match(str(value or "").strip())
    if not match:
        return ((0, 0, 0), 1, ())
    numbers = [int(part) for part in match.group(1).split(".")]
    while len(numbers) < 3:
        numbers.append(0)
    prerelease = match.group(2) or ""
    return (tuple(numbers), 0 if prerelease else 1, _prerelease_key(prerelease))


def is_newer(candidate: str, current: str = __version__) -> bool:
    return version_key(candidate) > version_key(current)


def current_version() -> str:
    return __version__


# ── Local state ─────────────────────────────────────────────────────────────

def repo_root(root: Optional[os.PathLike | str] = None) -> Path:
    if root:
        return Path(root).resolve()
    override = os.environ.get("DIZEL_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent.parent


def state_dir(root: Optional[os.PathLike | str] = None) -> Path:
    return repo_root(root) / ".dizel" / "updates"


def _state_path(root: Optional[os.PathLike | str] = None) -> Path:
    return state_dir(root) / "state.json"


def read_state(root: Optional[os.PathLike | str] = None) -> dict:
    defaults = {
        "version": STATE_VERSION,
        "auto_check": True,
        "last_check": 0.0,
        "last_version": "",
        "applied_version": "",
        "applied_at": 0.0,
    }
    try:
        stored = json.loads(_state_path(root).read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            defaults.update(stored)
    except (OSError, ValueError):
        pass
    return defaults


def write_state(root: Optional[os.PathLike | str] = None, **patch) -> dict:
    state = read_state(root)
    state.update(patch)
    state["version"] = STATE_VERSION
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return state


def auto_check_enabled(root: Optional[os.PathLike | str] = None) -> bool:
    return bool(read_state(root).get("auto_check", True))


def set_auto_check(enabled: bool, root: Optional[os.PathLike | str] = None) -> None:
    write_state(root, auto_check=bool(enabled))


def should_auto_check(
    root: Optional[os.PathLike | str] = None,
    interval: float = CHECK_INTERVAL_SECONDS,
) -> bool:
    state = read_state(root)
    if not state.get("auto_check", True):
        return False
    return time.time() - float(state.get("last_check") or 0) >= interval


# ── HTTP ────────────────────────────────────────────────────────────────────

def _httpx():
    try:
        import httpx  # noqa: WPS433 — optional at import time so `apply` stays lean
    except ImportError as exc:  # pragma: no cover - requirements include it
        raise UpdateError("Updates need httpx. Run: pip install -r requirements.txt") from exc
    return httpx


def _headers(url: str) -> dict:
    accept = (
        "application/vnd.github+json"
        if "api.github.com" in url
        else "application/octet-stream"
    )
    return {
        "Accept": accept,
        "User-Agent": f"{APP_NAME}/{__version__}",
    }


def _validate_url(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateError(f"Refusing a download from an untrusted host: {parsed.hostname or url}")
    return url


def _friendly_http_error(exc, url: str) -> UpdateError:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 404:
        return UpdateError(
            "No published GitHub release was found. The release workflow has not run yet."
        )
    if status == 403:
        return UpdateError(
            "GitHub rate-limited the update check. Try again later or open the releases page."
        )
    return UpdateError(f"GitHub returned HTTP {status or '?'} for {url}")


def _get_json(url: str, timeout: float = 15.0) -> dict:
    httpx = _httpx()
    last_error: Optional[Exception] = None
    for trust_env in (True, False):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers=_headers(url),
                trust_env=trust_env,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, dict) else {}
        except httpx.InvalidURL as exc:
            # Windows hosts sometimes carry a malformed NO_PROXY; retry direct.
            last_error = exc
            if trust_env:
                continue
            break
        except httpx.HTTPStatusError as exc:
            raise _friendly_http_error(exc, url) from exc
        except httpx.HTTPError as exc:
            raise UpdateError(f"Could not reach GitHub: {exc}") from exc
        except ValueError as exc:
            raise UpdateError(f"GitHub sent an unreadable response for {url}") from exc
    raise UpdateError(f"Could not reach GitHub: {last_error}")


def _digest(value: str) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _safe_segment(value: str, fallback: str = "update") -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "")).strip("._-")
    return text[:80] or fallback


def _asset_from_release(asset: dict) -> Optional[UpdateAsset]:
    name = str(asset.get("name") or "")
    url = str(asset.get("browser_download_url") or "")
    if not name or "/" in name or "\\" in name or not url:
        return None
    try:
        _validate_url(url)
    except UpdateError:
        return None
    return UpdateAsset(
        name=name,
        url=url,
        size=int(asset.get("size") or 0),
        sha256=_digest(asset.get("digest") or ""),
    )


def _asset_named(assets: dict, name: str, prefix: str) -> Optional[UpdateAsset]:
    raw = assets.get(name) if name else None
    if raw is None:
        matches = [item for item in assets.values() if item["name"].startswith(prefix)]
        if not matches:
            return None
        raw = sorted(matches, key=lambda item: item["name"])[-1]
    return _asset_from_release(raw)


def fetch_latest_release(timeout: float = 15.0) -> Optional[UpdateInfo]:
    """Return the latest stable release, or None when the repo has none."""
    payload = _get_json(LATEST_RELEASE_API, timeout=timeout)
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return None

    assets = {
        str(asset.get("name")): asset
        for asset in payload.get("assets", [])
        if isinstance(asset, dict) and asset.get("name")
    }

    manifest: dict = {}
    manifest_asset = assets.get(MANIFEST_ASSET)
    if manifest_asset:
        try:
            manifest = _get_json(str(manifest_asset.get("browser_download_url")), timeout=timeout)
        except UpdateError:
            manifest = {}

    gui_spec = manifest.get("gui") if isinstance(manifest.get("gui"), dict) else {}
    tui_spec = manifest.get("tui") if isinstance(manifest.get("tui"), dict) else {}
    gui = _asset_named(
        assets,
        str(gui_spec.get("asset") or ""),
        GUI_ASSET_PREFIX,
    )
    tui = _asset_named(
        assets,
        str(tui_spec.get("asset") or ""),
        TUI_ASSET_PREFIX,
    )

    if gui and gui_spec.get("sha256"):
        gui = UpdateAsset(gui.name, gui.url, int(gui_spec.get("size") or gui.size), _digest(gui_spec["sha256"]))
    if tui and tui_spec.get("sha256"):
        tui = UpdateAsset(tui.name, tui.url, int(tui_spec.get("size") or tui.size), _digest(tui_spec["sha256"]))

    version = normalize_version(str(manifest.get("version") or tag))
    notes = str(manifest.get("notes") or payload.get("body") or "").strip()
    return UpdateInfo(
        version=version,
        tag=tag,
        name=str(payload.get("name") or f"{APP_NAME} {tag}"),
        notes=notes,
        url=str(payload.get("html_url") or RELEASES_URL),
        published_at=str(payload.get("published_at") or ""),
        gui=gui,
        tui=tui,
        prerelease=bool(payload.get("prerelease")),
    )


def check_for_update(
    current: Optional[str] = None,
    root: Optional[os.PathLike | str] = None,
    timeout: float = 15.0,
    remember: bool = True,
) -> CheckResult:
    """Return whether a newer stable release exists.

    Raises UpdateError for network and API failures; callers doing a background
    check should swallow that and keep starting the app.
    """
    current = normalize_version(current or __version__)
    info = fetch_latest_release(timeout=timeout)
    available = bool(info and is_newer(info.version, current))
    if remember:
        write_state(
            root,
            last_check=time.time(),
            last_version=(info.version if info else ""),
        )
    return CheckResult(current=current, available=available, info=info)


# ── Downloads ───────────────────────────────────────────────────────────────

ProgressFn = Callable[[int, int], None]


def _stream_download(
    url: str,
    destination: Path,
    expected_size: int = 0,
    timeout: float = 300.0,
    progress: Optional[ProgressFn] = None,
) -> str:
    """Download to ``destination.part`` and return the SHA-256 hex digest."""
    httpx = _httpx()
    _validate_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    last_error: Optional[Exception] = None

    for trust_env in (True, False):
        digest = hashlib.sha256()
        received = 0
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers=_headers(url),
                trust_env=trust_env,
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length") or expected_size or 0)
                    with open(part, "wb") as handle:
                        for chunk in response.iter_bytes(256 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                            if progress:
                                progress(received, total)
                    os.replace(part, destination)
                    return digest.hexdigest()
        except httpx.InvalidURL as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if trust_env:
                continue
            break
        except httpx.HTTPStatusError as exc:
            part.unlink(missing_ok=True)
            raise _friendly_http_error(exc, url) from exc
        except httpx.HTTPError as exc:
            part.unlink(missing_ok=True)
            raise UpdateError(f"Download failed: {exc}") from exc
        except OSError as exc:
            part.unlink(missing_ok=True)
            raise UpdateError(f"Could not write the download: {exc}") from exc

    part.unlink(missing_ok=True)
    raise UpdateError(f"Download failed: {last_error}")


def download_asset(
    asset: UpdateAsset,
    destination: os.PathLike | str,
    progress: Optional[ProgressFn] = None,
) -> Path:
    target = Path(destination)
    if target.exists() and asset.sha256:
        if _sha256_file(target) == asset.sha256:
            return target
    digest = _stream_download(asset.url, target, asset.size, progress=progress)
    if asset.sha256 and digest != asset.sha256:
        target.unlink(missing_ok=True)
        raise UpdateError(
            f"{asset.name} failed its SHA-256 check. The download was discarded."
        )
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── TUI staging and apply ───────────────────────────────────────────────────

_APP_DIRS = ("api", "dizel_core", "model", "core", "utils", "inference/tui")
_APP_FILES = (
    "config.py",
    "capabilities.json",
    "requirements.txt",
    "README.md",
    "inference/__init__.py",
)
_SKIP_SUFFIXES = (".pyc", ".pyo")
_SKIP_PARTS = {"__pycache__", ".git", "node_modules"}
_PENDING = "pending.json"


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = member.filename.replace("\\", "/")
            if name.endswith("/"):
                continue
            if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
                raise UpdateError("The update archive contains an absolute path.")
            target = (destination / name).resolve()
            if not _inside(target, destination):
                raise UpdateError("The update archive tries to write outside its folder.")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)


def _remove_tree(path: Path, base: Path) -> None:
    if path.exists() and _inside(path, base):
        shutil.rmtree(path, ignore_errors=True)


def _normalise_stage(staging: Path) -> Path:
    children = [child for child in staging.iterdir() if child.name != "__MACOSX"]
    directories = [child for child in children if child.is_dir()]
    files = [child for child in children if child.is_file()]
    if len(directories) == 1 and not files:
        return directories[0]
    return staging


def _pending_path(root: Optional[os.PathLike | str] = None) -> Path:
    return state_dir(root) / _PENDING


def pending_update(root: Optional[os.PathLike | str] = None) -> Optional[dict]:
    try:
        payload = json.loads(_pending_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("version") or not payload.get("source"):
        return None
    source = Path(str(payload["source"]))
    if not source.is_dir():
        return None
    return payload


def stage_tui_update(
    archive: os.PathLike | str,
    info: UpdateInfo,
    root: Optional[os.PathLike | str] = None,
) -> Path:
    if not info.tui:
        raise UpdateError(f"Release {info.tag} does not include a TUI package.")
    archive = Path(archive)
    version = normalize_version(info.version)
    if not _VERSION_CORE.match(version):
        raise UpdateError(f"Release {info.tag} has an invalid version.")
    base = state_dir(root)
    staging = base / "staging" / _safe_segment(version)
    _remove_tree(staging, base)
    try:
        _extract_zip(archive, staging)
    except zipfile.BadZipFile as exc:
        _remove_tree(staging, base)
        raise UpdateError("The downloaded TUI package is not a valid ZIP archive.") from exc

    source = _normalise_stage(staging)
    if not (source / "dizel_core" / "updater.py").is_file():
        _remove_tree(staging, base)
        raise UpdateError("The update archive is missing dizel_core/updater.py.")
    if not (source / "inference" / "tui" / "main.py").is_file():
        _remove_tree(staging, base)
        raise UpdateError("The update archive is missing inference/tui/main.py.")

    payload = {
        "version": version,
        "tag": info.tag,
        "source": str(source),
        "archive": str(archive),
        "sha256": info.tui.sha256,
        "staged_at": time.time(),
    }
    pending = _pending_path(root)
    pending.parent.mkdir(parents=True, exist_ok=True)
    tmp = pending.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, pending)
    return pending


def download_tui_update(
    info: UpdateInfo,
    root: Optional[os.PathLike | str] = None,
    progress: Optional[ProgressFn] = None,
) -> Path:
    if not info.tui:
        raise UpdateError(f"Release {info.tag} does not include a TUI package.")
    if not info.tui.sha256:
        raise UpdateError(
            f"Release {info.tag} has no SHA-256 digest for the TUI package; refusing an unverified update."
        )
    name = str(info.tui.name)
    if not name or "/" in name or "\\" in name:
        raise UpdateError("The release asset has an unsafe filename.")
    archive = state_dir(root) / "downloads" / name
    if not (archive.exists() and info.tui.sha256 and _sha256_file(archive) == info.tui.sha256):
        download_asset(info.tui, archive, progress=progress)
    return stage_tui_update(archive, info, root=root)


def _allowed_relative(relative: Path) -> bool:
    name = relative.as_posix()
    if any(part in _SKIP_PARTS for part in relative.parts):
        return False
    if name in _APP_FILES:
        return True
    return any(name == folder or name.startswith(f"{folder}/") for folder in _APP_DIRS)


def _iter_payload(source: Path) -> Iterable[Path]:
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() not in _SKIP_SUFFIXES:
            yield path


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".dizel-new")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)


def _backup_current(root: Path, backup: Path) -> None:
    backup.mkdir(parents=True, exist_ok=True)
    for name in _APP_DIRS:
        source = root / name
        if source.is_dir():
            shutil.copytree(
                source,
                backup / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                dirs_exist_ok=True,
            )
    for name in _APP_FILES:
        source = root / name
        if source.is_file():
            shutil.copy2(source, backup / name)


def _restore_backup(root: Path, backup: Path) -> None:
    for name in _APP_DIRS:
        source = backup / name
        if not source.is_dir():
            continue
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file():
                _copy_file(path, target / path.relative_to(source))
    for name in _APP_FILES:
        source = backup / name
        if source.is_file():
            _copy_file(source, root / name)


def apply_pending_update(root: Optional[os.PathLike | str] = None) -> Optional[dict]:
    """Apply a staged TUI update. Safe to call when there is nothing pending."""
    root_path = repo_root(root)
    pending = pending_update(root)
    if not pending:
        return None

    version = normalize_version(str(pending["version"]))
    if not is_newer(version, __version__):
        _discard_pending(root)
        return None

    base = state_dir(root)
    source = Path(str(pending["source"])).resolve()
    if not _inside(source, base):
        _discard_pending(root)
        raise UpdateError("The staged update is outside Dizel's update folder.")

    backup = base / "backup" / f"{__version__}-{int(time.time())}"
    try:
        _backup_current(root_path, backup)
        for path in _iter_payload(source):
            relative = path.relative_to(source)
            if not _allowed_relative(relative):
                continue
            _copy_file(path, root_path / relative)
    except Exception as exc:
        try:
            _restore_backup(root_path, backup)
        except Exception:
            pass
        raise UpdateError(f"Could not apply update {version}: {exc}") from exc

    result = {
        "applied": version,
        "previous": __version__,
        "backup": str(backup),
    }
    write_state(
        root,
        applied_version=version,
        applied_at=time.time(),
        last_version=version,
    )
    _discard_pending(root)
    _prune_backups(base, keep=1)
    return result


def _discard_pending(root: Optional[os.PathLike | str] = None) -> None:
    base = state_dir(root)
    pending = _pending_path(root)
    try:
        payload = json.loads(pending.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    pending.unlink(missing_ok=True)
    source = payload.get("source")
    if source and _inside(Path(str(source)), base):
        staging = Path(str(source))
        while staging != base and _inside(staging, base):
            if staging.name == "staging":
                break
            staging = staging.parent
        _remove_tree(staging, base)
    archive = payload.get("archive")
    if archive and _inside(Path(str(archive)), base):
        Path(str(archive)).unlink(missing_ok=True)


def _prune_backups(base: Path, keep: int = 1) -> None:
    folder = base / "backup"
    if not folder.is_dir():
        return
    backups = sorted(
        (item for item in folder.iterdir() if item.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for old in backups[keep:]:
        _remove_tree(old, base)


# ── CLI used by dizel.cmd and CI ───────────────────────────────────────────

def _print_check(result: CheckResult) -> int:
    if not result.available or not result.info:
        print(f"{APP_NAME} {result.current} is up to date.")
        return 0
    info = result.info
    print(f"{APP_NAME} {info.version} is available (current {result.current}).")
    print(info.url)
    if info.tui:
        print(f"TUI package: {info.tui.name} · {info.tui.size:,} bytes")
    if info.gui:
        print(f"GUI installer: {info.gui.name} · {info.gui.size:,} bytes")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dizel_core.updater")
    parser.add_argument("--root", default="", help="Repository root (defaults to Dizel's own)")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Check GitHub for a newer release")
    check.add_argument("--timeout", type=float, default=15.0)

    apply_cmd = commands.add_parser("apply", help="Apply a staged TUI update")
    apply_cmd.add_argument("--quiet", action="store_true")
    apply_cmd.add_argument("--json", action="store_true")

    stage = commands.add_parser("stage", help="Stage a downloaded TUI archive")
    stage.add_argument("archive")
    stage.add_argument("--version", default="")
    stage.add_argument("--tag", default="")
    stage.add_argument("--sha256", default="")

    commands.add_parser("status", help="Show the local update state")

    args = parser.parse_args(argv)
    root = args.root or None

    try:
        if args.command == "check":
            return _print_check(check_for_update(root=root, timeout=args.timeout))
        if args.command == "apply":
            result = apply_pending_update(root=root)
            if args.json:
                print(json.dumps(result or {"applied": None}))
            elif result:
                print(f"Applied {APP_NAME} {result['applied']} (previous {result['previous']}).")
            elif not args.quiet:
                print("No staged update to apply.")
            return 0
        if args.command == "stage":
            version = normalize_version(args.version or __version__)
            asset = UpdateAsset(
                name=Path(args.archive).name,
                url=Path(args.archive).as_uri(),
                size=Path(args.archive).stat().st_size,
                sha256=args.sha256,
            )
            info = UpdateInfo(
                version=version,
                tag=args.tag or f"v{version}",
                name=f"{APP_NAME} {version}",
                notes="",
                url=RELEASES_URL,
                tui=asset,
            )
            pending = stage_tui_update(args.archive, info, root=root)
            print(f"Staged {version} at {pending}.")
            return 0
        if args.command == "status":
            print(json.dumps({"version": __version__, "state": read_state(root)}, indent=2))
            return 0
    except UpdateError as exc:
        print(f"Update error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
