from __future__ import annotations

import asyncio
import time
from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


def _root() -> str:
    from dizel_core import updater

    return str(updater.repo_root())


def _when(timestamp: float) -> str:
    if not timestamp:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def _download_busy(app: Any) -> bool:
    thread = getattr(app, "_update_thread", None)
    return bool(thread and thread.is_alive())


async def _check(app: Any, download: bool) -> str:
    from dizel_core import updater

    if download and _download_busy(app) and not updater.pending_update():
        return "An update is already downloading — try /update install again in a moment."

    try:
        result = await asyncio.to_thread(
            updater.check_for_update,
            root=_root(),
            timeout=15.0,
        )
    except updater.UpdateError as error:
        return f"Update check failed: {error}"

    if not result.available or not result.info:
        return f"Dizel {result.current} is up to date."

    info = result.info
    if not info.tui:
        return (
            f"Dizel {info.version} is available, but this release has no terminal "
            f"package. Desktop installer: {info.url}"
        )
    try:
        app.update_ready = info.version
    except Exception:
        pass
    if not download:
        return (
            f"Dizel {info.version} is available. Run /update install to download, "
            "apply, and restart, or leave automatic updates on."
        )

    last = [-25]

    def progress(received: int, total: int) -> None:
        if not total:
            return
        percent = int(received * 100 / total)
        if percent < last[0] + 25 and percent != 100:
            return
        last[0] = percent
        try:
            app.call_from_thread(
                app._log_system, f"Downloading Dizel {info.version} · {percent}%"
            )
        except Exception:
            pass

    try:
        await asyncio.to_thread(
            updater.download_tui_update,
            info,
            root=_root(),
            progress=progress,
        )
    except updater.UpdateError as error:
        return f"Update download failed: {error}"

    if not updater.pending_update():
        return "The update downloaded, but staging it failed. Try /update install again."

    return (
        f"Dizel {info.version} downloaded and staged. Run /update install to restart now; "
        "otherwise it applies when you quit."
    )


class UpdateCommand(Command):
    name = "update"
    help_text = "Check, download, stage, and install Dizel updates."
    category = "Runtime"
    usage = "/update [check|now|install|apply|status|auto on|off|notes]"
    palette_hint = "/update "
    aliases = ["upgrade"]
    examples = [
        "/update",
        "/update install",
        "/update check",
        "/update status",
        "/update auto off",
        "/update notes",
    ]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import updater

        action = (invocation.args[0].lower() if invocation.args else "check").strip()

        if action in ("check",):
            return await _check(app, download=False)

        if action in ("now", "download", "upgrade"):
            return await _check(app, download=True)

        if action in ("install", "apply", "restart"):
            return await self._install(app)

        if action == "status":
            return self._status()

        if action == "auto":
            value = (invocation.args[1].lower() if len(invocation.args) > 1 else "").strip()
            if value not in ("on", "off", "true", "false", "1", "0"):
                return "Use /update auto on or /update auto off."
            enabled = value in ("on", "true", "1")
            updater.set_auto_check(enabled)
            return (
                "Automatic update checks are on."
                if enabled
                else "Automatic update checks are off. /update check still works."
            )

        if action == "notes":
            return await self._notes()

        return self.usage

    async def _install(self, app: Any) -> str:
        from dizel_core import updater

        pending = updater.pending_update()
        if not pending:
            message = await _check(app, download=True)
            pending = updater.pending_update()
            if not pending:
                return message

        version = str(pending.get("version") or "update")
        app._log_system(f"Restarting Dizel to apply {version}…", "ok")
        app.request_update_restart()
        return f"Applying Dizel {version} and restarting…"

    def _status(self) -> str:
        from dizel_core import updater

        state = updater.read_state()
        pending = updater.pending_update()
        lines = [
            f"Dizel {updater.current_version()}",
            f"  automatic checks   {'on' if state.get('auto_check', True) else 'off'}",
            f"  last check         {_when(float(state.get('last_check') or 0))}",
        ]
        if pending:
            lines.append(
                f"  staged             {pending.get('version')} — /update install to restart"
            )
        else:
            lines.append("  staged             none")
        applied = state.get("applied_version")
        if applied:
            lines.append(
                f"  last applied       {applied} · {_when(float(state.get('applied_at') or 0))}"
            )
        return "\n".join(lines)

    async def _notes(self) -> str:
        from dizel_core import updater

        try:
            result = await asyncio.to_thread(
                updater.check_for_update,
                root=_root(),
                timeout=15.0,
            )
        except updater.UpdateError as error:
            return f"Could not read the release notes: {error}"
        info = result.info
        if not info:
            return "No published release found."
        notes = (info.notes or "").strip()
        if len(notes) > 1800:
            notes = notes[:1800].rstrip() + "…"
        body = f"{info.name}\n\n{notes or 'No release notes.'}\n\n{info.url}"
        if not result.available:
            body = f"Dizel {result.current} is up to date. Latest release: {info.version}\n\n{body}"
        return body


class CheckUpdateCommand(Command):
    name = "check-update"
    help_text = "Check GitHub for a newer Dizel release."
    category = "Runtime"
    usage = "/check-update"
    palette_hint = "/check-update"
    aliases = ["check-updates", "update-check"]
    examples = ["/check-update"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        return await _check(app, download=False)
