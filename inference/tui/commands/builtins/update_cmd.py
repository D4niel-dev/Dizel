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


class UpdateCommand(Command):
    name = "update"
    help_text = "Check, download, and stage a newer Dizel release."
    category = "Runtime"
    usage = "/update [check|now|status|auto on|off|apply|notes]"
    palette_hint = "/update "
    aliases = ["upgrade"]
    examples = [
        "/update",
        "/update now",
        "/update status",
        "/update auto off",
        "/update notes",
    ]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import updater

        action = (invocation.args[0].lower() if invocation.args else "check").strip()

        if action in ("check", "now", "download", "upgrade"):
            return await self._check(app, download=action in ("now", "download", "upgrade"))

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

        if action == "apply":
            pending = updater.pending_update()
            if not pending:
                return "No update is staged. Run /update now first."
            return (
                f"Dizel {pending.get('version')} is staged. Close this session and run "
                "dizel.cmd — it applies before the app starts. Your chats and settings are kept."
            )

        if action == "notes":
            return await self._notes()

        return self.usage

    async def _check(self, app: Any, download: bool) -> str:
        from dizel_core import updater

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
        if not download:
            if not info.tui:
                return (
                    f"Dizel {info.version} is available, but this release has no terminal "
                    f"package. Desktop installer: {info.url}"
                )
            return (
                f"Dizel {info.version} is available. Run /update now to download it, "
                "or leave automatic updates on."
            )

        if not info.tui:
            return (
                f"Release {info.tag} has no TUI package. Use the desktop installer: {info.url}"
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

        return (
            f"Dizel {info.version} downloaded and staged. It will apply the next time "
            "you start Dizel — chats and settings are kept."
        )

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
            lines.append(f"  staged             {pending.get('version')} — restart to apply")
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
