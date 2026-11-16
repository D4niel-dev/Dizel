from __future__ import annotations

from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class ReleaseNotesCommand(Command):
    name = "release-notes"
    help_text = "Read the release notes bundled with this build."
    category = "Core"
    usage = "/release-notes"
    palette_hint = "/release-notes"
    aliases = ["whats-new", "changelog"]
    examples = ["/release-notes"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import release_notes, updater
        from inference.tui.panels.modals.release_notes import ReleaseNotesModal

        text = release_notes.read_changelog()
        if not text.strip():
            return "No release notes are bundled with this build."
        app.push_screen(ReleaseNotesModal(text, updater.current_version()))
        return ""