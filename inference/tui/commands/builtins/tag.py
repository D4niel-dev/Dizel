from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation
from dizel_core.history_manager import add_tag_to_session


class TagCommand(Command):
    name = "tag"
    help_text = "Add a tag to the current session. Usage: /tag <tag_name>"
    category = "Session"
    usage = "/tag <tag_name>"
    palette_hint = "/tag "
    examples = ["/tag research", "/tag experiments"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            return "Usage: /tag <tag_name>"

        tag = invocation.args[0].lstrip("#")
        if not app.session_id:
            return "No active session to tag. Send a message first."

        if add_tag_to_session(app.session_id, tag):
            app.refresh_sessions()
            return f"Tagged this session #{tag}."
        return f"Could not tag this session (is “{tag}” already set?)."
