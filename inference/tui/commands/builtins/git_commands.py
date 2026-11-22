from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class ReviewCommand(Command):
    name = "review"
    help_text = "Ask the model to review the current working-tree diff (read-only)."
    category = "Project"
    usage = "/review"
    palette_hint = "/review "
    examples = ["/review"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import git_tools

        change = git_tools.working_diff()
        if change.get("error"):
            return change["error"]
        count = len(change.get("files") or [])
        display = f"Review the working-tree diff · {count} files"
        return app.send_composed(display, git_tools.review_prompt(change))


class DraftCommitCommand(Command):
    name = "commit"
    help_text = "Draft a commit message for the staged diff; it never commits."
    category = "Project"
    usage = "/commit"
    palette_hint = "/commit "
    examples = ["/commit"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import git_tools

        change = git_tools.staged_diff()
        if change.get("error"):
            return change["error"]
        count = len(change.get("files") or [])
        display = f"Draft a commit message · {count} staged files"
        return app.send_composed(display, git_tools.commit_prompt(change))
