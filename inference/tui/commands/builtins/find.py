from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class FindCommand(Command):
    name = "find"
    help_text = "Search the transcript; repeat to step to the next match."
    category = "Session"
    usage = "/find [text]"
    palette_hint = "/find "
    aliases = ["search"]
    examples = ["/find docker", "/find"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from inference.tui.panels.workspace import WorkspacePanel

        try:
            panel = app.query_one(WorkspacePanel)
        except Exception:
            return "The transcript is not open."

        query = " ".join(invocation.args).strip()
        result = panel.find(query)

        if not result["query"]:
            return f"Usage: {self.usage}"

        if not result["turns"]:
            return (
                f"No match for “{result['query']}” in this transcript.\n"
                "Only completed turns are searched."
            )

        return (
            f"“{result['query']}” · turn {result['index']} of {result['turns']} "
            f"· {result['matches']} "
            f"{'match' if result['matches'] == 1 else 'matches'}\n"
            "/find again for the next one."
        )
