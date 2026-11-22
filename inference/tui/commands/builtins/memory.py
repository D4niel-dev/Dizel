from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class MemoryCommand(Command):
    name = "memory"
    help_text = "View, add, or clear the project's durable memory and rules."
    category = "Runtime"
    usage = "/memory [add <text>|remove <id|index>|clear|rules]"
    palette_hint = "/memory "
    examples = ["/memory", "/memory add Uses CRLF for .cmd", "/memory rules"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import memory_store, project_context, tools

        root = tools._project_root()
        action = invocation.args[0].lower() if invocation.args else ""
        rest = invocation.args[1:]

        if action == "rules":
            context = project_context.load(root)
            app.show_debug_output(
                "Project rules", project_context.rules_summary(context)
            )
            return ""

        if action == "add":
            text = " ".join(rest).strip()
            if not text:
                return "Usage: /memory add <text>"
            try:
                entry = memory_store.add(text, project_root=root, source="user")
            except ValueError as error:
                return str(error)
            return f"Saved to project memory: {entry['text'][:80]}"

        if action == "remove":
            selector = rest[0] if rest else ""
            if not selector:
                return "Usage: /memory remove <id|index>"
            if memory_store.remove(selector, project_root=root):
                return "Removed from project memory."
            return "No such memory entry."

        if action == "clear":
            removed = memory_store.clear(project_root=root)
            return f"Cleared {removed} memory entries."

        context = project_context.load(root)
        lines = [
            f"Project: {context.root}",
            "",
            project_context.rules_summary(context),
            "",
            f"Memory — {len(context.memory)} entries",
        ]
        for row in context.memory:
            lines.append(f"  {row['id']}  {row['text']}")
        if not context.memory:
            lines.append("  (empty — /memory add <text>)")
        lines.extend(
            [
                "",
                "Commands: /memory add <text> · remove <id|index> · clear · rules",
            ]
        )
        app.show_debug_output("Project memory", "\n".join(lines))
        return ""
