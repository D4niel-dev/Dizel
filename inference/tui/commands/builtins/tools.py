from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation

# The single registry. The tools menu, /tools and the context rail all read
# this, so a tool cannot exist in one place and be missing in another.
# Two different mechanisms used to share the name "tool", which was confusing:
# `tools` gates the callable registry in dizel_core, while `deep_think` is a
# generation mode. `web_search` is no longer listed here — it is a real tool in
# that registry now, so offering it as a pre-search toggle was a second way to do
# the same thing.
AVAILABLE_TOOLS = {
    "tools": "Let the model call tools: read_file, grep, find_files, calc, web_search…",
    "deep_think": "Reason in <think>…</think> before answering",
}


class ToolsCommand(Command):
    name = "tools"
    help_text = "List or toggle tools. Usage: /tools [name] [on|off]"
    category = "Runtime"
    usage = "/tools [name] [on|off]"
    palette_hint = "/tools "
    examples = ["/tools", "/tools web_search on"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not hasattr(app, "_tool_states"):
            app._tool_states = {name: False for name in AVAILABLE_TOOLS}

        if not invocation.args:
            from inference.tui.panels.tools_menu import ToolsMenuModal

            app.push_screen(ToolsMenuModal())
            return ""

        tool_name = invocation.args[0].lower()
        if tool_name not in AVAILABLE_TOOLS:
            return f"Unknown tool: {tool_name}\nAvailable: {', '.join(AVAILABLE_TOOLS)}"

        if len(invocation.args) < 2:
            state = "on" if app._tool_states.get(tool_name, False) else "off"
            return f"{tool_name}: {state} — {AVAILABLE_TOOLS[tool_name]}"

        action = invocation.args[1].lower()
        if action in ("on", "enable", "true", "1"):
            app._tool_states[tool_name] = True
        elif action in ("off", "disable", "false", "0"):
            app._tool_states[tool_name] = False
        else:
            return f"Invalid action: {action}. Use 'on' or 'off'."

        if tool_name == "tools":
            # The registry is a module-level switch in dizel_core, so it has to be
            # set here rather than only in the app's own state.
            from dizel_core import tools as tool_registry

            tool_registry.set_enabled(app._tool_states[tool_name])

        app.refresh_context()
        state = "on" if app._tool_states[tool_name] else "off"
        if tool_name == "tools":
            from dizel_core import tools as tool_registry

            names = ", ".join(sorted(tool_registry.REGISTRY))
            return f"tools: {state}\nAvailable: {names}"
        return f"{tool_name}: {state}"
