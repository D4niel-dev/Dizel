from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command

VERBOSITIES = ("short", "normal", "detailed")

_HELP = (
    "Response length scales the reply's token budget.\n"
    "  short     answer only what was asked\n"
    "  normal    the default\n"
    "  detailed  room for reasoning, examples and caveats"
)


class VerbosityCommand(Command):
    name = "verbosity"
    help_text = "Set how long replies may run: short, normal or detailed."
    category = "Runtime"
    usage = "/verbosity [short|normal|detailed]"
    palette_hint = "/verbosity "
    aliases = ["length"]
    examples = ["/verbosity", "/verbosity short", "/verbosity detailed"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        current = str(getattr(app, "active_verbosity", "normal") or "normal")

        if not invocation.args:
            lines = [f"Response length: {current}", "", _HELP]
            if current != "normal":
                lines += ["", f"/verbosity normal to reset."]
            return "\n".join(lines)

        requested = invocation.args[0].strip().lower()
        if requested not in VERBOSITIES:
            return f"Unknown length: {requested}\n\n{_HELP}"
        if requested == current:
            return f"Response length is already {current}."

        app.active_verbosity = requested
        # Same key the desktop app writes: the choice survives a restart and
        # the two surfaces agree on what "detailed" means.
        try:
            from dizel_core.config_manager import ConfigManager

            settings = ConfigManager.load()
            budget = settings.get("token_budget")
            if not isinstance(budget, dict):
                budget = {}
            budget["verbosity"] = requested
            settings["token_budget"] = budget
            ConfigManager.save(settings)
        except Exception:
            # A read-only settings.json must not lose the choice for this run.
            return f"Response length: {requested} (could not be saved)"

        return f"Response length: {requested}. Applies to the next message."
