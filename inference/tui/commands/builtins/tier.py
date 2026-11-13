from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation
from inference.tui.constants import TIERS


class TierCommand(Command):
    name = "tier"
    help_text = "Pick the model tier (Flash or Pro) for Dizel models."
    category = "Runtime"
    usage = "/tier [flash|pro]"
    palette_hint = "/tier "
    examples = ["/tier", "/tier pro"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not app.is_dizel_model:
            return "Tier applies to Dizel models only. Use /mode for this model."

        if not invocation.args:
            from inference.tui.panels.modals import TierPickerModal

            app.push_screen(
                TierPickerModal(),
                lambda tier: app.set_tier(tier) if tier else None,
            )
            return ""

        new_tier = invocation.args[0].capitalize()
        if new_tier not in TIERS:
            return f"Unknown tier: {new_tier}\nAvailable: {', '.join(TIERS)}"

        app.set_tier(new_tier)
        return ""
