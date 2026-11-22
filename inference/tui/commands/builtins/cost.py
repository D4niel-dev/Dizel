from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


class CostCommand(Command):
    name = "cost"
    help_text = "Estimate provider usage and cost for the current 12-hour window."
    category = "Runtime"
    usage = "/cost"
    palette_hint = "/cost "
    examples = ["/cost"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core import usage_report

        app.show_debug_output(
            "Usage & cost",
            usage_report.report_text(usage_report.snapshot(app.usage_manager)),
        )
        return ""
