from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ReloadCommand(Command):
    name = "reload"
    help_text = "Reload configuration and providers."
    aliases = ["r"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        app.chat_bridge.manager.reload_provider()

        # Sync reactive state from the reloaded manager
        app.active_provider = app.chat_bridge.manager.active_provider_slug
        api_model = app.chat_bridge.manager.active_api_model
        if api_model:
            app.active_model = api_model

        return (
            f"Configuration reloaded.\n"
            f"  Provider: {app.active_provider}\n"
            f"  Model:    {app.active_model}"
        )
