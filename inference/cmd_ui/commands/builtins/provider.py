from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

VALID_PROVIDERS = ["local", "openai", "anthropic", "google", "ollama", "groq", "mistral", "xai", "ai21", "azure", "cohere", "meta"]

class ProviderCommand(Command):
    name = "provider"
    help_text = "Switch or list providers. Usage: /provider [name]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            lines = [f"Current provider: {app.active_provider}", "", "Available providers:"]
            for p in VALID_PROVIDERS:
                marker = " *" if p == app.active_provider else ""
                lines.append(f"  {p}{marker}")
            return "\n".join(lines)

        new_provider = invocation.args[0].lower()
        if new_provider not in VALID_PROVIDERS:
            return f"Unknown provider: {new_provider}\nAvailable: {', '.join(VALID_PROVIDERS)}"

        app.active_provider = new_provider

        # Update the chat manager's provider state
        manager = app.chat_bridge.manager
        from inference.dizel_ui.logic.config_manager import ConfigManager
        cfg = ConfigManager.load()
        api_cfg = cfg.get("api_router", {})
        api_cfg["provider"] = new_provider
        cfg["api_router"] = api_cfg
        ConfigManager.save(cfg)

        # Reload to pick up the change
        manager.reload_provider()

        return f"Provider switched to: {new_provider}"
