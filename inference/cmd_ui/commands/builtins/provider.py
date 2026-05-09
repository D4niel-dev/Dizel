from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

VALID_PROVIDERS = ["local", "openai", "anthropic", "google", "ollama", "groq", "mistral", "xai", "ai21", "azure", "cohere", "meta"]

class ProviderCommand(Command):
    name = "provider"
    help_text = "Switch or list providers. Usage: /provider [name] [model] [key]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            lines = [f"Current provider: {app.active_provider}"]
            mgr = app.chat_bridge.manager
            if mgr._api_model:
                lines.append(f"Current model: {mgr._api_model}")
            lines.append("")
            lines.append("Available providers:")
            for p in VALID_PROVIDERS:
                marker = " ←" if p == app.active_provider else ""
                lines.append(f"  {p}{marker}")
            lines.append("")
            lines.append("Usage:")
            lines.append("  /provider ollama llama3")
            lines.append("  /provider openai gpt-4o sk-...")
            lines.append("  /provider anthropic claude-sonnet-4-20250514 sk-ant-...")
            lines.append("  /provider local  (switch back to local model)")
            return "\n".join(lines)

        new_provider = invocation.args[0].lower()
        if new_provider not in VALID_PROVIDERS:
            return f"Unknown provider: {new_provider}\nAvailable: {', '.join(VALID_PROVIDERS)}"

        model_id = invocation.args[1] if len(invocation.args) > 1 else ""
        api_key = invocation.args[2] if len(invocation.args) > 2 else ""

        from inference.dizel_ui.logic.config_manager import ConfigManager, encrypt_key
        cfg = ConfigManager.load()
        api_cfg = cfg.get("api_router", {})
        api_cfg["provider"] = new_provider

        if model_id:
            api_cfg["model"] = model_id
        if api_key:
            api_cfg["api_key"] = encrypt_key(api_key)

        cfg["api_router"] = api_cfg
        ConfigManager.save(cfg)

        manager = app.chat_bridge.manager
        manager.reload_provider()

        app.active_provider = new_provider

        if new_provider == "local":
            app.active_model = "Dizel Lite"
            return "Switched to local model inference."

        actual_model = model_id or manager._api_model or "(none set)"
        app.active_model = actual_model

        # Validate connection for non-local providers
        if manager._api_provider:
            try:
                manager._api_provider.validate(key=manager._api_key)
                return f"Provider: {new_provider}\nModel: {actual_model}\n✓ Connection verified."
            except (ConnectionError, ValueError) as e:
                return f"Provider: {new_provider}\nModel: {actual_model}\n✗ Connection failed: {e}"
        else:
            return f"Provider set to {new_provider} but failed to load. Check dependencies."
