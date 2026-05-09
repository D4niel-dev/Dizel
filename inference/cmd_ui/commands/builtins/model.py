from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

LOCAL_MODELS = ["Dizel Lite", "Dizel Pro", "Mila Lite", "Mila Pro"]

class ModelCommand(Command):
    name = "model"
    help_text = "Switch models. Usage: /model [name]  — works for both local and API providers."

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        manager = app.chat_bridge.manager
        is_api = manager._provider_slug != "local" and manager._api_provider is not None

        if not invocation.args:
            lines = [f"Current model: {app.active_model}"]
            lines.append(f"Provider: {app.active_provider}")
            lines.append("")

            if is_api:
                lines.append("You are using an API provider.")
                lines.append("Set a model with:  /model <model-id>")
                lines.append("Example:  /model gpt-4o")
                lines.append("")
                # Try listing available models
                try:
                    models = manager._api_provider.list_models(key=manager._api_key)
                    if models:
                        lines.append("Available models from provider:")
                        for m in models[:20]:
                            marker = " ←" if m.id == manager._api_model else ""
                            lines.append(f"  {m.id} ({m.name}){marker}")
                except Exception as e:
                    lines.append(f"Could not list models: {e}")
            else:
                lines.append("Local models:")
                for m in LOCAL_MODELS:
                    marker = " ←" if m == app.active_model else ""
                    lines.append(f"  {m}{marker}")
            return "\n".join(lines)

        new_model = " ".join(invocation.args)

        if is_api:
            # For API providers, set the model ID directly
            manager._api_model = new_model
            app.active_model = new_model

            # Persist to config
            from inference.dizel_ui.logic.config_manager import ConfigManager
            cfg = ConfigManager.load()
            api_cfg = cfg.get("api_router", {})
            api_cfg["model"] = new_model
            cfg["api_router"] = api_cfg
            ConfigManager.save(cfg)

            return f"Model set to: {new_model} (via {app.active_provider})"
        else:
            # Local model: match against known profiles
            matched = None
            for m in LOCAL_MODELS:
                if m.lower() == new_model.lower():
                    matched = m
                    break

            if not matched:
                return f"Unknown local model: {new_model}\nAvailable: {', '.join(LOCAL_MODELS)}"

            app.active_model = matched
            try:
                manager.apply_profile(matched, app.active_mode)
            except Exception:
                pass
            return f"Model switched to: {matched}"
