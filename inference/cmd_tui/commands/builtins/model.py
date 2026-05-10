from typing import Any

from inference.cmd_tui.commands.parser import CommandInvocation
from inference.cmd_tui.commands.registry import Command

LOCAL_MODELS = ["Dizel Lite", "Dizel Pro", "Mila Lite", "Mila Pro"]


class ModelCommand(Command):
    name = "model"
    help_text = "Switch models. Works for both local and API providers."
    category = "Runtime"
    usage = "/model [name]"
    palette_hint = "/model "
    examples = ["/model Dizel Pro", "/model gpt-4o"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        manager = app.chat_bridge.manager
        is_api = manager._provider_slug != "local" and manager._api_provider is not None

        if not invocation.args:
            lines = [f"### Current Status", f"- **Model:** `{app.active_model}`", f"- **Provider:** `{app.active_provider}`", ""]

            if is_api:
                lines.append("You are using an **API provider**.")
                lines.append("Set a model with: `/model <model-id>`")
                lines.append("*Example: `/model gpt-4o`*")
                lines.append("")
                try:
                    models = manager._api_provider.list_models(key=manager._api_key)
                    if models:
                        lines.append("#### Available models from provider:")
                        for model in models[:20]:
                            marker = " **(current)**" if model.id == manager._api_model else ""
                            lines.append(f"- `{model.id}` ({model.name}){marker}")
                except Exception as exc:
                    lines.append(f"**Error:** Could not list models: {exc}")
            else:
                lines.append("#### Local models:")
                for model in LOCAL_MODELS:
                    marker = " **(current)**" if model == app.active_model else ""
                    lines.append(f"- `{model}`{marker}")
            return "\n".join(lines)

        new_model = " ".join(invocation.args)

        if is_api:
            manager._api_model = new_model
            app.active_model = new_model

            from inference.dizel_gui.logic.config_manager import ConfigManager

            cfg = ConfigManager.load()
            api_cfg = cfg.get("api_router", {})
            api_cfg["model"] = new_model
            cfg["api_router"] = api_cfg
            ConfigManager.save(cfg)

            return f"Model set to: {new_model} (via {app.active_provider})"

        matched = next((model for model in LOCAL_MODELS if model.lower() == new_model.lower()), None)
        if not matched:
            return f"Unknown local model: {new_model}\nAvailable: {', '.join(LOCAL_MODELS)}"

        app.active_model = matched
        try:
            manager.apply_profile(matched, app.active_mode)
        except Exception:
            pass
        return f"Model switched to: {matched}"
