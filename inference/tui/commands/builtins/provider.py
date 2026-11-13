from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command
from inference.tui.runtime_actions import (
    apply_model,
    apply_provider,
    provider_catalog,
    provider_meta,
    validate_provider_async,
)


class ProviderCommand(Command):
    name = "provider"
    help_text = "Pick or switch API providers."
    category = "Runtime"
    usage = "/provider [name] [model] [key]"
    palette_hint = "/provider "
    examples = ["/provider", "/provider local", "/provider ollama llama3"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            from inference.tui.panels.modals import ProviderPickerModal

            app.push_screen(
                ProviderPickerModal(),
                lambda slug: _picked(app, slug),
            )
            return ""

        new_provider = invocation.args[0].lower()
        if not provider_meta(new_provider):
            available = ", ".join(meta["slug"] for meta in provider_catalog())
            return f"Unknown provider: {new_provider}\nAvailable: {available}"

        model_id = invocation.args[1] if len(invocation.args) > 1 else ""
        api_key = invocation.args[2] if len(invocation.args) > 2 else ""

        try:
            message = apply_provider(app, new_provider, model_id, api_key)
        except ValueError as error:
            return str(error)

        if new_provider != "local" and app.active_provider == new_provider:
            validate_provider_async(app, new_provider)
        return message


def _picked(app: Any, slug: str | None) -> None:
    if not slug:
        return

    meta = provider_meta(slug)
    if meta is None:
        app._log_system(f"Unknown provider: {slug}")
        return

    if slug == "custom":
        from dizel_core.config_manager import ConfigManager

        if not str(ConfigManager.load().get("api_router", {}).get("custom_url", "")).strip():
            app._log_system("Custom endpoint: set a base URL in /settings → API first.")
            return

    manager = app.chat_bridge.manager
    needs_key = slug != "local" and bool(meta["requires_key"])
    has_key = manager._provider_slug == slug and bool(manager._api_key)

    if needs_key and not has_key:
        from inference.tui.panels.modals import ApiKeyModal

        def _with_key(key: str | None) -> None:
            if not key:
                app._log_system("Provider unchanged · no key entered.")
                return
            _finish(app, slug, key)

        app.push_screen(ApiKeyModal(slug), _with_key)
        return

    _finish(app, slug, "")


def _finish(app: Any, slug: str, key: str) -> None:
    try:
        message = apply_provider(app, slug, key=key)
    except Exception as error:
        app._log_system(f"Provider error: {error}")
        return

    app._log_system(message)

    if slug == "local" or app.active_provider != slug:
        return

    validate_provider_async(app, slug)

    if not app.active_model:
        from inference.tui.panels.modals import ModelPickerModal

        def _picked_model(name: str | None) -> None:
            if not name:
                return
            try:
                app._log_system(apply_model(app, name))
            except Exception as error:
                app._log_system(f"Model error: {error}")

        app.push_screen(ModelPickerModal(), _picked_model)
