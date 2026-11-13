from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation


class SettingsCommand(Command):
    name = "settings"
    help_text = "Open settings, or get/set one value by section and key."
    category = "Settings"
    usage = "/settings [section] [key] [value]"
    palette_hint = "/settings"
    examples = ["/settings", "/settings sampling temperature 0.7"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from dizel_core.config_manager import ConfigManager

        cfg = ConfigManager.load()
        args = invocation.args

        if not args:
            from inference.tui.panels.modals import SettingsModal

            app.push_screen(SettingsModal())
            return ""

        section = args[0].lower()

        if len(args) == 1:
            return self._render_section(app, cfg, section)

        if len(args) == 2:
            return self._render_key(cfg, section, args[1])

        return self._update_key(app, cfg, section, args[1], " ".join(args[2:]))

    def _render_section(self, app: Any, cfg: dict, section: str) -> str:
        if section == "runtime":
            api = cfg.get("api_router", {})
            lines = [
                f"Provider:   {api.get('provider', 'local')}",
                f"Model:      {api.get('model', '') or app.active_model}",
                f"Mode:       {app.active_mode}",
                f"Device:     {cfg.get('device', 'cpu')}",
                f"Checkpoint: {cfg.get('checkpoint', '(auto)')}",
            ]
            return "\n".join(lines)

        data = cfg.get(section)
        if data is None:
            available = [k for k in cfg if isinstance(cfg[k], dict)]
            return f"Unknown section: {section}\nAvailable: {', '.join(available)}"

        if isinstance(data, dict):
            lines = [f"{section}:"]
            for key, value in data.items():
                display = "••••••••" if "key" in key.lower() and value else value
                lines.append(f"  {key}: {display}")
            return "\n".join(lines)

        return f"{section}: {data}"

    def _render_key(self, cfg: dict, section: str, key: str) -> str:
        data = cfg.get(section)
        if data is None:
            return f"Unknown section: {section}"
        if isinstance(data, dict):
            if key not in data:
                return f"Unknown key: {section}.{key}\nAvailable: {', '.join(data.keys())}"
            value = data[key]
            if "key" in key.lower() and value:
                return f"{section}.{key}: ••••••••"
            return f"{section}.{key}: {value}"
        return f"{section}: {data}"

    def _update_key(self, app: Any, cfg: dict, section: str, key: str, value: str) -> str:
        from dizel_core.config_manager import ConfigManager, encrypt_key

        data = cfg.get(section)
        if data is None:
            return f"Unknown section: {section}"
        if not isinstance(data, dict):
            return f"Cannot update non-dict section: {section}"

        old = data.get(key)
        if old is None:
            return f"Unknown key: {section}.{key}\nAvailable: {', '.join(data.keys())}"

        if isinstance(old, bool):
            new_value = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(old, int):
            try:
                new_value = int(value)
            except ValueError:
                return f"Invalid integer: {value}"
        elif isinstance(old, float):
            try:
                new_value = float(value)
            except ValueError:
                return f"Invalid number: {value}"
        else:
            new_value = value

        if "key" in key.lower() and section == "api_router":
            data[key] = encrypt_key(new_value) if new_value else ""
        else:
            data[key] = new_value

        cfg[section] = data
        ConfigManager.save(cfg)

        applied = self._apply(app, section)
        display = "••••••••" if "key" in key.lower() else new_value
        return f"Updated {section}.{key} = {display}{applied}"

    def _apply(self, app: Any, section: str) -> str:
        """Push the change into the live engine instead of waiting for /reload."""
        manager = app.chat_bridge.manager
        try:
            if section == "sampling":
                manager.reload_sampling()
                return "  (applied)"
            if section == "api_router":
                manager.reload_provider()
                app.active_provider = manager.active_provider_slug
                if manager.active_api_model:
                    app.active_model = manager.active_api_model
                return "  (applied)"
            if section in ("checkpoint", "device"):
                return "  (run /reload to load it)"
        except Exception as error:
            return f"  (saved, but applying failed: {error})"
        return ""
