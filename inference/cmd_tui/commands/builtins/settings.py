from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation


class SettingsCommand(Command):
    name = "settings"
    help_text = "View or update settings. Usage: /settings [section] [key] [value]"
    category = "Settings"
    usage = "/settings [section] [key] [value]"
    palette_hint = "/settings"
    examples = ["/settings", "/settings sampling temperature 0.7"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        from inference.dizel_gui.logic.config_manager import ConfigManager

        cfg = ConfigManager.load()
        args = invocation.args

        # /settings — show full overview
        if not args:
            app.query_one("SettingsPanel").open()
            return None

        section = args[0].lower()

        # /settings <section> — show a specific section
        if len(args) == 1:
            return self._render_section(app, cfg, section)

        # /settings <section> <key> — show a specific key
        if len(args) == 2:
            return self._render_key(cfg, section, args[1])

        # /settings <section> <key> <value> — update a setting
        return self._update_key(cfg, section, args[1], " ".join(args[2:]))

    def _render_overview(self, app: Any, cfg: dict) -> str:
        mgr = app.chat_bridge.manager
        samp = cfg.get("sampling", {})
        api = cfg.get("api_router", {})
        budget = cfg.get("token_budget", {})
        nova = cfg.get("nova", {})

        provider = api.get("provider", "local")
        model = api.get("model", "") or app.active_model
        key_status = "✓ set" if api.get("api_key") else "✗ not set"

        lines = [
            "### Settings Overview",
            "",
            "#### Runtime",
            f"- **Provider:** `{provider}`",
            f"- **Model:** `{model}`",
            f"- **Mode:** `{app.active_mode}`",
            f"- **API Key:** `{key_status}`",
            f"- **Device:** `{cfg.get('device', 'cpu')}`",
            "",
            "#### Sampling",
            f"- **Temperature:** `{samp.get('temperature', 0.7)}`",
            f"- **Top-K:** `{samp.get('top_k', 50)}`",
            f"- **Top-P:** `{samp.get('top_p', 0.92)}`",
            f"- **Rep Penalty:** `{samp.get('repetition_penalty', 1.15)}`",
            f"- **Max Tokens:** `{samp.get('max_new_tokens', 200)}`",
            "",
            "#### Token Budget",
            f"- **Chat:** `{budget.get('chat_budget', 150)}`",
            f"- **Coding:** `{budget.get('coding_budget', 350)}`",
            f"- **Complex:** `{budget.get('complex_budget', 500)}`",
            f"- **Hard Limit:** `{budget.get('hard_output_limit', 600)}`",
            "",
            "#### Nova (Voice)",
            f"- **Model Size:** `{nova.get('model_size', 'base')}`",
            f"- **Language:** `{nova.get('language', 'auto')}`",
            f"- **Timeout:** `{nova.get('silence_timeout', 5)}s`",
            "",
            "---",
            "**Sections:** `sampling` · `token_budget` · `nova` · `api_router`",
            "**Usage:** `/settings sampling temperature 0.5`",
            "**Example:** `/settings api_router model gpt-4o`",
        ]
        return "\n".join(lines)

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
            for k, v in data.items():
                display = v
                if "key" in k.lower() and isinstance(v, str) and v:
                    display = "••••••••" if v else "(empty)"
                lines.append(f"  {k}: {display}")
            return "\n".join(lines)

        return f"{section}: {data}"

    def _render_key(self, cfg: dict, section: str, key: str) -> str:
        data = cfg.get(section)
        if data is None:
            return f"Unknown section: {section}"
        if isinstance(data, dict):
            val = data.get(key)
            if val is None:
                return f"Unknown key: {section}.{key}\nAvailable: {', '.join(data.keys())}"
            if "key" in key.lower() and isinstance(val, str) and val:
                return f"{section}.{key}: ••••••••"
            return f"{section}.{key}: {val}"
        return f"{section}: {data}"

    def _update_key(self, cfg: dict, section: str, key: str, value: str) -> str:
        from inference.dizel_gui.logic.config_manager import ConfigManager, encrypt_key

        data = cfg.get(section)
        if data is None:
            return f"Unknown section: {section}"
        if not isinstance(data, dict):
            return f"Cannot update non-dict section: {section}"

        old = data.get(key)
        if old is None:
            return f"Unknown key: {section}.{key}\nAvailable: {', '.join(data.keys())}"

        # Type coercion based on original type
        if isinstance(old, bool):
            new_val = value.lower() in ("true", "1", "yes")
        elif isinstance(old, int):
            try:
                new_val = int(value)
            except ValueError:
                return f"Invalid integer: {value}"
        elif isinstance(old, float):
            try:
                new_val = float(value)
            except ValueError:
                return f"Invalid number: {value}"
        else:
            new_val = value

        # Encrypt API keys before saving
        if "key" in key.lower() and section == "api_router":
            data[key] = encrypt_key(new_val) if new_val else ""
        else:
            data[key] = new_val

        cfg[section] = data
        ConfigManager.save(cfg)

        display = "••••••••" if "key" in key.lower() else new_val
        return f"**Updated:** `{section}.{key}` = `{display}`"
