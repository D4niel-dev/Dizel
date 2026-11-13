import copy
from typing import Any, Dict
from dizel_core.config_manager import ConfigManager

# MCP secret values live in settings.json and never travel over HTTP (spec A6/A9).
_SECRET_FIELDS = ("env", "headers")

# Sections owned by the Settings dialog. Resetting preferences restores these
# while leaving identity, provider, checkpoint, MCP and history untouched.
_PREF_SECTIONS = ("appearance", "sampling", "token_budget", "nova")


def redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy safe for HTTP: MCP env/header values are blanked.

    Key names are kept so clients know what is stored, but every value is
    replaced with an empty string.
    """
    redacted = copy.deepcopy(config)
    mcp = redacted.get("mcp")
    servers = mcp.get("servers") if isinstance(mcp, dict) else None
    if not isinstance(servers, list):
        return redacted
    for server in servers:
        if not isinstance(server, dict):
            continue
        for field in _SECRET_FIELDS:
            values = server.get(field)
            if isinstance(values, dict):
                server[field] = {str(key): "" for key in values}
            elif field in server:
                server[field] = {}
    return redacted


class ConfigService:
    """Service wrapper for application configuration"""

    @staticmethod
    def get_config_raw() -> Dict[str, Any]:
        """Unredacted config for in-process consumers (MCP startup/routes)."""
        return ConfigManager.load()

    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get the current application configuration (MCP secrets redacted)."""
        return redact_config(ConfigManager.load())

    @staticmethod
    def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the application configuration with partial changes.
        Returns the updated configuration (MCP secrets redacted).
        """
        config = ConfigManager.load()

        # Merge updates. The `mcp` section is owned by PUT /mcp/servers; a
        # generic patch must not replace it or validation could be bypassed.
        for key, val in updates.items():
            if key == "mcp":
                continue
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val

        ConfigManager.save(config)
        return redact_config(config)

    @staticmethod
    def reset_preferences() -> Dict[str, Any]:
        """Restore the preference sections to their built-in defaults."""
        config = ConfigManager.load()
        defaults = ConfigManager._deep_copy_defaults()
        for section in _PREF_SECTIONS:
            config[section] = defaults[section]
        ConfigManager.save(config)
        return redact_config(config)
