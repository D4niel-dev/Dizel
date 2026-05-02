from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation
from inference.dizel_ui.logic.config_manager import ConfigManager

class ConfigCommand(Command):
    name = "config"
    help_text = "Show or update settings. Usage: /config [key] [value]"
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        manager = ConfigManager()
        settings = manager.load()
        if not invocation.args:
            lines = ["Current Configuration:"]
            for k, v in settings.items():
                if isinstance(v, dict):
                    lines.append(f"{k}:")
                    for subk, subv in v.items():
                        lines.append(f"  {subk}: {subv}")
                else:
                    lines.append(f"{k}: {v}")
            return "\n".join(lines)
            
        key = invocation.args[0]
        if len(invocation.args) == 1:
            return f"{key}: {settings.get(key, 'Not found')}"
            
        value = " ".join(invocation.args[1:])
        settings[key] = value
        manager.save(settings)
        return f"Updated {key} = {value}"
