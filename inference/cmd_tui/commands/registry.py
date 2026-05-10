from typing import Any, Dict, List, Optional
from inference.cmd_tui.commands.parser import CommandInvocation


class Command:
    name: str = ""
    help_text: str = ""
    category: str = "General"
    usage: str = ""
    palette_hint: str = ""
    aliases: List[str] = []
    examples: List[str] = []
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        raise NotImplementedError

    @property
    def insert_text(self) -> str:
        return self.palette_hint or f"/{self.name} "


class CommandRegistry:
    def __init__(self):
        self._commands: Dict[str, Command] = {}
        
    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command
            
    def lookup(self, name: str) -> Optional[Command]:
        return self._commands.get(name)
        
    def complete(self, prefix: str) -> List[str]:
        return sorted(
            name
            for name, cmd in self._commands.items()
            if name.startswith(prefix) and name == cmd.name
        )
        
    def list_all(self) -> List[Command]:
        seen = set()
        commands = []
        for command in self._commands.values():
            if command.name in seen:
                continue
            seen.add(command.name)
            commands.append(command)
        return sorted(commands, key=lambda command: (command.category, command.name))

# Global registry
registry = CommandRegistry()
