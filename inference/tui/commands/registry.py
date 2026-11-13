from typing import Any, Dict, List, Optional
from inference.tui.commands.matching import command_score
from inference.tui.commands.parser import CommandInvocation


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

    def unregister(self, command: Command) -> None:
        for name in [name for name, entry in self._commands.items() if entry is command]:
            del self._commands[name]
            
    def lookup(self, name: str) -> Optional[Command]:
        return self._commands.get(name)
        
    def complete(self, prefix: str) -> List[str]:
        return sorted(
            name
            for name, cmd in self._commands.items()
            if name.startswith(prefix) and name == cmd.name
        )

    def search(self, query: str) -> List[Command]:
        """Fuzzy search over canonical names and aliases, best match first."""
        needle = query.strip().lower()
        if not needle:
            return self.list_all()
        scored = []
        for command in self.list_all():
            score = command_score(command.name, command.aliases, needle)
            if score < float("inf"):
                scored.append((score, command))
        scored.sort(key=lambda item: (item[0], item[1].category, item[1].name))
        return [command for _, command in scored]

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
