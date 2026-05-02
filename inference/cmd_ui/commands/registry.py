from typing import Dict, Any, List, Optional
from inference.cmd_ui.commands.parser import CommandInvocation

class Command:
    name: str = ""
    help_text: str = ""
    aliases: List[str] = []
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        raise NotImplementedError

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
        return [name for name, cmd in self._commands.items() if name.startswith(prefix) and name == cmd.name]
        
    def list_all(self) -> List[Command]:
        return list(set(self._commands.values()))

# Global registry
registry = CommandRegistry()
