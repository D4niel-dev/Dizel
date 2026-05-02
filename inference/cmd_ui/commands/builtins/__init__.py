from inference.cmd_ui.commands.registry import registry
from .help import HelpCommand
from .clear import ClearCommand
from .exit import ExitCommand
from .model import ModelCommand
from .mode import ModeCommand
from .provider import ProviderCommand
from .config import ConfigCommand
from .tools import ToolsCommand
from .context import ContextCommand
from .reload import ReloadCommand
from .session import SessionCommand
from .history import HistoryCommand
from .export import ExportCommand
from .reset import ResetCommand
from .load import LoadCommand

def register_builtins():
    registry.register(HelpCommand())
    registry.register(ClearCommand())
    registry.register(ExitCommand())
    registry.register(ModelCommand())
    registry.register(ModeCommand())
    registry.register(ProviderCommand())
    registry.register(ConfigCommand())
    registry.register(ToolsCommand())
    registry.register(ContextCommand())
    registry.register(ReloadCommand())
    registry.register(SessionCommand())
    registry.register(HistoryCommand())
    registry.register(ExportCommand())
    registry.register(ResetCommand())
    registry.register(LoadCommand())
