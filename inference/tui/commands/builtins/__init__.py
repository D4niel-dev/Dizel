from inference.tui.commands.registry import registry
from .help import HelpCommand
from .clear import ClearCommand
from .exit import ExitCommand
from .model import ModelCommand
from .mode import ModeCommand
from .tier import TierCommand
from .provider import ProviderCommand
from .config import ConfigCommand
from .tools import ToolsCommand
from .mcp import McpCommand
from .plugins import PluginsCommand
from .capabilities import CapabilitiesCommand
from .context import ContextCommand
from .reload import ReloadCommand
from .session import SessionCommand
from .history import HistoryCommand
from .export import ExportCommand
from .reset import ResetCommand
from .load import LoadCommand
from .settings import SettingsCommand
from .tag import TagCommand
from .diff import DiffCommand
from .macro import MacroCommand
from .update_cmd import UpdateCommand
from .skills import SkillsCommand, register_skills

def register_builtins():
    registry.register(HelpCommand())
    registry.register(ClearCommand())
    registry.register(ExitCommand())
    registry.register(ModelCommand())
    registry.register(ModeCommand())
    registry.register(TierCommand())
    registry.register(ProviderCommand())
    registry.register(ConfigCommand())
    registry.register(ToolsCommand())
    registry.register(McpCommand())
    registry.register(PluginsCommand())
    registry.register(CapabilitiesCommand())
    registry.register(ContextCommand())
    registry.register(ReloadCommand())
    registry.register(SessionCommand())
    registry.register(HistoryCommand())
    registry.register(ExportCommand())
    registry.register(ResetCommand())
    registry.register(LoadCommand())
    registry.register(SettingsCommand())
    registry.register(TagCommand())
    registry.register(DiffCommand())
    registry.register(MacroCommand())
    registry.register(UpdateCommand())
    registry.register(SkillsCommand())
    register_skills()
