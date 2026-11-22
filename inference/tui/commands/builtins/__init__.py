from inference.tui.commands.registry import registry
from .help import HelpCommand
from .clear import ClearCommand
from .exit import ExitCommand
from .model import ModelCommand
from .mode import ModeCommand
from .verbosity import VerbosityCommand
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
from .find import FindCommand
from .export import ExportCommand
from .copy import CopyCommand
from .reset import ResetCommand
from .load import LoadCommand
from .settings import SettingsCommand
from .tag import TagCommand
from .diff import DiffCommand
from .macro import MacroCommand
from .update_cmd import CheckUpdateCommand, UpdateCommand
from .debug import DebugCommand
from .doctor import DoctorCommand
from .scorecard import ScorecardCommand
from .cost import CostCommand
from .memory import MemoryCommand
from .git_commands import ReviewCommand, DraftCommitCommand
from .plan import PlanCommand
from .edit import EditCommand
from .release_notes import ReleaseNotesCommand
from .open_gui import OpenInGuiCommand
from .skills import SkillsCommand, register_skills

def register_builtins():
    registry.register(HelpCommand())
    registry.register(ClearCommand())
    registry.register(ExitCommand())
    registry.register(ModelCommand())
    registry.register(ModeCommand())
    registry.register(VerbosityCommand())
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
    registry.register(FindCommand())
    registry.register(ExportCommand())
    registry.register(CopyCommand())
    registry.register(ResetCommand())
    registry.register(LoadCommand())
    registry.register(SettingsCommand())
    registry.register(TagCommand())
    registry.register(DiffCommand())
    registry.register(MacroCommand())
    registry.register(UpdateCommand())
    registry.register(CheckUpdateCommand())
    registry.register(DebugCommand())
    registry.register(DoctorCommand())
    registry.register(ScorecardCommand())
    registry.register(CostCommand())
    registry.register(MemoryCommand())
    registry.register(ReviewCommand())
    registry.register(DraftCommitCommand())
    registry.register(PlanCommand())
    registry.register(EditCommand())
    registry.register(ReleaseNotesCommand())
    registry.register(OpenInGuiCommand())
    registry.register(SkillsCommand())
    register_skills()
