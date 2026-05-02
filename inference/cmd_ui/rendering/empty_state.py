from textual.widgets import Static
from textual.app import ComposeResult
from textual.containers import Container

ASCII_LOGO = """
[b #93C5FD]██████╗ ██╗███████╗███████╗██╗         █████╗ ██╗[/]
[b #60A5FA]██╔══██╗██║╚══███╔╝██╔════╝██║        ██╔══██╗██║[/]
[b #3B82F6]██║  ██║██║  ███╔╝ █████╗  ██║        ███████║██║[/]
[b #2563EB]██║  ██║██║ ███╔╝  ██╔══╝  ██║        ██╔══██║██║[/]
[b #1D4ED8]██████╔╝██║███████╗███████╗███████╗   ██║  ██║██║[/]
[b #1E3A8A]╚═════╝ ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝  ╚═╝╚═╝[/]

[#A1A1AA]Dizel - Your personal AI agent
Type your prompt in the input bar and press Enter to start![/]

[#71717A]Press [#3B82F6 b]ctrl+r[/] to toggle code in context panel
[#3B82F6 b]/help[/] for commands, [#3B82F6 b]/exit[/] to close the CMD UI![/]
"""

class EmptyStateBlock(Container):
    def compose(self) -> ComposeResult:
        yield Static(ASCII_LOGO, classes="empty-state-logo")
