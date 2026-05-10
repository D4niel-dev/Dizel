from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation

class ClearCommand(Command):
    name = "clear"
    help_text = "Clear the current workspace."
    category = "Workspace"
    usage = "/clear"
    palette_hint = "/clear"
    aliases = ["c"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        app.action_clear_workspace()
        
        from textual.widgets import Static
        from inference.cmd_tui.rendering.message_block import MessageBlock
        
        try:
            workspace = app.query_one("WorkspacePanel")
            msg_widget = Static(MessageBlock("SYSTEM", "Workspace cleared."))
            workspace.mount(msg_widget)
            workspace.scroll_end(animate=False)
            
            app.set_timer(10.0, lambda: msg_widget.remove())
        except Exception:
            pass
            
        return None
