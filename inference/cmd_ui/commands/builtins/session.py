from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class SessionCommand(Command):
    name = "session"
    help_text = "Manage sessions. Usage: /session new|rename|pin|delete [args]"
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            return f"Current session: {app.session_id or 'None'}"
            
        action = invocation.args[0]
        
        if action == "new":
            app.session_bridge.create()
            app.query_one("WorkspacePanel").clear_workspace()
            return None
            
        elif action == "rename":
            title = " ".join(invocation.args[1:])
            if not title:
                return "Please provide a title."
            app.session_bridge.save(title=title)
            return f"Session renamed to: {title}"
            
        elif action == "pin":
            if not app.session_id:
                return "No active session to pin."
            app.session_bridge.pin(app.session_id)
            return "Toggled pin status."
            
        elif action == "delete":
            target = invocation.args[1] if len(invocation.args) > 1 else app.session_id
            if target:
                app.session_bridge.delete(target)
                if target == app.session_id:
                    app.query_one("WorkspacePanel").clear_workspace()
                    return None
                return f"Deleted session {target}"
            return "No session specified."
            
        return f"Unknown action: {action}"
