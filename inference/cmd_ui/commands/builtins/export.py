from typing import Any
import json
import os
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ExportCommand(Command):
    name = "export"
    help_text = "Export current session. Usage: /export [json|txt|md]"
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not app.session_bridge.current_messages:
            return "No messages to export."
            
        fmt = invocation.args[0].lower() if invocation.args else "json"
        filename = f"export_{app.session_id}.{fmt}"
        
        with open(filename, "w", encoding="utf-8") as f:
            if fmt == "json":
                json.dump(app.session_bridge.current_messages, f, indent=2)
            elif fmt in ("md", "txt"):
                for m in app.session_bridge.current_messages:
                    f.write(f"[{m['role'].upper()}]\n{m['content']}\n\n")
            else:
                return f"Unsupported format: {fmt}"
                
        return f"Exported to {os.path.abspath(filename)}"
