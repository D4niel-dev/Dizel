from typing import Any
import json
import os
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation

class ExportCommand(Command):
    name = "export"
    help_text = "Export current session. Usage: /export [json|txt|md]"
    category = "Session"
    usage = "/export [json|txt|md] [path]"
    palette_hint = "/export "
    examples = ["/export md", "/export json exports/session.json"]
    
    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not app.session_bridge.current_messages:
            return "No messages to export."
            
        fmt = invocation.args[0].lower() if invocation.args else "json"
        if fmt not in ("json", "md", "txt"):
            return f"Unsupported format: {fmt}\nAvailable: json, md, txt"

        filename = (
            " ".join(invocation.args[1:])
            if len(invocation.args) > 1
            else f"export_{app.session_id or 'session'}.{fmt}"
        )
        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.{fmt}"
        
        directory = os.path.dirname(os.path.abspath(filename))
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            if fmt == "json":
                json.dump(app.session_bridge.current_messages, f, indent=2)
            elif fmt in ("md", "txt"):
                for m in app.session_bridge.current_messages:
                    f.write(f"[{m['role'].upper()}]\n{m['content']}\n\n")
                
        return f"Exported to {os.path.abspath(filename)}"
