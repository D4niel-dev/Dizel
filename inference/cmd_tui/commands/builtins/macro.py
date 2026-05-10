from typing import Any
from inference.cmd_tui.commands.registry import Command
from inference.cmd_tui.commands.parser import CommandInvocation, parse_command
from inference.dizel_gui.logic.config_manager import ConfigManager


class MacroCommand(Command):
    name = "macro"
    help_text = "Define or run command sequences. Usage: /macro [define|run|list|delete] [name] [commands...]"
    category = "Utility"
    usage = "/macro [define|run|list|delete] [name] [commands...]"
    palette_hint = "/macro "
    examples = [
        "/macro define audit /mode Review | /tag audit",
        "/macro run audit",
        "/macro list"
    ]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            return self._render_list()
            
        action = invocation.args[0].lower()
        
        if action == "list":
            return self._render_list()
            
        if action == "define":
            if len(invocation.args) < 3:
                return "**Error:** Usage: `/macro define <name> <cmd1> | <cmd2> ...`"
            name = invocation.args[1]
            # Reconstruct the raw arguments as a single string, then split by `|`
            raw_cmds = " ".join(invocation.args[2:])
            cmds = [c.strip() for c in raw_cmds.split("|") if c.strip()]
            
            cfg = ConfigManager.load()
            macros = cfg.get("macros", {})
            macros[name] = cmds
            cfg["macros"] = macros
            ConfigManager.save(cfg)
            
            return f"**Saved macro:** `{name}` -> `{cmds}`"
            
        if action == "delete":
            if len(invocation.args) < 2:
                return "**Error:** Usage: `/macro delete <name>`"
            name = invocation.args[1]
            cfg = ConfigManager.load()
            macros = cfg.get("macros", {})
            if name in macros:
                del macros[name]
                cfg["macros"] = macros
                ConfigManager.save(cfg)
                return f"**Deleted macro:** `{name}`"
            return f"**Error:** Macro `{name}` not found."
            
        if action == "run":
            if len(invocation.args) < 2:
                return "**Error:** Usage: `/macro run <name>`"
            name = invocation.args[1]
            cfg = ConfigManager.load()
            macros = cfg.get("macros", {})
            
            if name not in macros:
                return f"**Error:** Macro `{name}` not found."
                
            cmds = macros[name]
            app._log_system(f"Running macro `{name}` with {len(cmds)} commands...")
            
            # Execute them sequentially
            from inference.cmd_tui.commands.registry import registry
            workspace = app.query_one("WorkspacePanel")
            from textual.widgets import Static
            from inference.cmd_tui.rendering.message_block import MessageBlock
            
            for cmd_str in cmds:
                if not cmd_str.startswith("/"):
                    cmd_str = "/" + cmd_str
                invoc = parse_command(cmd_str)
                cmd_obj = registry.lookup(invoc.name)
                if cmd_obj:
                    try:
                        result = await cmd_obj.execute(app, invoc)
                        if result:
                            workspace.mount(Static(MessageBlock("SYSTEM", result)))
                    except Exception as e:
                        workspace.mount(Static(MessageBlock("SYSTEM", f"Macro Error in `{cmd_str}`: {e}")))
                else:
                    workspace.mount(Static(MessageBlock("SYSTEM", f"Macro Error: Unknown command `{cmd_str}`")))
                    
            workspace.scroll_end(animate=False)
            return None
            
        return f"**Error:** Unknown action `{action}`. Use define, run, list, or delete."

    def _render_list(self) -> str:
        cfg = ConfigManager.load()
        macros = cfg.get("macros", {})
        if not macros:
            return "**Macros:** None defined. Create one with `/macro define <name> <cmd1> | <cmd2>`"
            
        lines = ["### Saved Macros"]
        for name, cmds in macros.items():
            lines.append(f"- **`{name}`**: " + " `|` ".join(f"`{c}`" for c in cmds))
        lines.append("")
        lines.append("**Usage:** `/macro run <name>`")
        return "\n".join(lines)
