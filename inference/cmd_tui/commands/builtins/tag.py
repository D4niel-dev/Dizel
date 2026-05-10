from textual.app import App
from inference.cmd_tui.commands.registry import Command
from inference.dizel_gui.logic.history_manager import add_tag_to_session

class TagCommand(Command):
    name = "tag"
    help_text = "Add a tag (e.g. #work) to the current session. Usage: /tag <tag_name>"
    usage = "/tag <tag_name>"

    async def execute(self, app: App, invocation) -> str:
        if not invocation.args:
            return "Usage: /tag <tag_name>"
            
        tag = invocation.args[0]
        if not app.session_id:
            return "No active session to tag. Generate a message first to start a session."
            
        success = add_tag_to_session(app.session_id, tag)
        if success:
            app.session_bridge.refresh()
            return f"Added tag #{tag} to current session."
        else:
            return "Failed to add tag to session."
