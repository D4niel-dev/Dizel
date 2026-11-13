from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command
from inference.tui.runtime_actions import apply_model, load_checkpoint


class ModelCommand(Command):
    name = "model"
    help_text = "Pick or switch models. Works for local and API providers."
    category = "Runtime"
    usage = "/model [name]"
    palette_hint = "/model "
    examples = ["/model", "/model Dizel 1.5 Distill", "/model gpt-4o"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        local = app.active_provider == "local"

        if not invocation.args:
            if local:
                from inference.tui.panels.modals import CheckpointPickerModal

                def _picked(path: str | None) -> None:
                    if not path:
                        return
                    try:
                        app._log_system(load_checkpoint(app, path))
                    except ValueError as error:
                        app._log_system(str(error), "error")

                app.push_screen(CheckpointPickerModal(), _picked)
                return ""

            from inference.tui.panels.modals import ModelPickerModal

            def _picked_model(name: str | None) -> None:
                if not name:
                    return
                try:
                    app._log_system(apply_model(app, name))
                except Exception as error:
                    app._log_system(f"Model error: {error}", "error")

            app.push_screen(ModelPickerModal(), _picked_model)
            return ""

        try:
            return apply_model(app, " ".join(invocation.args))
        except ValueError as error:
            return str(error)
