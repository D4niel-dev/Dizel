from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation
from inference.tui.constants import FAST_SPEEDS, MODES, THINKING_EFFORTS


class ModeCommand(Command):
    name = "mode"
    help_text = "Pick the response mode (Fast, Planning, Coding, Thinking)."
    category = "Runtime"
    usage = "/mode [fast|planning|coding|thinking [low|medium|high|xhigh|deep]]"
    palette_hint = "/mode "
    examples = ["/mode", "/mode coding", "/mode thinking high", "/mode deep"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            from inference.tui.panels.modals import ModePickerModal

            def picked(mode: str | None) -> None:
                if not mode:
                    return
                if mode == "Thinking":
                    app.set_mode("Thinking")
                    self._open_efforts(app)
                else:
                    app.set_mode(mode)

            app.push_screen(ModePickerModal(), picked)
            return ""

        words = (
            " ".join(invocation.args).strip().lower().replace("-", " ").split()
        )
        if not words:
            return f"Available: {', '.join(MODES)}"
        if words in (["deep"], ["deep", "think"]):
            app.set_mode("Thinking", "deep")
            return ""
        requested = words[0]
        modes = {mode.lower(): mode for mode in MODES}
        mode = modes.get(requested)
        if not mode:
            return (
                f"Unknown mode: {' '.join(words)}\nAvailable: {', '.join(MODES)}"
            )
        effort = None
        speed = None
        if len(words) > 1:
            if mode == "Fast":
                requested_speed = " ".join(words[1:]).replace("-", "").replace(" ", "")
                if requested_speed not in FAST_SPEEDS:
                    return (
                        f"Unknown speed: {requested_speed}\n"
                        f"Available: {', '.join(FAST_SPEEDS)}"
                    )
                speed = requested_speed
            elif mode != "Thinking":
                return f"{mode} does not take an effort or speed."
            else:
                effort = " ".join(words[1:]).replace(" ", "")
            if effort not in THINKING_EFFORTS:
                return (
                    f"Unknown effort: {effort}\n"
                    f"Available: {', '.join(THINKING_EFFORTS)}"
                )
        if mode == "Thinking" and effort is None:
            app.set_mode("Thinking")
            self._open_efforts(app)
            return ""
        app.set_mode(mode, effort)
        if speed is not None:
            app.set_fast_speed(speed, announce=False)
        return ""

    def _open_efforts(self, app: Any) -> None:
        from inference.tui.panels.modals import ThinkingEffortPickerModal

        def picked(effort: str | None) -> None:
            if effort:
                app.set_thinking_effort(effort)

        app.push_screen(ThinkingEffortPickerModal(), picked)
