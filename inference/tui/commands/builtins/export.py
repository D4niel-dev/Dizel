import json
import os
from typing import Any

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


def _write_export(app: Any, fmt: str, filename: str) -> str:
    messages = app.session_bridge.current_messages
    if fmt not in ("json", "md", "txt"):
        return f"Unsupported format: {fmt}\nAvailable: json, md, txt"

    if not os.path.splitext(filename)[1]:
        filename = f"{filename}.{fmt}"

    directory = os.path.dirname(os.path.abspath(filename))
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        with open(filename, "w", encoding="utf-8") as handle:
            if fmt == "json":
                json.dump(messages, handle, indent=2, ensure_ascii=False)
            else:
                for message in messages:
                    handle.write(f"[{message['role'].upper()}]\n{message['content']}\n\n")
    except OSError as error:
        return f"Export failed: {error}"

    return f"Exported to {os.path.abspath(filename)}"


class ExportCommand(Command):
    name = "export"
    help_text = "Export the current transcript as Markdown, JSON or text."
    category = "Session"
    usage = "/export [json|txt|md] [path]"
    palette_hint = "/export "
    examples = ["/export", "/export md", "/export json exports/session.json"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not app.session_bridge.current_messages:
            return "No messages to export."

        if not invocation.args:
            from inference.tui.panels.modals import ExportFormatModal

            def _chosen(fmt: str | None) -> None:
                if not fmt:
                    return
                default = f"export_{app.session_id or 'session'}.{fmt}"
                app._log_system(_write_export(app, fmt, default))

            app.push_screen(
                ExportFormatModal(len(app.session_bridge.current_messages)),
                _chosen,
            )
            return ""

        fmt = invocation.args[0].lower()
        filename = (
            " ".join(invocation.args[1:])
            if len(invocation.args) > 1
            else f"export_{app.session_id or 'session'}.{fmt}"
        )
        return _write_export(app, fmt, filename)
