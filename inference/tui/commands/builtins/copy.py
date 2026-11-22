from typing import Any, Dict, List

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


def _replies(messages: List[Dict]) -> List[str]:
    """Assistant turns only, oldest first. Empty replies are not copied."""
    return [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "assistant" and str(message.get("content") or "").strip()
    ]


class CopyCommand(Command):
    name = "copy"
    help_text = "Copy the last reply (or the whole transcript) to the clipboard."
    category = "Session"
    usage = "/copy [n|all]"
    palette_hint = "/copy "
    examples = ["/copy", "/copy 2", "/copy all"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        bridge = getattr(app, "session_bridge", None)
        messages = list(getattr(bridge, "current_messages", []) or [])
        target = (invocation.args[0].lower() if invocation.args else "1")

        if target in ("all", "transcript"):
            body = "\n\n".join(
                f"{'You' if message.get('role') == 'user' else 'Dizel'}: "
                f"{message.get('content') or ''}"
                for message in messages
                if str(message.get("content") or "").strip()
            )
            label = "transcript"
        else:
            replies = _replies(messages)
            if not replies:
                return "Nothing to copy yet — no replies in this session."
            try:
                index = int(target)
            except ValueError:
                return f"Unknown target: {target}\nUsage: {self.usage}"
            if index < 1 or index > len(replies):
                return (
                    f"No reply {index}. This session has {len(replies)} "
                    f"{'reply' if len(replies) == 1 else 'replies'}."
                )
            body = replies[-index]
            label = "reply" if len(replies) == 1 else f"reply {index} of {len(replies)}"

        if not body.strip():
            return "That turn is empty — nothing was copied."

        app.copy_to_clipboard(body)
        # OSC 52 is the only clipboard a terminal has. Most terminals honour it,
        # a few (macOS Terminal.app, some tmux builds) silently drop it, so the
        # copy claims what it did rather than what the user will see.
        return (
            f"Copied {label} · {len(body):,} characters. "
            "Sent with OSC 52 — if nothing lands in your clipboard, this terminal "
            "blocks it."
        )
