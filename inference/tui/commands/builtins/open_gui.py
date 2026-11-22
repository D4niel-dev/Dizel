import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from inference.tui.commands.parser import CommandInvocation
from inference.tui.commands.registry import Command


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _open_protocol(session_id: str) -> bool:
    """Hand the id to the OS handler. Returns False when there is no handler."""
    url = f"dizel://session/{session_id}"
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        if sys.platform == "darwin":
            subprocess.Popen(["open", url])
            return True
        subprocess.Popen(["xdg-open", url])
        return True
    except Exception:
        return False


def _open_dev_session(session_id: str) -> Optional[str]:
    """Fallback for a checkout where the protocol is not registered yet."""
    gui = _repo_root() / "inference" / "gui"
    if not (gui / "node_modules").is_dir():
        return "node_modules is missing — run npm install in inference/gui."
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        return "npx was not found on PATH."

    env = dict(os.environ)
    env.setdefault("DIZEL_PROD", "1")
    command = [npx, "electron", ".", f"--session={session_id}"]
    try:
        if sys.platform.startswith("win"):
            # A console window would steal focus from the terminal; electron is a
            # GUI process, so hide the shell that launched it.
            subprocess.Popen(
                command,
                cwd=str(gui),
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            subprocess.Popen(command, cwd=str(gui), env=env)
        return None
    except Exception as error:
        return str(error)


class OpenInGuiCommand(Command):
    name = "open"
    help_text = "Open this conversation in the desktop app."
    category = "Session"
    usage = "/open"
    examples = ["/open"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        session_id = str(getattr(app, "session_id", "") or "")
        if not session_id:
            return "This chat has no saved id yet — send a message first."

        try:
            from dizel_core.history_manager import load_session

            if not load_session(session_id):
                return (
                    "This chat has not been saved yet. Send a message, then /open."
                )
        except Exception:
            pass

        if _open_protocol(session_id):
            return f"Opening {session_id} in the desktop app."

        error = _open_dev_session(session_id)
        if error:
            return f"Could not open the desktop app: {error}"
        return (
            f"Opening {session_id} in the desktop app from this checkout. "
            "Register the dizel:// protocol in a packaged install to skip the launcher."
        )
