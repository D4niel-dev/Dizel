import glob
import os
from typing import Any

from inference.tui.commands.registry import Command
from inference.tui.commands.parser import CommandInvocation
from inference.tui.runtime_actions import project_root

_CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.gguf", "*.safetensors", "*.bin", "*.ckpt")


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    root = project_root()
    candidate = root / path
    if candidate.exists():
        return str(candidate)
    alt = root / "checkpoints" / path
    if alt.exists():
        return str(alt)
    return str(candidate)


def _available() -> list[str]:
    directory = project_root() / "checkpoints"
    if not directory.is_dir():
        return []
    names: list[str] = []
    for pattern in _CHECKPOINT_PATTERNS:
        names.extend(os.path.basename(p) for p in glob.glob(str(directory / pattern)))
    return sorted(set(names))


class LoadCommand(Command):
    name = "load"
    help_text = "Pick a checkpoint, or load one by path."
    category = "Runtime"
    usage = "/load [path|auto]"
    palette_hint = "/load "
    examples = ["/load", "/load auto", "/load checkpoints/model.pt"]

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args:
            from inference.tui.panels.modals import CheckpointPickerModal

            def _picked(path: str | None) -> None:
                if not path:
                    return
                resolved = _resolve(path)
                if not os.path.exists(resolved):
                    app._log_system(f"Checkpoint not found: {path}")
                    return
                app._load_model_async(resolved, app._device)

            app.push_screen(CheckpointPickerModal(), _picked)
            return ""

        if invocation.args[0] == "auto":
            checkpoint = app._discover_checkpoint()
            if not checkpoint:
                return "No checkpoint found in checkpoints/ directory."
            if not app._load_model_async(checkpoint, app._device):
                return ""
            return f"Auto-loading: {os.path.basename(checkpoint)}…"

        path = _resolve(" ".join(invocation.args))
        if not os.path.exists(path):
            available = _available()
            if available:
                listing = "\n".join(f"  {name}" for name in available[:20])
                return f"File not found: {path}\n\nAvailable checkpoints:\n{listing}"
            return f"File not found: {path}"

        if not app._load_model_async(path, app._device):
            return ""
        return f"Loading: {os.path.basename(path)}…"
