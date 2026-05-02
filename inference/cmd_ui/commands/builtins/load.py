import os
import glob
from typing import Any
from inference.cmd_ui.commands.registry import Command
from inference.cmd_ui.commands.parser import CommandInvocation


class LoadCommand(Command):
    name = "load"
    help_text = "Load a model checkpoint. Usage: /load [path|auto]"

    async def execute(self, app: Any, invocation: CommandInvocation) -> str:
        if not invocation.args or invocation.args[0] == "auto":
            # Auto-discover
            checkpoint = app._discover_checkpoint()
            if not checkpoint:
                return "No checkpoint found in checkpoints/ directory."
            app._load_model_async(checkpoint, app._device)
            return f"Auto-loading: {os.path.basename(checkpoint)}..."

        path = " ".join(invocation.args)

        # Resolve relative paths
        if not os.path.isabs(path):
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(proj_root, path)

        if not os.path.exists(path):
            # Try checkpoints/ subdir
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            alt = os.path.join(proj_root, "checkpoints", path)
            if os.path.exists(alt):
                path = alt
            else:
                # List available checkpoints for help
                ckpt_dir = os.path.join(proj_root, "checkpoints")
                if os.path.isdir(ckpt_dir):
                    pts = [os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, "*.pt"))]
                    if pts:
                        return f"File not found: {path}\n\nAvailable checkpoints:\n" + "\n".join(f"  {p}" for p in pts)
                return f"File not found: {path}"

        app._load_model_async(path, app._device)
        return f"Loading: {os.path.basename(path)}..."
