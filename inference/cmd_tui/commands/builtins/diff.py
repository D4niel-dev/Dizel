import os
import difflib
from textual.app import App
from inference.cmd_tui.commands.registry import Command
from rich.syntax import Syntax

class DiffCommand(Command):
    name = "diff"
    help_text = "Compare two files and show syntax-highlighted diff. Usage: /diff <file1> <file2>"
    usage = "/diff <file1> <file2>"

    async def execute(self, app: App, invocation):
        if len(invocation.args) != 2:
            return "Usage: /diff <file1> <file2>"
            
        file1, file2 = invocation.args
        
        if not os.path.exists(file1):
            return f"File not found: {file1}"
        if not os.path.exists(file2):
            return f"File not found: {file2}"
            
        with open(file1, "r", encoding="utf-8") as f:
            lines1 = f.readlines()
        with open(file2, "r", encoding="utf-8") as f:
            lines2 = f.readlines()
            
        diff_lines = list(difflib.unified_diff(
            lines1, lines2, 
            fromfile=file1, tofile=file2,
            n=3
        ))
        
        if not diff_lines:
            return f"No differences between {file1} and {file2}"
            
        diff_text = "".join(diff_lines)
        return Syntax(diff_text, "diff", theme="monokai", word_wrap=True, background_color="default")
