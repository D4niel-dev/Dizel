import re
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static
from rich.syntax import Syntax


class ArtifactPanel(Container):
    """A slide-out panel to view code blocks generated in chat."""
    
    def compose(self) -> ComposeResult:
        yield Static("CODE ARTIFACTS", classes="panel-header")
        with VerticalScroll(id="artifact-container"):
            yield Static(id="artifact-content")

    def toggle_panel(self) -> None:
        if self.has_class("-visible"):
            self.remove_class("-visible")
        else:
            self.add_class("-visible")

    def update_artifacts(self, text: str) -> None:
        """Parse text for code blocks and update the panel."""
        blocks = []
        pattern = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
        for match in pattern.finditer(text):
            lang = match.group(1) or "text"
            code = match.group(2).strip()
            blocks.append((lang, code))
            
        if not blocks:
            return
            
        content = []
        for lang, code in blocks:
            content.append(Syntax(code, lexer=lang, theme="monokai", word_wrap=True))
            
        # Update UI
        from rich.console import Group
        self.query_one("#artifact-content").update(Group(*content))
        if not self.has_class("-visible"):
            self.add_class("-visible")
