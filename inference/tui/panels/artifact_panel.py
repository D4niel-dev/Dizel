import re

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static

from inference.tui.theme import CODE_THEME

_CODE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


class ArtifactPanel(Container):
    """Code blocks collected from the transcript.

    Closed by default and opened with ctrl+a. It used to open itself on the
    first code block, which silently took 40 columns away from the answer the
    reader had just asked for.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._seen: set[tuple[str, str]] = set()

    def compose(self) -> ComposeResult:
        yield Static("ARTIFACTS", classes="panel-header")
        with VerticalScroll(id="artifact-container"):
            yield Static("No code yet.", id="artifact-empty")

    def add_artifacts(self, text: str) -> None:
        """Append any new code blocks. Repeats are ignored, order is kept."""
        container = self.query_one("#artifact-container", VerticalScroll)
        added = False
        for match in _CODE_RE.finditer(text):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            key = (language, code)
            if not code or key in self._seen:
                continue
            self._seen.add(key)
            added = True
            container.mount(
                Static(
                    Syntax(code, language, theme=CODE_THEME, word_wrap=True),
                    classes="artifact-block",
                )
            )
        if added:
            self.query_one("#artifact-empty", Static).display = False

    def clear(self) -> None:
        self._seen.clear()
        for block in list(self.query(".artifact-block")):
            block.remove()
        self.query_one("#artifact-empty", Static).display = True
