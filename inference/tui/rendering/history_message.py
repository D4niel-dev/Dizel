from rich.markdown import Markdown

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from inference.tui.rendering.message_block import (
    MessageBlock,
    _parse_think_blocks,
    message_style,
)
from inference.tui.theme import CODE_THEME


class HistoryMessage(Vertical):
    """A reloaded assistant turn, with its reasoning folded like a live one."""

    def __init__(self, content: str, ts: float | None = None):
        super().__init__(classes="assistant-msg")
        self._content = content
        self._ts = ts

    def compose(self):
        main_text, thought_text = _parse_think_blocks(self._content)
        if thought_text:
            body = Static(Markdown(thought_text, code_theme=CODE_THEME), id="hist-think-body")
            yield Collapsible(
                body,
                title=f"Thought · {len(thought_text):,} chars",
                collapsed=True,
                classes="asm-collapsible",
            )
        yield Static(
            MessageBlock(
                "ASSISTANT",
                main_text or "…",
                **message_style(self.app, timestamp=self._ts),
            )
        )