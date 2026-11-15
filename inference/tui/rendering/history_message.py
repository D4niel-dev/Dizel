from rich.markdown import Markdown

from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from inference.tui.rendering.message_block import (
    MessageBlock,
    _parse_think_blocks,
    message_style,
    reasoning_summary,
)
from inference.tui.theme import CODE_THEME, TEXT_3


class HistoryMessage(Vertical):
    """A reloaded assistant turn, with its reasoning folded like a live one."""

    def __init__(self, content: str, ts: float | None = None):
        super().__init__(classes="assistant-msg")
        self._content = content
        self._ts = ts

    def compose(self):
        main_text, thought_text = _parse_think_blocks(self._content)
        if thought_text:
            title, thought_body = reasoning_summary(thought_text)
            label = (
                f"Thought: {title}"
                if title
                else f"Thought · {len(thought_text):,} chars"
            )
            body = Static(
                Markdown(thought_body, code_theme=CODE_THEME, style=TEXT_3),
                id="hist-think-body",
            )
            yield Collapsible(
                body,
                title=label,
                collapsed=True,
                collapsed_symbol="+",
                expanded_symbol="-",
                classes="asm-collapsible",
            )
        if main_text:
            yield Static(
                MessageBlock(
                    "ASSISTANT",
                    main_text,
                    **message_style(self.app, timestamp=self._ts),
                )
            )