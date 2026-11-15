"""Explicit approval for a built-in file write or edit."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from dizel_core.tools import FileChange
from inference.tui.theme import (
    DIFF_ADD,
    DIFF_ADD_BG,
    DIFF_DEL,
    DIFF_DEL_BG,
    DIFF_HUNK,
    TEXT_2,
    TEXT_3,
    WARN,
)


class ToolApprovalModal(ModalScreen[bool]):
    """Review one file diff, then approve or reject the disk change."""

    BINDINGS = [
        Binding("escape", "reject", "Reject"),
        Binding("n", "reject", "Reject", show=False),
        Binding("y", "approve", "Approve", show=False),
    ]

    def __init__(self, change: FileChange, **kwargs) -> None:
        super().__init__(**kwargs)
        self.change = change

    def compose(self) -> ComposeResult:
        action = {
            "create": "Create file",
            "overwrite": "Overwrite file",
            "edit": "Edit file",
        }.get(self.change.action, "Write file")
        with Vertical(classes="modal-card approval-card"):
            with Horizontal(classes="modal-head"):
                yield Static(action, classes="modal-title")
                yield Static("y / n", classes="modal-kbd")
            yield Static(self.change.path, classes="approval-path")
            yield Static(
                f"[{DIFF_ADD}]+{self.change.additions}[/] "
                f"[{DIFF_DEL}]-{self.change.deletions}[/] "
                f"[{TEXT_3}]· {self.change.tool}[/]",
                classes="approval-stats",
            )
            with VerticalScroll(classes="approval-diff"):
                yield Static(self._diff_text(), classes="approval-diff-text")
            yield Static(
                "This changes the file on disk. Review the diff before approving.",
                classes="approval-warning",
            )
            with Horizontal(classes="modal-actions"):
                yield Button("Reject", id="approval-reject")
                yield Button("Approve", id="approval-ok", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#approval-reject", Button).focus()

    def _diff_text(self) -> Text:
        if not self.change.diff:
            return Text("(no textual diff — the file will be created or unchanged).", style=TEXT_3)
        text = Text()
        for line in self.change.diff.splitlines():
            if line.startswith(("--- ", "+++ ")):
                text.append(line + "\n", style=TEXT_3)
            elif line.startswith("@@"):
                text.append(line + "\n", style=f"bold {DIFF_HUNK}")
            elif line.startswith("+"):
                text.append(line + "\n", style=f"{DIFF_ADD} on {DIFF_ADD_BG}")
            elif line.startswith("-"):
                text.append(line + "\n", style=f"{DIFF_DEL} on {DIFF_DEL_BG}")
            else:
                text.append(line + "\n", style=TEXT_2)
        return text

    def action_reject(self) -> None:
        self.dismiss(False)

    def action_approve(self) -> None:
        self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "approval-ok")
