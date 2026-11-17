"""Explicit approval for a built-in file write, edit, or delete."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from dizel_core import tool_trust
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

_BUTTON_DECISIONS = {
    "approval-reject": tool_trust.DENY,
    "approval-ok": tool_trust.ALLOW_ONCE,
    "approval-once": tool_trust.ALLOW_ONCE,
    "approval-session": tool_trust.ALWAYS_SESSION,
    "approval-persistent": tool_trust.ALWAYS_PERSISTENT,
}


class ToolApprovalModal(ModalScreen[str]):
    """Review one file diff, then choose how far the approval reaches."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("n", "reject", "Reject", show=False),
        Binding("y", "allow_once", "Allow once", show=False),
        Binding("s", "allow_session", "Always this session", show=False),
        Binding("a", "allow_persistent", "Always allow", show=False),
    ]

    def __init__(self, change: FileChange, **kwargs) -> None:
        super().__init__(**kwargs)
        self.change = change
        self._armed: str | None = None

    def compose(self) -> ComposeResult:
        action = {
            "create": "Create file",
            "overwrite": "Overwrite file",
            "edit": "Edit file",
            "delete": "Delete file",
        }.get(self.change.action, "Write file")
        # Delete is always a per-call decision: this gate never offers a standing
        # "always allow", and no trust tier may auto-approve it.
        is_delete = self.change.action == "delete"
        warning = (
            "This moves the file to the recycle bin. Every deletion needs your "
            "explicit approval — it can never be auto-approved."
            if is_delete
            else "This changes the file on disk. Review the diff before approving."
        )
        with Vertical(classes="modal-card approval-card"):
            with Horizontal(classes="modal-head"):
                yield Static(
                    Text(action, style=DIFF_DEL if is_delete else ""),
                    classes="modal-title",
                )
                yield Static("y / n" if is_delete else "y / s / a / n", classes="modal-kbd")
            yield Static(Text(self.change.path), classes="approval-path")
            yield Static(
                f"[{DIFF_ADD}]+{self.change.additions}[/] "
                f"[{DIFF_DEL}]-{self.change.deletions}[/] "
                f"[{TEXT_3}]· {self.change.tool}[/]",
                classes="approval-stats",
            )
            with VerticalScroll(classes="approval-diff"):
                yield Static(self._diff_text(), classes="approval-diff-text")
            yield Static(
                Text(warning, style=DIFF_DEL if is_delete else ""),
                classes="approval-warning",
            )
            if not is_delete:
                yield Static(
                    Text(
                        '"Always allow" stays on until you revoke it in Settings.',
                        style=TEXT_3,
                    ),
                    classes="approval-trust-hint",
                )
            yield Static("", id="approval-confirm", classes="approval-confirm")
            with Horizontal(classes="modal-actions"):
                yield Button("Reject", id="approval-reject")
                if is_delete:
                    yield Button("Approve", id="approval-ok", variant="primary")
                else:
                    yield Button("Allow once", id="approval-once", variant="primary")
                    yield Button("Always this session", id="approval-session")
                    yield Button("Always allow", id="approval-persistent")

    def on_mount(self) -> None:
        self.query_one("#approval-confirm", Static).display = False
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
        self.dismiss(tool_trust.DENY)

    def action_back(self) -> None:
        if self._armed:
            self._disarm()
            return
        self.dismiss(tool_trust.DENY)

    def action_allow_once(self) -> None:
        self.dismiss(tool_trust.ALLOW_ONCE)

    def action_allow_session(self) -> None:
        self._arm_or_confirm(tool_trust.ALWAYS_SESSION, "s")

    def action_allow_persistent(self) -> None:
        self._arm_or_confirm(tool_trust.ALWAYS_PERSISTENT, "a")

    def _disarm(self) -> None:
        self._armed = None
        confirm = self.query_one("#approval-confirm", Static)
        confirm.update("")
        confirm.display = False

    def _arm_or_confirm(self, decision: str, key: str) -> None:
        if self.change.action == "delete":
            return
        if self._armed == decision:
            self.dismiss(decision)
            return
        self._armed = decision
        confirm = self.query_one("#approval-confirm", Static)
        if decision == tool_trust.ALWAYS_PERSISTENT:
            copy = (
                "Every write in every project will apply without asking until you "
                f"revoke it in Settings → Chat → File writes. Press {key} again to "
                "confirm — esc goes back."
            )
            style = WARN
        else:
            copy = (
                "Every write in this chat will apply without asking until you close "
                f"it. Press {key} again to confirm — esc goes back."
            )
            style = TEXT_2
        confirm.update(Text(copy, style=style))
        confirm.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        decision = _BUTTON_DECISIONS[event.button.id]
        if decision in (tool_trust.ALWAYS_SESSION, tool_trust.ALWAYS_PERSISTENT):
            key = "s" if decision == tool_trust.ALWAYS_SESSION else "a"
            self._arm_or_confirm(decision, key)
            return
        self.dismiss(decision)
