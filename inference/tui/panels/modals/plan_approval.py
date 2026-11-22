"""Two-outcome approval for a model plan: execute the steps, or discard them."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from dizel_core.tools import PlanChange


class PlanApprovalModal(ModalScreen[bool]):
    """Review a plan; there is no standing approval for plans."""

    BINDINGS = [
        Binding("escape", "discard", "Discard"),
        Binding("n", "discard", "Discard", show=False),
        Binding("y", "execute", "Execute", show=False),
    ]

    def __init__(self, change: PlanChange, **kwargs) -> None:
        super().__init__(**kwargs)
        self.change = change

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card plan-card"):
            with Horizontal(classes="modal-head"):
                yield Static("Plan approval", classes="modal-title")
                yield Static("y / n", classes="modal-kbd")
            if self.change.task:
                yield Static(self.change.task, classes="plan-task")
            with VerticalScroll(classes="plan-steps"):
                for index, step in enumerate(self.change.steps, start=1):
                    yield Static(f"{index}. {step}", classes="plan-step")
            yield Static(
                "The model runs these steps with its tools; every write still "
                "needs approval.",
                classes="modal-sub",
            )
            with Horizontal(classes="modal-actions"):
                yield Button("Discard", id="plan-discard")
                yield Button("Execute plan", id="plan-execute", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#plan-discard", Button).focus()

    def action_discard(self) -> None:
        self.dismiss(False)

    def action_execute(self) -> None:
        self.dismiss(True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "plan-execute")
