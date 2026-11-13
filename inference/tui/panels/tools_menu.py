from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from inference.tui.commands.builtins.tools import AVAILABLE_TOOLS


class ToolsMenuModal(ModalScreen[None]):
    """Toggle the tools that run before a prompt reaches the model."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("space", "toggle", "Toggle", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card -compact"):
            with Horizontal(classes="modal-head"):
                yield Static("Tools", classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static("", id="tools-sub", classes="modal-sub")
            yield OptionList(classes="modal-list", id="tools-list")
            yield Static(
                "space / enter toggle · esc close",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        if not hasattr(self.app, "_tool_states"):
            self.app._tool_states = {name: False for name in AVAILABLE_TOOLS}
        self._populate()
        self.query_one("#tools-list", OptionList).focus()

    def _populate(self, keep: str = "") -> None:
        if not keep:
            options = self.query_one("#tools-list", OptionList)
            if options.highlighted is not None and options.option_count:
                keep = getattr(options.get_option_at_index(options.highlighted), "tool_name", "")
        enabled = sum(1 for state in self.app._tool_states.values() if state)
        total = len(AVAILABLE_TOOLS)
        self.query_one("#tools-sub", Static).update(
            f"{enabled} of {total} enabled · applied to the next message only"
        )

        option_list = self.query_one("#tools-list", OptionList)
        option_list.clear_options()
        target = 0
        for index, (name, description) in enumerate(AVAILABLE_TOOLS.items()):
            state = self.app._tool_states.get(name, False)
            mark = "●" if state else "○"
            label = name.replace("_", " ").title()
            option = Option(
                Content.assemble(
                    (f"{mark} ", ""),
                    (label, "bold"),
                    (f"\n  {description}", ""),
                ),
                id=name,
            )
            option_list.add_option(option)
            option.tool_name = name  # type: ignore[attr-defined]
            if name == keep:
                target = index
        if option_list.option_count:
            option_list.highlighted = target
            option_list.scroll_to_highlight()

    def _toggle(self, name: str) -> None:
        self.app._tool_states[name] = not self.app._tool_states.get(name, False)
        self.app.refresh_context()
        self.call_after_refresh(self._populate, name)

    def action_toggle(self) -> None:
        options = self.query_one("#tools-list", OptionList)
        if options.highlighted is None:
            return
        option = options.get_option_at_index(options.highlighted)
        name = getattr(option, "tool_name", None) or option.id
        if name in AVAILABLE_TOOLS:
            self._toggle(name)

    @on(OptionList.OptionSelected, "#tools-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        name = getattr(event.option, "tool_name", None) or event.option.id
        if name in AVAILABLE_TOOLS:
            self._toggle(name)

    def action_close(self) -> None:
        self.dismiss(None)
