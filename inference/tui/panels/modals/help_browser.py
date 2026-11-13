"""Searchable command reference with a live detail pane."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from inference.tui.commands.registry import Command, registry
from inference.tui.panels.modals.common import NavInput, clip, plain_text

GLOBAL_KEYS = (
    ("ctrl+k", "Command palette"),
    ("ctrl+y", "Multi-line editor"),
    ("ctrl+t", "Sessions panel"),
    ("ctrl+r", "Context rail"),
    ("ctrl+a", "Artifacts panel"),
    ("ctrl+l", "Clear transcript"),
    ("esc", "Interrupt generation"),
    ("ctrl+c", "Interrupt / quit"),
)


class HelpBrowserModal(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card help-card"):
            with Horizontal(classes="modal-head"):
                yield Static("Help", classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static("Commands and global keys", classes="modal-sub")
            yield NavInput(placeholder="Search commands…", classes="modal-input")
            with Horizontal(classes="help-body"):
                yield OptionList(classes="modal-list help-list")
                with VerticalScroll(classes="help-detail-wrap"):
                    yield Static("", id="help-detail")
                    yield self._keys_block()
            yield Static(
                "↑↓ browse · enter insert into prompt · esc close",
                classes="modal-footer",
            )

    def _keys_block(self) -> Static:
        rows = []
        for index in range(0, len(GLOBAL_KEYS), 2):
            left = GLOBAL_KEYS[index]
            right = GLOBAL_KEYS[index + 1] if index + 1 < len(GLOBAL_KEYS) else ("", "")
            rows.append(f"{left[0]:<8} {left[1]:<17}  {right[0]:<8} {right[1]}")
        return Static("[b]Global keys[/]\n" + "\n".join(rows), classes="help-keys")

    def on_mount(self) -> None:
        self._commands: dict[str, Command] = {}
        self._last_query: str | None = None
        self._populate("")
        self.query_one(NavInput).focus()

    @property
    def _input(self) -> NavInput:
        return self.query_one(NavInput)

    @property
    def _list(self) -> OptionList:
        return self.query_one(OptionList)

    def _populate(self, query: str) -> None:
        options = self._list
        preserve = query == self._last_query
        previous = None
        if preserve and options.highlighted is not None and options.option_count:
            previous = options.get_option_at_index(options.highlighted).id
        self._last_query = query

        options.clear_options()
        self._commands.clear()

        q = query.lower().strip()
        matches: list[tuple[int, int, Command]] = []
        for index, command in enumerate(registry.list_all()):
            extras = " ".join([command.usage, " ".join(command.aliases), " ".join(command.examples)])
            haystack = f"{command.name} {command.help_text} {command.category} {extras}".lower()
            if q and q not in haystack:
                continue
            name = command.name.lower()
            if not q or name.startswith(q):
                score = 0
            elif q in name:
                score = 1
            elif q in command.help_text.lower():
                score = 2
            elif q in command.category.lower():
                score = 3
            else:
                score = 4
            matches.append((score, index, command))

        for _, _, command in sorted(
            matches,
            key=lambda entry: (entry[2].name == "exit", entry[0], entry[1]),
        ):
            self._commands[command.name] = command
            label = Content.assemble(
                (f"/{command.name}", "bold"),
                (f"  {clip(command.help_text, 23)}", ""),
            )
            options.add_option(Option(label, id=command.name))

        if not self._commands:
            options.add_option(
                Option(plain_text("  No matching commands"), id="__empty__", disabled=True)
            )
            self.query_one("#help-detail", Static).update(
                plain_text("No command matches that search.")
            )
            return

        target = 0
        if previous in self._commands:
            for index, name in enumerate(self._commands):
                if name == previous:
                    target = index
                    break
        options.highlighted = target
        options.scroll_to_highlight()
        command = self._commands.get(list(self._commands)[target])
        if command:
            self._show_detail(command)

    def _show_detail(self, command: Command) -> None:
        parts: list[tuple[str, str]] = [
            (f"/{command.name}", "bold"),
            (f"   {command.category}\n", ""),
            (command.help_text, ""),
        ]
        if command.usage:
            parts.extend([("\n\nUsage", "bold"), (f"\n  {command.usage}", "")])
        if command.aliases:
            aliases = ", ".join(f"/{alias}" for alias in command.aliases)
            parts.extend([("\n\nAliases", "bold"), (f"\n  {aliases}", "")])
        if command.examples:
            parts.append(("\n\nExamples", "bold"))
            for example in command.examples:
                parts.append((f"\n  {example}", ""))
        self.query_one("#help-detail", Static).update(Content.assemble(*parts))

    def _choose(self) -> None:
        options = self._list
        if options.highlighted is None:
            return
        option = options.get_option_at_index(options.highlighted)
        command = self._commands.get(option.id or "")
        if command:
            self.dismiss(command.insert_text)

    @on(NavInput.Navigate)
    def _navigate(self, event: NavInput.Navigate) -> None:
        event.stop()
        options = self._list
        if options.option_count == 0:
            return
        current = options.highlighted or 0
        index = max(0, min(options.option_count - 1, current + event.direction))
        options.highlighted = index
        options.scroll_to_highlight()
        option = options.get_option_at_index(index)
        command = self._commands.get(option.id or "")
        if command:
            self._show_detail(command)

    @on(NavInput.Submitted)
    def _submitted(self, event: NavInput.Submitted) -> None:
        event.stop()
        self._choose()

    @on(OptionList.OptionHighlighted)
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        command = self._commands.get(event.option.id or "")
        if command:
            self._show_detail(command)

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        command = self._commands.get(event.option.id or "")
        if command:
            self.dismiss(command.insert_text)

    @on(NavInput.Changed)
    def _changed(self, event: NavInput.Changed) -> None:
        self._populate(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)
