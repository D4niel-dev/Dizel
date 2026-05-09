from rich.markup import escape
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from inference.cmd_ui.commands.registry import Command, registry


class CommandPalette(Container):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._commands: dict[str, Command] = {}

    def compose(self) -> ComposeResult:
        yield Static("COMMANDS", id="palette-title")
        yield Input(placeholder="Search commands...", id="palette-input")
        yield OptionList(id="palette-options")

    def on_mount(self):
        self.display = False
        self.populate_options("")

    def populate_options(self, query: str) -> None:
        options_list = self.query_one("#palette-options", OptionList)
        options_list.clear_options()
        self._commands.clear()

        q = query.lower().strip()
        for command in registry.list_all():
            if not self._matches(command, q):
                continue
            option_id = f"cmd_{command.name}"
            self._commands[option_id] = command
            options_list.add_option(Option(self._format_command(command), id=option_id))

        if not self._commands:
            options_list.add_option(Option("  No matching commands", id="__empty__", disabled=True))
        else:
            options_list.highlighted = 0

    def _matches(self, command: Command, query: str) -> bool:
        if not query:
            return True
        parts = [
            command.name,
            command.help_text,
            command.usage,
            command.category,
            " ".join(command.aliases),
            " ".join(command.examples),
        ]
        return query in " ".join(parts).lower()

    def _format_command(self, command: Command) -> str:
        aliases = f" aliases: {', '.join(command.aliases)}" if command.aliases else ""
        usage = command.usage or f"/{command.name}"
        return (
            f"[#60A5FA]{escape(usage)}[/]  "
            f"[#F4F4F5]{escape(command.help_text)}[/]  "
            f"[dim]{escape(command.category + aliases)}[/]"
        )

    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self.populate_options(event.value)

    @on(Input.Submitted, "#palette-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        options = self.query_one("#palette-options", OptionList)
        if options.highlighted is None and self._commands:
            options.highlighted = 0
        options.focus()

    @on(OptionList.OptionSelected, "#palette-options")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        command = self._commands.get(event.option.id or "")
        if not command:
            return

        prompt = self.app.query_one("#prompt-input", Input)
        prompt.value = command.insert_text
        prompt.cursor_position = len(prompt.value)
        self.close()

    def toggle(self) -> None:
        if self.display:
            self.close()
        else:
            self.open()

    def open(self) -> None:
        self.display = True
        inp = self.query_one("#palette-input", Input)
        inp.value = ""
        self.populate_options("")
        inp.focus()

    def close(self) -> None:
        self.display = False
        self.query_one("#palette-input", Input).value = ""
        self.app.query_one("#prompt-input", Input).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.close()
