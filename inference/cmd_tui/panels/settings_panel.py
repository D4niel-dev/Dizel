from rich.markup import escape
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class SettingsPanel(Container):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._commands: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("SETTINGS", id="settings-title")
        yield Input(placeholder="Search settings commands...", id="settings-input")
        yield OptionList(id="settings-options")

    def on_mount(self):
        self.display = False
        self.populate_options("")

    def open(self) -> None:
        self.display = True
        inp = self.query_one("#settings-input", Input)
        inp.value = ""
        self.populate_options("")
        inp.focus()

    def close(self) -> None:
        self.display = False
        self.query_one("#settings-input", Input).value = ""
        self.app.query_one("#prompt-input", Input).focus()

    def populate_options(self, query: str) -> None:
        option_list = self.query_one("#settings-options", OptionList)
        option_list.clear_options()
        self._commands.clear()

        q = query.lower()
        for index, item in enumerate(self._build_items()):
            haystack = f"{item['command']} {item['label']} {item['detail']}".lower()
            if q and q not in haystack:
                continue
            option_id = f"setting_{index}"
            self._commands[option_id] = item["command"]
            label = self._format_option(item["command"], item["label"], item["detail"])
            option_list.add_option(Option(label, id=option_id))

        if not self._commands:
            option_list.add_option(Option("  No matching settings commands", id="__empty__", disabled=True))
        else:
            option_list.highlighted = 0

    def _build_items(self) -> list[dict[str, str]]:
        from inference.dizel_gui.logic.config_manager import ConfigManager

        cfg = ConfigManager.load()
        items = [
            {
                "command": "/settings runtime",
                "label": "Runtime",
                "detail": "provider, model, mode, device, checkpoint",
            },
            {
                "command": "/settings sampling",
                "label": "Sampling",
                "detail": "temperature, top-k, top-p, repetition, max tokens",
            },
            {
                "command": "/settings token_budget",
                "label": "Token Budget",
                "detail": "chat, coding, complex, hard output limits",
            },
            {
                "command": "/settings nova",
                "label": "Nova Voice",
                "detail": "voice model, language, silence timeout",
            },
            {
                "command": "/settings api_router",
                "label": "API Router",
                "detail": "provider, model, encrypted key",
            },
        ]

        for section, data in cfg.items():
            if not isinstance(data, dict):
                continue
            for key, value in data.items():
                command = f"/settings {section} {key}"
                detail = self._format_value(key, value)
                items.append({"command": command, "label": f"{section}.{key}", "detail": detail})

        return items

    def _format_value(self, key: str, value: object) -> str:
        if "key" in key.lower() and value:
            return "current: ********"
        if value in ("", None):
            return "current: empty"
        return f"current: {value}"

    def _format_option(self, command: str, label: str, detail: str) -> str:
        padded_cmd = f"{command:<35}"
        padded_label = f"{label:<30}"
        return f"[#60A5FA]{escape(padded_cmd)}[/] [#F4F4F5]{escape(padded_label)}[/] [dim]{escape(detail)}[/]"

    @on(Input.Changed, "#settings-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self.populate_options(event.value)

    @on(Input.Submitted, "#settings-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        option_list = self.query_one("#settings-options", OptionList)
        if option_list.highlighted is None and self._commands:
            option_list.highlighted = 0
        option_list.focus()

    @on(OptionList.OptionSelected, "#settings-options")
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        command = self._commands.get(event.option.id or "")
        if not command:
            return

        prompt = self.app.query_one("#prompt-input", Input)
        prompt.value = f"{command} "
        prompt.cursor_position = len(prompt.value)
        self.close()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.close()
