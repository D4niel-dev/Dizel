from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Input, OptionList
from textual import on, events
from textual.widgets.option_list import Option
from inference.cmd_ui.commands.registry import registry

class CommandPalette(Container):
    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search commands... (e.g. model, mode, session)", id="palette-input")
        yield OptionList(id="palette-options")
        
    def on_mount(self):
        self.display = False
        self.populate_options("")
        
    def populate_options(self, query: str):
        options_list = self.query_one("#palette-options", OptionList)
        options_list.clear_options()
        
        cmds = registry.list_all()
        q = query.lower()
        
        for cmd in cmds:
            if q in cmd.name.lower() or q in cmd.help_text.lower():
                options_list.add_option(Option(f"/{cmd.name}  -  {cmd.help_text}", id=f"cmd_{cmd.name}"))
                
    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed):
        self.populate_options(event.value)
        
    @on(Input.Submitted, "#palette-input")
    def on_input_submitted(self, event: Input.Submitted):
        self.query_one("#palette-options").focus()

    @on(OptionList.OptionSelected, "#palette-options")
    def on_option_selected(self, event: OptionList.OptionSelected):
        cmd_id = event.option.id
        if cmd_id and cmd_id.startswith("cmd_"):
            cmd_name = cmd_id[4:]
            input_bar = self.app.query_one("InputBar").query_one("Input")
            input_bar.value = f"/{cmd_name} "
            input_bar.focus()
            input_bar.cursor_position = len(input_bar.value)
            
        self.display = False
        self.query_one("#palette-input").value = ""
        
    def toggle(self):
        self.display = not self.display
        if self.display:
            inp = self.query_one("#palette-input")
            inp.value = ""
            self.populate_options("")
            inp.focus()
        else:
            self.app.query_one("InputBar").query_one("Input").focus()
            
    def on_key(self, event: events.Key):
        if event.key == "escape":
            self.display = False
            self.app.query_one("InputBar").query_one("Input").focus()
