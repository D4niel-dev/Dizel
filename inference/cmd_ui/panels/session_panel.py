from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from textual import on

class SessionPanel(Container):
    def compose(self) -> ComposeResult:
        yield Static("SESSIONS", classes="panel-header")
        yield OptionList(id="session-list")

    def on_mount(self):
        self.update_sessions()
        self.set_interval(5.0, self.update_sessions)

    def update_sessions(self):
        sessions = self.app.session_bridge.get_all()
        option_list = self.query_one("#session-list", OptionList)
        option_list.clear_options()

        for s in sessions[:15]:
            prefix = "> " if s["id"] == self.app.session_id else "  "
            pin = "* " if s.get("pinned") else ""
            label = f"{prefix}{pin}{s['title'][:20]}"
            option_list.add_option(Option(label, id=s["id"]))

    @on(OptionList.OptionSelected, "#session-list")
    def on_session_selected(self, event: OptionList.OptionSelected):
        session_id = event.option.id
        if session_id and session_id != self.app.session_id:
            self.app.load_session_to_workspace(session_id)
            self.update_sessions()
