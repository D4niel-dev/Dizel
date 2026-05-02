from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

class StatusBar(Container):
    def compose(self) -> ComposeResult:
        yield Static(id="status-text")

    def on_mount(self):
        self.update_status()
        self.watch(self.app, "active_mode", self.update_status)
        self.watch(self.app, "active_provider", self.update_status)

    def update_status(self, *args) -> None:
        app = self.app
        # Show mode, provider, and helpful hints
        text = f"[{app.active_mode.upper()}] {app.active_provider}  •  esc interrupt  tab switch agent  ctrl+k commands  ctrl+h sessions  ctrl+r right panel"
        self.query_one("#status-text").update(text)
