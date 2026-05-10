from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

class StatusBar(Container):
    def compose(self) -> ComposeResult:
        yield Static(id="status-text")
        yield Static(id="status-session-info")
        yield Static("v2.0.3", id="status-version")

    def on_mount(self):
        self.update_status()
        self.watch(self.app, "active_mode", self.update_status)
        self.watch(self.app, "active_provider", self.update_status)
        self.watch(self.app, "session_id", self.update_session_info)

    def update_session_info(self, *args) -> None:
        try:
            if not self.app.session_id:
                info_text = "No active session"
            else:
                # Find title if possible
                title = "Unknown"
                for s in self.app.session_bridge.get_all():
                    if s["id"] == self.app.session_id:
                        title = s["title"]
                        break
                if len(title) > 15:
                    title = title[:12] + "..."
                info_text = f"Sess: {title}"
                
            self.query_one("#status-session-info").update(info_text)
        except Exception:
            pass

    def update_status(self, *args) -> None:
        app = self.app
        text = (
            f"{app.active_provider}  |  esc interrupt  tab agent  "
            "ctrl+k cmd  ctrl+t sess  ctrl+a arti  ctrl+r ctx"
        )
        self.query_one("#status-text").update(text)
