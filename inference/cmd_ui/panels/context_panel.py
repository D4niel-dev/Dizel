from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

class ContextPanel(Container):
    def compose(self) -> ComposeResult:
        yield Static("SYSTEM", classes="panel-header")
        yield Static(id="ctx-model", classes="ctx-item")
        yield Static(id="ctx-mode", classes="ctx-item")
        yield Static(id="ctx-provider", classes="ctx-item")
        yield Static(id="ctx-status", classes="ctx-item")
        yield Static("CONTEXT", classes="panel-header")
        yield Static(id="ctx-tokens", classes="ctx-item")
        yield Static(id="ctx-budget", classes="ctx-item")
        
    def on_mount(self):
        self.update_panel()
        self.watch(self.app, "active_model", self.update_panel)
        self.watch(self.app, "active_mode", self.update_panel)
        self.watch(self.app, "active_provider", self.update_panel)
        self.watch(self.app, "generation_state", self.update_panel)
        self.watch(self.app, "context_tokens", self.update_panel)
        self.watch(self.app, "budget_tokens", self.update_panel)

    def update_panel(self, *args) -> None:
        self.query_one("#ctx-model").update(f"  Model: {self.app.active_model}")
        self.query_one("#ctx-mode").update(f"  Mode: {self.app.active_mode}")
        self.query_one("#ctx-provider").update(f"  Provider: {self.app.active_provider}")
        self.query_one("#ctx-status").update(f"  Status: {self.app.generation_state}")
        
        max_cap = int(self.app.usage_manager.max_capacity) if hasattr(self.app, 'usage_manager') else 4096
        self.query_one("#ctx-tokens").update(f"  Tokens: {int(self.app.context_tokens)}/{max_cap}")
        self.query_one("#ctx-budget").update(f"  Budget: {self.app.budget_tokens}")
