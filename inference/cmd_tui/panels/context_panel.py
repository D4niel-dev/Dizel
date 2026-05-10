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
        yield Static("\nTOOLS", classes="panel-header")
        yield Static(id="ctx-tools", classes="ctx-item")
        yield Static("\nMONITOR", classes="panel-header")
        yield Static(id="ctx-cpu", classes="ctx-item")
        yield Static(id="ctx-ram", classes="ctx-item")
        yield Static(id="ctx-speed", classes="ctx-item")
        
    def on_mount(self):
        self.update_panel()
        self.watch(self.app, "active_model", self.update_panel)
        self.watch(self.app, "active_mode", self.update_panel)
        self.watch(self.app, "active_provider", self.update_panel)
        self.watch(self.app, "generation_state", self.update_panel)
        self.watch(self.app, "context_tokens", self.update_panel)
        self.watch(self.app, "budget_tokens", self.update_panel)
        self.set_interval(1.0, self.update_panel)  # To catch tool changes
        self.set_interval(2.0, self.update_monitor)

    def update_panel(self, *args) -> None:
        self.query_one("#ctx-model").update(f"  Model: {self.app.active_model}")
        self.query_one("#ctx-mode").update(f"  Mode: {self.app.active_mode}")
        self.query_one("#ctx-provider").update(f"  Provider: {self.app.active_provider}")
        self.query_one("#ctx-status").update(f"  Status: {self.app.generation_state}")
        
        max_cap = int(self.app.usage_manager.max_capacity) if hasattr(self.app, 'usage_manager') else 4096
        self.query_one("#ctx-tokens").update(f"  Tokens: {int(self.app.context_tokens)}/{max_cap}")
        self.query_one("#ctx-budget").update(f"  Budget: {self.app.budget_tokens}")
        
        # Tools Overview
        tool_states = getattr(self.app, "_tool_states", {})
        active_tools = [name.replace('_', ' ').title() for name, state in tool_states.items() if state]
        if not active_tools:
            tools_str = "  No active tools"
        else:
            tools_str = "\n".join([f"  • {t}" for t in active_tools])
        self.query_one("#ctx-tools").update(tools_str)

    def update_monitor(self) -> None:
        try:
            import psutil
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            def make_bar(percent):
                blocks = "▏▎▍▌▋▊▉█"
                full = int(percent / 10)
                rem = int((percent % 10) / 1.25)
                bar = ("█" * full) + (blocks[rem] if rem < len(blocks) else "")
                bar = bar.ljust(10, " ")
                color = "green" if percent < 60 else "yellow" if percent < 85 else "red"
                return f"[{color}]{bar}[/]"
                
            self.query_one("#ctx-cpu").update(f"  CPU: {cpu:04.1f}% {make_bar(cpu)}")
            self.query_one("#ctx-ram").update(f"  RAM: {ram:04.1f}% {make_bar(ram)}")
        except ImportError:
            self.query_one("#ctx-cpu").update("  CPU: [dim]pip install psutil[/]")
            self.query_one("#ctx-ram").update("  RAM: [dim]pip install psutil[/]")

        # Update Speed
        speed_text = "  Speed: -- t/s"
        if getattr(self.app, "generation_state", "") == "STREAMING":
            speed = getattr(self.app.chat_bridge.manager, "_last_tps", 0.0)
            if speed > 0:
                speed_text = f"  Speed: {speed:.1f} t/s"
        try:
            self.query_one("#ctx-speed").update(speed_text)
        except Exception:
            pass
