from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from inference.tui.theme import ACCENT, DANGER, TEXT_2, TEXT_3, WARN


def _meter(percent: float, width: int = 8) -> str:
    """A block meter whose colour tracks the value, not the theme."""
    filled = max(0, min(width, round(percent / 100 * width)))
    color = ACCENT if percent < 60 else WARN if percent < 85 else DANGER
    return f"[{color}]{'━' * filled}[/][{TEXT_3}]{'─' * (width - filled)}[/]"


class ContextPanel(Container):
    """Opt-in right rail: context budget, tool state, host load.

    State and model live in the header and status line; repeating them here
    was the third copy of the same sentence.
    """

    def compose(self) -> ComposeResult:
        yield Static("CONTEXT", classes="panel-header")
        yield Static(id="ctx-tokens", classes="ctx-item")
        yield Static(id="ctx-meter", classes="ctx-item")
        yield Static(id="ctx-budget", classes="ctx-item")
        yield Static("TOOLS", classes="panel-header")
        yield Static(id="ctx-tools", classes="ctx-item")
        yield Static("HOST", classes="panel-header", id="ctx-host-head")
        yield Static(id="ctx-cpu", classes="ctx-item")
        yield Static(id="ctx-ram", classes="ctx-item")
        yield Static(id="ctx-speed", classes="ctx-item")

    def on_mount(self):
        self.update_panel()
        for prop in ("context_tokens", "budget_tokens", "tools_revision"):
            self.watch(self.app, prop, self.update_panel)
        self._prime_monitor()
        self.update_monitor()
        self.set_interval(2.0, self.update_monitor)

    def _prime_monitor(self) -> None:
        """psutil reports 0.0% on its first non-blocking sample."""
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            self._hide_host()

    def _hide_host(self) -> None:
        for selector in ("#ctx-host-head", "#ctx-cpu", "#ctx-ram", "#ctx-speed"):
            for widget in self.query(selector):
                widget.display = False

    def update_panel(self, *args) -> None:
        app = self.app
        cap = int(getattr(app.usage_manager, "max_capacity", 0) or 0)
        used = int(app.context_tokens)
        pct = (used / cap * 100) if cap else 0.0
        self.query_one("#ctx-tokens").update(
            f"[{TEXT_2}]{used:,}[/][{TEXT_3}] / {cap:,}[/]"
        )
        self.query_one("#ctx-meter").update(_meter(pct))
        self.query_one("#ctx-budget").update(
            f"[{TEXT_3}]budget[/] [{TEXT_2}]{int(app.budget_tokens)}[/]"
        )

        states = getattr(app, "_tool_states", {})
        if states:
            lines = []
            for name, enabled in states.items():
                mark = f"[{ACCENT}]●[/]" if enabled else f"[{TEXT_3}]○[/]"
                label = name.replace("_", " ")
                color = TEXT_2 if enabled else TEXT_3
                lines.append(f"{mark} [{color}]{label}[/]")
            self.query_one("#ctx-tools").update("\n".join(lines))
        else:
            self.query_one("#ctx-tools").update(f"[{TEXT_3}]none[/]")

    def update_monitor(self) -> None:
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except Exception:
            self._hide_host()
            return

        self.query_one("#ctx-cpu").update(f"[{TEXT_3}]cpu[/] {cpu:>5.1f}% {_meter(cpu, 6)}")
        self.query_one("#ctx-ram").update(f"[{TEXT_3}]ram[/] {ram:>5.1f}% {_meter(ram, 6)}")

        speed_text = f"[{TEXT_3}]speed  --[/]"
        if getattr(self.app, "generation_state", "") == "STREAMING":
            speed = getattr(self.app.chat_bridge.manager, "_last_tps", 0.0)
            if speed > 0:
                speed_text = f"[{TEXT_3}]speed[/]  [{TEXT_2}]{speed:.1f}[/] [{TEXT_3}]t/s[/]"
        self.query_one("#ctx-speed").update(speed_text)
