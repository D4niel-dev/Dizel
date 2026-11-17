from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from inference.tui.theme import ACCENT, DANGER, TEXT_2, TEXT_3, WARN


def _meter(percent: float, width: int = 8) -> str:
    """A block meter whose colour tracks the value, not the theme."""
    filled = max(0, min(width, round(percent / 100 * width)))
    color = ACCENT if percent < 60 else WARN if percent < 85 else DANGER
    return f"[{color}]{'━' * filled}[/][{TEXT_3}]{'─' * (width - filled)}[/]"


class ContextPanel(VerticalScroll):
    """Chat-session right rail: context, tools, plugins, and host load.

    Hidden on the welcome, docked once a transcript starts. It is not an
    overlay: ctrl+r focuses it, and there is no close path from inside a
    session. Long tool and plugin lists scroll inside the rail.
    """

    can_focus = True

    def compose(self) -> ComposeResult:
        yield Static("CONTEXT", classes="panel-header")
        yield Static(id="ctx-tokens", classes="ctx-item")
        yield Static(id="ctx-meter", classes="ctx-item")
        yield Static(id="ctx-budget", classes="ctx-item")
        yield Static("TOOLS", classes="panel-header")
        yield Static(id="ctx-tools", classes="ctx-item")
        yield Static("PLUGINS", classes="panel-header")
        yield Static(id="ctx-plugins", classes="ctx-item")
        yield Static("HOST", classes="panel-header", id="ctx-host-head")
        yield Static(id="ctx-cpu", classes="ctx-item")
        yield Static(id="ctx-ram", classes="ctx-item")
        yield Static(id="ctx-speed", classes="ctx-item")

    def on_mount(self):
        self.update_panel()
        self.update_plugins()
        for prop in ("context_tokens", "budget_tokens", "tools_revision"):
            self.watch(self.app, prop, self.update_panel)
        self._prime_monitor()
        self.update_monitor()
        self.set_interval(2.0, self.update_monitor)
        self.set_interval(2.0, self.update_plugins)

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
        manager = getattr(getattr(app, "chat_bridge", None), "manager", None)
        cap = int(getattr(manager, "effective_ctx_len", 0) or 0)
        if not cap:
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

    def update_plugins(self, *_: object) -> None:
        """Mirror the GUI's plugin catalog: server state plus tool count."""
        try:
            from dizel_core.mcp.manager import get_manager

            manager = get_manager()
            servers = manager.configured()
            states = {row.get("id"): row for row in manager.status()}
        except Exception:
            self.query_one("#ctx-plugins").update(f"[{TEXT_3}]unavailable[/]")
            return

        if not servers:
            self.query_one("#ctx-plugins").update(f"[{TEXT_3}]none configured[/]")
            return

        width = self.size.width or 32
        name_limit = max(12, width - 14)
        lines = []
        for server in servers:
            server_id = str(server.get("id") or "")
            name = " ".join(str(server.get("name") or server_id).split())
            if len(name) > name_limit:
                name = name[: name_limit - 1].rstrip() + "…"
            enabled = bool(server.get("enabled"))
            state = states.get(server_id, {})
            status = str(state.get("state") or ("starting" if enabled else "disabled"))
            if not enabled:
                status = "disabled"
            tools = state.get("tools") or []

            if status == "ready":
                glyph, color = "●", ACCENT
                count = len(tools)
                suffix = f"{count} tool" + ("" if count == 1 else "s") if count else "connected"
                name_color = TEXT_2
            elif status == "starting":
                glyph, color = "◐", WARN
                suffix = "connecting"
                name_color = TEXT_2
            elif status == "error":
                glyph, color = "×", DANGER
                suffix = "error"
                name_color = TEXT_3
            else:
                glyph, color = "○", TEXT_3
                suffix = "off"
                name_color = TEXT_3

            lines.append(
                f"[{color}]{glyph}[/] [{name_color}]{escape(name)}[/]"
                f"  [{TEXT_3}]{escape(suffix)}[/]"
            )
            error = " ".join(str(state.get("error") or "").split())
            if status == "error" and error:
                room = max(12, width - 2)
                if len(error) > room:
                    error = error[: room - 1].rstrip() + "…"
                lines.append(f"  [{TEXT_3}]{escape(error)}[/]")

        self.query_one("#ctx-plugins").update("\n".join(lines))

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
