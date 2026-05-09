import os
import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from inference.cmd_ui.panels.status_bar import StatusBar
from inference.cmd_ui.panels.session_panel import SessionPanel
from inference.cmd_ui.panels.workspace import WorkspacePanel
from inference.cmd_ui.panels.context_panel import ContextPanel
from inference.cmd_ui.panels.input_bar import InputBar
from inference.cmd_ui.panels.command_palette import CommandPalette
from inference.cmd_ui.panels.settings_panel import SettingsPanel
from inference.cmd_ui.commands.builtins import register_builtins
from inference.cmd_ui.bridge.chat_bridge import ChatBridge
from inference.cmd_ui.rendering.message_block import MessageBlock
from inference.dizel_ui.logic.usage_manager import UsageManager


class DizelCMDApp(App):
    """The Dizel CMD UI."""

    CSS_PATH = "cmd_ui.tcss"

    BINDINGS = [
        ("ctrl+c", "interrupt_or_quit", "Interrupt/Quit"),
        ("escape", "interrupt", "Interrupt"),
        ("ctrl+l", "clear_workspace", "Clear"),
        Binding("ctrl+k", "toggle_palette", "Command Palette", priority=True),
        Binding("ctrl+t", "toggle_session", "Toggle Sessions", priority=True),
        Binding("ctrl+h", "toggle_session", "Toggle Sessions", priority=True),
        Binding("ctrl+r", "toggle_context", "Toggle Context", priority=True),
        Binding("ctrl+y", "multiline_input", "Multi-line Input", priority=True),
    ]

    # Reactive state
    active_model = reactive("Dizel Lite")
    active_mode = reactive("Fast")
    active_provider = reactive("local")
    generation_state = reactive("IDLE")
    session_id = reactive("")
    context_tokens = reactive(0)
    budget_tokens = reactive(0)

    def __init__(self, checkpoint: str = "", device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self._checkpoint = checkpoint
        self._device = device
        self.usage_manager = UsageManager()
        register_builtins()
        self.chat_bridge = ChatBridge(self)
        from inference.cmd_ui.bridge.tool_bridge import ToolBridge
        self.tool_bridge = ToolBridge(self)
        from inference.cmd_ui.bridge.session_bridge import SessionBridge
        self.session_bridge = SessionBridge(self)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def on_mount(self) -> None:
        # Start each run in a fresh session; history stays available in the session panel.
        self.session_bridge.create()

        # Sync provider state from config
        mgr = self.chat_bridge.manager
        self.active_provider = mgr.active_provider_slug
        if mgr.active_api_model:
            self.active_model = mgr.active_api_model
        
        # If API provider is active, validate connection on startup
        if self.active_provider != "local" and mgr._api_provider:
            try:
                mgr._api_provider.validate(key=mgr._api_key)
                self._log_system(f"✓ {self.active_provider} connected · model: {mgr._api_model or '(not set)'}")
            except Exception as e:
                self._log_system(f"✗ {self.active_provider} unreachable: {e}")

        # Auto-load model
        self.call_after_refresh(self._auto_load_model)
        self.call_after_refresh(self._focus_prompt)

    def _auto_load_model(self) -> None:
        """Discover and load a checkpoint, matching the GUI flow."""
        from inference.dizel_ui.logic.config_manager import ConfigManager

        checkpoint = self._checkpoint
        device = self._device

        # Read from config if not provided via CLI
        if not checkpoint:
            cfg = ConfigManager.load()
            checkpoint = cfg.get("checkpoint", "")
            saved_device = cfg.get("device", "cpu")
            if saved_device in ("cpu", "cuda"):
                device = saved_device

        # If using an API provider, no local model needed
        if self.active_provider != "local":
            model_info = self.active_model or "(no model set — use /model <id>)"
            self._log_system(f"Using {self.active_provider} provider · {model_info}")
            self._log_system("Type /model to see available models, or /provider to change.")
            return

        if not checkpoint:
            # Try auto-discovering checkpoints
            checkpoint = self._discover_checkpoint()

        if not checkpoint:
            self._log_system("No checkpoint found. Use /load <path> or configure a provider.")
            return

        self._checkpoint = checkpoint
        self._device = device
        self._load_model_async(checkpoint, device)

    def _discover_checkpoint(self) -> str:
        """Look in the checkpoints/ directory for any .pt files."""
        import glob
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_dir = os.path.join(proj_root, "checkpoints")

        if not os.path.isdir(ckpt_dir):
            return ""

        # Prefer sft-best > sft > any
        all_pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        if not all_pts:
            return ""

        for p in all_pts:
            fn = os.path.basename(p).lower()
            if "sft-best" in fn:
                return p

        for p in all_pts:
            fn = os.path.basename(p).lower()
            if "sft" in fn:
                return p

        return all_pts[0]

    def _load_model_async(self, checkpoint: str, device: str) -> None:
        """Load model in a background thread with a single updating status line."""
        self.generation_state = "LOADING"

        # Mount a single static widget for loader status
        try:
            workspace = self.query_one("WorkspacePanel")
            self._loader_widget = Static(MessageBlock("SYSTEM", f"Loading {os.path.basename(checkpoint)}..."))
            workspace.mount(self._loader_widget)
            workspace.scroll_end(animate=False)
        except Exception:
            self._loader_widget = None

        def _worker():
            try:
                mgr = self.chat_bridge.manager
                mgr.load_model(
                    checkpoint_path=checkpoint,
                    device=device,
                    on_progress=lambda msg: self.call_from_thread(self._update_loader_status, msg),
                )
                self.call_from_thread(self._on_model_ready)
            except Exception as exc:
                self.call_from_thread(self._on_model_error, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_loader_status(self, msg: str) -> None:
        """Update the single loader line in-place."""
        if self._loader_widget:
            self._loader_widget.update(MessageBlock("SYSTEM", msg))
            try:
                self.query_one("WorkspacePanel").scroll_end(animate=False)
            except Exception:
                pass

    def _on_model_ready(self) -> None:
        self.generation_state = "IDLE"
        mgr = self.chat_bridge.manager
        info = mgr.model_info
        if info:
            self.context_tokens = 0
            ctx_len = info.get('ctx_len', 1024)
            self.usage_manager.set_capacity(ctx_len)
            msg = (
                f"Model ready -- {info.get('params', '?')} params, "
                f"d={info.get('d_model', '?')}, L={info.get('n_layers', '?')}, "
                f"ctx={ctx_len}, device={info.get('device', '?')}"
            )
            self._update_loader_status(msg)
        else:
            self._update_loader_status("Model loaded.")
            
        if self._loader_widget:
            widget = self._loader_widget
            self.set_timer(10.0, lambda: widget.remove())
        self._loader_widget = None

    def _on_model_error(self, msg: str) -> None:
        self.generation_state = "IDLE"
        self._update_loader_status(f"Load error: {msg}")
        if self._loader_widget:
            widget = self._loader_widget
            self.set_timer(10.0, lambda: widget.remove())
        self._loader_widget = None
        self._loader_widget = None

    def _log_system(self, msg: str) -> None:
        """Post a SYSTEM message to the workspace."""
        try:
            workspace = self.query_one("WorkspacePanel")
            workspace.mount(Static(MessageBlock("SYSTEM", msg)))
            workspace.scroll_end(animate=False)
        except Exception:
            pass  # Workspace not yet mounted

    def _focus_prompt(self) -> None:
        try:
            self.query_one("#prompt-input").focus()
        except Exception:
            pass

    # ── Layout ─────────────────────────────────────────────────────────

    def load_session_to_workspace(self, session_id: str):
        if self.session_bridge.load(session_id):
            workspace = self.query_one("WorkspacePanel")
            workspace.clear_workspace(show_empty=False)
            for msg in self.session_bridge.current_messages:
                workspace.mount(Static(MessageBlock(msg["role"].upper(), msg["content"])))
                workspace.scroll_end(animate=False)

    def compose(self) -> ComposeResult:
        from textual.containers import Container
        with Horizontal(id="main-layout"):
            yield SessionPanel()
            with Container(id="workspace-container"):
                yield WorkspacePanel()
                yield InputBar()
                yield StatusBar()
            yield ContextPanel()
        yield CommandPalette()
        yield SettingsPanel()

    # ── Actions ────────────────────────────────────────────────────────

    def action_clear_workspace(self) -> None:
        workspace = self.query_one("WorkspacePanel")
        workspace.clear_workspace()

    def action_multiline_input(self) -> None:
        from inference.cmd_ui.panels.multiline_input import MultilineInputModal
        from textual.widgets import Input

        try:
            inp = self.query_one("#prompt-input", Input)
            current_text = inp.value
        except Exception:
            return

        def _on_dismiss(text: str | None) -> None:
            if text is not None:
                try:
                    inp = self.query_one("#prompt-input", Input)
                    inp.value = text
                    inp.focus()
                    # Optionally, we could call inp.action_submit() here if we want Ctrl+S to auto-send,
                    # but bringing it back to the input bar is safer for review.
                except Exception:
                    pass

        self.push_screen(MultilineInputModal(current_text), _on_dismiss)

    def action_interrupt_or_quit(self):
        if self.generation_state in ("STREAMING", "THINKING"):
            self.chat_bridge.stop()
            self.generation_state = "IDLE"
            workspace = self.query_one("WorkspacePanel")
            workspace.mount(Static(MessageBlock("SYSTEM", "INTERRUPTED")))
            workspace.scroll_end(animate=False)
        else:
            self.chat_bridge.stop()
            self.exit()

    def action_interrupt(self):
        if self.generation_state in ("STREAMING", "THINKING"):
            self.chat_bridge.stop()
            self.generation_state = "IDLE"
            workspace = self.query_one("WorkspacePanel")
            workspace.mount(Static(MessageBlock("SYSTEM", "INTERRUPTED")))
            workspace.scroll_end(animate=False)

    def action_toggle_palette(self):
        palette = self.query_one("CommandPalette")
        palette.toggle()

    def action_toggle_session(self):
        panel = self.query_one(SessionPanel)
        panel.toggle_panel()

    def action_toggle_context(self):
        panel = self.query_one("ContextPanel")
        if panel.styles.display == "none":
            panel.styles.display = "block"
        else:
            panel.styles.display = "none"

    def action_switch_agent(self):
        modes = ["Fast", "Planning", "Coding", "Review"]
        current = self.active_mode
        if current in modes:
            next_idx = (modes.index(current) + 1) % len(modes)
            self.active_mode = modes[next_idx]
        else:
            self.active_mode = modes[0]

    def action_switch_agent_back(self):
        modes = ["Fast", "Planning", "Coding", "Review"]
        current = self.active_mode
        if current in modes:
            prev_idx = (modes.index(current) - 1) % len(modes)
            self.active_mode = modes[prev_idx]
        else:
            self.active_mode = modes[0]


if __name__ == "__main__":
    app = DizelCMDApp()
    app.run()
