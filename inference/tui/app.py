import faulthandler
import os
import queue
import threading
import time

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.widgets import Static

from inference.tui.panels.status_bar import StatusBar
from inference.tui.panels.notice_line import NoticeLine
from inference.tui.panels.session_panel import SessionPanel
from inference.tui.panels.session_tabs import SessionTabs
from inference.tui.panels.workspace import WorkspacePanel
from inference.tui.panels.context_panel import ContextPanel
from inference.tui.panels.input_bar import InputBar
from inference.tui.commands.builtins import register_builtins
from inference.tui.commands.builtins.tools import AVAILABLE_TOOLS
from inference.tui.bridge.chat_bridge import ChatBridge
from inference.tui.constants import MODES, TIERS
from inference.tui.rendering.history_message import HistoryMessage
from inference.tui.rendering.message_block import MessageBlock, message_style
from inference.tui.settings_apply import chat_pref
from dizel_core.usage_manager import UsageManager
from dizel_core.token_budget import model_family as detect_model_family
from model.naming import model_display_name

# Below this width the session sidebar would leave the transcript too narrow
# to read; ctrl+t still opens it on demand.
SIDEBAR_MIN_WIDTH = 100
# A side rail is a wider commitment than the sidebar, so the sidebar steps
# aside sooner when one is open.
RAIL_SIDEBAR_MIN_WIDTH = 120
# A load that reports nothing for this long gets every thread's stack dumped.
# The UI thread cannot report a wedged loader; a daemon thread can.
LOAD_STALL_SECONDS = 20


def _log_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "logs", name)


def _open_load_trace():
    """A dedicated handle so the loader can write even if the app is stuck."""
    from datetime import datetime

    try:
        path = _log_path("tui_load.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle = open(path, "a", encoding="utf-8", buffering=1)
        handle.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} ===\n")
        return handle
    except Exception:
        return None


def _trace(handle, message: str) -> None:
    import time

    if handle is None:
        return
    try:
        handle.write(f"{time.monotonic():12.3f}  {message}\n")
        handle.flush()
    except Exception:
        pass


class PanelOverlay(ModalScreen):
    """A panel floating over the workspace instead of narrowing it.

    The desktop keeps these parked; a terminal can't afford a permanent column,
    so ctrl+r / ctrl+a raise them over the transcript and escape puts them away.
    Panels are built fresh here and filled after mount, which keeps the layout
    instance free of state that would otherwise have to be shared.
    """

    BINDINGS = [Binding("escape", "close_overlay", "Close")]

    def __init__(self, panel, filler=None) -> None:
        super().__init__()
        self._panel = panel
        self._filler = filler

    def compose(self) -> ComposeResult:
        with Vertical(id="overlay-dock"):
            yield self._panel

    def on_mount(self) -> None:
        if self._filler is not None:
            self._filler(self._panel)
        self._panel.focus()

    def action_close_overlay(self) -> None:
        self.dismiss()


class DizelCMDApp(App):
    """The Dizel terminal client."""

    TITLE = "Dizel"
    SUB_TITLE = "terminal"
    CSS_PATH = "cmd_ui.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+c", "interrupt_or_quit", "Interrupt / quit", priority=True),
        ("escape", "interrupt", "Interrupt"),
        ("ctrl+l", "clear_workspace", "Clear"),
        Binding("ctrl+k", "toggle_palette", "Command palette", priority=True),
        Binding("ctrl+f", "toggle_palette", "All commands", priority=True),
        Binding("ctrl+t", "toggle_session", "Sessions", priority=True),
        Binding("ctrl+r", "toggle_context", "Context rail", priority=True),
        Binding("ctrl+y", "multiline_input", "Multi-line", priority=True),
        Binding("ctrl+a", "toggle_artifacts", "Artifacts", priority=True),
    ]

    # Reactive state
    active_model = reactive("")
    active_tier = reactive("Flash")
    active_mode = reactive("Fast")
    active_provider = reactive("local")
    generation_state = reactive("IDLE")
    session_id = reactive("")
    context_tokens = reactive(0)
    budget_tokens = reactive(0)
    tools_revision = reactive(0)

    def __init__(
        self,
        checkpoint: str = "",
        device: str = "cpu",
        preloaded=None,
        resume_session: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._checkpoint = checkpoint
        self._device = device
        self._resume_session = resume_session
        self._resumed = False
        self._resume_failed = False
        self.open_tabs: list[str] = []
        self._tab_messages: dict[str, list[dict]] = {}
        self._loader_events = None
        self._loader_timer = None
        self._load_watchdog_stop = None
        self._load_last_event_at = 0.0
        self._sessions_auto = True
        self._update_thread = None
        self.usage_manager = UsageManager()
        self._tool_states = {name: False for name in AVAILABLE_TOOLS}
        register_builtins()
        self.chat_bridge = ChatBridge(self, manager=preloaded)
        from inference.tui.bridge.tool_bridge import ToolBridge
        from inference.tui.bridge.session_bridge import SessionBridge

        self.tool_bridge = ToolBridge(self)
        self.session_bridge = SessionBridge(self)

        if checkpoint:
            self.active_model = model_display_name(checkpoint)

    # ── Lifecycle ──────────────────────────────────────────────────────
    def on_mount(self) -> None:
        if self._resume_session and self.session_bridge.load(self._resume_session):
            self._resumed = True
        else:
            self._resume_failed = bool(self._resume_session)
            self.session_bridge.create()
        self.open_tabs = [self.session_id] if self.session_id else []
        if self.session_id:
            self._tab_messages[self.session_id] = list(self.session_bridge.current_messages)

        # Terminal-only preferences. Applied here so a saved choice survives a
        # restart, not just the session that changed it.
        try:
            from inference.tui.settings_apply import apply_terminal_prefs

            apply_terminal_prefs(self)
        except Exception:
            pass
        self._apply_layout()

        # OpenCode opens full width with no session list — the chat is the
        # screen. ctrl+t brings the list in when it is wanted.
        self._sessions_auto = False
        try:
            self.query_one(SessionPanel).set_sidebar_visible(False)
        except Exception:
            pass

        mgr = self.chat_bridge.manager
        self.active_provider = mgr.active_provider_slug
        if self.active_provider != "local":
            self.active_model = mgr.active_api_model or ""
        else:
            # A preloaded manager already has weights in memory, but the app only
            # knows the checkpoint if it was passed on the command line. Ask the
            # manager, otherwise the UI claims "no model" while one is loaded.
            path = self._checkpoint or getattr(mgr, "checkpoint_path", "")
            if path:
                self._checkpoint = path
                self.active_model = model_display_name(path)

        self._apply_profile()

        from dizel_core.config_manager import ConfigManager

        cfg = ConfigManager.load()
        saved_prompt = str(cfg.get("system_prompt", "")).strip()
        default_prompt = str(ConfigManager._DEFAULTS.get("system_prompt", "")).strip()
        if saved_prompt and saved_prompt != default_prompt:
            mgr.system_prompt = saved_prompt

        if self.active_provider != "local" and mgr._api_provider:
            try:
                mgr._api_provider.validate(key=mgr._api_key)
                self._log_system(
                    f"{self.active_provider} connected · {mgr._api_model or '(no model set)'}",
                    "ok",
                )
            except Exception as error:
                self._log_system(f"{self.active_provider} unreachable: {error}", "error")

        self.call_after_refresh(self._auto_load_model)
        self.call_after_refresh(self._focus_prompt)
        self.call_after_refresh(self._start_mcp)
        if self._resumed:
            self.call_after_refresh(self._restore_resumed_session)
        if self._resume_failed:
            self._log_system("No saved chat to resume — started a new one.", "error")
        self.refresh_tabs()
        # Let the first screen settle before a network check competes with it.
        self.set_timer(2.0, self._start_update_check)
        # A long-running terminal session should still notice releases.
        self.set_interval(6 * 60 * 60, self._start_update_check)

    def _restore_resumed_session(self) -> None:
        if self.session_id:
            self.load_session_to_workspace(self.session_id, announce=False)

    def _start_mcp(self) -> None:
        from dizel_core.config_manager import ConfigManager
        from dizel_core.mcp import config as mcp_config
        from dizel_core.mcp.manager import get_manager

        manager = get_manager()
        # Spawned servers inherit stderr; letting them paint sys.stderr would
        # corrupt the Textual screen.  Route child logs to a file instead.
        try:
            path = _log_path("mcp.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            manager.set_errlog(
                open(path, "a", encoding="utf-8", buffering=1), owned=True
            )
        except Exception:
            pass
        manager.apply_async(mcp_config.load(ConfigManager.load()))

    def _stop_mcp(self) -> None:
        from dizel_core.mcp.manager import get_manager

        get_manager().shutdown()

    def on_unmount(self) -> None:
        self._stop_mcp()

    async def _on_resize(self, event: events.Resize) -> None:
        # App stops Resize before it reaches an `on_resize` handler, so the
        # sidebar breakpoint has to hook the event here.
        await super()._on_resize(event)
        self._apply_layout(event.size.width)

    def _apply_layout(self, width: int | None = None) -> None:
        if not self._sessions_auto:
            return
        width = self.size.width if width is None else width
        try:
            threshold = RAIL_SIDEBAR_MIN_WIDTH if self._rail_open() else SIDEBAR_MIN_WIDTH
            self.query_one(SessionPanel).set_sidebar_visible(width >= threshold)
        except Exception:
            # Resize can arrive before the first compose; the next pass wins.
            pass

    def _rail_open(self) -> bool:
        from inference.tui.panels.artifact_panel import ArtifactPanel

        return self.query_one(ArtifactPanel).has_class("-visible") or self.query_one(
            ContextPanel
        ).has_class("-visible")

    def _auto_load_model(self) -> None:
        try:
            self._auto_load_model_impl()
        except Exception as error:
            self._log_system(f"Startup load failed: {error}", "error")

    def _auto_load_model_impl(self) -> None:
        from dizel_core.config_manager import ConfigManager

        if self.chat_bridge.manager.is_ready:
            # main() loaded it before the UI existed; nothing left to load.
            path = getattr(self.chat_bridge.manager, "checkpoint_path", "")
            if path:
                self._checkpoint = path
                self.active_model = model_display_name(path)
            self._log_system("Model ready.")
            return

        checkpoint = self._checkpoint
        device = self._device

        if not checkpoint:
            cfg = ConfigManager.load()
            checkpoint = cfg.get("checkpoint", "")
            saved_device = cfg.get("device", "cpu")
            if saved_device in ("cpu", "cuda"):
                device = saved_device

        if self.active_provider != "local":
            self._log_system(
                f"Using {self.active_provider} · {self.active_model or '(no model set)'}. "
                "Type /model to change it."
            )
            return

        if not checkpoint:
            checkpoint = self._discover_checkpoint()

        if not checkpoint:
            self._log_system(
                "No checkpoint found. Use /load <path> or configure a provider.", "error"
            )
            return

        self._load_model_async(checkpoint, device)

    def _discover_checkpoint(self) -> str:
        """Look in checkpoints/ for a .pt, .gguf or .safetensors artifact."""
        import glob

        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_dir = os.path.join(proj_root, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return ""

        files = []
        for pattern in ("*.pt", "*.gguf", "*.safetensors"):
            files.extend(glob.glob(os.path.join(ckpt_dir, pattern)))
        if not files:
            return ""

        def rank(path: str) -> tuple:
            name = os.path.basename(path).lower()
            order = 0
            for index, token in enumerate(("sft-best", "distill-best", "sft", "distill", "best")):
                if token in name:
                    order = index
                    break
            return (order, name)

        return sorted(files, key=rank)[0]

    def _load_model_async(self, checkpoint: str, device: str) -> bool:
        """Start a background load. Returns False when it was refused.

        Refusal is not an error state: the previously loaded model is still
        usable, so the caller only needs to stop and let the notice explain.
        """
        manager = self.chat_bridge.manager
        if getattr(manager, "is_generating", False):
            self._log_system(
                "Still generating — press esc to stop before loading a model.", "error"
            )
            return False
        if self._loader_events is not None:
            self._log_system("A model load is already in progress.", "error")
            return False

        # The load is accepted: this is now the checkpoint the UI reports. If
        # it fails, _on_model_error restores whatever is still loaded.
        self._checkpoint = checkpoint
        self._device = device
        self.active_model = model_display_name(checkpoint)
        self.generation_state = "LOADING"
        self._log_system(f"Loading {model_display_name(checkpoint)}…")
        try:
            self.refresh(layout=True)
        except Exception:
            pass

        # Progress is marshalled through a queue drained on the UI thread.
        # Calling Textual from the loader thread can deadlock on Windows and
        # leaves the loader stuck at an arbitrary progress message.
        events: queue.Queue = queue.Queue()
        self._loader_events = events
        self._load_started = time.monotonic()
        self._load_last_event_at = self._load_started
        self._load_tick = -1
        self._load_last = "starting"
        self._loader_timer = self.set_interval(0.05, self._poll_loader_events)
        self._start_load_watchdog()

        def _worker():
            # The loader writes its own record, straight to disk, from inside the
            # thread. If the UI thread is wedged we still learn how far the load
            # got — an empty log only proves the UI never printed, not that
            # nothing happened.
            trace = _open_load_trace()
            _trace(trace, f"worker started · {checkpoint} · device={device}")
            self._load_last_event_at = time.monotonic()

            def report(msg):
                self._load_last_event_at = time.monotonic()
                _trace(trace, f"progress · {msg}")
                events.put(("progress", msg))

            try:
                _trace(trace, "calling ChatManager.load_model()")
                manager.load_model(
                    checkpoint_path=checkpoint,
                    device=device,
                    on_progress=report,
                )
                _trace(trace, "load_model() returned OK")
                events.put(("ready", None))
            except Exception as exc:
                _trace(trace, f"load_model() raised {type(exc).__name__}: {exc}")
                events.put(("error", str(exc)))
            finally:
                self._load_last_event_at = time.monotonic()
                _trace(trace, "worker exiting")
                try:
                    trace.close()
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True, name="dizel-model-loader").start()
        return True

    def _start_load_watchdog(self) -> None:
        """Dump every thread's stack if the loader stops reporting progress.

        A wedged native call leaves the UI alive but silent, and no Python-level
        handler can report it. ``faulthandler`` can, from a daemon thread, and
        its dump is the only evidence that distinguishes a blocked native load
        from a lost worker thread.
        """
        stop = threading.Event()
        self._load_watchdog_stop = stop
        path = _log_path("tui_load_stacks.log")

        def _watch() -> None:
            dumped_at = 0.0
            while not stop.wait(1.0):
                idle = time.monotonic() - self._load_last_event_at
                if idle < LOAD_STALL_SECONDS or idle - dumped_at < LOAD_STALL_SECONDS:
                    continue
                dumped_at = idle
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(
                            f"\n=== stalled load · {time.strftime('%Y-%m-%dT%H:%M:%S')}"
                            f" · {idle:.0f}s without progress ===\n"
                        )
                        handle.flush()
                        faulthandler.dump_traceback(file=handle, all_threads=True)
                except Exception:
                    pass

        threading.Thread(target=_watch, daemon=True, name="dizel-load-watchdog").start()

    def _stop_load_watchdog(self) -> None:
        stop = self._load_watchdog_stop
        self._load_watchdog_stop = None
        if stop is not None:
            stop.set()

    def _poll_loader_events(self) -> None:
        events = self._loader_events
        if events is None:
            self._stop_loader_poll()
            return

        finished = False
        while not finished:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break

            if kind == "progress":
                self._load_last = str(payload)
                # Timestamped to the log file: if a load hangs on someone
                # else's machine, this is the record of where it stopped.
                print(
                    f"[load] {time.monotonic() - self._load_started:7.2f}s  {payload}",
                    flush=True,
                )
                self._log_system(self._load_last)
            elif kind == "ready":
                finished = True
                self._on_model_ready()
            elif kind == "error":
                finished = True
                self._on_model_error(str(payload))

        if finished:
            self._stop_loader_poll()
            self._loader_events = None
            return

        # A silent loader looks like a frozen app. Counting the seconds on screen
        # proves the UI is alive even when the load is grinding through 1 GB.
        elapsed = int(time.monotonic() - self._load_started)
        if elapsed != self._load_tick:
            self._load_tick = elapsed
            self._log_system(f"{self._load_last} · {elapsed}s")

    def _stop_loader_poll(self) -> None:
        self._stop_load_watchdog()
        timer = self._loader_timer
        self._loader_timer = None
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass

    def _on_model_ready(self) -> None:
        self.generation_state = "IDLE"
        info = self.chat_bridge.manager.model_info or {}

        requested_device = str(getattr(self, "_device", "cpu"))
        actual_device = info.get("device")
        if actual_device:
            self._device = str(actual_device)
        if (
            requested_device == "cuda"
            and self._device == "cpu"
            and str(info.get("provider", "")).startswith("llama.cpp")
        ):
            self._log_system(
                "GPU offload unavailable in this llama.cpp build — running on CPU.", "error"
            )

        if self._checkpoint:
            self.active_model = model_display_name(self._checkpoint)

        ctx_len = int(info.get("ctx_len", 2048) or 2048)
        self.context_tokens = 0
        self.budget_tokens = 0
        self.usage_manager.set_capacity(ctx_len)
        # The new checkpoint may belong to a different family (Dizel, Mila or
        # a generic model), so re-resolve the tier/mode profile now.
        self._apply_profile()

        label = self.active_model or "Model"
        self._log_system(f"{label} ready · ctx {ctx_len:,} · {self._device}", "ok")

    def _on_model_error(self, message: str) -> None:
        self.generation_state = "IDLE"
        manager = self.chat_bridge.manager
        previous = getattr(manager, "checkpoint_path", "") or ""
        if previous and getattr(manager, "is_ready", False):
            # The failed load never touched the old backend, so put the UI back
            # on it instead of leaving the new filename in the header.
            self._checkpoint = previous
            self.active_model = model_display_name(previous)
            self._apply_profile()
            self._log_system(
                f"Load error: {message} — previous model is still loaded.", "error"
            )
            return
        self._log_system(f"Load error: {message}", "error")

    # ── Tier and mode ──────────────────────────────────────────────────
    @property
    def model_family(self) -> str:
        """Profile family for the loaded local model; empty for providers."""
        if self.active_provider != "local":
            return ""
        return detect_model_family(self.active_model)

    @property
    def is_dizel_model(self) -> bool:
        return self.model_family == "Dizel"

    @property
    def response_label(self) -> str:
        """What the footer calls the active response profile."""
        return self.active_tier if self.is_dizel_model else self.active_mode

    def _profile_family(self) -> str:
        family = self.model_family
        if not family:
            return ""
        return self.active_tier if family == "Dizel" else family

    def _apply_profile(self) -> None:
        """Tiers and modes are local-model concepts; API models keep their own."""
        if self.active_provider != "local":
            return
        family = self._profile_family()
        if not family:
            return
        try:
            self.chat_bridge.manager.apply_profile(family, self.active_mode)
        except Exception:
            pass

    def set_tier(self, tier: str, announce: bool = True) -> None:
        if not self.is_dizel_model:
            if announce:
                self._log_system(
                    "Tier applies to Dizel models only — use /mode for this model.",
                    "error",
                )
            return
        if tier not in TIERS:
            tier = TIERS[0]
        self.active_tier = tier
        self._apply_profile()
        if announce:
            self._log_system(f"{tier} · {self.active_mode} mode")

    def set_mode(self, mode: str, announce: bool = True) -> None:
        """Change the response mode and apply its prompt/sampling profile."""
        if mode not in MODES:
            mode = MODES[0]
        self.active_mode = mode
        self._apply_profile()
        if announce:
            label = f"{self.active_tier} · {mode} mode" if self.is_dizel_model else f"{mode} mode"
            self._log_system(label)

    def action_switch_agent(self) -> None:
        current = self.active_mode if self.active_mode in MODES else MODES[0]
        self.set_mode(MODES[(MODES.index(current) + 1) % len(MODES)])

    def action_switch_agent_back(self) -> None:
        current = self.active_mode if self.active_mode in MODES else MODES[0]
        self.set_mode(MODES[(MODES.index(current) - 1) % len(MODES)])

    # ── Helpers used by commands ───────────────────────────────────────
    def refresh_sessions(self) -> None:
        try:
            self.query_one(SessionPanel).update_sessions(force=True)
        except Exception:
            pass
        self.refresh_tabs()

    def refresh_tabs(self) -> None:
        try:
            self.query_one(SessionTabs).refresh_tabs()
        except Exception:
            pass

    def tab_title(self, session_id: str) -> str:
        text = ""
        try:
            for session in self.session_bridge.get_all():
                if session.get("id") == session_id:
                    text = " ".join(str(session.get("title") or "").split())
                    break
        except Exception:
            pass
        if (not text or text == "Untitled") and session_id == self.session_id:
            for message in self.session_bridge.current_messages:
                if message.get("role") == "user" and message.get("content"):
                    text = " ".join(str(message["content"]).split())
                    break
        if not text or text == "Untitled":
            return "New chat"
        return text[:22] + ("…" if len(text) > 22 else "")

    def close_active_tab(self) -> None:
        self.close_tab()

    def close_tab(self, session_id: str | None = None) -> None:
        if self._modal_open():
            return
        if self.generation_state in ("STREAMING", "THINKING"):
            self._log_system("Stop the current reply before closing a tab.", "error")
            return
        target = session_id or self.session_id
        if not target or target not in self.open_tabs:
            return
        was_active = target == self.session_id
        index = self.open_tabs.index(target)
        self.open_tabs.remove(target)
        if not was_active:
            self.refresh_tabs()
            return
        if not self.open_tabs:
            self.action_new_session()
            return
        next_id = self.open_tabs[min(index, len(self.open_tabs) - 1)]
        self.load_session_to_workspace(next_id)
        self.refresh_tabs()

    def forget_tab(self, session_id: str) -> None:
        if session_id in self.open_tabs:
            self.open_tabs.remove(session_id)
        self._tab_messages.pop(session_id, None)
        self.refresh_tabs()

    def refresh_context(self) -> None:
        self.tools_revision += 1

    def _log_system(self, message: str, level: str = "info") -> None:
        """A transient app notice, shown above the composer, never in the transcript."""
        try:
            self.query_one(NoticeLine).show(message, level)
        except Exception:
            pass

    def _start_update_check(self) -> None:
        """Check once per interval and stage a TUI update in the background."""
        try:
            from dizel_core import updater
        except Exception:
            return

        root = str(updater.repo_root())
        try:
            pending = updater.pending_update(root)
            if pending:
                self._log_system(
                    f"Update {pending.get('version')} is staged — restart via dizel.cmd to apply it.",
                    "ok",
                )
        except Exception:
            pass

        try:
            if not updater.should_auto_check(root):
                return
        except Exception:
            return
        if self._update_thread and self._update_thread.is_alive():
            return

        def notify(message: str, level: str = "info") -> None:
            try:
                self.call_from_thread(self._log_system, message, level)
            except Exception:
                pass

        def worker() -> None:
            try:
                result = updater.check_for_update(root=root, timeout=12.0)
                if not result.available or not result.info:
                    return
                info = result.info
                if not info.tui:
                    notify(
                        f"Dizel {info.version} is available, but this release has no terminal package."
                    )
                    return
                notify(f"Dizel {info.version} is available — downloading in the background…")
                last = [-20]

                def progress(received: int, total: int) -> None:
                    if not total:
                        return
                    percent = int(received * 100 / total)
                    if percent < last[0] + 20 and percent != 100:
                        return
                    last[0] = percent
                    notify(f"Downloading Dizel {info.version} · {percent}%")

                updater.download_tui_update(info, root=root, progress=progress)
                notify(
                    f"Update {info.version} ready — restart via dizel.cmd to apply it.",
                    "ok",
                )
            except Exception:
                # A background check must never interrupt a working session.
                try:
                    updater.write_state(root, last_check=time.time())
                except Exception:
                    pass

        thread = threading.Thread(target=worker, daemon=True, name="dizel-update-check")
        self._update_thread = thread
        thread.start()

    def _focus_prompt(self) -> None:
        if not chat_pref(self, "auto_focus_composer", True):
            return
        try:
            self.query_one("#prompt-input").focus()
        except Exception:
            pass

    # ── Layout ─────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        from inference.tui.panels.artifact_panel import ArtifactPanel

        yield SessionTabs()
        with Horizontal(id="main-layout"):
            yield SessionPanel()
            with Container(id="workspace-container"):
                # The transcript is held to the same measure as the composer, so
                # the two read as one column instead of two unrelated widths.
                with Horizontal(id="transcript-row"):
                    yield WorkspacePanel()
                yield NoticeLine()
                # A row exists purely to centre the composer: a vertical
                # container stretches children, so `align` has nothing to centre.
                from inference.tui.panels.slash_drawer import SlashDrawer

                # Above the composer, not beside it — the drawer belongs to the
                # prompt the way a menu belongs to a menu button.
                with Horizontal(id="drawer-row"):
                    yield SlashDrawer()
                with Horizontal(id="composer-row"):
                    yield InputBar()
            yield ArtifactPanel()
            yield ContextPanel()
        yield StatusBar()

    def load_session_to_workspace(self, session_id: str, announce: bool = True) -> bool:
        if not session_id:
            return False
        if session_id != self.session_id and self.generation_state in ("STREAMING", "THINKING"):
            self._log_system("Stop the current reply before switching tabs.", "error")
            return False
        if self.session_id and self.session_id != session_id:
            self._tab_messages[self.session_id] = list(self.session_bridge.current_messages)
        loaded = self.session_bridge.load(session_id)
        if not loaded:
            cached = self._tab_messages.get(session_id)
            if cached is None:
                return False
            # A new chat has no file until its first message. The tab keeps it
            # alive in memory so switching away and back is lossless.
            self.session_id = session_id
            self.session_bridge.current_messages = list(cached)
        else:
            self._tab_messages[session_id] = list(self.session_bridge.current_messages)
        if session_id not in self.open_tabs:
            self.open_tabs.append(session_id)

        messages = self.session_bridge.current_messages
        self.context_tokens = sum(
            max(1, len(str(m.get("content", ""))) // 4) for m in messages
        )
        self.budget_tokens = 0
        # Keep the model's own context in step with the transcript on screen.
        try:
            self.chat_bridge.manager.load_history(
                [{"role": m["role"], "content": m["content"]} for m in messages]
            )
        except Exception:
            pass

        workspace = self.query_one(WorkspacePanel)
        # An empty tab is still a new chat: keep the welcome visible. The
        # transcript only replaces it once the session has messages.
        workspace.clear_workspace(show_empty=not messages)
        full_assistant_text = []
        for msg in messages:
            if msg["role"] == "assistant":
                workspace.mount(HistoryMessage(msg["content"], ts=msg.get("ts")))
                full_assistant_text.append(msg["content"])
            else:
                workspace.mount(
                    Static(
                        MessageBlock(
                            msg["role"].upper(),
                            msg["content"],
                            **message_style(self, timestamp=msg.get("ts")),
                        )
                    )
                )
        workspace.scroll_end(animate=False)

        try:
            from inference.tui.panels.artifact_panel import ArtifactPanel

            self.query_one(ArtifactPanel).add_artifacts("\n".join(full_assistant_text))
        except Exception:
            pass

        if announce and messages:
            self._log_system(f"Loaded {len(messages)} messages from this session.")
        self.refresh_tabs()
        self._focus_prompt()
        return True

    # ── Actions ────────────────────────────────────────────────────────
    def action_new_session(self) -> None:
        if self._modal_open():
            return
        if self.generation_state in ("STREAMING", "THINKING"):
            self._log_system("Stop the current reply first (esc).", "error")
            return
        self.create_empty_session()
        self._focus_prompt()

    def create_empty_session(self) -> None:
        try:
            self.chat_bridge.manager.load_history([])
        except Exception:
            pass
        self.session_bridge.create()
        if self.session_id:
            self._tab_messages[self.session_id] = []
            if self.session_id not in self.open_tabs:
                self.open_tabs.append(self.session_id)
        self.query_one(WorkspacePanel).clear_workspace()
        self._clear_artifacts()
        self.refresh_sessions()

    def _clear_artifacts(self) -> None:
        try:
            from inference.tui.panels.artifact_panel import ArtifactPanel

            self.query_one(ArtifactPanel).clear()
        except Exception:
            pass

    def action_clear_workspace(self) -> None:
        if self._modal_open():
            return
        self.query_one(WorkspacePanel).clear_workspace()
        self._clear_artifacts()

    def action_multiline_input(self, send: bool = False) -> None:
        from inference.tui.panels.multiline_input import MultilineInputModal
        from inference.tui.panels.input_bar import PromptInput

        if self._modal_open():
            return

        try:
            current_text = self.query_one("#prompt-input", PromptInput).value
        except Exception:
            return

        def _on_dismiss(text: str | None) -> None:
            if text is None:
                return
            try:
                inp = self.query_one("#prompt-input", PromptInput)
                if send:
                    # Leave the draft in place: on_submit clears it only once
                    # the send is accepted, so a busy rejection loses nothing.
                    inp.value = text
                    inp.focus()
                    inp.post_message(PromptInput.SubmittedFromEditor(inp, text))
                else:
                    inp.value = text
                    inp.cursor_position = len(text)
                    inp.focus()
            except Exception:
                pass

        self.push_screen(MultilineInputModal(current_text, send_on_save=send), _on_dismiss)

    def action_interrupt_or_quit(self) -> None:
        if self.generation_state in ("STREAMING", "THINKING") or self.chat_bridge.manager.is_generating:
            self.action_interrupt()
            return

        # Quitting mid-thought is the one keystroke nobody means to press twice.
        # Armed once, it needs a second press inside three seconds.
        from inference.tui.settings_apply import pref

        if pref(self, "confirm_quit", True):
            now = time.monotonic()
            if now - getattr(self, "_quit_armed", 0.0) > 3.0:
                self._quit_armed = now
                try:
                    self.query_one("StatusBar").set_hint("press ctrl+c again to quit")
                except Exception:
                    pass
                return

        self.chat_bridge.stop()
        self.exit()

    def action_interrupt(self) -> None:
        if self.generation_state not in ("STREAMING", "THINKING"):
            return
        self.chat_bridge.stop()
        self.generation_state = "IDLE"
        self._log_system("Interrupted.")

    def _modal_open(self) -> bool:
        from textual.screen import ModalScreen

        return isinstance(self.screen, ModalScreen)

    def insert_prompt(self, text: str) -> None:
        """Put text in the composer and leave the cursor at the end."""
        from textual.widgets import Input

        try:
            prompt = self.query_one("#prompt-input", Input)
            prompt.value = text
            prompt.cursor_position = len(text)
            prompt.focus()
        except Exception:
            pass

    def action_toggle_palette(self) -> None:
        from inference.tui.panels.modals.command_palette import CommandPaletteModal

        if isinstance(self.screen, CommandPaletteModal):
            self.screen.dismiss(None)
            return
        if self._modal_open():
            return
        self.push_screen(
            CommandPaletteModal(),
            lambda text: self.insert_prompt(text) if text else None,
        )

    def action_toggle_session(self) -> None:
        if self._modal_open():
            return
        # A manual toggle wins over the width breakpoint for the rest of the session.
        self._sessions_auto = False
        self.query_one(SessionPanel).toggle_panel()

    def _transcript_text(self) -> str:
        """Every assistant reply in the open chat, for rebuilding a panel."""
        try:
            messages = self.session_bridge.current_messages
        except Exception:
            return ""
        return "\n".join(
            m.get("content", "")
            for m in messages
            if m.get("role") == "assistant" and m.get("content")
        )

    def _dismiss_overlay(self) -> bool:
        """Close an open overlay; True when there was one."""
        if isinstance(self.screen, PanelOverlay):
            self.screen.dismiss()
            return True
        return False

    def action_toggle_context(self) -> None:
        from inference.tui.panels.context_panel import ContextPanel

        if self._dismiss_overlay():
            return
        if self._modal_open():
            return
        self.push_screen(PanelOverlay(ContextPanel()))

    def action_toggle_artifacts(self) -> None:
        from inference.tui.panels.artifact_panel import ArtifactPanel

        if self._dismiss_overlay():
            return
        if self._modal_open():
            return
        text = self._transcript_text()
        self.push_screen(
            PanelOverlay(ArtifactPanel(), lambda panel: panel.add_artifacts(text))
        )


if __name__ == "__main__":
    app = DizelCMDApp()
    app.run()
