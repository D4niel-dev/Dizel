# dizel_ui/main.py

import argparse
import os
import sys
import threading

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QProgressBar, QFileDialog
from PySide6.QtCore import Qt, QTimer, QSize, Slot, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QPixmap

# ── Make project root importable ──────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_DIR = os.path.dirname(_HERE)
_PROJ_ROOT     = os.path.dirname(_INFERENCE_DIR)

if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

from dizel_ui.ui.chat_window     import ChatWindow
from dizel_ui.ui.sidebar         import Sidebar
from dizel_ui.ui.secondary_sidebar import SecondarySidebar
from dizel_ui.ui.input_panel     import InputPanel
from dizel_ui.ui.settings_dialog import SettingsDialog
from dizel_ui.ui.profile_dialog  import ProfileDialog
from dizel_ui.ui.command_palette import CommandPalette
from dizel_ui.logic.chat_manager import ChatManager
from dizel_ui.logic.history_manager import (
    save_session, load_session, list_sessions,
    delete_session, new_session_id, toggle_pin_session
)
from dizel_ui.logic.config_manager import ConfigManager
from dizel_ui.logic.usage_manager import UsageManager
from dizel_ui.theme.colors import (
    BG_ROOT, BG_CHAT, BG_INPUT, ACCENT, TEXT_PRIMARY, TEXT_DIM,
    ACTION_PILL, SIDEBAR_BTN_HOVER, SIDEBAR_BORDER, ACCENT_LIGHT, resolve
)
from dizel_ui.theme.fonts import LABEL, BTN_LABEL, LABEL_SM
from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.theme_manager import Theme
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style

try:
    from model.dizel_info import MODEL_NAME, VERSION
except ImportError:
    MODEL_NAME = "Dizel"
    VERSION = "v1.0"

class DizelApp(QMainWindow):
    MIN_W = 860
    MIN_H = 560
    _invoke_in_main = Signal(object)  # thread-safe main-thread dispatch

    def __init__(self, checkpoint: str = "", device: str = "cpu"):
        super().__init__()
        self._invoke_in_main.connect(lambda fn: fn())

        self._chat_mgr = ChatManager()
        self._checkpoint = checkpoint
        self._device = device
        self._session_id = new_session_id()
        self._model_loaded = False
        self._usage_mgr = UsageManager()

        self.setWindowTitle("Dizel AI")
        self.resize(1100, 700)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self.setStyleSheet(f"QMainWindow {{ background-color: {resolve(BG_ROOT)}; }}")
        
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._animate_loading)
        self._loading_dots = 0
        self._is_loading = False
        
        try:
            ico_path = os.path.join(_HERE, "assets", "app", "Dizel.ico")
            png_path = os.path.join(_HERE, "assets", "app", "Dizel.png")
            if os.path.exists(png_path):
                self.setWindowIcon(QIcon(png_path))
            elif os.path.exists(ico_path):
                self.setWindowIcon(QIcon(ico_path))
        except Exception as e:
            print(f"Failed to load application window icon: {e}")

        cfg = ConfigManager.load()
        if not checkpoint:
            checkpoint = cfg.get("checkpoint", "")
        if device == "cpu" and cfg.get("device") in ["cpu", "cuda"]:
            device = cfg.get("device")

        self._checkpoint = checkpoint
        self._device = device
        
        app_cfg = cfg.get("appearance", {})
        cm = app_cfg.get("color_mode", "system").lower().replace(" ", "_").replace("dark_blue", "dark_blue")
        if cm in ["system", "light", "dark", "dark_blue"]:
            Theme.set_mode(cm)
        else:
            Theme.set_mode("dark")

        # Set up drop target natively
        self.setAcceptDrops(True)

        self._build_layout()
        self._refresh_history()

        if self._checkpoint:
            QTimer.singleShot(200, lambda: self._load_model_async(self._checkpoint, self._device))
        else:
            self._show_status("  No checkpoint loaded — open Settings to select one.", dim=True)

    def _build_layout(self):
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._sidebar = Sidebar(
            on_new_chat=self._new_chat,
            on_toggle_chats=self._toggle_secondary_sidebar,
            on_workspace_view=self._on_workspace_view,
            on_session_select=self._load_session,
            on_session_delete=self._delete_session,
            on_pin=self._on_pin,
            on_settings=self._open_settings,
            on_profile_click=self._on_profile_click,
            on_action=lambda msg: self._show_status(msg),
            parent=self.central_widget
        )
        main_layout.addWidget(self._sidebar)
        
        # Secondary Context Sidebar
        self._secondary_sidebar = SecondarySidebar(
            on_session_select=self._load_session,
            on_session_delete=self._delete_session,
            on_pin=self._on_pin,
            on_import=self._import_session,
            parent=self.central_widget
        )
        main_layout.addWidget(self._secondary_sidebar)

        # Right Area
        right_area = QFrame(self.central_widget)
        right_area.setStyleSheet(get_frame_style(BG_CHAT, radius=0))
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Status Clip Container (Animated Height)
        self._status_clip = QWidget(right_area)
        self._status_clip.setMaximumHeight(48)
        clip_lyt = QVBoxLayout(self._status_clip)
        clip_lyt.setContentsMargins(0, 0, 0, 0)
        clip_lyt.setSpacing(0)
        
        self._status_bar = QFrame(self._status_clip)
        self._status_bar.setFixedHeight(48)
        self._status_bar.setStyleSheet("background: transparent;")
        clip_lyt.addWidget(self._status_bar, alignment=Qt.AlignBottom)
        
        status_layout = QHBoxLayout(self._status_bar)
        status_layout.setContentsMargins(24, 12, 24, 0)
        
        left_pill = QFrame(self._status_bar)
        left_pill.setFixedHeight(32)
        left_pill.setStyleSheet(get_frame_style(ACTION_PILL, radius=16))
        left_pill_layout = QHBoxLayout(left_pill)
        left_pill_layout.setContentsMargins(12, 0, 12, 0)
        
        self._status_lbl = QLabel(f"{MODEL_NAME} {VERSION} ⌵", left_pill)
        self._status_lbl.setFont(LABEL_SM)
        self._status_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent;")
        left_pill_layout.addWidget(self._status_lbl)
        
        self._model_info_lbl = QLabel("", left_pill)
        self._model_info_lbl.setFont(LABEL_SM)
        self._model_info_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent;")
        left_pill_layout.addWidget(self._model_info_lbl)
        
        status_layout.addWidget(left_pill)
        status_layout.addStretch(1)

        # Context Counter (Token Usage)
        self._ctx_widget = QFrame(self._status_bar)
        self._ctx_widget.setFixedHeight(32)
        self._ctx_widget.setStyleSheet(get_frame_style(BG_INPUT, radius=16))
        ctx_layout = QHBoxLayout(self._ctx_widget)
        ctx_layout.setContentsMargins(12, 0, 12, 0)
        ctx_layout.setSpacing(8)
        
        ctx_ico_lbl = QLabel(self._ctx_widget)
        ctx_ico = get_icon("cpu", size=(14, 14), color=TEXT_DIM)
        if ctx_ico: ctx_ico_lbl.setPixmap(ctx_ico.pixmap(14, 14))
        ctx_layout.addWidget(ctx_ico_lbl)
        
        self._token_progress = QProgressBar(self._ctx_widget)
        self._token_progress.setFixedWidth(80)
        self._token_progress.setFixedHeight(4)
        self._token_progress.setTextVisible(False)
        self._token_progress.setRange(0, 100)
        self._token_progress.setValue(self._usage_mgr.percentage)
        self._token_progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {resolve(BG_ROOT)};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {resolve(ACCENT_LIGHT)};
                border-radius: 2px;
            }}
        """)
        ctx_layout.addWidget(self._token_progress)
        
        self._token_lbl = QLabel(f"{self._usage_mgr.percentage}%", self._ctx_widget)
        self._token_lbl.setFont(LABEL_SM)
        self._token_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent;")
        ctx_layout.addWidget(self._token_lbl)
        
        status_layout.addWidget(self._ctx_widget)
        
        status_layout.addSpacing(16)
        
        export_btn = QPushButton(" Export  ", self._status_bar)
        export_btn.setFixedHeight(32)
        export_btn.setLayoutDirection(Qt.RightToLeft)
        export_ico = get_icon("external-link", size=(14, 14), color=TEXT_PRIMARY)
        if export_ico: export_btn.setIcon(export_ico)
        export_btn.setFont(LABEL_SM)
        export_btn.setStyleSheet(get_button_style(ACTION_PILL, SIDEBAR_BTN_HOVER, TEXT_PRIMARY, radius=16))
        export_btn.setCursor(Qt.PointingHandCursor)
        export_btn.clicked.connect(self._export_chat)
        status_layout.addWidget(export_btn)
        
        config_btn = QPushButton(" Configuration  ", self._status_bar)
        config_btn.setFixedHeight(32)
        config_btn.setLayoutDirection(Qt.RightToLeft)
        config_ico = get_icon("settings", size=(14, 14), color=TEXT_PRIMARY)
        if config_ico: config_btn.setIcon(config_ico)
        config_btn.setFont(LABEL_SM)
        config_btn.setStyleSheet(get_button_style(ACTION_PILL, SIDEBAR_BTN_HOVER, TEXT_PRIMARY, radius=16))
        config_btn.setCursor(Qt.PointingHandCursor)
        config_btn.clicked.connect(self._open_settings)
        status_layout.addWidget(config_btn)
        
        right_layout.addWidget(self._status_clip)

        # Toggle Chevron
        self._status_toggle_wrap = QWidget(right_area)
        self._status_toggle_wrap.setFixedHeight(24)
        toggle_lyt = QHBoxLayout(self._status_toggle_wrap)
        toggle_lyt.setContentsMargins(0, 0, 0, 0)
        toggle_lyt.setSpacing(0)
        
        self._status_toggle_btn = QPushButton(self._status_toggle_wrap)
        self._status_toggle_btn.setFixedSize(24, 24)
        chevron_ico = get_icon("chevron-up", size=(16, 16), color=TEXT_DIM)
        if chevron_ico: self._status_toggle_btn.setIcon(chevron_ico)
        self._status_toggle_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, "transparent", radius=12))
        self._status_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._status_toggle_btn.clicked.connect(self._toggle_status_bar)
        
        # Center horizontally, move down slightly by taking top spacing from right layout
        toggle_lyt.addWidget(self._status_toggle_btn, alignment=Qt.AlignCenter)
        right_layout.addWidget(self._status_toggle_wrap)
        
        # State & Animations for Status Bar
        self._status_is_open = True
        self.status_anim = QPropertyAnimation(self._status_clip, b"maximumHeight")
        self.status_anim.setEasingCurve(QEasingCurve.InOutQuart)
        self.status_anim.setDuration(250)

        # Text Input Area
        self._chat_window = ChatWindow(
            on_quick_action=self._on_quick_action,
            on_regenerate=self._on_regenerate,
            parent=right_area
        )
        right_layout.addWidget(self._chat_window, stretch=1)

        # Input Panel
        self._input_panel = InputPanel(
            on_send=self._on_send,
            on_stop=self._on_stop,
            on_settings=self._open_settings,
            on_attach=self._do_attach,
            on_voice=self._do_voice,
            parent=right_area
        )
        right_layout.addWidget(self._input_panel)
        
        # Load user profile UI details
        self._update_profile_ui()
        
        # Removed Avatar typing state tracking

        main_layout.addWidget(right_area, stretch=1)

    # ── PySide6 Drag & Drop natively ───────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self._input_panel.add_attachment(url.toLocalFile())

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_N:
                self._new_chat()
            elif event.key() == Qt.Key_Comma:
                self._open_settings()
            elif event.key() == Qt.Key_W:
                self.close()
            elif event.key() == Qt.Key_K:
                self._open_command_palette()
        super().keyPressEvent(event)

    # ── Helper for thread-safety ──────────────────────────────────────────
    def run_in_main(self, fn):
        """Schedule fn to run on the main/GUI thread (safe from any thread)."""
        self._invoke_in_main.emit(fn)

    def _open_command_palette(self):
        commands = [
            {"label": "New Chat", "icon": "plus-square", "hook": self._new_chat},
            {"label": "Settings", "icon": "settings", "hook": self._open_settings},
            {"label": "Clear Chat Output", "icon": "trash-2", "hook": self._chat_window.clear},
            {"label": "Quit", "icon": "x-circle", "hook": self.close},
        ]
        
        palette = CommandPalette(commands, parent=self)
        # Center over the main window
        geom = self.geometry()
        x = geom.x() + (geom.width() - palette.width()) // 2
        y = geom.y() + (geom.height() - palette.height()) // 4
        palette.move(x, y)
        palette.exec()

    def _show_status(self, msg: str, dim: bool = False):
        color = TEXT_DIM if dim else TEXT_PRIMARY
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color: {resolve(color)}; background: transparent;")

    def _update_model_info(self):
        info = self._chat_mgr.model_info
        if not info:
            self._model_info_lbl.setText("")
            return
        txt = (
            f"{MODEL_NAME} {VERSION}  •  {info.get('params','?')}  •  "
            f"d={info.get('d_model','?')}  L={info.get('n_layers','?')}  •  "
            f"{self._chat_mgr._device.upper()}"
        )
        self._model_info_lbl.setText(txt)

    # ── Model Loading ─────────────────────────────────────────────────────
    def _animate_loading(self):
        if not self._is_loading:
            self._loading_timer.stop()
            return
        self._loading_dots = (self._loading_dots + 1) % 4
        dots = "." * self._loading_dots
        self._show_status(f"Loading model{dots}")

    def _load_model_async(self, checkpoint: str, device: str):
        self._is_loading = True
        self._show_status("Loading model")
        self._loading_timer.start(500)

        def _worker():
            try:
                self._chat_mgr.load_model(
                    checkpoint_path=checkpoint,
                    device=device,
                    on_progress=lambda msg: self.run_in_main(lambda m=msg: self._show_status(m)),
                )
                self._model_loaded = True
                self.run_in_main(self._on_model_ready)
            except Exception as exc:
                self.run_in_main(lambda e=exc: self._on_model_error(str(e)))
            finally:
                self._is_loading = False

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_ready(self):
        self._show_status("Model ready.", dim=True)
        self._update_model_info()
        
        # Detect context capacity from model info
        info = self._chat_mgr.model_info
        if info:
            ctx_len = info.get("ctx_len", 1024)
            self._usage_mgr.set_capacity(ctx_len)
            self._update_usage_display()
            
        self._input_panel.focus_input()

    def _on_model_error(self, msg: str):
        self._show_status(f"Error: {msg}")
        self._chat_window.show_error(msg)
        
    def _do_attach(self):
        from PySide6.QtWidgets import QFileDialog
        fpath, _ = QFileDialog.getOpenFileName(self, "Select File to Attach")
        if fpath:
            self._input_panel.add_attachment(fpath)

    def _do_voice(self):
        self._show_status("Listening... (Speak now)", dim=False)
        QTimer.singleShot(2000, lambda: self._input_panel._input.append("Hello Dizel, how are you?"))
        QTimer.singleShot(2000, lambda: self._show_status("Transcribed.", dim=True))

    def _on_quick_action(self, prompt: str):
        self._on_send(prompt)

    def _on_send(self, text: str, files: list = None):
        if not self._chat_mgr.is_ready:
            self._show_status("⚠ Model not loaded — open Settings (⚙) to select a checkpoint.")
            return
        if self._chat_mgr.is_generating:
            self._show_status("⚠ Still generating — please wait or press Stop.")
            return

        self._input_panel.set_generating(True)

        # Read active tool modes from UI
        active = self._input_panel.get_active_contexts()

        from core.tool_state import ToolState
        state = ToolState.from_ui(active, text, files)

        def _do_generate():
            if state.has_active_tools():
                threading.Thread(
                    target=self._tool_pipeline, args=(state,), daemon=True
                ).start()
            else:
                self._show_status("Generating…")
                self._start_generation(text, files)
                
        self._chat_window.begin_response_flow(text, files, on_ready=_do_generate)

        # Since it takes time for the animation, we shouldn't immediately load tools
        # until the flow reaches the processing step (handled by begin_response_flow)

    def _tool_pipeline(self, state):
        """Background thread: run tool preprocessing → build prompt → generate."""
        try:
            from core.router import route_request
            from core.prompt_builder import build_tool_prompt

            # Step 1: File parsing
            if state.parse_files_enabled and state.uploaded_files:
                self.run_in_main(lambda: self._show_status("Parsing files…"))
                self._usage_mgr.add_usage(50.0 * len(state.uploaded_files)) # Complexity: 50 per file
                self.run_in_main(self._update_usage_display)

            # Step 2: Web search
            if state.web_search_enabled:
                self.run_in_main(lambda: self._show_status("Searching the web…"))
                self._usage_mgr.add_usage(80.0) # Complexity: 80 per search
                self.run_in_main(self._update_usage_display)

            # Execute all tool pipelines
            route_request(state)

            # Build final prompt with tool context
            final_prompt = build_tool_prompt(state)

            # Generate with tool awareness
            self.run_in_main(
                lambda p=final_prompt, dt=state.deep_think_enabled, f=state.uploaded_files: (
                    self._start_generation_with_tools(p, dt, f)
                )
            )

        except Exception as e:
            self.run_in_main(lambda m=str(e): self._generation_error(m))

    def _start_generation(self, text: str, files: list = None):
        """Direct generation — no tools active."""
        self._show_status("Generating…")
        on_token, on_done, on_error = self._make_gen_callbacks(deep_think=False)
        self._chat_mgr.send_message(
            user_text=text,
            attachments=files,
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

    def _start_generation_with_tools(self, text: str, deep_think: bool, files: list = None):
        """Tool-augmented generation — may apply Deep Think overrides."""
        self._show_status("Generating…")
        on_token, on_done, on_error = self._make_gen_callbacks(deep_think=deep_think)
        self._chat_mgr.send_message_with_tools(
            augmented_text=text,
            deep_think=deep_think,
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

    def _make_gen_callbacks(self, deep_think: bool = False):
        """Create the standard token/done/error callbacks."""
        def on_token(piece: str):
            def _update_tokens():
                # Usage logic: Normal=0.5, DeepThink=1.0 units per token
                cost = 1.0 if deep_think else 0.5
                self._usage_mgr.add_usage(cost)
                self._update_usage_display()
                
            self.run_in_main(_update_tokens)
            self.run_in_main(lambda p=piece: self._chat_window.append_token(p))

        def on_done(full_text: str):
            self.run_in_main(lambda t=full_text: self._finish_generation(t))

        def on_error(msg: str):
            self.run_in_main(lambda m=msg: self._generation_error(m))

        return on_token, on_done, on_error

    def _update_usage_display(self):
        """Update the status bar usage percentage and progress bar."""
        perc = self._usage_mgr.percentage
        self._token_progress.setValue(perc)
        self._token_lbl.setText(f"{perc}%")
        
        # Color coding: Green -> Yellow -> Red
        if perc < 60:
            color = resolve(ACCENT_LIGHT)
        elif perc < 85:
            color = "#FFB302" # Warning Yellow
        else:
            color = "#FF4D4D" # Danger Red
            
        self._token_progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {resolve(BG_ROOT)};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)

    def _finish_generation(self, full_text: str):
        self._chat_window.finish_assistant_message(full_text)
        self._input_panel.set_generating(False)
        self._show_status("", dim=True)
        self._autosave_session()

    def _generation_error(self, msg: str):
        self._chat_window.show_error(msg)
        self._input_panel.set_generating(False)
        self._show_status(f"Error: {msg}")

    def _on_regenerate(self):
        if not self._chat_mgr.is_ready:
            self._show_status("Model not loaded.")
            return
            
        self._chat_window.pop_last_assistant_message()
        self._input_panel.set_generating(True)
        
        on_token, on_done, on_error = self._make_gen_callbacks()
        self._chat_mgr.regenerate_last(
            on_token=on_token,
            on_done=on_done,
            on_error=on_error
        )

    def _on_stop(self):
        self._chat_mgr.stop_generation()
        self._show_status("Stopping…", dim=True)

    def _new_chat(self):
        self._autosave_session()
        self._chat_mgr.new_session()
        self._session_id = new_session_id()
        cfg = ConfigManager.load()
        username = cfg.get("user_profile", {}).get("username", "User")
        self._chat_window.clear(username)
        self._show_status("New chat started.", dim=True)
        self._refresh_history()
        self._input_panel.focus_input()

    def _autosave_session(self):
        history = self._chat_mgr.get_history()
        if not history: return
        try:
            save_session(history, session_id=self._session_id)
            self._refresh_history()
        except Exception:
            pass

    def _load_session(self, session_id: str):
        data = load_session(session_id)
        if not data: return
        self._autosave_session()
        messages = data.get("messages", [])
        self._chat_mgr.new_session()
        self._chat_mgr.load_history(messages)
        self._session_id = session_id

        self._chat_window.clear()
        self._chat_window.set_skip_animations(True)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                self._chat_window.add_user_message_instant(content)
            elif role == "assistant":
                self._chat_window.add_assistant_message_instant(content)
        self._chat_window.set_skip_animations(False)
        self._show_status("Session loaded.", dim=True)

    def _import_session(self, filepath: str):
        import os, json
        messages = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, dict) and "messages" in data:
                        messages = data["messages"]
                    elif isinstance(data, list):
                        messages = data
                elif filepath.endswith('.md'):
                    content = f.read()
                    messages = [{"role": "user", "content": "## Imported Markdown\n\n" + content}]
        except Exception as e:
            self._show_status(f"Import failed: {e}", dim=True)
            return
            
        if messages:
            new_id = new_session_id()
            save_session(messages, session_id=new_id, title=os.path.basename(filepath))
            self._load_session(new_id)
            self._refresh_history()
            self._show_status("Chat imported successfully.", dim=True)
        else:
            self._show_status("No messages found to import.", dim=True)
        self._chat_window.set_skip_animations(False)

        self._show_status(f"Loaded: {data.get('title','')[:40]}", dim=True)

    def _delete_session(self, session_id: str):
        delete_session(session_id)
        if session_id == self._session_id:
            self._chat_mgr.new_session()
            self._session_id = new_session_id()
            cfg = ConfigManager.load()
            username = cfg.get("user_profile", {}).get("username", "User")
            self._chat_window.clear(username)
        self._refresh_history()

    def _on_pin(self, session_id: str):
        toggle_pin_session(session_id)
        self._refresh_history()

    def _toggle_secondary_sidebar(self):
        self._secondary_sidebar.toggle()

    def _on_workspace_view(self, view_name: str):
        self._secondary_sidebar.switch_view(view_name)

    def _refresh_history(self):
        sessions = list_sessions()
        self._sidebar.refresh_history(sessions[:5])
        self._secondary_sidebar.refresh_data(sessions)
    def _export_chat(self):
        history = self._chat_mgr.get_history()
        if not history:
            self._show_status("Nothing to export — start a conversation first.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Chat",
            os.path.join(os.path.expanduser("~"), "Desktop", "dizel_chat"),
            "Markdown (*.md);;JSON (*.json);;Plain Text (*.txt)"
        )
        if not path:
            return

        try:
            if path.endswith(".json"):
                import json
                data = {
                    "id": self._session_id,
                    "messages": history,
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                lines = []
                for msg in history:
                    role = msg.get("role", "unknown").capitalize()
                    content = msg.get("content", "")
                    if path.endswith(".md"):
                        lines.append(f"### {role}\n\n{content}\n")
                    else:
                        lines.append(f"[{role}]\n{content}\n")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

            self._show_status(f"Exported to {os.path.basename(path)}")
        except Exception as e:
            self._show_status(f"Export failed: {e}")

    def _toggle_status_bar(self):
        self._status_is_open = not self._status_is_open
        target_h = 48 if self._status_is_open else 0
        
        self.status_anim.setStartValue(self._status_clip.maximumHeight())
        self.status_anim.setEndValue(target_h)
        self.status_anim.start()
        
        # Flop the icon
        ico_name = "chevron-up" if self._status_is_open else "chevron-down"
        ico = get_icon(ico_name, size=(16, 16), color=TEXT_DIM)
        if ico: self._status_toggle_btn.setIcon(ico)

    def _open_settings(self):
        old_mode = Theme.mode
        dlg = SettingsDialog(parent=self, chat_mgr=self._chat_mgr, on_reload=self._reload_model)
        dlg.exec()
        if Theme.mode != old_mode:
            self._apply_theme()

    def _on_profile_click(self):
        cfg = ConfigManager.load()
        current_profile = cfg.get("user_profile", {"username": "User", "avatar": ""})
        dlg = ProfileDialog(current_profile, parent=self)
        if dlg.exec():
            new_profile = dlg.get_profile()
            cfg["user_profile"] = new_profile
            ConfigManager.save(cfg)
            self._update_profile_ui()

    def _update_profile_ui(self):
        cfg = ConfigManager.load()
        profile = cfg.get("user_profile", {"username": "User", "avatar": ""})
        username = profile.get("username", "User")
        
        avatar_path = ""
        avatar_file = profile.get("avatar", "")
        if avatar_file:
            _HERE = os.path.dirname(os.path.abspath(__file__))
            _DATA_DIR = os.path.join(_HERE, ".dizel")
            avatar_path = os.path.join(_DATA_DIR, avatar_file)
            
        if hasattr(self, '_sidebar') and self._sidebar:
            self._sidebar.set_profile(username, avatar_path)
            
        if hasattr(self, '_chat_window') and self._chat_window:
            if getattr(self._chat_window, "_welcome_shown", False):
                self._chat_window.clear(username)

    def _apply_theme(self):
        old_central = self.centralWidget()
        self.setStyleSheet(f"QMainWindow {{ background-color: {resolve(BG_ROOT)}; }}")
        
        # Suspend history load during layout rebuild
        self._build_layout()
        if old_central:
            old_central.deleteLater()
            
        self._refresh_history()
        
        # Restore active session content
        if self._session_id:
            data = load_session(self._session_id)
            if data:
                messages = data.get("messages", [])
                self._chat_window.clear()
                self._chat_window.set_skip_animations(True)
                for msg in messages:
                    if msg.get("role") == "user":
                        self._chat_window.add_user_message_instant(msg.get("content", ""))
                    elif msg.get("role") == "assistant":
                        self._chat_window.add_assistant_message_instant(msg.get("content", ""))
                self._chat_window.set_skip_animations(False)

    def _reload_model(self):
        ckpt = getattr(self._chat_mgr, "_pending_checkpoint", "")
        device = getattr(self._chat_mgr, "_device", "cpu")
        if ckpt:
            self._checkpoint = ckpt
            self._device = device
            self._load_model_async(ckpt, device)
        else:
            self._show_status("Settings applied.", dim=True)


def parse_args():
    p = argparse.ArgumentParser(description="Dizel AI Desktop Interface")
    p.add_argument("--checkpoint", "-c", type=str, default="", help="Path to .pt model checkpoint")
    p.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    
    # We can apply generic PySide6 app settings here if needed
    
    window = DizelApp(checkpoint=args.checkpoint, device=args.device)
    window.showMaximized()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
