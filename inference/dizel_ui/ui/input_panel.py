# dizel_ui/ui/input_panel.py

import os
from PySide6.QtWidgets import (QFrame, QTextEdit, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QScrollArea, QWidget, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QTimer
from PySide6.QtGui import QIcon, QPixmap, QKeyEvent

from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.colors import (
    BG_INPUT, BG_INPUT_FIELD, SEND_BTN, SEND_BTN_HOVER,
    BORDER, BORDER_FOCUS, TEXT_PRIMARY, TEXT_DIM, TEXT_SECONDARY,
    ACCENT, ACCENT_HOVER, WELCOME_CARD_HOVER, resolve
)
from dizel_ui.theme.fonts import INPUT_TEXT, BTN_LABEL, LABEL_DIM, NAV_ITEM
from dizel_ui.theme.stylesheets import get_frame_style, get_button_style, get_scrollbar_style
from dizel_ui.ui.action_menu import ActionMenu
from dizel_ui.ui.animated_button import AnimatedIconButton

class ModelSelectorPopup(QFrame):
    selected = Signal(str)
    
    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(200)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        container = QFrame(self)
        base_style = get_frame_style(BG_INPUT, radius=12, border_color=BORDER)
        container.setStyleSheet(base_style + "\nQLabel { border: none; background: transparent; }")
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(6, 6, 6, 6)
        c_layout.setSpacing(4)
        
        # Format: Name, Icon
        models = [
            ("Dizel Lite", "zap"),
            ("Dizel Pro", "cpu"),
            ("Mila Lite", "zap"),
            ("Mila Pro", "cpu")
        ]
        for name, ico_name in models:
            btn = QPushButton(container)
            btn.setFixedHeight(36)
            is_active = (name == current)
            
            b_layout = QHBoxLayout(btn)
            b_layout.setContentsMargins(10, 0, 10, 0)
            b_layout.setSpacing(10)
            
            c_text = resolve(ACCENT) if is_active else resolve(TEXT_PRIMARY)
            
            # Icon
            ico_lbl = QLabel(btn)
            i_obj = get_icon(ico_name, size=(16, 16), color=ACCENT if is_active else TEXT_DIM)
            if i_obj: ico_lbl.setPixmap(i_obj.pixmap(16, 16))
            ico_lbl.setStyleSheet("background: transparent;")
            b_layout.addWidget(ico_lbl)
            
            # Text
            t_lbl = QLabel(name, btn)
            t_lbl.setFont(NAV_ITEM)
            t_lbl.setStyleSheet(f"color: {c_text}; background: transparent;")
            b_layout.addWidget(t_lbl)
            
            b_layout.addStretch(1)
            
            # Checkmark
            if is_active:
                chk = QLabel(btn)
                c_obj = get_icon("check", size=(16, 16), color=ACCENT)
                if c_obj: chk.setPixmap(c_obj.pixmap(16, 16))
                chk.setStyleSheet("background: transparent;")
                b_layout.addWidget(chk)
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    outline: none;
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    background-color: {resolve(WELCOME_CARD_HOVER)};
                }}
            """)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, n=name: self._on_choose(n))
            c_layout.addWidget(btn)
            
        layout.addWidget(container)

    def showEvent(self, event):
        super().showEvent(event)
        self.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(150)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def _on_choose(self, name):
        self.selected.emit(name)
        self._out_anim = QPropertyAnimation(self, b"windowOpacity")
        self._out_anim.setDuration(100)
        self._out_anim.setStartValue(self.windowOpacity())
        self._out_anim.setEndValue(0.0)
        self._out_anim.finished.connect(lambda: self.close())
        self._out_anim.start()

class ModelModePopup(QFrame):
    selected = Signal(str)
    
    def __init__(self, current: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(320)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        container = QFrame(self)
        base_style = get_frame_style(BG_INPUT, radius=12, border_color=BORDER)
        container.setStyleSheet(base_style + "\nQLabel { border: none; background: transparent; }")
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(6, 6, 6, 6)
        c_layout.setSpacing(4)
        
        # Format: Name, Icon, Description
        modes = [
            ("Fast", "zap", "For standard queries and chat"),
            ("Planning", "command", "Deep analytical thinking and planning")
        ]
        for name, ico_name, desc in modes:
            btn = QPushButton(container)
            btn.setFixedHeight(48)
            is_active = (name == current)
            
            b_layout = QHBoxLayout(btn)
            b_layout.setContentsMargins(10, 8, 10, 8)
            b_layout.setSpacing(12)
            
            c_text = resolve(ACCENT) if is_active else resolve(TEXT_PRIMARY)
            
            # Icon
            ico_lbl = QLabel(btn)
            i_obj = get_icon(ico_name, size=(18, 18), color=ACCENT if is_active else TEXT_DIM)
            if i_obj: ico_lbl.setPixmap(i_obj.pixmap(18, 18))
            ico_lbl.setStyleSheet("background: transparent;")
            ico_lbl.setAlignment(Qt.AlignTop)
            b_layout.addWidget(ico_lbl)
            
            # Text container
            t_cont = QWidget(btn)
            t_cont.setStyleSheet("background: transparent;")
            t_layout = QVBoxLayout(t_cont)
            t_layout.setContentsMargins(0, 0, 0, 0)
            t_layout.setSpacing(2)
            
            t_lbl = QLabel(name, t_cont)
            t_lbl.setFont(NAV_ITEM)
            t_lbl.setStyleSheet(f"color: {c_text}; background: transparent; font-weight: 600;")
            t_layout.addWidget(t_lbl)
            
            d_lbl = QLabel(desc, t_cont)
            d_lbl.setFont(LABEL_DIM)
            d_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent;")
            t_layout.addWidget(d_lbl)
            
            b_layout.addWidget(t_cont)
            b_layout.addStretch(1)
            
            # Checkmark
            if is_active:
                chk = QLabel(btn)
                c_obj = get_icon("check", size=(16, 16), color=ACCENT)
                if c_obj: chk.setPixmap(c_obj.pixmap(16, 16))
                chk.setStyleSheet("background: transparent;")
                chk.setAlignment(Qt.AlignTop)
                b_layout.addWidget(chk)
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: none;
                    outline: none;
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    background-color: {resolve(WELCOME_CARD_HOVER)};
                }}
            """)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, n=name: self._on_choose(n))
            c_layout.addWidget(btn)
            
        layout.addWidget(container)

    def showEvent(self, event):
        super().showEvent(event)
        self.setWindowOpacity(0.0)
        self._anim = QPropertyAnimation(self, b"windowOpacity")
        self._anim.setDuration(150)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def _on_choose(self, name):
        self.selected.emit(name)
        self._out_anim = QPropertyAnimation(self, b"windowOpacity")
        self._out_anim.setDuration(100)
        self._out_anim.setStartValue(self.windowOpacity())
        self._out_anim.setEndValue(0.0)
        self._out_anim.finished.connect(lambda: self.close())
        self._out_anim.start()

PLACEHOLDER = "Message Dizel…"

class _InputTextEdit(QTextEdit):
    enter_pressed = Signal(bool) # True if shift held
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(INPUT_TEXT)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        c_text = resolve(TEXT_PRIMARY)
        # Transparent background so it blends with the container
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent; 
                color: {c_text};
                border: none;
            }}
        """)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() == Qt.Key_Return or e.key() == Qt.Key_Enter:
            if e.modifiers() & Qt.ShiftModifier:
                # Insert newline
                super().keyPressEvent(e)
            else:
                self.enter_pressed.emit(False)
            return
        super().keyPressEvent(e)

class InputPanel(QFrame):
    def __init__(self, on_send, on_stop, on_settings=lambda: None, on_attach=lambda: None, 
                 on_options=lambda: None, on_voice=lambda: None, parent=None):
        super().__init__(parent)
        self._on_send_msg = on_send
        self._on_stop = on_stop
        self._on_attach_cb = on_attach # keep reference if needed
        self._on_voice = on_voice
        self._generating = False
        self._attachments = []
        self._active_contexts = set()
        self._current_model = "Dizel Lite"
        self._current_mode = "Fast"

        self.setStyleSheet(f"background-color: transparent; border: none;")
        self._build()
        self._setup_action_menu()

    def _build(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 0, 24, 24)
        main_layout.setSpacing(8)

        # 1. Floating Box
        self.box = QFrame(self)
        self.box.setObjectName("InputBox")
        self._apply_box_style(focused=False)
        
        box_layout = QVBoxLayout(self.box)
        box_layout.setContentsMargins(12, 12, 12, 12)
        box_layout.setSpacing(4)

        # 1a. Attachment Chips Row (inline)
        self._preview_area = QScrollArea(self.box)
        self._preview_area.setFixedHeight(32)
        self._preview_area.setWidgetResizable(True)
        self._preview_area.setStyleSheet(get_scrollbar_style(BG_INPUT, BORDER, ACCENT) + "QScrollArea { border: none; background: transparent; }")
        
        self._preview_content = QWidget()
        self._preview_content.setStyleSheet("background: transparent;")
        self._preview_layout = QHBoxLayout(self._preview_content)
        self._preview_layout.setContentsMargins(0, 0, 0, 0)
        self._preview_layout.setSpacing(8)
        self._preview_layout.setAlignment(Qt.AlignLeft)
        self._preview_area.setWidget(self._preview_content)
        
        self._preview_area.hide()
        box_layout.addWidget(self._preview_area)

        # 1b. Active Tool Chips Row
        self._chip_row = QFrame(self.box)
        self._chip_row.setStyleSheet("background: transparent; border: none;")
        self._chip_layout = QHBoxLayout(self._chip_row)
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_layout.setSpacing(8)
        self._chip_layout.setAlignment(Qt.AlignLeft)
        self._chip_row.hide()
        box_layout.addWidget(self._chip_row)

        # 1c. Text input
        self._input = _InputTextEdit(self.box)
        self._input.setPlaceholderText(PLACEHOLDER)
        self._input.setFixedHeight(44)
        self._input.textChanged.connect(self._on_text_changed)
        self._input.enter_pressed.connect(self._on_return)
        
        # Focus events for glow ring
        self._input.focusInEvent = self._wrap_focus_in(self._input.focusInEvent)
        self._input.focusOutEvent = self._wrap_focus_out(self._input.focusOutEvent)
        
        box_layout.addWidget(self._input)

        # 1d. Action Row
        self._action_row = QFrame(self.box)
        self._action_row.setStyleSheet("background: transparent; border: none;")
        action_layout = QHBoxLayout(self._action_row)
        action_layout.setContentsMargins(0, 4, 0, 0)

        # Plus Menu Button
        self._plus_btn = AnimatedIconButton("", self._action_row)
        ico_plus = get_icon("plus", size=(20, 20), color=TEXT_DIM)
        if ico_plus: self._plus_btn.set_custom_icon(ico_plus, 20)
        self._plus_btn.setFixedSize(32, 32)
        self._plus_btn.setStyleSheet(get_button_style("transparent", WELCOME_CARD_HOVER, TEXT_DIM, radius=16))
        self._plus_btn.setCursor(Qt.PointingHandCursor)
        self._plus_btn.clicked.connect(self._toggle_action_menu)
        action_layout.addWidget(self._plus_btn)

        # Model Selection toggle (Standalone, modern)
        self._model_btn = QPushButton(f"  {self._current_model}", self._action_row)
        ico_mod = get_icon("chevron-up", size=(14,14), color=TEXT_DIM)
        if ico_mod: self._model_btn.setIcon(ico_mod)
        self._model_btn.setFont(BTN_LABEL)
        self._model_btn.setStyleSheet(get_button_style("transparent", WELCOME_CARD_HOVER, TEXT_DIM, radius=8))
        self._model_btn.setCursor(Qt.PointingHandCursor)
        self._model_btn.clicked.connect(self._show_model_menu)
        action_layout.addWidget(self._model_btn)

        # Model Mode Selection toggle
        self._mode_btn = QPushButton(f"  {self._current_mode}", self._action_row)
        ico_name = "command" if self._current_mode == "Planning" else "zap"
        ico_mode = get_icon(ico_name, size=(14,14), color=TEXT_DIM)
        if ico_mode: self._mode_btn.setIcon(ico_mode)
        self._mode_btn.setFont(BTN_LABEL)
        self._mode_btn.setStyleSheet(get_button_style("transparent", WELCOME_CARD_HOVER, TEXT_DIM, radius=8))
        self._mode_btn.setCursor(Qt.PointingHandCursor)
        self._mode_btn.clicked.connect(self._show_mode_menu)
        action_layout.addWidget(self._mode_btn)

        action_layout.addStretch(1)

        # Right actions
        self._voice_btn = QPushButton("", self._action_row)
        v_ico = get_icon("mic", size=(18,18), color=TEXT_DIM)
        if v_ico: self._voice_btn.setIcon(v_ico)
        self._voice_btn.setFixedSize(32, 32)
        self._voice_btn.setStyleSheet(get_button_style("transparent", WELCOME_CARD_HOVER, TEXT_DIM, radius=16))
        self._voice_btn.setCursor(Qt.PointingHandCursor)
        self._voice_btn.clicked.connect(self._on_voice)
        action_layout.addWidget(self._voice_btn)

        self._send_btn = AnimatedIconButton("", self._action_row)
        s_ico = get_icon("arrow-up", size=(20,20), color="#ffffff")
        if s_ico: self._send_btn.set_custom_icon(s_ico, 20)
        self._send_btn.setFixedSize(32, 32)
        self._send_btn.setStyleSheet(get_button_style(SEND_BTN, SEND_BTN_HOVER, "#ffffff", radius=16))
        self._send_btn.setCursor(Qt.PointingHandCursor)
        self._send_btn.clicked.connect(self._do_submit)
        
        # Send btn press anim
        self._send_btn.pressed.connect(self._animate_send_press)
        self._send_btn.released.connect(self._animate_send_release)
        action_layout.addWidget(self._send_btn)

        self._stop_btn = QPushButton("", self._action_row)
        st_ico = get_icon("square", size=(16,16), color="#f87171")
        if st_ico: self._stop_btn.setIcon(st_ico)
        self._stop_btn.setFixedSize(32, 32)
        self._stop_btn.setStyleSheet(get_button_style("#3a1a1a", "#5a2a2a", "#ffffff", radius=16))
        self._stop_btn.setCursor(Qt.PointingHandCursor)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.hide()
        action_layout.addWidget(self._stop_btn)

        box_layout.addWidget(self._action_row)
        main_layout.addWidget(self.box)

        # 2. Footer hints
        footer = QFrame(self)
        footer.setStyleSheet("background: transparent;")
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(8, 0, 8, 0)
        
        self._hint_lbl = QLabel("Enter to send  •  Shift+Enter for new line  •  Dizel can make mistakes. Please verify important information.", footer)
        self._hint_lbl.setFont(LABEL_DIM)
        self._hint_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)};")
        
        self._counter_lbl = QLabel("", footer)
        self._counter_lbl.setFont(LABEL_DIM)
        self._counter_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)};")

        f_layout.addWidget(self._hint_lbl, alignment=Qt.AlignLeft)
        f_layout.addWidget(self._counter_lbl, alignment=Qt.AlignRight)
        main_layout.addWidget(footer)

    def _setup_action_menu(self):
        self._action_menu = ActionMenu(self.window())
        self._action_menu.tool_toggled.connect(self._on_tool_toggled_from_menu)
        self._action_menu.action_triggered.connect(self._on_menu_action)
        self._action_menu.menu_hidden.connect(self._on_menu_hidden)

    def _apply_box_style(self, focused: bool):
        bg = resolve(BG_INPUT)
        border = resolve(BORDER_FOCUS) if focused else resolve(BORDER)
        style = get_frame_style(bg, radius=20, border_color=border)
        # Thicker border on focus for glow effect
        if focused:
            style += f" QFrame#InputBox {{ border: 1.5px solid {border}; }}"
        self.box.setStyleSheet(style.replace("QFrame", "QFrame#InputBox"))

    def _wrap_focus_in(self, original_event_handler):
        def focusInEvent(event):
            self._apply_box_style(focused=True)
            original_event_handler(event)
        return focusInEvent

    def _wrap_focus_out(self, original_event_handler):
        def focusOutEvent(event):
            self._apply_box_style(focused=False)
            original_event_handler(event)
        return focusOutEvent

    def _animate_send_press(self):
        self._send_btn.animate_scale(0.85, duration=100)

    def _animate_send_release(self):
        self._send_btn.animate_scale(1.0, duration=150)

    def _toggle_action_menu(self):
        if self._action_menu.isVisible():
            self._action_menu.close()
            # _on_menu_hidden handles the icon reset
        else:
            self._action_menu.sync_state(self._active_contexts)
            self._action_menu.show_above(self._plus_btn)
            # Rotate plus into an X
            self._plus_btn.animate_rotation(45.0, duration=150)
            
    def _on_menu_hidden(self):
        # Reset rotation back to plus sign
        self._plus_btn.animate_rotation(0.0, duration=150)
    
    def _on_tool_toggled_from_menu(self, tool_id: str, active: bool):
        if active:
            self._active_contexts.add(tool_id)
        else:
            self._active_contexts.discard(tool_id)
            
        self._rebuild_active_chips()

    def _on_menu_action(self, action_id: str):
        if action_id == "upload_image" or action_id == "upload_file":
            # Fire old attach callback which triggers file dialog in main.py
            if self._on_attach_cb:
                self._on_attach_cb()
        elif action_id == "reset_all":
            self.clear_attachments()
            self._active_contexts.clear()
            self._rebuild_active_chips()
            
    def _rebuild_active_chips(self):
        # Clear existing
        while self._chip_layout.count():
            item = self._chip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not self._active_contexts:
            self._chip_row.hide()
            return
            
        self._chip_row.show()
        
        labels = {
            "web": ("  Web Search", "globe"),
            "deep": ("  Deep Think", "cpu"),
            "thinking": ("  Thinking", "light-bulb"),
            "files": ("  Parse Files", "file-text"),
        }
        
        for cid in sorted(self._active_contexts):
            if cid not in labels: continue
            lbl, ico_name = labels[cid]
            
            chip = QFrame(self._chip_row)
            chip.setStyleSheet(f"background: {resolve(ACCENT)}; border-radius: 12px;")
            chip.setFixedHeight(24)
            
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(8, 2, 8, 2)
            cl.setSpacing(4)
            
            ico = get_icon(ico_name, size=(12,12), color="#ffffff")
            if ico:
                i_lbl = QLabel(chip)
                i_lbl.setPixmap(ico.pixmap(12,12))
                i_lbl.setStyleSheet("background: transparent;")
                cl.addWidget(i_lbl)
                
            t_lbl = QLabel(lbl, chip)
            t_lbl.setFont(LABEL_DIM)
            t_lbl.setStyleSheet("color: #ffffff; background: transparent;")
            cl.addWidget(t_lbl)
            
            close_btn = QPushButton("✕", chip)
            close_btn.setFixedSize(16, 16)
            close_btn.setStyleSheet("""
                QPushButton { background: transparent; color: #ffffff; border: none; }
                QPushButton:hover { color: #f87171; }
            """)
            close_btn.setFont(LABEL_DIM)
            close_btn.setCursor(Qt.PointingHandCursor)
            close_btn.clicked.connect(lambda _, x=cid: self._remove_tool_chip(x))
            cl.addWidget(close_btn)
            
            self._chip_layout.addWidget(chip)
            
        self._chip_layout.addStretch(1)

    def _remove_tool_chip(self, cid):
        self._active_contexts.discard(cid)
        self._rebuild_active_chips()
        if self._action_menu.isVisible():
            self._action_menu.sync_state(self._active_contexts)

    def _on_text_changed(self):
        text = self._input.toPlainText()
        n = len(text)
        self._counter_lbl.setText(f"{n} chars" if n > 0 else "")
        
        lines = text.count("\n") + 1
        h = max(44, min(lines * 22 + 20, 150))
        self._input.setFixedHeight(h)

    def _on_return(self, shift):
        if not shift:
            self._do_submit()

    def _do_submit(self):
        if self._generating:
            return
            
        text = self._input.toPlainText().strip()
        files = self.get_attachments()
        
        if not text and not files:
            return
            
        self.clear()
        self.set_generating(True)
        if self._on_send_msg:
            self._on_send_msg(text, files)

    def clear(self):
        self._input.clear()
        self._input.setFixedHeight(44)
        self.clear_attachments()
        self._counter_lbl.setText("")

    def add_attachment(self, file_path: str):
        if not self._attachments:
            self._preview_area.show()
            
        self._attachments.append(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        name = os.path.basename(file_path)
        
        pill = QFrame(self._preview_content)
        pill.setFixedHeight(26)
        pill.setStyleSheet(get_frame_style(BG_INPUT_FIELD, radius=13, border_color=BORDER))
        
        p_layout = QHBoxLayout(pill)
        p_layout.setContentsMargins(8,0,8,0)
        p_layout.setSpacing(6)
        
        ico_lbl = QLabel(pill)
        ico_lbl.setStyleSheet("background: transparent;")
        if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            ico = get_icon("image", size=(14,14), color=ACCENT)
        else:
            i_name = "archive" if ext in ['.zip', '.tar', '.gz', '.rar', '.7z'] else "file"
            ico = get_icon(i_name, size=(14,14), color=ACCENT)
        if ico: ico_lbl.setPixmap(ico.pixmap(14,14))
        p_layout.addWidget(ico_lbl)
        
        t_lbl = QLabel(name if len(name) < 15 else name[:12]+"...", pill)
        t_lbl.setFont(LABEL_DIM)
        t_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent;")
        p_layout.addWidget(t_lbl)
        
        rm_btn = QPushButton("✕", pill)
        rm_btn.setFixedSize(16, 16)
        rm_btn.setFont(LABEL_DIM)
        rm_btn.setStyleSheet("""
            QPushButton { background: transparent; color: """ + resolve(TEXT_DIM) + """; border: none; }
            QPushButton:hover { color: #f87171; }
        """)
        rm_btn.setCursor(Qt.PointingHandCursor)
        rm_btn.clicked.connect(lambda: self._remove_attachment(pill, file_path))
        p_layout.addWidget(rm_btn)

        self._preview_layout.addWidget(pill)

    def _remove_attachment(self, pill, file_path):
        pill.deleteLater()
        if file_path in self._attachments:
            self._attachments.remove(file_path)
        if not self._attachments:
            self._preview_area.hide()

    def get_attachments(self) -> list:
        return self._attachments.copy()

    def clear_attachments(self):
        while self._preview_layout.count():
            item = self._preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._attachments.clear()
        self._preview_area.hide()

    def get_active_contexts(self) -> set:
        return self._active_contexts.copy()

    def _show_model_menu(self):
        self._popup = ModelSelectorPopup(self._current_model, self.window())
        self._popup.selected.connect(self._on_model_selected)
        
        btn_pos = self._model_btn.mapToGlobal(self._model_btn.rect().topLeft())
        self._popup.move(btn_pos.x(), btn_pos.y() - self._popup.sizeHint().height() - 5)
        self._popup.show()

    def _on_model_selected(self, model_name):
        self._current_model = model_name
        self._model_btn.setText(f"  {model_name}")
        brand = "Mila" if "Mila" in model_name else "Dizel"
        self._input.setPlaceholderText(f"Message {brand}…")

    def _show_mode_menu(self):
        self._mode_popup = ModelModePopup(self._current_mode, self.window())
        self._mode_popup.selected.connect(self._on_mode_selected)
        
        btn_pos = self._mode_btn.mapToGlobal(self._mode_btn.rect().topLeft())
        self._mode_popup.move(btn_pos.x(), btn_pos.y() - self._mode_popup.sizeHint().height() - 5)
        self._mode_popup.show()

    def _on_mode_selected(self, mode_name):
        self._current_mode = mode_name
        self._mode_btn.setText(f"  {mode_name}")
        ico_name = "command" if mode_name == "Planning" else "zap"
        ico_mode = get_icon(ico_name, size=(14,14), color=TEXT_DIM)
        if ico_mode: self._mode_btn.setIcon(ico_mode)


    def set_generating(self, generating: bool):
        self._generating = generating
        if generating:
            self._send_btn.hide()
            self._stop_btn.show()
            self._plus_btn.setDisabled(True)
            self._input.setReadOnly(True)
        else:
            self._stop_btn.hide()
            self._send_btn.show()
            self._plus_btn.setDisabled(False)
            self._input.setReadOnly(False)

    def focus_input(self):
        self._input.setFocus()

    @property
    def current_model(self) -> str:
        return self._current_model

    @property
    def current_mode(self) -> str:
        return self._current_mode
