# dizel_ui/ui/message_bubble.py

import datetime
from PySide6.QtWidgets import (QFrame, QLabel, QTextEdit, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QApplication, QSizePolicy, QGraphicsOpacityEffect, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QRect
from PySide6.QtGui import QTextOption, QTextCursor

from dizel_gui.theme.colors import (
    BUBBLE_USER, BUBBLE_ASST, BUBBLE_USER_TXT, BUBBLE_ASST_TXT,
    BG_CHAT, BG_INPUT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, ACCENT, ACCENT_LIGHT, ACCENT_HOVER, WELCOME_CARD_HOVER, BORDER, resolve
)
from dizel_gui.theme.fonts import MSG_TEXT, MSG_META, LABEL_SM
from dizel_gui.logic.config_manager import ConfigManager
from dizel_gui.theme.stylesheets import get_frame_style, get_button_style
from dizel_gui.utils.icons import get_icon
from dizel_gui.utils.anim_helpers import AnimHelpers
import re

class _CodeBlockWidget(QFrame):
    def __init__(self, code: str, lang: str = "", parent=None):
        super().__init__(parent)
        self.code_text = code
        
        # Style the container
        bg_color = resolve(BG_INPUT)
        border_color = resolve(BORDER)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {resolve(BORDER)};
                border-radius: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header bar
        header = QFrame(self)
        header.setStyleSheet(f"background-color: rgba(0,0,0,0.2); border-bottom: 1px solid {border_color}; border-top-left-radius: 8px; border-top-right-radius: 8px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 8, 6)
        
        lang_lbl = QLabel(lang if lang else "code")
        lang_lbl.setFont(MSG_META)
        lang_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent; border: none;")
        header_layout.addWidget(lang_lbl)
        
        header_layout.addStretch(1)
        
        self.copy_btn = QPushButton(" Copy")
        cp_ico = get_icon("copy", size=(14,14), color=TEXT_DIM)
        if cp_ico: self.copy_btn.setIcon(cp_ico)
        self.copy_btn.setFont(MSG_META)
        self.copy_btn.setStyleSheet(get_button_style("transparent", WELCOME_CARD_HOVER, TEXT_DIM, radius=4) + " border: none;")
        self.copy_btn.setCursor(Qt.PointingHandCursor)
        self.copy_btn.setFixedHeight(24)
        self.copy_btn.clicked.connect(self._copy_code)
        header_layout.addWidget(self.copy_btn)
        
        layout.addWidget(header)
        
        # Code content
        self.text_edit = _AutoResizingTextEdit('transparent', TEXT_PRIMARY, self)
        self.text_edit.setPlainText(code)
        # Use a monospaced font for code
        from PySide6.QtGui import QFontDatabase
        code_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        code_font.setPointSize(10)
        self.text_edit.setFont(code_font)
        self.text_edit.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent; border: none; padding: 8px;")
        layout.addWidget(self.text_edit)
        
    def _copy_code(self):
        QApplication.clipboard().setText(self.code_text)
        self.copy_btn.setText(" Copied")
        chk = get_icon("check", size=(14,14), color=ACCENT)
        if chk: self.copy_btn.setIcon(chk)
        QTimer.singleShot(1500, self._reset_copy)
        
    def _reset_copy(self):
        self.copy_btn.setText(" Copy")
        cp = get_icon("copy", size=(14,14), color=TEXT_DIM)
        if cp: self.copy_btn.setIcon(cp)


class _ThoughtWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: transparent;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 4)
        layout.setSpacing(4)
        
        self.toggle_btn = QPushButton("  Thinking...")
        ico = get_icon("cpu", size=(14, 14), color=TEXT_DIM)
        if ico: self.toggle_btn.setIcon(ico)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setFont(MSG_META)
        self.toggle_btn.setStyleSheet(get_button_style("transparent", WELCOME_CARD_HOVER, TEXT_DIM, radius=4) + " border: none;")
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.setFixedHeight(24)
        
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0,0,0,0)
        btn_layout.addWidget(self.toggle_btn, alignment=Qt.AlignLeft)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)
        
        # Add a left-border frame for the actual text
        self.content_frame = QFrame(self)
        self.content_frame.setStyleSheet(f"border-left: 2px solid {resolve(BORDER)}; margin-left: 6px;")
        cf_layout = QVBoxLayout(self.content_frame)
        cf_layout.setContentsMargins(12, 0, 0, 8)
        
        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setFont(LABEL_SM)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent; border: none;")
        cf_layout.addWidget(self.text_label)
        
        layout.addWidget(self.content_frame)
        
        self.toggle_btn.toggled.connect(self._on_toggle)
        
    def _on_toggle(self, checked):
        self.content_frame.setVisible(checked)
        if checked:
            self.toggle_btn.setText("  Thought process (completed)" if getattr(self, "_finished", False) else "  Thinking...")
        else:
            self.toggle_btn.setText("  Thought process hidden")
            
    def set_text(self, text: str, is_finished: bool = False):
        self._finished = is_finished
        self.text_label.setText(text.strip())
        if self.toggle_btn.isChecked():
            self.toggle_btn.setText("  Thought process (completed)" if is_finished else "  Thinking...")

    def collapse(self):
        self.toggle_btn.setChecked(False)

class _AutoResizingTextEdit(QTextEdit):
    """
    A read-only QTextEdit that automatically resizes its height to fit its content.
    """
    def __init__(self, fg_color: str, text_color: str, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFrameStyle(QFrame.NoFrame)
        self.setFont(MSG_TEXT)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        bg = resolve(fg_color)
        txt = resolve(text_color)
        
        bg_css = bg if bg != 'transparent' else 'transparent'
        
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg_css};
                color: {txt};
                border: none;
            }}
        """)
        
        self.document().documentLayout().documentSizeChanged.connect(self._adjust_height)

    def _adjust_height(self):
        doc_height = self.document().size().height()
        # Add a little padding to the height
        self.setFixedHeight(int(doc_height) + 10)

class MessageBubble(QFrame):
    """
    A single chat message rendered as a styled PySide6 bubble.
    """

    _PROVIDER_AVATARS = {
        "ollama": "ollama.png", "openai": "chatgpt.png",
        "anthropic": "claude.png", "google": "gemini.png",
        "groq": "Groq.png", "mistral": "mistral-ai.png",
        "xai": "xai.png", "ai21": "ai21-labs.png",
        "azure": "microsoft-azure-openaI.png",
        "cohere": "cohere.png", "meta": "meta.png",
    }

    def __init__(self, role: str, content: str, meta: str = "", attachments: list = None,
                 bubble_width: int = 500, on_regenerate=None, on_edit_submit=None,
                 provider_slug: str = "local", model_name: str = "",
                 parent=None):
        super().__init__(parent)
        self._on_regenerate = on_regenerate
        self._on_edit_submit = on_edit_submit

        self._role = role
        self._content = content
        self._is_user = (role == "user")
        
        bg_color = BUBBLE_USER if self._is_user else BUBBLE_ASST
        txt_color = BUBBLE_USER_TXT if self._is_user else BUBBLE_ASST_TXT

        # Determine the display name for assistant
        if self._is_user:
            role_label = "You"
        elif model_name:
            role_label = model_name
        else:
            role_label = "Dizel"

        # Bubble alignment:
        # User = right aligned, Assistant = left aligned
        outer_layout = QHBoxLayout(self)
        # Give the AI response more breathing room by reducing the right margin clamp
        outer_layout.setContentsMargins(12 if not self._is_user else 100, 2, 100 if not self._is_user else 12, 2)
        
        # Spacer for alignment
        if self._is_user:
            outer_layout.addStretch(1)

        # Bubble Container
        self._bubble_frame = QFrame(self)
        self._bubble_frame.setStyleSheet(get_frame_style(bg_color, radius=16))
        self._bubble_frame.setSizePolicy(QSizePolicy.Expanding if self._is_user else QSizePolicy.Fixed, QSizePolicy.Minimum)
        
        # Enforce minimum width so the AI bubble never feels cramped
        if not self._is_user:
            self._bubble_frame.setMinimumWidth(bubble_width)
            self._bubble_frame.setMaximumWidth(bubble_width)
        
        self._bubble_layout = QVBoxLayout(self._bubble_frame)
        self._bubble_layout.setContentsMargins(16, 12, 16, 12)
        self._bubble_layout.setSpacing(4)

        # Role + Avatar Container
        role_container = QFrame(self._bubble_frame)
        role_container.setStyleSheet("background: transparent;")
        role_layout = QHBoxLayout(role_container)
        role_layout.setContentsMargins(0, 0, 0, 0)
        role_layout.setSpacing(8)
        
        self._avatar_lbl = None
        if not self._is_user:
            import os
            from PySide6.QtGui import QPixmap, QPainter, QPainterPath
            self._avatar_lbl = QLabel(role_container)
            self._avatar_lbl.setFixedSize(24, 24)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            # Choose avatar based on provider
            avatar_path = None
            needs_rounding = False
            if provider_slug and provider_slug != "local":
                fname = self._PROVIDER_AVATARS.get(provider_slug)
                if fname:
                    candidate = os.path.join(base_dir, "assets", "avatars", "providers", fname)
                    if os.path.exists(candidate):
                        avatar_path = candidate
                        needs_rounding = True

            if not avatar_path:
                avatar_path = os.path.join(base_dir, "assets", "app", "Dizel.png")

            if os.path.exists(avatar_path):
                original = QPixmap(avatar_path)
                if not original.isNull():
                    if needs_rounding:
                        # Render as circle
                        size = 24
                        scaled = original.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                        result = QPixmap(size, size)
                        result.fill(Qt.transparent)
                        painter = QPainter(result)
                        painter.setRenderHint(QPainter.Antialiasing)
                        clip = QPainterPath()
                        clip.addEllipse(0, 0, size, size)
                        painter.setClipPath(clip)
                        painter.drawPixmap(0, 0, scaled)
                        painter.end()
                        self._avatar_lbl.setPixmap(result)
                    else:
                        self._avatar_lbl.setPixmap(original.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            role_layout.addWidget(self._avatar_lbl)

        role_lbl = QLabel(role_label, role_container)
        role_lbl.setFont(LABEL_SM)
        role_lbl_color = resolve(TEXT_PRIMARY if self._is_user else ACCENT)
        role_lbl.setStyleSheet(f"color: {role_lbl_color}; background: transparent; border: none;")
        role_layout.addWidget(role_lbl, alignment=Qt.AlignLeft)
        role_layout.addStretch(1)
        
        self._bubble_layout.addWidget(role_container)

        self._thought_widget = None
        self._block_widgets = []  # Store dynamically generated code/text blocks
        self._main_content_container = QFrame(self._bubble_frame)
        self._main_content_container.setStyleSheet("background: transparent; border: none;")
        self._main_content_layout = QVBoxLayout(self._main_content_container)
        self._main_content_layout.setContentsMargins(0, 0, 0, 0)
        self._main_content_layout.setSpacing(8)
        self._bubble_layout.addWidget(self._main_content_container)

        # Attachments Preview
        if attachments:
            import os
            from PySide6.QtGui import QPixmap 
            
            att_scroll = QFrame(self._bubble_frame)
            att_scroll.setStyleSheet("background: transparent; border: none;")
            att_layout = QHBoxLayout(att_scroll)
            att_layout.setContentsMargins(0, 4, 0, 8)
            att_layout.setSpacing(12)
            
            has_valid = False
            for path in attachments:
                if not os.path.exists(path):
                    continue
                has_valid = True
                ext = os.path.splitext(path)[1].lower()
                name = os.path.basename(path)
                
                if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
                    pi = QPixmap(path)
                    if not pi.isNull():
                        # Restrict to reasonable standard UI sizes while preserving ratios
                        pi = pi.scaled(240, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        lbl = QLabel(att_scroll)
                        lbl.setPixmap(pi)
                        lbl.setStyleSheet(f"border-radius: 16px; border: 1px solid {resolve(ACCENT_LIGHT)}; background: transparent;")
                        att_layout.addWidget(lbl)
                else:
                    # Generic File Pill
                    pill = QFrame(att_scroll)
                    pill.setFixedHeight(36)
                    pill.setStyleSheet(get_frame_style('transparent', radius=8, border_color=ACCENT_LIGHT))
                    pill_lyt = QHBoxLayout(pill)
                    pill_lyt.setContentsMargins(10, 0, 12, 0)
                    pill_lyt.setSpacing(8)
                    
                    i_name = "archive" if ext in ['.zip', '.tar', '.gz', '.rar', '.7z'] else "file"
                    ico = get_icon(i_name, size=(16,16), color=TEXT_PRIMARY)
                    if ico:
                        il = QLabel(pill)
                        il.setPixmap(ico.pixmap(16,16))
                        il.setStyleSheet("background: transparent;")
                        pill_lyt.addWidget(il)
                        
                    tl = QLabel(name if len(name)<30 else name[:27]+"...", pill)
                    tl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent; border: none;")
                    tl.setFont(LABEL_SM)
                    pill_lyt.addWidget(tl)
                    att_layout.addWidget(pill)
                    
            if has_valid:
                att_layout.addStretch(1)
                self._bubble_layout.insertWidget(1, att_scroll)  # Insert right after role container

        # Message text (Used during streaming)
        self._streaming_textbox = _AutoResizingTextEdit(bg_color, txt_color, self._main_content_container)
        self._main_content_layout.addWidget(self._streaming_textbox)
        
        self._parse_and_update(content)

        # Bottom row (action bar)
        self._action_bar = QFrame(self._bubble_frame)
        self._action_bar.setStyleSheet("background: transparent; border: none;")
        bottom_layout = QHBoxLayout(self._action_bar)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(6)
        
        # Action bar starts hidden
        self._action_bar.hide()
            
        self._copy_btn = None
        def create_btn(lbl, icon_name, is_copy=False):
            btn = QPushButton(f" {lbl}" if lbl else "", self._action_bar)
            ico = get_icon(icon_name, size=(16, 16), color=TEXT_SECONDARY)
            if ico: btn.setIcon(ico)
            btn.setFont(MSG_META)
            hover_color = BUBBLE_USER if self._is_user else WELCOME_CARD_HOVER
            btn.setStyleSheet(get_button_style("transparent", hover_color, TEXT_SECONDARY, radius=4))
            btn.setCursor(Qt.PointingHandCursor)
            
            if is_copy:
                btn.clicked.connect(self._copy_text)
                self._copy_btn = btn
                
            bottom_layout.addWidget(btn)
            return btn
            
        create_btn("Copy", "copy", is_copy=True)
        
        if not self._is_user:
            regen_btn = create_btn("Regenerate", "refresh-cw")
            if self._on_regenerate:
                regen_btn.clicked.connect(self._on_regenerate)
            
            bottom_layout.addStretch(1)
            create_btn("", "thumbs-up")
            create_btn("", "thumbs-down")
            create_btn("Share", "share")
        else:
            self._edit_btn = create_btn("Edit", "edit-2")
            self._edit_btn.clicked.connect(self._toggle_edit_mode)
            bottom_layout.addStretch(1)

        # Meta line (token count / latency)
        self._meta_lbl = None
        if not self._is_user:
            self._meta_lbl = QLabel(meta, self._action_bar)
            self._meta_lbl.setFont(MSG_META)
            self._meta_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)};")
            bottom_layout.addWidget(self._meta_lbl)

        # Timestamp
        app_cfg = ConfigManager.load().get("appearance", {})
        if app_cfg.get("show_timestamps", True):
            ts = datetime.datetime.now().strftime("%H:%M")
            ts_lbl = QLabel(ts, self._action_bar)
            ts_lbl.setFont(MSG_META)
            ts_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)};")
            bottom_layout.addWidget(ts_lbl)

        self._bubble_layout.addWidget(self._action_bar)
        
        # Edit Action Bar (hidden by default)
        self._edit_action_bar = QFrame(self._bubble_frame)
        self._edit_action_bar.setStyleSheet("background: transparent; border: none;")
        e_layout = QHBoxLayout(self._edit_action_bar)
        e_layout.setContentsMargins(0, 8, 0, 0)
        e_layout.setSpacing(8)
        
        save_btn = QPushButton("Save & Regenerate", self._edit_action_bar)
        save_btn.setFixedHeight(32)
        save_btn.setFont(MSG_META)
        save_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, "#ffffff", radius=6))
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self._submit_edit)
        e_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel", self._edit_action_bar)
        cancel_btn.setFixedHeight(32)
        cancel_btn.setFont(MSG_META)
        cancel_btn.setStyleSheet(get_button_style("transparent", BUBBLE_USER, TEXT_PRIMARY, radius=6))
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self._cancel_edit)
        e_layout.addWidget(cancel_btn)
        e_layout.addStretch(1)
        
        self._edit_action_bar.hide()
        self._bubble_layout.addWidget(self._edit_action_bar)
        
        outer_layout.addWidget(self._bubble_frame)
        
        if not self._is_user:
            outer_layout.addStretch(1)
            
        self._is_editing = False

    def enterEvent(self, event):
        # On Hover: show action bar
        if self._is_user and not self._is_editing:
            self._action_bar.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # On Leave: hide user action bar
        if self._is_user and not self._is_editing:
            self._action_bar.hide()
        super().leaveEvent(event)
        
    def _toggle_edit_mode(self):
        self._is_editing = True
        self._action_bar.hide()
        self._edit_action_bar.show()
        
        # Change streaming textbox to editable
        self._streaming_textbox.setReadOnly(False)
        self._streaming_textbox.setStyleSheet(f"QTextEdit {{ background-color: {resolve(BG_CHAT)}; color: {resolve(BUBBLE_USER_TXT)}; border: 1px solid {resolve(BORDER)}; border-radius: 6px; padding: 4px; }}")
        self._streaming_textbox.setFocus()
        self._streaming_textbox._adjust_height()
        
    def _cancel_edit(self):
        self._is_editing = False
        self._edit_action_bar.hide()
        self._action_bar.show()
        
        # Revert text and readonly
        self._streaming_textbox.setPlainText(self._content)
        bg = resolve(BUBBLE_USER) if self._is_user else resolve(BUBBLE_ASST)
        self._streaming_textbox.setStyleSheet(f"QTextEdit {{ background-color: {bg if bg != 'transparent' else 'transparent'}; color: {resolve(BUBBLE_USER_TXT)}; border: none; }}")
        self._streaming_textbox.setReadOnly(True)
        self._streaming_textbox._adjust_height()
        
    def _submit_edit(self):
        if not self._on_edit_submit: return
        new_text = self._streaming_textbox.toPlainText().strip()
        if not new_text: return
        
        self._cancel_edit() # Reset UI state
        self._content = new_text
        self._streaming_textbox.setPlainText(new_text)
        self._on_edit_submit(self, new_text)

    def _copy_text(self):
        QApplication.clipboard().setText(self._content)
        self._copy_btn.setText("Copied")
        import dizel_gui.utils.icons as icons
        chk_icon = icons.get_icon("check", size=(16,16), color=TEXT_SECONDARY)
        if chk_icon: self._copy_btn.setIcon(chk_icon)
        
        def _reset():
            self._copy_btn.setText(" Copy")
            cp_icon = icons.get_icon("copy", size=(16,16), color=TEXT_SECONDARY)
            if cp_icon: self._copy_btn.setIcon(cp_icon)
            
        QTimer.singleShot(1500, _reset)

    def _parse_and_update(self, full_text: str, is_final: bool = False):
        self._content = full_text
        
        # Extract <think> blocks natively
        thoughts = re.findall(r'<think>(.*?)(?:</think>|$)', full_text, flags=re.DOTALL)
        if thoughts and not self._is_user:
            thought_text = "\n\n".join([t.strip() for t in thoughts if t.strip()])
            
            if not self._thought_widget and thought_text:
                self._thought_widget = _ThoughtWidget(self._bubble_frame)
                # Insert it right below the role label (index 1)
                self._bubble_layout.insertWidget(1, self._thought_widget)
                
            if self._thought_widget:
                self._thought_widget.set_text(thought_text, is_finished=is_final)
                
            # Strip thought blocks to show only pure answer in main textbox
            main_text = re.sub(r'<think>.*?(?:</think>|$)', '', full_text, flags=re.DOTALL).strip()
        else:
            main_text = full_text.strip()
            
        if not is_final:
            self._streaming_textbox.setPlainText(main_text)
            self._streaming_textbox.show()
            for w in self._block_widgets:
                w.hide()
            self._streaming_textbox._adjust_height()
        else:
            self._streaming_textbox.hide()
            self._render_final_blocks(main_text)

    def _render_final_blocks(self, main_text: str):
        # Clear previous block widgets
        for w in self._block_widgets:
            self._main_content_layout.removeWidget(w)
            w.deleteLater()
        self._block_widgets.clear()
        
        # Split by markdown code blocks
        parts = re.split(r'(```[\s\S]*?```)', main_text)
        
        for part in parts:
            if not part: continue
            
            if part.startswith('```') and part.endswith('```'):
                # Extract lang and code
                inner = part[3:-3]
                lines = inner.split('\n', 1)
                lang = lines[0].strip() if len(lines) > 1 and lines[0].strip() else ""
                code = lines[1] if len(lines) > 1 else lines[0]
                
                # For inline ```` that isn't multiline
                if len(lines) == 1 and lang:
                    code = lang
                    lang = ""
                    
                code_w = _CodeBlockWidget(code.strip("\n"), lang, self._main_content_container)
                self._main_content_layout.addWidget(code_w)
                self._block_widgets.append(code_w)
            else:
                # Normal text
                txt_w = _AutoResizingTextEdit('transparent', BUBBLE_USER_TXT if self._is_user else BUBBLE_ASST_TXT, self._main_content_container)
                # Apply simple markdown (PySide6 supports basic HTML parsing from Markdown via setMarkdown)
                txt_w.setMarkdown(part.strip())
                self._main_content_layout.addWidget(txt_w)
                self._block_widgets.append(txt_w)

    def append_text(self, piece: str):
        new_text = self._content + piece
        self._parse_and_update(new_text)

    def finalise(self, full_text: str, meta: str = "", skip_anim: bool = False):
        self._parse_and_update(full_text, is_final=True)
        
        if self._thought_widget:
            self._thought_widget.collapse()
            
        if not self._is_user:
            if self._meta_lbl:
                self._meta_lbl.setText(meta)
            self._action_bar.show()

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "_animated") and self._animated:
            return
            
        # Subtle slide-up and fade-in
        self._animated = True
        
        # We need an opacity effect to fade in
        eff = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(eff)
        
        # Opacity anim
        self._op_anim = QPropertyAnimation(eff, b"opacity")
        self._op_anim.setDuration(250)
        self._op_anim.setStartValue(0.0)
        self._op_anim.setEndValue(1.0)
        self._op_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        # We can't easily animate layout position of a QFrame within a layout without 
        # custom spacing/margins, so we'll just use the opacity which still gives a 
        # very premium feel.
        self._op_anim.start()

