# dizel_ui/ui/chat_window.py

import os
import time
import datetime
import random
from enum import Enum
from typing import Optional
from PySide6.QtWidgets import (QFrame, QScrollArea, QVBoxLayout, QWidget, 
                               QLabel, QHBoxLayout, QGridLayout, QPushButton, QSizePolicy)
from PySide6.QtGui import QPixmap, QPainter, QLinearGradient, QColor
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

from dizel_ui.utils.anim_helpers import AnimHelpers

from .message_bubble import MessageBubble
from .typing_indicator import TypingIndicator
from dizel_ui.theme.colors import (
    BG_CHAT, ACCENT, ACCENT_LIGHT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    WELCOME_CARD, WELCOME_CARD_HOVER, ACTION_PILL, SCROLLBAR, resolve,
    BG_CARD, BORDER, ACCENT_HOVER
)
from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.fonts import (
    WELCOME_TITLE, WELCOME_SUB, CARD_TITLE, CARD_BODY, BTN_LABEL
)
from dizel_ui.logic.config_manager import ConfigManager
from dizel_ui.theme.stylesheets import get_scrollbar_style, get_frame_style

# ── Action Pills ──────────────────────────────────────────────────────────────
ACTION_PILLS = [
    ("image", "Create Image", "Generate an image of a futuristic city."),
    ("zap", "Brainstorm", "Give me 5 ideas for a new web project."),
    ("file-text", "Make a plan", "Create a structured study plan for Python."),
]

class ChatState(Enum):
    IDLE = "idle"
    TRANSITIONING = "transitioning"     # welcome fading out
    PROCESSING = "processing"           # typing indicator shown, no bubble
    STREAMING = "streaming"             # tokens flowing into bubble
    RESPONSE_COMPLETE = "complete"      # generation done, action bar visible

class ChatWindow(QFrame):
    def __init__(self, on_quick_action: callable, on_regenerate: callable = None, parent=None):
        super().__init__(parent)
        self._on_quick_action = on_quick_action
        self._on_regenerate = on_regenerate
        
        self.setStyleSheet(get_frame_style(BG_CHAT, radius=0))
        
        self._active_bubble: Optional[MessageBubble] = None
        self._typing_indicator: Optional[TypingIndicator] = None
        self._welcome_frame: Optional[QFrame] = None
        self._welcome_shown = False
        self._gen_start_time: float = 0.0
        self._token_count: int = 0
        self._state = ChatState.IDLE
        self._skip_animations = False
        
        self._build()



    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameStyle(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(get_scrollbar_style(BG_CHAT, SCROLLBAR, ACCENT) + "QScrollArea { background: transparent; }")
        
        self._content_widget = QWidget()
        self._content_widget.setStyleSheet(f"background-color: transparent;")
        
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setAlignment(Qt.AlignTop)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        
        self._scroll.setWidget(self._content_widget)
        layout.addWidget(self._scroll)
        
        self.show_welcome()

    def show_welcome(self, username: str = "User"):
        self._clear_messages()
        self._welcome_shown = True
        
        self._welcome_frame = QFrame(self._content_widget)
        self._welcome_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        welcome_layout = QVBoxLayout(self._welcome_frame)
        welcome_layout.setAlignment(Qt.AlignCenter)
        welcome_layout.setContentsMargins(0, 0, 0, 0)
        welcome_layout.setSpacing(24)
        
        welcome_layout.addStretch(1)

        # Static Avatar
        logo_lbl = QLabel(self._welcome_frame)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        avatar_path = os.path.join(base_dir, "assets", "avatars", "Diszi_beta2.png")
        if os.path.exists(avatar_path):
            original = QPixmap(avatar_path)
            if not original.isNull():
                cropped = original.copy(240, 240, 600, 600) # Static crop out the massive void padding without erasing glowing bounds
                logo_lbl.setPixmap(cropped.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        welcome_layout.addWidget(logo_lbl, 0, Qt.AlignCenter)

        # Dynamic Greeting based on time
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            greeting = f"Good morning, {username}!"
        elif 12 <= hour < 18:
            greeting = f"Good afternoon, {username}!"
        else:
            greeting = f"Good evening, {username}!"

        # Central Title
        titles = [
            greeting,
            "Ready to Create Something New?",
            "What shall we build today?",
            "Start a new masterpiece.",
            "Let's spark some creativity.",
            "Your next big idea starts here.",
            "How can I help you today?",
            "How shall I assist you today?",
            "Ready to architect the future?",
            "Let's brainstorm something brilliant.",
            "What's on your mind?",
            "Design, code, create.",
            "Let's bring your ideas to life.",
            "Ready when you are.",
            "Unleash your imagination.",
            "The blank canvas awaits you.",
            "Code with passion, create with purpose.",
            "Transforming thoughts into reality.",
            "Where logic meets creativity.",
            "Dive into the world of Dizel.",
            "Let's make some magic happen.",
            "Innovate. Iterate. Implement.",
            "Your creative companion is ready.",
            "Dream big, build bigger.",
            "Solving problems, one prompt at a time.",
            "Beyond expectations, into reality.",
            "Your vision, powered by Dizel.",
            "Crafting the future, together.",
            "What's the next big milestone?",
            "Let's solve the unsolvable.",
            "Empowering your workflow.",
            "Simple prompts, complex achievements.",
            "Ignite your workflow.",
            "Code, create, conquer.",
            "Let's build something extraordinary.",
            "Thinking outside the context window.",
            "From concept to perfection.",
            "Unlocking new possibilities.",
            "Let's turn coffee into code.",
            "Bringing your boldest ideas to life.",
            "Redefining the development experience.",
            "Accelerate your creative engine.",
            "Your intelligent coding partner.",
            "Making the impossible, achievable.",
            "Start your next masterpiece here."
        ]
        title_lbl = QLabel(random.choice(titles), self._welcome_frame)
        title_lbl.setFont(WELCOME_TITLE)
        title_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
        title_lbl.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(title_lbl)

        # Action Pills Row
        pills_row = QFrame(self._welcome_frame)
        pills_layout = QHBoxLayout(pills_row)
        pills_layout.setAlignment(Qt.AlignCenter)
        pills_layout.setSpacing(16)
        
        for (icon_name, label, prompt) in ACTION_PILLS:
            btn = QPushButton(f"{label}  ", pills_row)
            btn.setLayoutDirection(Qt.RightToLeft)
            ico = get_icon(icon_name, size=(16, 16), color=TEXT_PRIMARY)
            if ico: btn.setIcon(ico)
            btn.setFont(BTN_LABEL)
            # PySide styled buttons
            c_text = resolve(TEXT_PRIMARY)
            c_bg = resolve(BG_CARD)
            c_border = resolve(BORDER)
            c_hover = resolve(WELCOME_CARD_HOVER)
            c_hover_border = resolve(BORDER)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c_bg};
                    color: {c_text};
                    border: 1px solid {c_border};
                    border-radius: 18px;
                    padding: 10px 20px;
                    font-weight: 500;
                }}
                QPushButton:hover {{ 
                    background-color: {c_hover}; 
                    border: 1px solid {c_hover_border};
                    color: #ffffff;
                }}
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, p=prompt: self._on_quick_action(p))
            pills_layout.addWidget(btn)
            
        welcome_layout.addWidget(pills_row)
        
        welcome_layout.addStretch(1)
        
        self._content_layout.addWidget(self._welcome_frame)



    def _dismiss_welcome(self, callback=None):
        if not self._welcome_frame:
            if callback:
                callback()
            return
            
        def _finally_remove():
            if self._welcome_frame:
                self._content_layout.removeWidget(self._welcome_frame)
                self._welcome_frame.deleteLater()
                self._welcome_frame = None
            self._welcome_shown = False
            if callback:
                callback()
                
        if self._skip_animations:
            _finally_remove()
        else:
            AnimHelpers.fade_out(self._welcome_frame, duration=250, on_finished=_finally_remove)

    def _clear_messages(self):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._active_bubble = None
        self._typing_indicator = None
        self._welcome_frame = None
        self._state = ChatState.IDLE

    def clear(self, username: str = "User"):
        self._clear_messages()
        self.show_welcome(username)

    def _bubble_width(self) -> int:
        return max(400, int(self.width() * 0.88))

    def set_skip_animations(self, skip: bool):
        self._skip_animations = skip

    def begin_response_flow(self, text: str, attachments: list, on_ready: callable):
        """Sequenced orchestrator for new messages."""
        if self._state != ChatState.IDLE:
            return # Ignore if already processing
            
        self._state = ChatState.TRANSITIONING
        
        def _step_2_user_bubble():
            # Create user bubble
            bubble = MessageBubble(role="user", content=text, attachments=attachments, bubble_width=self._bubble_width(), parent=self._content_widget)
            self._content_layout.addWidget(bubble)
            
            if not self._skip_animations:
                AnimHelpers.slide_in_from_bottom(bubble, duration=200, on_finished=_step_3_processing)
            else:
                _step_3_processing()
                
            self._scroll_to_bottom()
            
        def _step_3_processing():
            self._state = ChatState.PROCESSING
            self._gen_start_time = time.time()
            self._token_count = 0
            
            # Show ONLY the typing indicator
            self._typing_indicator = TypingIndicator(self._content_widget)
            ind_cont = QFrame(self._content_widget)
            ind_cont.setStyleSheet("background: transparent;")
            ind_layout = QHBoxLayout(ind_cont)
            ind_layout.setContentsMargins(12, 4, 80, 0)
            ind_layout.addWidget(self._typing_indicator, alignment=Qt.AlignLeft)
            ind_layout.addStretch(1)
            self._content_layout.addWidget(ind_cont)
            
            self._typing_indicator_container = ind_cont
            self._typing_indicator.start()
            
            if not self._skip_animations:
                AnimHelpers.fade_in(ind_cont, duration=150)
                
            self._scroll_to_bottom()
            
            # Now trigger generation
            if on_ready:
                on_ready()

        # Start sequence
        self._dismiss_welcome(callback=_step_2_user_bubble)

    def add_user_message_instant(self, text: str, attachments: list = None) -> MessageBubble:
        """For loading history instantly without triggering inference flow."""
        self._dismiss_welcome()
        bubble = MessageBubble(role="user", content=text, attachments=attachments, bubble_width=self._bubble_width(), parent=self._content_widget)
        self._content_layout.addWidget(bubble)
        self._scroll_to_bottom()
        return bubble

    def add_assistant_message_instant(self, text: str) -> MessageBubble:
        """For loading history instantly."""
        self._dismiss_welcome()
        bubble = MessageBubble(
            role="assistant", 
            content=text, 
            bubble_width=self._bubble_width(), 
            on_regenerate=self._on_regenerate,
            parent=self._content_widget
        )
        bubble.finalise(text, "") # Skip action bar fade in for history
        self._content_layout.addWidget(bubble)
        self._scroll_to_bottom()
        return bubble

    def append_token(self, piece: str):
        if self._state == ChatState.PROCESSING:
            # First token transition
            self._state = ChatState.STREAMING
            
            if self._typing_indicator:
                self._typing_indicator.stop()
                
                def _spawn_bubble():
                    if self._typing_indicator_container:
                        self._content_layout.removeWidget(self._typing_indicator_container)
                        self._typing_indicator_container.deleteLater()
                        self._typing_indicator = None
                        self._typing_indicator_container = None
                    
                    self._active_bubble = MessageBubble(role="assistant", content="", bubble_width=self._bubble_width(), parent=self._content_widget)
                    self._content_layout.addWidget(self._active_bubble)
                    self._active_bubble.append_text(piece)
                    self._token_count += 1
                    
                    if not self._skip_animations:
                        AnimHelpers.slide_in_from_bottom(self._active_bubble, offset=15, duration=200)
                    self._scroll_to_bottom()
                    
                if not self._skip_animations:
                    # Minor fade out of dots before spawning bubble
                    AnimHelpers.fade_out(self._typing_indicator_container, duration=100, on_finished=_spawn_bubble)
                else:
                    _spawn_bubble()
            else:
                self._active_bubble = MessageBubble(
                    role="assistant", 
                    content="", 
                    bubble_width=self._bubble_width(), 
                    on_regenerate=self._on_regenerate,
                    parent=self._content_widget
                )
                self._content_layout.addWidget(self._active_bubble)
                self._active_bubble.append_text(piece)
                self._token_count += 1
                self._scroll_to_bottom()
                
        elif self._active_bubble:
            self._active_bubble.append_text(piece)
            self._token_count += 1
            self._scroll_to_bottom()

    def finish_assistant_message(self, full_text: str):
        if self._typing_indicator:
            self._typing_indicator.stop()
            self._content_layout.removeWidget(self._typing_indicator_container)
            self._typing_indicator_container.deleteLater()
            self._typing_indicator = None
            self._typing_indicator_container = None

        elapsed = time.time() - self._gen_start_time
        tps = self._token_count / max(elapsed, 0.01)
        meta = f"{self._token_count} tokens  •  {elapsed:.1f}s  •  {tps:.0f} tok/s"

        if self._active_bubble:
            self._active_bubble.finalise(full_text, meta, skip_anim=self._skip_animations)
            self._active_bubble = None

        self._state = ChatState.IDLE
        self._scroll_to_bottom()

    def show_error(self, msg: str):
        if self._typing_indicator:
            self._typing_indicator.stop()
            self._content_layout.removeWidget(self._typing_indicator_container)
            self._typing_indicator_container.deleteLater()
            self._typing_indicator = None
            self._typing_indicator_container = None

        if self._active_bubble:
            self._active_bubble.finalise(f"⚠ Error: {msg}")
            self._active_bubble = None
            
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        # We need a slight delay to allow layouts to compute their bounds
        QTimer.singleShot(30, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()))

    def pop_last_assistant_message(self):
        """Removes the last message widget from the chat view."""
        cnt = self._content_layout.count()
        if cnt > 0:
            item = self._content_layout.takeAt(cnt - 1)
            widget = item.widget()
            if widget:
                widget.deleteLater()
