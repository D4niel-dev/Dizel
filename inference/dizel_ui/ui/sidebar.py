# dizel_ui/ui/sidebar.py

import os
from typing import Callable, List, Dict, Optional
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QWidget, QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PySide6.QtGui import QIcon, QPixmap

from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.colors import (
    BG_SIDEBAR, ACCENT, ACCENT_HOVER, ACCENT_LIGHT,
    SIDEBAR_BTN_HOVER, SIDEBAR_BTN_ACTIVE, SIDEBAR_TEXT,
    SIDEBAR_TEXT_DIM, SIDEBAR_BORDER, TEXT_PRIMARY, TEXT_DIM, BG_ROOT, SIDEBAR_PREMIUM_BG, resolve
)
from dizel_ui.theme.fonts import LOGO, NAV_ITEM, NAV_ITEM_SM, BTN_LABEL, LABEL_SM, LABEL_DIM
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style, get_scrollbar_style
from dizel_ui.ui.secondary_sidebar import HistoryItem

SIDEBAR_W_OPEN = 240
SIDEBAR_W_CLOSED = 56

class Sidebar(QFrame):
    def __init__(self, on_new_chat: Callable[[], None], on_toggle_chats: Callable[[], None],
                 on_workspace_view: Callable[[str], None],
                 on_session_select: Callable[[str], None], on_session_delete: Callable[[str], None],
                 on_pin: Callable[[str], None],
                 on_settings: Callable[[], None],
                 on_profile_click: Callable[[], None],
                 on_action: Optional[Callable[[str], None]] = None, parent=None):
        super().__init__(parent)
        self._on_new_chat = on_new_chat
        self._on_toggle_chats = on_toggle_chats
        self._on_workspace_view = on_workspace_view
        self._on_session_select = on_session_select
        self._on_session_delete = on_session_delete
        self._on_pin = on_pin
        self._on_settings = on_settings
        self._on_profile_click = on_profile_click
        self._on_action = on_action
        
        self._is_open = True
        self._workspaces_open = False
        self._feature_btns = []
        self._workspace_btns = []
        
        self.setMinimumWidth(SIDEBAR_W_OPEN)
        self.setMaximumWidth(SIDEBAR_W_OPEN)
        self.setStyleSheet(get_frame_style(BG_SIDEBAR, radius=0))
        
        self.anim_group = QParallelAnimationGroup(self)
        self.anim_min = QPropertyAnimation(self, b"minimumWidth")
        self.anim_max = QPropertyAnimation(self, b"maximumWidth")
        self.anim_min.setEasingCurve(QEasingCurve.InOutQuart)
        self.anim_max.setEasingCurve(QEasingCurve.InOutQuart)
        self.anim_min.setDuration(250)
        self.anim_max.setDuration(250)
        self.anim_group.addAnimation(self.anim_min)
        self.anim_group.addAnimation(self.anim_max)

        self._build()

    def _build(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Top Bar (Logo + Toggle)
        top_bar = QFrame(self)
        top_bar.setFixedHeight(64)
        top_bar.setStyleSheet("background: transparent;")
        self.top_layout = QHBoxLayout(top_bar)
        self.top_layout.setContentsMargins(20, 18, 12, 18)
        
        _UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(_UI_DIR, "assets", "app", "Dizel.png")
        
        self._logo_container = QWidget(top_bar)
        logo_layout = QHBoxLayout(self._logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        
        self._logo_ico = QLabel()
        if os.path.exists(logo_path):
            self._logo_ico.setPixmap(QPixmap(logo_path).scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self._logo_ico.setText("⬡")
            self._logo_ico.setStyleSheet(f"color: {resolve(ACCENT)}; font-size: 24px;")
            
        self._logo_lbl = QLabel("  Dizel")
        self._logo_lbl.setFont(LOGO)
        self._logo_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
        
        logo_layout.addWidget(self._logo_ico)
        logo_layout.addWidget(self._logo_lbl)
        self.top_layout.addWidget(self._logo_container)
        
        self.top_layout.addStretch(1)
        
        self._icon_open = get_icon("sidebar", size=(18, 18), color=SIDEBAR_TEXT_DIM)
        self._icon_closed = get_icon("menu", size=(18, 18), color=SIDEBAR_TEXT_DIM)
        
        self._toggle_btn = QPushButton(top_bar)
        self._toggle_btn.setFixedSize(28, 28)
        if self._icon_open: self._toggle_btn.setIcon(self._icon_open)
        self._toggle_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, "transparent", radius=8))
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.clicked.connect(self.toggle)
        self.top_layout.addWidget(self._toggle_btn)
        
        self.main_layout.addWidget(top_bar)

        # Main Scrollable Area
        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameStyle(QFrame.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet(get_scrollbar_style(BG_SIDEBAR, SIDEBAR_BORDER, ACCENT) + "QScrollArea { background: transparent; }")
        
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background: transparent;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 16)
        self.scroll_layout.setSpacing(2)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        
        # New Chat Button
        self.new_btn_wrap = QWidget(self.scroll_content)
        new_layout = QVBoxLayout(self.new_btn_wrap)
        new_layout.setContentsMargins(0, 12, 0, 0)
        self._new_btn = QPushButton("  New Chat", self.new_btn_wrap)
        self._new_btn.setFixedHeight(34)
        from PySide6.QtCore import QSize
        plus_ico = get_icon("edit-3", size=(20, 20), color=SIDEBAR_TEXT)
        if plus_ico:
            self._new_btn.setIcon(plus_ico)
            self._new_btn.setIconSize(QSize(20, 20))
        self._new_btn.setFont(BTN_LABEL)
        self._new_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
        self._new_btn.setCursor(Qt.PointingHandCursor)
        self._new_btn.clicked.connect(self._on_new_chat)
        new_layout.addWidget(self._new_btn)
        self.scroll_layout.addWidget(self.new_btn_wrap)

        # Chats Toggle
        self._chat_container = QWidget(self.scroll_content)
        chat_col = QVBoxLayout(self._chat_container)
        chat_col.setContentsMargins(0, 0, 0, 0)
        chat_col.setSpacing(0)
        
        self._chat_btn = QPushButton("  Chats", self._chat_container)
        self._chat_btn.setFixedHeight(34)
        chat_ico = get_icon("new-chat-2", size=(22, 22), color=SIDEBAR_TEXT)
        if chat_ico:
            self._chat_btn.setIcon(chat_ico)
            self._chat_btn.setIconSize(QSize(22, 22))
        self._chat_btn.setFont(NAV_ITEM)
        self._chat_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
        self._chat_btn.setCursor(Qt.PointingHandCursor)
        self._chat_btn.clicked.connect(lambda: self._on_workspace_view("Chats"))
        chat_col.addWidget(self._chat_btn)
        
        self.scroll_layout.addWidget(self._chat_container)

        # Workspaces Toggle Button
        self._work_btn = QPushButton("  Workspaces", self.scroll_content)
        self._work_btn.setFixedHeight(34)
        work_ico = get_icon("workspace-2", size=(20, 20), color=SIDEBAR_TEXT)
        if work_ico:
            self._work_btn.setIcon(work_ico)
            self._work_btn.setIconSize(QSize(20, 20))
        self._work_btn.setFont(NAV_ITEM)
        self._work_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
        self._work_btn.setCursor(Qt.PointingHandCursor)
        self._work_btn.clicked.connect(self._toggle_workspaces)
        self.scroll_layout.addWidget(self._work_btn)
        
        # Workspaces Container
        self._work_container = QWidget(self.scroll_content)
        self._work_container.setMaximumHeight(0) # Collapsed by default
        work_col = QVBoxLayout(self._work_container)
        work_col.setContentsMargins(0, 0, 0, 0)
        work_col.setSpacing(2)
        self._work_container.hide()
        
        self._work_anim = QPropertyAnimation(self._work_container, b"maximumHeight")
        self._work_anim.setDuration(300)
        self._work_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._work_anim.finished.connect(self._hide_work_container_on_finish)
        
        workspaces = [
            ("image", "Image"),
            ("layout", "Presentation"), ("search", "Riset"),
            ("archive", "Archived"), ("book", "Library")
        ]
        for icon_name, text in workspaces:
            btn = QPushButton(f"  {text}", self._work_container)
            btn.setFixedHeight(32)
            ico = get_icon(icon_name, size=(20, 20), color=SIDEBAR_TEXT)
            if ico:
                btn.setIcon(ico)
                btn.setIconSize(QSize(20, 20))
            btn.setFont(NAV_ITEM)
            btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 19px; }")
            btn.setCursor(Qt.PointingHandCursor)
            
            # Switch to this view AND toggle it open if needed
            btn.clicked.connect(lambda _, t=text: self._on_workspace_view(t))
            
            work_col.addWidget(btn)
            self._workspace_btns.append((btn, text))
            
        self.scroll_layout.addWidget(self._work_container)

        # Separator (moves down)
        self._sep1_wrap = QWidget(self.scroll_content)
        sep_lyt = QVBoxLayout(self._sep1_wrap)
        sep_lyt.setContentsMargins(19, 16, 19, 8)
        sep_line = QFrame()
        sep_line.setFixedHeight(1)
        sep_line.setStyleSheet(f"background-color: {resolve(SIDEBAR_BORDER)};")
        sep_lyt.addWidget(sep_line)
        self.scroll_layout.addWidget(self._sep1_wrap)

        # Recent Chats Title
        self._recent_lbl_wrap = QWidget(self.scroll_content)
        rlbl_lyt = QVBoxLayout(self._recent_lbl_wrap)
        rlbl_lyt.setContentsMargins(19, 0, 19, 4)
        self._recent_lbl = QLabel("Recent Chats", self._recent_lbl_wrap)
        self._recent_lbl.setFont(LABEL_DIM)
        self._recent_lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)};")
        rlbl_lyt.addWidget(self._recent_lbl)
        self.scroll_layout.addWidget(self._recent_lbl_wrap)
        
        # History List
        self._hist_wrap = QWidget(self.scroll_content)
        hist_wrap_lyt = QVBoxLayout(self._hist_wrap)
        hist_wrap_lyt.setContentsMargins(0, 0, 0, 0)
        
        self._hist_list = QWidget(self._hist_wrap)
        self._hist_layout = QVBoxLayout(self._hist_list)
        self._hist_layout.setContentsMargins(19, 0, 0, 0)
        self._hist_layout.setSpacing(0)
        
        hist_wrap_lyt.addWidget(self._hist_list)
        self.scroll_layout.addWidget(self._hist_wrap)

        self.scroll_layout.addStretch(1)

        # Profile Pill
        self._profile_pill_container = QWidget(self.scroll_content)
        self._profile_pill_layout = QVBoxLayout(self._profile_pill_container)
        self._profile_pill_layout.setContentsMargins(12, 0, 12, 0) # Create horizontal padding for the pill look
        
        self._profile_pill = QPushButton(self._profile_pill_container)
        self._profile_pill.setFixedHeight(48)
        self._profile_pill.setFocusPolicy(Qt.NoFocus)
        self._profile_pill.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=24) + "\nQPushButton { text-align: left; }")
        self._profile_pill.setCursor(Qt.PointingHandCursor)
        self._profile_pill.clicked.connect(self._on_profile_click)
        
        self._profile_inner_layout = QHBoxLayout(self._profile_pill)
        self._profile_inner_layout.setContentsMargins(12, 4, 12, 4)
        self._profile_inner_layout.setSpacing(12)
        
        self._profile_avatar = QLabel(self._profile_pill)
        self._profile_avatar.setFixedSize(32, 32)
        self._profile_avatar.setStyleSheet(f"background-color: {resolve(BG_ROOT)}; border-radius: 16px; border: 1px solid {resolve(SIDEBAR_BORDER)};")
        self._profile_avatar.setAlignment(Qt.AlignCenter)
        self._profile_avatar.setFont(LABEL_SM)
        self._profile_inner_layout.addWidget(self._profile_avatar)
        
        self._profile_name = QLabel("User", self._profile_pill)
        self._profile_name.setFont(NAV_ITEM)
        self._profile_name.setStyleSheet(f"background: transparent; border: none; color: {resolve(SIDEBAR_TEXT)};")
        self._profile_inner_layout.addWidget(self._profile_name, stretch=1)
        
        self._profile_pill_layout.addWidget(self._profile_pill)
        self.scroll_layout.addWidget(self._profile_pill_container)
        
        # Bottom Spacing
        bot_spacer = QWidget()
        bot_spacer.setFixedHeight(8)
        bot_spacer.setStyleSheet("background: transparent;")
        self.scroll_layout.addWidget(bot_spacer)

        self._scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self._scroll_area)

    def set_profile(self, username: str, avatar_path: str):
        """Update the profile pill UI."""
        self._profile_name.setText(username)
        if avatar_path and os.path.exists(avatar_path):
            pix = QPixmap(avatar_path)
            if not pix.isNull():
                from PySide6.QtGui import QPainter, QPainterPath
                size = 32
                pix = pix.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                out = QPixmap(size, size)
                out.fill(Qt.transparent)
                painter = QPainter(out)
                painter.setRenderHint(QPainter.Antialiasing)
                path = QPainterPath()
                path.addEllipse(0, 0, size, size)
                painter.setClipPath(path)
                x = (size - pix.width()) // 2
                y = (size - pix.height()) // 2
                painter.drawPixmap(x, y, pix)
                painter.end()
                
                self._profile_avatar.setPixmap(out)
                self._profile_avatar.setStyleSheet("background-color: transparent;")
                self._profile_avatar.setText("")
                return
        
        # Fallback to initials
        initial = username[0].upper() if username else "U"
        self._profile_avatar.setPixmap(QPixmap())  # Clear any image
        self._profile_avatar.setText(initial)

    def refresh_history(self, sessions: List[Dict]):
        while self._hist_layout.count():
            item = self._hist_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        if not sessions:
            lbl = QLabel("No recent chats yet...", self._hist_list)
            lbl.setFont(NAV_ITEM_SM)
            lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; padding: 8px 0px;")
            self._hist_layout.addWidget(lbl)
            return

        for session in sessions:
            item = HistoryItem(session, self._on_session_select, self._on_session_delete, on_pin=self._on_pin, parent=self._hist_list)
            self._hist_layout.addWidget(item)

    def _toggle_history(self):
        if not self._is_open:
            self.toggle()
            self._hist_open = True
            self._hist_wrap.show()
            return
            
        self._hist_open = not self._hist_open
        if self._hist_open:
            self._hist_wrap.show()
        else:
            self._hist_wrap.hide()

    def toggle(self):
        self._is_open = not self._is_open
        
        target_w = SIDEBAR_W_OPEN if self._is_open else SIDEBAR_W_CLOSED
        self.anim_min.setStartValue(self.minimumWidth())
        self.anim_min.setEndValue(target_w)
        self.anim_max.setStartValue(self.maximumWidth())
        self.anim_max.setEndValue(target_w)
        
        if self._icon_open:
            self._toggle_btn.setIcon(self._icon_open if self._is_open else self._icon_closed)

        if self._is_open:
            self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._logo_container.show()
            self.top_layout.setContentsMargins(20, 18, 12, 18)
            
            self._new_btn.setText("  New Chat")
            self._new_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
            
            self._chat_btn.setText("  Chats")
            self._chat_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
            
            self._sep1_wrap.show()
            self._recent_lbl_wrap.show()
            self._hist_wrap.show()
            self._work_btn.setText("  Workspaces")
            self._work_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
            
            for btn, text in self._workspace_btns:
                btn.setText(f"  {text}")
                btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 19px; }")
                
            self._profile_name.show()
            self._profile_pill_layout.setContentsMargins(12, 0, 12, 0)
            self._profile_inner_layout.setContentsMargins(12, 4, 12, 4)
            self._profile_pill.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=24) + "\nQPushButton { text-align: left; }")
        else:
            self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._logo_container.hide()
            self.top_layout.setContentsMargins(0, 18, 14, 18) # Align toggle btn exactly vertically with the columns (56 - 14 - 28 = 14 => icon at 14+5=19)
            
            self._new_btn.setText("")
            self._new_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_ACTIVE, "transparent", radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
            
            self._chat_btn.setText("")
            self._chat_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
            
            self._sep1_wrap.hide()
            self._recent_lbl_wrap.hide()
            self._hist_wrap.hide()
            self._work_btn.setText("")
            self._work_btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 17px; }")
            self._work_container.hide()
            
            for btn, _ in self._workspace_btns:
                btn.setText("")
                btn.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=0) + "\nQPushButton { text-align: left; padding-left: 19px; }")
                
            self._profile_name.hide()
            self._profile_pill_layout.setContentsMargins(0, 0, 0, 0)
            self._profile_inner_layout.setContentsMargins(8, 4, 12, 4) # Tweak this to move avatar further left
            self._profile_pill.setStyleSheet(get_button_style("transparent", SIDEBAR_BTN_HOVER, SIDEBAR_TEXT, radius=24) + "\nQPushButton { text-align: left; padding-left: 12px; }")
            
        self.anim_group.start()

    def _toggle_history(self):
        if self._on_toggle_chats:
            self._on_toggle_chats()

    def _toggle_workspaces(self):
        if not self._is_open:
            self.toggle()
        self._workspaces_open = not self._workspaces_open
        
        # 5 buttons * 32px + 4 spaces * 2px = 168
        target_h = 168 if self._workspaces_open else 0
        
        self._work_anim.stop()
        if self._workspaces_open:
            self._work_container.show()
            self._work_anim.setStartValue(self._work_container.maximumHeight())
            self._work_anim.setEndValue(target_h)
            self._work_anim.start()
        else:
            self._work_anim.setStartValue(self._work_container.maximumHeight())
            self._work_anim.setEndValue(0)
            self._work_anim.start()

    def _hide_work_container_on_finish(self):
        if not self._workspaces_open:
            self._work_container.hide()
