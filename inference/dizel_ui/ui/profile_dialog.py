# dizel_ui/ui/profile_dialog.py
import os
import shutil
import uuid
from typing import Dict, Any

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QFileDialog, QFrame, QSizePolicy, QWidget,
    QComboBox
)
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QTimer, QEvent
from PySide6.QtGui import QPixmap, QIcon

from dizel_ui.theme.colors import (
    BG_ROOT, BG_CHAT, ACCENT, ACCENT_HOVER, 
    TEXT_PRIMARY, TEXT_DIM, BORDER, resolve, ACTION_PILL, ACCENT_LIGHT
)
from dizel_ui.theme.fonts import WELCOME_TITLE, NAV_ITEM, BTN_LABEL, WELCOME_SUB
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style, get_input_style
from dizel_ui.utils.icons import get_icon

_HERE = os.path.dirname(os.path.abspath(__file__))
_UI_DIR = os.path.dirname(_HERE)
_DATA_DIR = os.path.join(_UI_DIR, ".dizel")

class ProfileDialog(QDialog):
    def __init__(self, current_profile: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._profile = current_profile.copy()
        if not self._profile.get("display_name"):
            self._profile["display_name"] = self._profile.get("username", "User")
        if not self._profile.get("username"):
            self._profile["username"] = "@user"
            
        self.setWindowTitle("Edit Profile")
        self.setFixedSize(400, 620)
        
        # Transparent Frameless Window
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.new_avatar_path = None

        main_lyt = QVBoxLayout(self)
        main_lyt.setContentsMargins(0, 0, 0, 0)
        
        # Background Container
        self.bg_frame = QFrame(self)
        self.bg_frame.setObjectName("BgFrame")
        self.bg_frame.setStyleSheet(f"""
            QFrame#BgFrame {{
                background-color: {resolve(BG_ROOT)};
                border-radius: 16px;
                border: 1px solid {resolve(BORDER)};
                /* add slight shadow feel with darker base */
            }}
        """)
        main_lyt.addWidget(self.bg_frame)
        
        layout = QVBoxLayout(self.bg_frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header Banner
        header = QFrame(self.bg_frame)
        header.setFixedHeight(120)
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {resolve(ACCENT)}, stop:1 {resolve(ACCENT_LIGHT)});
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
            }}
        """)
        header_lyt = QHBoxLayout(header)
        header_lyt.setContentsMargins(24, 16, 24, 16)
        
        close_btn = QPushButton("✕", header)
        close_btn.setFixedSize(30, 30)
        close_btn.setFont(BTN_LABEL)
        close_btn.setStyleSheet(get_button_style("transparent", "rgba(255,255,255,0.2)", "white", radius=15))
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.reject)
        
        header_lyt.addStretch(1)
        header_lyt.addWidget(close_btn, alignment=Qt.AlignTop | Qt.AlignRight)
        
        layout.addWidget(header)

        # Body Layout
        body_lyt = QVBoxLayout()
        body_lyt.setContentsMargins(32, 0, 32, 24)
        body_lyt.setSpacing(0)
        
        # --- Avatar Offset Section ---
        # Status Ring Frame
        self.ring_frame = QFrame(self.bg_frame)
        self.ring_frame.setFixedSize(132, 132)
        self.ring_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {resolve(ACCENT)}, stop:1 {resolve(ACCENT_LIGHT)});
                border-radius: 66px;
            }}
        """)
        self.ring_frame.move((400 - 132) // 2, 54) # Offset to align with avatar overlap

        self.avatar_btn = QPushButton(self.bg_frame)
        self.avatar_btn.setFixedSize(120, 120)
        self.avatar_btn.setCursor(Qt.PointingHandCursor)
        self.avatar_btn.move( (400 - 120) // 2, 60 ) # Center horizontally, offset vertically to overlap banner
        
        self._update_avatar_preview()
        self.avatar_btn.clicked.connect(self._select_avatar)

        self.avatar_btn.installEventFilter(self)
        
        self.avatar_overlay = QLabel(self.bg_frame)
        self.avatar_overlay.setFixedSize(120, 120)
        self.avatar_overlay.move((400 - 120) // 2, 60)
        self.avatar_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 60px;")
        self.avatar_overlay.setAlignment(Qt.AlignCenter)
        self.avatar_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        icon = get_icon("user-cap", (32, 32), "white")
        if icon:
            self.avatar_overlay.setPixmap(icon.pixmap(32, 32))
        self.avatar_overlay.hide()

        # Spacer to account for the overlapping avatar
        layout.addSpacing(70) 

        helper = QLabel("Click to change avatar", self.bg_frame)
        helper.setFont(WELCOME_SUB)
        helper.setStyleSheet(f"color: {resolve(TEXT_DIM)}; margin-top: 10px;")
        layout.addWidget(helper, alignment=Qt.AlignCenter)
        
        layout.addSpacing(20)

        # --- Fields Section ---
        fields_container = QWidget()
        fields_lyt = QVBoxLayout(fields_container)
        fields_lyt.setContentsMargins(32, 0, 32, 0)
        fields_lyt.setSpacing(16)
        
        # Display Name
        disp_lyt = QVBoxLayout()
        disp_lyt.setSpacing(8)
        name_lbl = QLabel("Display Name")
        name_lbl.setFont(NAV_ITEM)
        name_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-weight: bold;")
        disp_lyt.addWidget(name_lbl)
        
        self.name_input = QLineEdit(self._profile.get("display_name", self._profile.get("username", "User")))
        self.name_input.setFixedHeight(48)
        self.name_input.setFont(NAV_ITEM)
        self.name_input.setPlaceholderText("Enter your display name...")
        self.name_input.setStyleSheet(get_input_style(BG_CHAT, BORDER, ACCENT, TEXT_PRIMARY, radius=12) + " QLineEdit { padding: 0 16px; }")
        disp_lyt.addWidget(self.name_input)
        fields_lyt.addLayout(disp_lyt)
        
        # Username
        usr_lyt = QVBoxLayout()
        usr_lyt.setSpacing(8)
        usr_lbl = QLabel("Username")
        usr_lbl.setFont(NAV_ITEM)
        usr_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-weight: bold;")
        usr_lyt.addWidget(usr_lbl)
        
        self.username_input = QLineEdit(self._profile.get("username", "@user"))
        self.username_input.setFixedHeight(48)
        self.username_input.setFont(NAV_ITEM)
        self.username_input.setPlaceholderText("Choose a unique username...")
        self.username_input.setStyleSheet(get_input_style(BG_CHAT, BORDER, ACCENT, TEXT_PRIMARY, radius=12) + " QLineEdit { padding: 0 16px; }")
        usr_lyt.addWidget(self.username_input)
        fields_lyt.addLayout(usr_lyt)
        
        # Theme Toggle
        theme_lyt = QVBoxLayout()
        theme_lyt.setSpacing(8)
        theme_lbl = QLabel("Theme Preference")
        theme_lbl.setFont(NAV_ITEM)
        theme_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-weight: bold;")
        theme_lyt.addWidget(theme_lbl)
        
        self.theme_combo = QComboBox()
        self.theme_combo.setFixedHeight(48)
        self.theme_combo.setFont(NAV_ITEM)
        self.theme_combo.addItems(["light", "dark", "blue_dark"])
        
        current_theme = self._profile.get("theme", "dark")
        idx = self.theme_combo.findText(current_theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
            
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {resolve(BG_CHAT)};
                border: 1px solid {resolve(BORDER)};
                border-radius: 12px;
                color: {resolve(TEXT_PRIMARY)};
                padding: 0 16px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox:focus {{
                border: 1px solid {resolve(ACCENT)};
            }}
            QComboBox QAbstractItemView {{
                background-color: {resolve(BG_CHAT)};
                color: {resolve(TEXT_PRIMARY)};
                selection-background-color: {resolve(ACCENT)};
                selection-color: white;
                border: 1px solid {resolve(BORDER)};
                border-radius: 4px;
            }}
        """)
        theme_lyt.addWidget(self.theme_combo)
        fields_lyt.addLayout(theme_lyt)
        
        layout.addWidget(fields_container)

        layout.addStretch(1)

        # --- Buttons Section ---
        btn_container = QWidget()
        btn_lyt = QHBoxLayout(btn_container)
        btn_lyt.setContentsMargins(32, 0, 32, 32)
        btn_lyt.setSpacing(12)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(44)
        self.cancel_btn.setFont(BTN_LABEL)
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.setStyleSheet(get_button_style("transparent", BG_CHAT, TEXT_DIM, radius=12, border_color=BORDER))
        self.cancel_btn.clicked.connect(self.reject)
        
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.setFixedHeight(44)
        self.save_btn.setFont(BTN_LABEL)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, "#FFFFFF", radius=12))
        self.save_btn.clicked.connect(self._save_profile)
        
        btn_lyt.addWidget(self.cancel_btn, stretch=1)
        btn_lyt.addWidget(self.save_btn, stretch=2)
        
        layout.addWidget(btn_container)

    def _update_avatar_preview(self):
        avatar_path = None
        # Did the user pick a new one?
        if self.new_avatar_path:
            avatar_path = self.new_avatar_path
        elif self._profile.get("avatar"):
            # Existing relative path resolving
            avatar_path = os.path.join(_DATA_DIR, self._profile["avatar"])

        if avatar_path and os.path.exists(avatar_path):
            pix = QPixmap(avatar_path)
            if not pix.isNull():
                from PySide6.QtGui import QPainter, QPainterPath
                size = 120
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
                
                icon = QIcon(out)
                self.avatar_btn.setIcon(icon)
                self.avatar_btn.setIconSize(QSize(120, 120))
                self.avatar_btn.setStyleSheet(f"""
                    QPushButton {{
                        border-radius: 60px;
                        background-color: {resolve(BG_ROOT)};
                        border: 4px solid {resolve(BG_ROOT)};
                    }}
                    QPushButton:hover {{
                        border-color: {resolve(ACCENT)};
                    }}
                """)
                self.avatar_btn.setText("")
                return
                
        # Default empty fallback
        self.avatar_btn.setIcon(QIcon())
        self.avatar_btn.setText(self._profile.get("username", "U")[0].upper())
        self.avatar_btn.setFont(WELCOME_TITLE)
        self.avatar_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {resolve(BG_CHAT)};
                border: 4px solid {resolve(BG_ROOT)};
                border-radius: 60px;
                color: {resolve(TEXT_DIM)};
            }}
            QPushButton:hover {{
                border-color: {resolve(ACCENT)};
            }}
        """)

    def _select_avatar(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Avatar Image", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if file_path:
            self.new_avatar_path = file_path
            self._update_avatar_preview()

    def _save_profile(self):
        self.save_btn.setText("✓ Saved!")
        self.save_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, "#FFFFFF", radius=12).replace(resolve(ACCENT), "#10b981"))
        self.save_btn.repaint()
        
        display_name = self.name_input.text().strip()
        if display_name:
            self._profile["display_name"] = display_name
            
        username = self.username_input.text().strip()
        if username:
            self._profile["username"] = username
            
        self._profile["theme"] = self.theme_combo.currentText()
            
        if self.new_avatar_path:
            os.makedirs(_DATA_DIR, exist_ok=True)
            ext = os.path.splitext(self.new_avatar_path)[1].lower()
            if not ext:
                ext = ".png"
            new_filename = f"avatar_{uuid.uuid4().hex[:8]}{ext}"
            target_path = os.path.join(_DATA_DIR, new_filename)
            try:
                shutil.copy2(self.new_avatar_path, target_path)
                
                # Delete old avatar if it existed and is different
                old_avatar = self._profile.get("avatar")
                if old_avatar:
                    old_path = os.path.join(_DATA_DIR, old_avatar)
                    if os.path.exists(old_path) and os.path.isfile(old_path):
                        os.remove(old_path)
                        
                self._profile["avatar"] = new_filename
            except Exception as e:
                print(f"Error copying avatar: {e}")

        QTimer.singleShot(400, self.accept)

    def get_profile(self) -> Dict[str, Any]:
        return self._profile

    def showEvent(self, event):
        super().showEvent(event)
        self.setWindowOpacity(0.0)
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(200)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_anim.start()

    def reject(self):
        self._fade_out(lambda: QDialog.reject(self))
        
    def accept(self):
        self._fade_out(lambda: QDialog.accept(self))
        
    def _fade_out(self, callback):
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(150)
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.InCubic)
        self._fade_anim.finished.connect(callback)
        self._fade_anim.start()

    def eventFilter(self, obj, event):
        if obj == self.avatar_btn:
            if event.type() == QEvent.Enter:
                self.avatar_overlay.show()
            elif event.type() == QEvent.Leave:
                self.avatar_overlay.hide()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event)

    # --- Frameless Window Dragging ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
