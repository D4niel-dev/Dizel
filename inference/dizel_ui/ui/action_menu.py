# dizel_ui/ui/action_menu.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect, QPoint
from PySide6.QtGui import QColor, QFont, QPaintEvent, QPainter, QPainterPath

from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.colors import (
    BG_ROOT, BG_INPUT, BG_INPUT_FIELD, ACCENT, TEXT_PRIMARY, TEXT_DIM, BORDER, resolve, WELCOME_CARD_HOVER
)
from dizel_ui.theme.fonts import LABEL, NAV_ITEM

class _SectionLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFont(LABEL)
        color = resolve(TEXT_DIM)
        self.setStyleSheet(f"color: {color}; padding: 4px 8px;")

class _Separator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)
        bg = resolve(BORDER)
        self.setStyleSheet(f"background-color: {bg}; margin: 4px 0;")

class _MenuItem(QPushButton):
    """Base item with hover styling."""
    def __init__(self, text: str, icon_name: str, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setCursor(Qt.PointingHandCursor)
        self.setText(text)
        self.setFont(NAV_ITEM)
        
        self.icon_name = icon_name
        self.base_color = resolve(TEXT_PRIMARY)
        
        # We handle painting the icon manually or setting it as stylesheet padding.
        # But for QPushButtons, setting setIcon is easiest.
        if icon_name:
            self.setIcon(get_icon(icon_name, color=self.base_color))
            self.setIconSize(self.fontMetrics().boundingRect("W").size() * 1.2) # approx sizing

        # Set stylesheet
        text_color = resolve(TEXT_PRIMARY)
        hover_bg = resolve(WELCOME_CARD_HOVER)
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {text_color};
                text-align: left;
                padding-left: 8px;
                padding-right: 8px;
                border: none;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {hover_bg};
            }}
        """)

class _ToggleItem(_MenuItem):
    """Item that acts like a toggle (checkbox context)."""
    toggled_state = Signal(str, bool) # id, state
    
    def __init__(self, text: str, icon_name: str, item_id: str, is_active: bool = False, parent=None):
        super().__init__(text, icon_name, parent)
        self.item_id = item_id
        self.is_active = is_active
        self.clicked.connect(self._toggle)
        self._update_style()
        
    def _toggle(self):
        self.is_active = not self.is_active
        self._update_style()
        self.toggled_state.emit(self.item_id, self.is_active)
        
    def _update_style(self):
        # We can simulate active state by changing text/icon color or adding a check
        color = resolve(ACCENT) if self.is_active else resolve(TEXT_PRIMARY)
        hover_bg = resolve(WELCOME_CARD_HOVER)
        
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {color};
                text-align: left;
                padding-left: 8px;
                padding-right: 8px;
                border: none;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {hover_bg};
            }}
        """)
        if self.icon_name:
            self.setIcon(get_icon(self.icon_name, color=color))

class ActionMenu(QWidget):
    tool_toggled = Signal(str, bool)
    action_triggered = Signal(str)
    menu_hidden = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setFixedWidth(200)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12) # Padding for shadow
        
        self.container = QFrame(self)
        self.container.setObjectName("MenuContainer")
        
        bg = resolve(BG_INPUT)
        border = resolve(BORDER)
        self.container.setStyleSheet(f"""
            QFrame#MenuContainer {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
        """)
        
        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 8)
        self.container.setGraphicsEffect(shadow)
        
        layout.addWidget(self.container)
        
        c_layout = QVBoxLayout(self.container)
        c_layout.setContentsMargins(6, 6, 6, 6)
        c_layout.setSpacing(2)
        
        # --- Tools Section ---
        c_layout.addWidget(_SectionLabel("Tools"))
        
        self.toggles = {}
        
        self._add_toggle("Web Search", "globe", "web", c_layout)
        self._add_toggle("Deep Think", "cpu", "deep", c_layout)
        self._add_toggle("Thinking", "zap", "thinking", c_layout)
        self._add_toggle("Parse Files", "file-text", "files", c_layout)
        
        c_layout.addWidget(_Separator())
        
        # --- Attachments Section ---
        c_layout.addWidget(_SectionLabel("Attachments"))
        self._add_action("Upload Image", "images", "upload_image", c_layout)
        self._add_action("Upload File", "file-plus", "upload_file", c_layout)
        
        c_layout.addWidget(_Separator())
        
        self._add_action("Reset All", "x-circle", "reset_all", c_layout)
        
        # Animation properties
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity")
        
    def _add_toggle(self, text, icon, item_id, layout):
        item = _ToggleItem(text, icon, item_id)
        item.toggled_state.connect(self._on_toggle)
        layout.addWidget(item)
        self.toggles[item_id] = item
        
    def _add_action(self, text, icon, action_id, layout):
        item = _MenuItem(text, icon)
        item.clicked.connect(lambda _, a=action_id: self._on_action(a))
        layout.addWidget(item)
        
    def _on_toggle(self, item_id: str, state: bool):
        # Mutual exclusivity
        if state:
            if item_id == "thinking" and self.toggles["deep"].is_active:
                self.toggles["deep"].is_active = False
                self.toggles["deep"]._update_style()
                self.tool_toggled.emit("deep", False)
            elif item_id == "deep" and self.toggles["thinking"].is_active:
                self.toggles["thinking"].is_active = False
                self.toggles["thinking"]._update_style()
                self.tool_toggled.emit("thinking", False)
                
        self.tool_toggled.emit(item_id, state)
        
    def _on_action(self, action_id: str):
        self.action_triggered.emit(action_id)
        self.close()

    def sync_state(self, active_contexts: set):
        """Syncs the menu toggles with external state."""
        for tid, t_item in self.toggles.items():
            t_item.is_active = (tid in active_contexts)
            t_item._update_style()

    def show_above(self, widget: QWidget):
        """Show the menu positioned above a specific widget, aligned to its left."""
        pos = widget.mapToGlobal(QPoint(0, 0))
        # Determine actual menu size. We need to format layout first
        self.layout().activate()
        self.adjustSize()
        height = self.height()
        
        # Position slightly above the widget
        target_pos = pos - QPoint(10, height - 10) 
        self.move(target_pos)
        
        # Fade animation
        self.setWindowOpacity(0.0)
        self._opacity_anim.setDuration(150)
        self._opacity_anim.setStartValue(0.0)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        self.show()
        self._opacity_anim.start()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.menu_hidden.emit()
