from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout, QWidget
from PySide6.QtCore import QTimer, Qt, QPropertyAnimation, Property, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath

from dizel_ui.theme.colors import BUBBLE_ASST, TEXT_SECONDARY, TYPING_DOT, resolve, ACCENT, ACCENT_LIGHT
from dizel_ui.theme.fonts import TYPING_TEXT
from dizel_ui.logic.config_manager import ConfigManager
from dizel_ui.theme.stylesheets import get_frame_style

class ThinkingSpinner(QWidget):
    """A premium, smooth glowing arc spinner mimicking modern AI loaders."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._angle = 0
        
        # We animate this custom property
        self.anim = QPropertyAnimation(self, b"angle")
        self.anim.setDuration(1200)
        self.anim.setStartValue(0)
        self.anim.setEndValue(360)
        self.anim.setLoopCount(-1) # infinite
        self.anim.setEasingCurve(QEasingCurve.Type.InOutSine) # Gives that fast-slow spinning feel
        
    def get_angle(self):
        return self._angle
        
    def set_angle(self, angle):
        self._angle = angle
        self.update()
        
    angle = Property(int, get_angle, set_angle)
    
    def start(self):
        if not self.anim.state() == QPropertyAnimation.State.Running:
            self.anim.start()
            
    def stop(self):
        self.anim.stop()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Base faint ring
        pen_bg = QPen(QColor(resolve(TEXT_SECONDARY)).darker(150))
        pen_bg.setWidth(2)
        pen_bg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(2, 2, self.width()-4, self.height()-4, 0, 360 * 16)
        
        # Glowing spinning arc
        pen_fg = QPen(QColor(resolve(ACCENT_LIGHT)))
        pen_fg.setWidth(3)
        pen_fg.setCapStyle(Qt.RoundCap)
        painter.setPen(pen_fg)
        
        # Draw arc: angle in 1/16th of a degree
        span_angle = 120 # arc length
        painter.drawArc(2, 2, self.width()-4, self.height()-4, -self._angle * 16, span_angle * 16)
        painter.end()


class TypingIndicator(QFrame):
    """
    Animated typing indicator container shown while generating.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        bg = resolve(BUBBLE_ASST)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border-radius: 14px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)
        
        self.spinner = ThinkingSpinner(self)
        
        self.lbl = QLabel("Thinking")
        self.lbl.setFont(TYPING_TEXT)
        self.lbl.setStyleSheet(f"color: {resolve(TEXT_SECONDARY)}; border: none; background: transparent;")
        
        layout.addWidget(self.spinner)
        layout.addWidget(self.lbl)
        layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def start(self):
        app_cfg = ConfigManager.load().get("appearance", {})
        if not app_cfg.get("animations", True):
            self.lbl.setText("Thinking...")
            self.spinner.hide()
            return
            
        self.spinner.start()

    def stop(self):
        self.spinner.stop()
