# dizel_ui/ui/waveform_widget.py

import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath

from dizel_ui.theme.colors import resolve, ACCENT, ACCENT_HOVER

class WaveformWidget(QWidget):
    """Animated audio waveform bars driven by amplitude data."""

    def __init__(self, bar_count=24, parent=None):
        super().__init__(parent)
        self.bar_count = bar_count
        self.setFixedHeight(40)
        self.amplitudes = [0.1] * bar_count
        self._target_amp = 0.0
        self._current_amp = 0.0

        # Animation timer to smooth out the transition
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_animation)
        self.timer.start(30) # ~30fps

    def set_amplitude(self, value: float):
        """Set current amplitude (0.0 to 1.0)"""
        self._target_amp = min(1.0, max(0.01, value))

    def _update_animation(self):
        # Smoothly interpolate towards target
        diff = self._target_amp - self._current_amp
        self._current_amp += diff * 0.3
        
        # Shift all bars to the left
        self.amplitudes.pop(0)
        
        # Add new amplitude with some noise for visual flair
        import random
        noise = random.uniform(0.8, 1.2)
        val = min(1.0, max(0.05, self._current_amp * noise))
        self.amplitudes.append(val)
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        
        bar_w = 4
        spacing = 4
        total_width = self.bar_count * (bar_w + spacing) - spacing
        start_x = (w - total_width) / 2

        color = QColor(resolve(ACCENT))
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)

        for i, val in enumerate(self.amplitudes):
            # val is 0.0 to 1.0, scale to height
            # keep min height so it looks like a flat line during silence
            bar_h = max(4, int(val * h))
            
            x = start_x + (i * (bar_w + spacing))
            y = (h - bar_h) / 2
            
            painter.drawRoundedRect(x, y, bar_w, bar_h, 2, 2)
