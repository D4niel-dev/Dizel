import math
import os
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Property, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap

class AvatarWidget(QWidget):
    """
    Renders the visual AI Avatar (Dizel).
    Handles completely smooth Qt-level drawing rather than pixmaps for clean vector scaling and property modifications.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        
        # Load custom Diszi PNG
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            avatar_path = os.path.join(base_dir, "assets", "avatars", "Diszi_beta2.png")
            if os.path.exists(avatar_path):
                original = QPixmap(avatar_path)
                if not original.isNull():
                    # Statically crop exactly the 600x600 core. This perfectly captures all glowing rings
                    # while trimming off the massive 1080x1080 blank void that ruined spacing!
                    cropped = original.copy(240, 240, 600, 600)
                    self._avatar_pixmap = cropped.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:
            pass
        
        self._scale = 1.0
        self._opacity = 1.0
        self._glow_intensity = 0.0
        self._ring_rotation = 0.0
        self._ring_opacity = 0.0
        self._base_color = QColor("#0ea5e9") # ACCENT

    # -----------------------------
    # Properties for Animation
    # -----------------------------
    def get_scale(self): return self._scale
    def set_scale(self, var): 
        self._scale = var
        self.update()
    scale_prop = Property(float, get_scale, set_scale)

    def get_glow(self): return self._glow_intensity
    def set_glow(self, var): 
        self._glow_intensity = var
        self.update()
    glow_prop = Property(float, get_glow, set_glow)

    def get_rotation(self): return self._ring_rotation
    def set_rotation(self, var): 
        self._ring_rotation = var
        self.update()
    rotation_prop = Property(float, get_rotation, set_rotation)

    def get_ring_op(self): return self._ring_opacity
    def set_ring_op(self, var): 
        self._ring_opacity = var
        self.update()
    ring_op_prop = Property(float, get_ring_op, set_ring_op)

    def get_opacity(self): return self._opacity
    def set_opacity(self, var): 
        self._opacity = var
        self.update()
    opacity_prop = Property(float, get_opacity, set_opacity)

    def get_color_shift(self): return self._base_color
    def set_color_shift(self, var): 
        self._base_color = var
        self.update()
    color_prop = Property(QColor, get_color_shift, set_color_shift)

    # -----------------------------
    # Render Logic
    # -----------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        painter.setOpacity(self._opacity)
        
        center = self.rect().center()
        
        # Main body sizing
        base_radius = 48
        radius = base_radius * self._scale
        
        # 1. Background Glow
        if self._glow_intensity > 0:
            glow_radius = radius + (14 * self._glow_intensity)
            glow_color = QColor(self._base_color)
            glow_color.setAlpha(int(100 * self._glow_intensity))
            painter.setBrush(QBrush(glow_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, glow_radius, glow_radius)

        # 2. Orbital Thinking Ring
        if self._ring_opacity > 0:
            painter.setOpacity(self._opacity * self._ring_opacity)
            pen = QPen(self._base_color)
            pen.setWidthF(1.5)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            
            painter.translate(center)
            painter.rotate(self._ring_rotation)
            
            ring_padding = 6
            arc_radius = radius + ring_padding
            arc_rect = QRectF(-arc_radius, -arc_radius, arc_radius*2, arc_radius*2)
            
            arc_len = 16 * 60
            painter.drawArc(arc_rect, 0, arc_len)
            painter.drawArc(arc_rect, 16 * 120, arc_len)
            painter.drawArc(arc_rect, 16 * 240, arc_len)
            
            painter.rotate(-self._ring_rotation)
            painter.translate(-center)
            
            painter.setOpacity(self._opacity)

        # 3. Core Character Avatar (Replaces the blank blue orb)
        if hasattr(self, "_avatar_pixmap") and not self._avatar_pixmap.isNull():
            # Calculate the target rectangle based on radius
            target_rect = QRectF(center.x() - radius, center.y() - radius, radius*2, radius*2)
            painter.drawPixmap(target_rect.toRect(), self._avatar_pixmap)
        else:
            # Fallback if image fails to load
            painter.setBrush(QBrush(self._base_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(center, radius, radius)
