# dizel_ui/ui/animated_button.py

from PySide6.QtWidgets import QPushButton, QStyleOptionButton, QStylePainter, QStyle
from PySide6.QtCore import Property, QPropertyAnimation, QEasingCurve, Qt
from PySide6.QtGui import QPainter, QIcon

class AnimatedIconButton(QPushButton):
    """
    A QPushButton subclass that animates its icon rotation and scale smoothly.
    To use, call `set_custom_icon(icon)` instead of `setIcon(icon)`.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._icon_rotation = 0.0
        self._icon_scale = 1.0
        self._custom_icon = None
        self._custom_icon_size = 20

    def set_custom_icon(self, icon: QIcon, size=20):
        """Sets the icon to be rendered natively centered and animatable."""
        self._custom_icon = icon
        self._custom_icon_size = size
        self.update()

    def get_icon_rotation(self): 
        return self._icon_rotation
        
    def set_icon_rotation(self, val):
        self._icon_rotation = val
        self.update()
        
    icon_rotation = Property(float, get_icon_rotation, set_icon_rotation)

    def get_icon_scale(self): 
        return self._icon_scale
        
    def set_icon_scale(self, val):
        self._icon_scale = val
        self.update()
        
    icon_scale = Property(float, get_icon_scale, set_icon_scale)

    def animate_rotation(self, end_angle: float, duration: int = 150):
        anim = QPropertyAnimation(self, b"icon_rotation", self)
        anim.setStartValue(self._icon_rotation)
        anim.setEndValue(end_angle)
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.InOutQuart)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def animate_scale(self, end_scale: float, duration: int = 100):
        anim = QPropertyAnimation(self, b"icon_scale", self)
        anim.setStartValue(self._icon_scale)
        anim.setEndValue(end_scale)
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def paintEvent(self, event):
        # 1. Let the button draw its background and standard styling using QStylePainter
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.icon = QIcon() # Clear the icon so default drawControl doesn't draw it natively
        painter.drawControl(QStyle.CE_PushButton, option)

        # 2. Draw our custom icon centrally with applied transforms
        if self._custom_icon:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            cx = self.rect().width() / 2.0
            cy = self.rect().height() / 2.0
            painter.translate(cx, cy)
            
            if self._icon_rotation != 0.0:
                painter.rotate(self._icon_rotation)
                
            if self._icon_scale != 1.0:
                painter.scale(self._icon_scale, self._icon_scale)
            
            # Extract pixmap
            # Use current state modes for pixmap styling (e.g. Disabled/Active)
            mode = QIcon.Normal if self.isEnabled() else QIcon.Disabled
            state = QIcon.On if self.isChecked() else QIcon.Off
            pm = self._custom_icon.pixmap(self._custom_icon_size, self._custom_icon_size, mode, state)
            
            # Offset to center using precise logical coordinates to handle high-DPI (Retina) screens
            w = pm.width() / pm.devicePixelRatio()
            h = pm.height() / pm.devicePixelRatio()
            from PySide6.QtCore import QRectF
            painter.drawPixmap(QRectF(-w/2, -h/2, w, h), pm, QRectF(pm.rect()))
