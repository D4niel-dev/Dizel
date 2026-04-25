from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPainterPath, QPen
from PySide6.QtCore import Qt, QRect, QPropertyAnimation, QEasingCurve, QRectF, Property

class TutorialOverlay(QWidget):
    """
    A full-window overlay that dims the background and cuts out a 
    spotlight (rounded rectangle) over a target widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make the overlay visually present but completely transparent to all mouse clicks
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet("background: transparent;")
        
        self._target_rect = QRectF(0, 0, 0, 0)
        self._current_rect = QRectF(0, 0, 0, 0)
        
        self._anim = QPropertyAnimation(self, b"currentRectF")
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.InOutQuart)
        
    @Property(QRectF)
    def currentRectF(self) -> QRectF:
        return self._current_rect
        
    @currentRectF.setter
    def currentRectF(self, rect: QRectF):
        self._current_rect = rect
        self.update()

    def set_target(self, widget: QWidget, padding: int = 8, animate: bool = True) -> QRectF:
        # Calculate target rect relative to this overlay
        if widget and widget.isVisible():
            # Use global coordinates to bridge across top-level window boundaries (like dialogs)
            global_pt = widget.mapToGlobal(widget.rect().topLeft())
            pt = self.parentWidget().mapFromGlobal(global_pt)
            rect = QRectF(pt.x() - padding, pt.y() - padding, 
                         widget.width() + padding*2, widget.height() + padding*2)
        else:
            # If no actual target (e.g., welcome screen), center a dummy 0-size rect
            rect = QRectF(self.width() / 2, self.height() / 2, 0, 0)

        if not animate or self._current_rect.isEmpty():
            self.currentRectF = rect
        else:
            self._anim.stop()
            self._anim.setStartValue(self._current_rect)
            self._anim.setEndValue(rect)
            self._anim.start()
            
        return rect

    def set_no_spotlight(self, animate: bool = True):
        # Just a point in center, radius 0 so it's fully dim
        rect = QRectF(self.width() / 2, self.height() / 2, 0, 0)
        if not animate or self._current_rect.isEmpty():
            self.currentRectF = rect
        else:
            self._anim.stop()
            self._anim.setStartValue(self._current_rect)
            self._anim.setEndValue(rect)
            self._anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Full overlay dim color
        dim_color = QColor(0, 0, 0, 160)

        path = QPainterPath()
        path.addRect(QRectF(self.rect()))

        if self._current_rect.width() > 0 and self._current_rect.height() > 0:
            # Add cutout
            cutout_path = QPainterPath()
            cutout_path.addRoundedRect(self._current_rect, 12, 12)
            # Subtract cutout from background
            path = path.subtracted(cutout_path)

        painter.fillPath(path, dim_color)

        # Draw a subtle glow/border around the cutout if it exists
        if self._current_rect.width() > 0 and self._current_rect.height() > 0:
            pen = QPen(QColor(138, 43, 226, 180)) # Accent color roughly
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRoundedRect(self._current_rect, 12, 12)


