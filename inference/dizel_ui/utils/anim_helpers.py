"""
Animation helpers for Dizel UI.
Provides reusable functions to create common UI transitions.
"""

from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QObject, QPoint
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect


class AnimHelpers:
    """Helper methods for PySide6 UI animations."""

    @staticmethod
    def fade_out(widget: QWidget, duration: int = 250, on_finished=None) -> QPropertyAnimation:
        """Fades out a widget by animating opacity from 1.0 to 0.0."""
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        
        effect.setOpacity(1.0)
        
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        if on_finished:
            anim.finished.connect(on_finished)
            
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        return anim

    @staticmethod
    def fade_in(widget: QWidget, duration: int = 200, on_finished=None) -> QPropertyAnimation:
        """Fades in a widget by animating opacity from 0.0 to 1.0."""
        # Ensure widget is visible before fading in
        widget.show()
        
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        
        effect.setOpacity(0.0)
        
        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        if on_finished:
            anim.finished.connect(on_finished)
            
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        return anim

    @staticmethod
    def slide_in_from_bottom(widget: QWidget, offset: int = 20, duration: int = 200, on_finished=None) -> QPropertyAnimation:
        """Slides a widget in from the bottom while fading it in."""
        widget.show()
        
        # Setup opacity effect
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        
        # Start state
        effect.setOpacity(0.0)
        start_rect = widget.geometry()
        end_rect = widget.geometry()
        start_rect.translate(0, offset)
        widget.setGeometry(start_rect)
        
        # Parent object to hold parallel animations
        group = QObject(widget)
        group.anim_finished_count = 0
        
        def _check_finished():
            group.anim_finished_count += 1
            if group.anim_finished_count == 2:
                if on_finished:
                    on_finished()
                group.deleteLater()
        
        anim_opacity = QPropertyAnimation(effect, b"opacity", group)
        anim_opacity.setDuration(duration)
        anim_opacity.setStartValue(0.0)
        anim_opacity.setEndValue(1.0)
        anim_opacity.setEasingCurve(QEasingCurve.Type.OutQuad)
        anim_opacity.finished.connect(_check_finished)
        
        anim_pos = QPropertyAnimation(widget, b"geometry", group)
        anim_pos.setDuration(duration)
        anim_pos.setStartValue(start_rect)
        anim_pos.setEndValue(end_rect)
        anim_pos.setEasingCurve(QEasingCurve.Type.OutQuad)
        anim_pos.finished.connect(_check_finished)
        
        # Keep reference to avoid garbage collection
        widget._entrance_anim = group 
        
        anim_opacity.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        anim_pos.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        
        return group

    @staticmethod
    def scale_in(widget: QWidget, duration: int = 250, on_finished=None) -> QPropertyAnimation:
        """Pops a widget into view with a slight spring/bounce effect."""
        widget.show()
        
        # PySide6 doesn't easily animate standard widget scale without QGraphicsView / QTransform
        # So we'll fake it by animating the geometry from center outwards if possible, 
        # but for simple avatars standard fade + minor geometry shift is often enough.
        # Let's do a combo of geometry expand + opacity.
        
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
            
        effect.setOpacity(0.0)
        end_rect = widget.geometry()
        
        # Start at 0 size, centered
        start_rect = widget.geometry()
        center = start_rect.center()
        start_rect.setWidth(0)
        start_rect.setHeight(0)
        start_rect.moveCenter(center)
        
        widget.setGeometry(start_rect)
        
        group = QObject(widget)
        group.anim_finished_count = 0
        
        def _check_finished():
            group.anim_finished_count += 1
            if group.anim_finished_count == 2:
                if on_finished:
                    on_finished()
                group.deleteLater()
                
        anim_opacity = QPropertyAnimation(effect, b"opacity", group)
        anim_opacity.setDuration(duration)
        anim_opacity.setStartValue(0.0)
        anim_opacity.setEndValue(1.0)
        anim_opacity.setEasingCurve(QEasingCurve.Type.OutBack)
        anim_opacity.finished.connect(_check_finished)
        
        anim_pos = QPropertyAnimation(widget, b"geometry", group)
        anim_pos.setDuration(duration)
        anim_pos.setStartValue(start_rect)
        anim_pos.setEndValue(end_rect)
        # OutBack gives that springy pop-in feel
        anim_pos.setEasingCurve(QEasingCurve.Type.OutBack)
        anim_pos.finished.connect(_check_finished)
        
        # Keep reference
        widget._scale_anim = group
        
        anim_opacity.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        anim_pos.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        
        return group
