from PySide6.QtCore import QPropertyAnimation, QParallelAnimationGroup, QSequentialAnimationGroup, QEasingCurve
from PySide6.QtGui import QColor

def make_fade_in(widget, duration=800):
    group = QParallelAnimationGroup(widget)
    
    op = QPropertyAnimation(widget, b"opacity_prop")
    op.setStartValue(0.0)
    op.setEndValue(1.0)
    op.setDuration(duration)
    op.setEasingCurve(QEasingCurve.InOutQuad)
    
    scale = QPropertyAnimation(widget, b"scale_prop")
    scale.setStartValue(0.8)
    scale.setEndValue(1.0)
    scale.setDuration(duration)
    scale.setEasingCurve(QEasingCurve.OutCubic)
    
    group.addAnimation(op)
    group.addAnimation(scale)
    return group

def make_idle(widget):
    # Smooth continuous subtle breathing
    anim = QPropertyAnimation(widget, b"scale_prop")
    anim.setStartValue(1.0)
    anim.setKeyValueAt(0.5, 1.04)
    anim.setEndValue(1.0)
    anim.setDuration(2400)
    anim.setEasingCurve(QEasingCurve.InOutSine)
    anim.setLoopCount(-1) # infinite
    return anim

def make_listening(widget):
    group = QParallelAnimationGroup(widget)
    
    glow = QPropertyAnimation(widget, b"glow_prop")
    glow.setStartValue(widget.get_glow())
    glow.setEndValue(1.0)
    glow.setDuration(600)
    glow.setEasingCurve(QEasingCurve.InOutSine)
    
    color = QPropertyAnimation(widget, b"color_prop")
    color.setStartValue(widget.get_color_shift())
    color.setEndValue(QColor("#38bdf8")) # brighter accent glow
    color.setDuration(600)
    
    group.addAnimation(glow)
    group.addAnimation(color)
    return group

def make_thinking(widget, spin_duration=1800):
    group = QParallelAnimationGroup(widget)
    
    # 1. Fade in the orbital ring
    ring_op = QPropertyAnimation(widget, b"ring_op_prop")
    ring_op.setStartValue(widget.get_ring_op())
    ring_op.setEndValue(1.0)
    ring_op.setDuration(400)
    
    # 2. Infinite rotation
    spin = QPropertyAnimation(widget, b"rotation_prop")
    spin.setStartValue(0.0)
    spin.setEndValue(360.0)
    spin.setDuration(spin_duration)
    spin.setLoopCount(-1)
    
    # 3. Compress the core slightly to highlight the ring
    scale = QPropertyAnimation(widget, b"scale_prop")
    scale.setStartValue(widget.get_scale())
    scale.setEndValue(0.92)
    scale.setDuration(400)
    scale.setEasingCurve(QEasingCurve.InOutSine)
    
    group.addAnimation(ring_op)
    group.addAnimation(spin)
    group.addAnimation(scale)
    return group

def make_responding(widget):
    group = QParallelAnimationGroup(widget)
    
    # Gentle flicker glow
    glow = QPropertyAnimation(widget, b"glow_prop")
    glow.setStartValue(0.4)
    glow.setKeyValueAt(0.5, 1.0)
    glow.setEndValue(0.4)
    glow.setDuration(1200)
    glow.setEasingCurve(QEasingCurve.InOutSine)
    glow.setLoopCount(-1)
    
    # Pulsing scale
    scale = QPropertyAnimation(widget, b"scale_prop")
    scale.setStartValue(1.0)
    scale.setKeyValueAt(0.5, 1.08)
    scale.setEndValue(1.0)
    scale.setDuration(1200)
    scale.setEasingCurve(QEasingCurve.InOutSine)
    scale.setLoopCount(-1)
    
    group.addAnimation(glow)
    group.addAnimation(scale)
    return group

def make_error(widget):
    group = QSequentialAnimationGroup(widget)
    
    c1 = QPropertyAnimation(widget, b"color_prop")
    c1.setStartValue(widget.get_color_shift())
    c1.setEndValue(QColor("#f43f5e")) # red
    c1.setDuration(200)
    
    c2 = QPropertyAnimation(widget, b"color_prop")
    c2.setStartValue(QColor("#f43f5e"))
    c2.setEndValue(QColor("#0ea5e9")) # normal theme
    c2.setDuration(800)
    
    group.addAnimation(c1)
    group.addAnimation(c2)
    return group

def make_cleanup(widget):
    """
    Creates a transition group to cleanly reset unmanaged properties back to defaults
    before a new state locks into loops.
    """
    group = QParallelAnimationGroup(widget)
    
    # Safely animate them to default
    s = QPropertyAnimation(widget, b"scale_prop")
    s.setEndValue(1.0); s.setDuration(300); s.setEasingCurve(QEasingCurve.InOutQuad)
    
    g = QPropertyAnimation(widget, b"glow_prop")
    g.setEndValue(0.0); g.setDuration(300)
    
    r = QPropertyAnimation(widget, b"ring_op_prop")
    r.setEndValue(0.0); r.setDuration(300)
    
    c = QPropertyAnimation(widget, b"color_prop")
    c.setEndValue(QColor("#0ea5e9")); c.setDuration(300)
    
    group.addAnimation(s); group.addAnimation(g); group.addAnimation(r); group.addAnimation(c)
    return group
