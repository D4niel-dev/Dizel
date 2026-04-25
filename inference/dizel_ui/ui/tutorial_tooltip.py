from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QRect, QPoint
from PySide6.QtGui import QColor

from dizel_ui.theme.colors import resolve, BG_INPUT, BORDER, ACCENT, TEXT_PRIMARY, TEXT_DIM, SEND_BTN, SEND_BTN_HOVER
from dizel_ui.theme.fonts import LABEL_SM, NAV_ITEM, LABEL_DIM
from dizel_ui.theme.stylesheets import get_frame_style, get_button_style
from dizel_ui.utils.icons import get_icon
from dizel_ui.logic.tutorial_manager import TutorialStep

class TutorialTooltip(QFrame):
    """
    A floating instruction panel that shows the current tutorial step.
    Provides Next, Back, Skip, and Rating controls.
    """
    next_clicked = Signal()
    prev_clicked = Signal()
    skip_clicked = Signal()
    finish_clicked = Signal(int) # rating

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedWidth(400) # Increased slightly to account for shadow margins
        
        # Outer layout to hold the container and shadow margins
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(20, 20, 20, 30) # Bottom needs 30px for shadow
        
        self.container = QFrame(self)
        bg = resolve(BG_INPUT)
        border = resolve(BORDER)
        self.container.setStyleSheet(get_frame_style(bg, radius=16, border_color=border))
        
        # Shadow applied to container, not the top-level window
        shadow = QGraphicsDropShadowEffect(self.container)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 8)
        self.container.setGraphicsEffect(shadow)
        
        outer_layout.addWidget(self.container)

        self._build_ui()
        
        self.current_step_index = 0
        self.total_steps = 1
        
        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def _build_ui(self):
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Header (Step count + Title)
        self.step_lbl = QLabel("Step 1 of X", self.container)
        self.step_lbl.setFont(LABEL_SM)
        self.step_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent; border: none;")
        layout.addWidget(self.step_lbl)
        
        self.title_lbl = QLabel("Title", self.container)
        self.title_lbl.setFont(NAV_ITEM)
        self.title_lbl.setStyleSheet(f"color: {resolve(ACCENT)}; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(self.title_lbl)
        
        self.body_lbl = QLabel("Body text goes here...", self.container)
        self.body_lbl.setFont(LABEL_DIM)
        self.body_lbl.setWordWrap(True)
        self.body_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent; border: none;")
        layout.addWidget(self.body_lbl)
        
        # Rating container (hidden by default)
        self.rating_container = QFrame(self.container)
        self.rating_container.setStyleSheet("background: transparent; border: none;")
        r_layout = QHBoxLayout(self.rating_container)
        r_layout.setContentsMargins(0, 8, 0, 8)
        self.stars = []
        self._current_rating = 0
        for i in range(1, 6):
            btn = QPushButton("☆", self.rating_container)
            btn.setFixedSize(32, 32)
            btn.setFont(NAV_ITEM)
            btn.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent; border: none;")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, r=i: self._on_star_click(r))
            self.stars.append(btn)
            r_layout.addWidget(btn)
        r_layout.addStretch()
        self.rating_container.hide()
        layout.addWidget(self.rating_container)
        
        # Bottom Actions
        self.action_layout = QHBoxLayout()
        self.action_layout.setContentsMargins(0, 8, 0, 0)
        
        self.skip_btn = QPushButton("Skip", self.container)
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent; border: none;")
        self.skip_btn.clicked.connect(self.skip_clicked.emit)
        
        self.back_btn = QPushButton("← Back", self.container)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent; border: none;")
        self.back_btn.clicked.connect(self.prev_clicked.emit)
        
        self.next_btn = QPushButton("Next →", self.container)
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setFixedHeight(32)
        self.next_btn.setStyleSheet(get_button_style(SEND_BTN, SEND_BTN_HOVER, "#ffffff", radius=16))
        self.next_btn.clicked.connect(self.next_clicked.emit)
        
        self.finish_btn = QPushButton("Finish", self.container)
        self.finish_btn.setCursor(Qt.PointingHandCursor)
        self.finish_btn.setFixedHeight(32)
        self.finish_btn.setStyleSheet(get_button_style(SEND_BTN, SEND_BTN_HOVER, "#ffffff", radius=16))
        self.finish_btn.clicked.connect(lambda: self.finish_clicked.emit(self._current_rating))
        self.finish_btn.hide()
        
        self.action_layout.addWidget(self.skip_btn)
        self.action_layout.addStretch()
        self.action_layout.addWidget(self.back_btn)
        self.action_layout.addSpacing(8)
        self.action_layout.addWidget(self.next_btn)
        self.action_layout.addWidget(self.finish_btn)
        
        layout.addLayout(self.action_layout)

    def set_step(self, step: TutorialStep, index: int, total: int):
        self.current_step_index = index
        self.total_steps = total
        
        self.step_lbl.setText(f"Step {index + 1} of {total}")
        self.title_lbl.setText(step.title)
        self.body_lbl.setText(step.body)
        
        self.back_btn.setVisible(index > 0)
        
        is_last = (index == total - 1)
        self.next_btn.setVisible(not is_last and not getattr(step, "require_action", False))
        self.finish_btn.setVisible(is_last)
        self.skip_btn.setVisible(not is_last)
        
        if is_last:
            self.rating_container.show()
            self._set_stars(0) # reset
        else:
            self.rating_container.hide()
            
        self.adjustSize()

    def _on_star_click(self, rating: int):
        self._current_rating = rating
        self._set_stars(rating)
        
    def _set_stars(self, rating: int):
        for i, btn in enumerate(self.stars):
            if i < rating:
                btn.setText("★")
                btn.setStyleSheet(f"color: {resolve(ACCENT)}; background: transparent; border: none; font-size: 20px;")
            else:
                btn.setText("☆")
                btn.setStyleSheet(f"color: {resolve(TEXT_DIM)}; background: transparent; border: none; font-size: 20px;")

    def move_adjacent_to(self, target_rect: QRect, margin: int = 24):
        # Determine best placement
        if not self.parentWidget(): return
        
        parent = self.parentWidget()
        parent_rect = parent.rect()
        w = self.width()
        h = self.height()
        
        # If the target is tall and narrow (e.g. sidebar), place to the right of it
        if target_rect.height() > target_rect.width() * 2:
            x = target_rect.right() + margin
            y = target_rect.center().y() - h // 2
        else:
            # Standard horizontal alignment
            x = target_rect.center().x() - w // 2
            
            # Try below
            y = target_rect.bottom() + margin
            if y + h > parent_rect.bottom():
                # Try above
                y = target_rect.top() - h - margin
            
        # Clamp to bounds of the parent window (but mapped to global later)
        min_x = 16
        max_x = parent_rect.width() - w - 16
        x = max(min_x, min(x, max_x))
        
        min_y = 16
        max_y = parent_rect.height() - h - 16
        y = max(min_y, min(y, max_y))
        
        # Convert local point to global screen coordinates so the top-level window follows it
        local_pt = QPoint(x, y)
        global_pt = parent.mapToGlobal(local_pt)
        
        if self.pos() == QPoint(0,0):
            self.move(global_pt)
        else:
            self._anim.stop()
            self._anim.setStartValue(self.pos())
            self._anim.setEndValue(global_pt)
            self._anim.start()
