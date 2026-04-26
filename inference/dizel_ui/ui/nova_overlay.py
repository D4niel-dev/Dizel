# dizel_ui/ui/nova_overlay.py

import os
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QWidget, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap

from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.colors import (BG_INPUT, BG_INPUT_FIELD, TEXT_PRIMARY, TEXT_DIM, 
                                   ACCENT, resolve, WELCOME_CARD_HOVER, BG_ROOT, BORDER, BORDER_FOCUS)
from dizel_ui.theme.fonts import CARD_TITLE, INPUT_TEXT, LABEL_SM, LABEL_DIM
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style
from dizel_ui.ui.waveform_widget import WaveformWidget

class NovaOverlay(QFrame):
    text_ready = Signal(str)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet(get_frame_style(BG_INPUT_FIELD, radius=13, border_color=BORDER))
        self.hide()
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        # 1. Icon
        self.status_icon = QLabel()
        mic_pix = get_icon("mic", size=(14, 14), color=ACCENT)
        if mic_pix: self.status_icon.setPixmap(mic_pix.pixmap(14, 14))
        self.status_icon.setStyleSheet("background: transparent;")
        layout.addWidget(self.status_icon)
        
        # 2. Status Label
        self.status_label = QLabel("Listening...")
        self.status_label.setFont(LABEL_DIM)
        self.status_label.setStyleSheet(f"color: {resolve(ACCENT)}; font-weight: 600; background: transparent;")
        layout.addWidget(self.status_label)
        
        # 3. Waveform
        self.waveform = WaveformWidget(bar_count=20, parent=self)
        self.waveform.setFixedHeight(12)
        layout.addWidget(self.waveform, stretch=1)
        
        # 4. Cancel
        self.btn_cancel = QPushButton("✕", self)
        self.btn_cancel.setFixedSize(16, 16)
        self.btn_cancel.setFont(LABEL_DIM)
        self.btn_cancel.setStyleSheet("""
            QPushButton { background: transparent; color: """ + resolve(TEXT_DIM) + """; border: none; }
            QPushButton:hover { color: #f87171; }
        """)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self.btn_cancel)
        
        # 5. Done
        self.btn_done = QPushButton("✓", self)
        self.btn_done.setFixedSize(16, 16)
        self.btn_done.setFont(LABEL_DIM)
        self.btn_done.setStyleSheet("""
            QPushButton { background: transparent; color: """ + resolve(ACCENT) + """; border: none; }
            QPushButton:hover { color: """ + resolve(TEXT_PRIMARY) + """; }
        """)
        self.btn_done.setCursor(Qt.PointingHandCursor)
        self.btn_done.clicked.connect(self._on_done)
        layout.addWidget(self.btn_done)

    def show_listening(self):
        self.status_label.setText("Listening...")
        self.show()

    def update_waveform(self, amplitude: float):
        if self.isVisible():
            self.waveform.set_amplitude(amplitude)

    def update_transcript(self, text: str):
        # We don't display transcript here anymore; main.py updates input directly
        pass

    def update_status(self, status: str):
        self.status_label.setText(status)

    def hide_overlay(self):
        self.hide()

    def _on_cancel(self):
        self.cancelled.emit()
        self.hide_overlay()

    def _on_done(self):
        self.text_ready.emit("") # Final text handled by thread
