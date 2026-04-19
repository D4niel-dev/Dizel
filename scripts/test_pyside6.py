"""Simple PySide6 test — just a window with a button and label."""

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout,
    QWidget, QPushButton, QLabel
)
from PySide6.QtCore import Qt


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 Test")
        self.setFixedSize(400, 250)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Click the button!")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 18px; color: #333;")
        layout.addWidget(self.label)

        self.count = 0
        btn = QPushButton("Click me")
        btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 24px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        btn.clicked.connect(self.on_click)
        layout.addWidget(btn)

    def on_click(self):
        self.count += 1
        self.label.setText(f"Clicked {self.count} time{'s' if self.count != 1 else ''}!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TestWindow()
    win.show()
    sys.exit(app.exec())
