# dizel_ui/ui/command_palette.py

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QListWidget, 
                               QListWidgetItem, QGraphicsDropShadowEffect, QFrame)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeyEvent

from dizel_ui.theme.colors import (
    BG_ROOT, BG_INPUT_FIELD, BORDER_FOCUS, TEXT_PRIMARY, TEXT_DIM, ACCENT, ACTION_PILL, resolve
)
from dizel_ui.theme.fonts import INPUT_TEXT, LABEL
from dizel_ui.utils.icons import get_icon

class CommandPalette(QDialog):
    """
    A global command palette similar to Spotlight or VSCode's Ctrl+K.
    It takes a list of command dicts:
    [
        {"label": "New Chat", "icon": "message-square", "hook": func},
        ...
    ]
    """
    def __init__(self, commands: list, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._commands = commands
        self._filtered = commands.copy()
        
        self.setFixedSize(550, 400)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: transparent;
            }}
            QFrame#PaletteFrame {{
                background-color: {resolve(BG_ROOT)};
                border: 1px solid {resolve(BORDER_FOCUS)};
                border-radius: 12px;
            }}
            QLineEdit {{
                background-color: transparent;
                color: {resolve(TEXT_PRIMARY)};
                border: none;
                padding: 10px;
            }}
            QListWidget {{
                background-color: transparent;
                color: {resolve(TEXT_PRIMARY)};
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                padding: 12px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background-color: {resolve(ACTION_PILL)};
                color: {resolve(ACCENT)};
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self._frame = QFrame(self)
        self._frame.setObjectName("PaletteFrame")
        
        # Add slight shadow
        shadow = QGraphicsDropShadowEffect(self._frame)
        shadow.setBlurRadius(20)
        shadow.setColor(resolve("#000000"))
        shadow.setOffset(0, 4)
        self._frame.setGraphicsEffect(shadow)
        
        f_layout = QVBoxLayout(self._frame)
        f_layout.setContentsMargins(12, 12, 12, 12)
        f_layout.setSpacing(8)
        
        self._input = QLineEdit(self)
        self._input.setPlaceholderText("Search commands... (e.g., Settings, New Chat)")
        self._input.setFont(INPUT_TEXT)
        self._input.textChanged.connect(self._filter_commands)
        # Override key press to navigate list when typing in line edit
        self._input.installEventFilter(self)
        
        f_layout.addWidget(self._input)
        
        self._list = QListWidget(self)
        self._list.setFont(LABEL)
        self._list.setIconSize(QSize(18, 18))
        self._list.itemClicked.connect(self._on_item_clicked)
        f_layout.addWidget(self._list)
        
        main_layout.addWidget(self._frame)
        self._populate()
        
    def _populate(self):
        self._list.clear()
        for cmd in self._filtered:
            item = QListWidgetItem(self._list)
            # Add icon if exists
            ico_name = cmd.get("icon", "terminal")
            ico = get_icon(ico_name, color=TEXT_DIM)
            if ico:
                item.setIcon(ico)
            item.setText(cmd["label"])
            item.setData(Qt.UserRole, cmd)
            self._list.addItem(item)
            
        if self._list.count() > 0:
            self._list.setCurrentRow(0)

    def _filter_commands(self, text: str):
        query = text.lower()
        if not query:
            self._filtered = self._commands.copy()
        else:
            self._filtered = [c for c in self._commands if query in c["label"].lower()]
        self._populate()

    def eventFilter(self, obj, event):
        if obj == self._input and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                row = self._list.currentRow()
                if row < self._list.count() - 1:
                    self._list.setCurrentRow(row + 1)
                return True
            elif key == Qt.Key_Up:
                row = self._list.currentRow()
                if row > 0:
                    self._list.setCurrentRow(row - 1)
                return True
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                self._on_item_clicked(self._list.currentItem())
                return True
            elif key == Qt.Key_Escape:
                self.reject()
                return True
        return super().eventFilter(obj, event)

    def _on_item_clicked(self, item):
        if not item: return
        cmd = item.data(Qt.UserRole)
        self.accept()
        if "hook" in cmd and cmd["hook"]:
            cmd["hook"]()

    def showEvent(self, event):
        super().showEvent(event)
        self._input.clear()
        self._input.setFocus()
