# dizel_ui/ui/command_palette.py

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, 
                               QListWidgetItem, QGraphicsDropShadowEffect, QFrame, QLabel,
                               QWidget)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QKeyEvent, QColor

from dizel_ui.theme.colors import (
    BG_ROOT, BG_CARD, BG_INPUT, BG_INPUT_FIELD, BORDER, BORDER_FOCUS,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, ACCENT, ACCENT_LIGHT, ACTION_PILL, resolve
)
from dizel_ui.theme.fonts import INPUT_TEXT, LABEL, LABEL_SM
from dizel_ui.utils.icons import get_icon
from PySide6.QtWidgets import QSpacerItem, QSizePolicy


class CommandPalette(QDialog):
    """
    A global command palette similar to Spotlight or VSCode's Ctrl+K.
    It takes a list of command dicts:
    [
        {"label": "New Chat", "icon": "message-square", "shortcut": "Ctrl+N", "hook": func},
        ...
    ]
    """
    def __init__(self, commands: list, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self._commands = commands
        self._filtered = commands.copy()
        
        self.setFixedSize(520, 420)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: transparent;
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self._frame = QFrame(self)
        self._frame.setObjectName("PaletteFrame")
        self._frame.setStyleSheet(f"""
            QFrame#PaletteFrame {{
                background-color: {resolve(BG_ROOT)};
                border: 1px solid {resolve(BORDER_FOCUS)};
                border-radius: 14px;
            }}
        """)
        
        # Shadow
        shadow = QGraphicsDropShadowEffect(self._frame)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 6)
        self._frame.setGraphicsEffect(shadow)
        
        f_layout = QVBoxLayout(self._frame)
        f_layout.setContentsMargins(14, 14, 14, 14)
        f_layout.setSpacing(8)
        
        # Search input area
        search_frame = QFrame()
        search_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {resolve(BG_INPUT)};
                border: 1px solid {resolve(BORDER)};
                border-radius: 10px;
            }}
        """)
        search_layout = QHBoxLayout(search_frame)
        search_layout.setContentsMargins(12, 0, 12, 0)
        search_layout.setSpacing(8)
        
        search_ico = QLabel()
        ico_pm = get_icon("search", (16, 16), TEXT_DIM)
        if ico_pm:
            search_ico.setPixmap(ico_pm.pixmap(16, 16))
        search_ico.setFixedSize(18, 18)
        search_ico.setStyleSheet("background: transparent; border: none;")
        search_layout.addWidget(search_ico)
        
        self._input = QLineEdit()
        self._input.setPlaceholderText("Search commands...")
        self._input.setFont(INPUT_TEXT)
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                color: {resolve(TEXT_PRIMARY)};
                border: none;
                padding: 10px 0px;
            }}
        """)
        self._input.textChanged.connect(self._filter_commands)
        self._input.installEventFilter(self)
        search_layout.addWidget(self._input)
        
        keys_layout = QHBoxLayout()
        keys_layout.setSpacing(4)
        
        lbl_ctrl = QLabel("Ctrl + K")
        
        key_style = f"""
            QLabel {{
                background-color: transparent;
                color: {resolve(TEXT_DIM)};
                border: none;
                padding: 4px 6px;
                font-size: 10px;
                font-weight: bold;
            }}
        """
        lbl_ctrl.setStyleSheet(key_style)
        keys_layout.addWidget(lbl_ctrl)
        search_layout.addLayout(keys_layout)
        
        f_layout.addWidget(search_frame)
        
        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {resolve(BORDER)};")
        f_layout.addWidget(sep)
        
        # List
        self._list = QListWidget()
        self._list.setFont(LABEL)
        self._list.setIconSize(QSize(18, 18))
        self._list.setStyleSheet(f"""
            QListWidget {{
                background-color: transparent;
                color: {resolve(TEXT_PRIMARY)};
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                padding: 0px;
                border-radius: 8px;
                margin: 1px 0px;
            }}
            QListWidget::item:selected {{
                background-color: {resolve(ACTION_PILL)};
                color: {resolve(ACCENT)};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {resolve(BG_CARD)};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {resolve(BORDER_FOCUS)};
                min-height: 20px;
                border-radius: 3px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)
        self._list.itemClicked.connect(self._on_item_clicked)
        f_layout.addWidget(self._list)
        
        # Footer hint
        footer = QLabel("↑↓ Navigate  ↵ Select  Esc Close")
        footer.setFont(LABEL_SM)
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(f"color: {resolve(TEXT_DIM)}; font-size: 11px; padding: 4px 0;")
        f_layout.addWidget(footer)
        
        main_layout.addWidget(self._frame)
        self._populate()
        
    def _populate(self):
        self._list.clear()
        for cmd in self._filtered:
            item = QListWidgetItem(self._list)
            # We don't set internal padding, we'll let the widget breathe 
            item.setData(Qt.UserRole, cmd)
            self._list.addItem(item)
            
            # Create a transparent widget for the layout
            w = QWidget()
            w.setStyleSheet("background: transparent;")
            # The custom widget padding replaces the item's padding
            l = QHBoxLayout(w)
            l.setContentsMargins(12, 10, 12, 10)
            l.setSpacing(12)
            
            # Left icon
            ico_name = cmd.get("icon", "terminal")
            ico = get_icon(ico_name, color=TEXT_DIM)
            ico_lbl = QLabel()
            if ico:
                ico_lbl.setPixmap(ico.pixmap(18, 18))
            ico_lbl.setFixedSize(18, 18)
            l.addWidget(ico_lbl)
            
            # Label
            text_lbl = QLabel(cmd["label"])
            text_lbl.setFont(LABEL)
            text_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
            l.addWidget(text_lbl)
            
            # Spacer
            l.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
            
            # Right shortcut
            shortcut = cmd.get("shortcut", "")
            if shortcut:
                sh_lbl = QLabel(shortcut)
                sh_lbl.setFont(LABEL_SM)
                sh_lbl.setStyleSheet(f"color: {resolve(ACCENT)}; font-weight: bold;")
                l.addWidget(sh_lbl)
                
            # Set the fixed height of item based on hint
            item.setSizeHint(w.sizeHint())
            self._list.setItemWidget(item, w)
            
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
