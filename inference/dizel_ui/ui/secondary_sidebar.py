# dizel_ui/ui/secondary_sidebar.py

import os
from typing import Callable, List, Dict, Optional
from PySide6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QWidget, QScrollArea, QSizePolicy, QLineEdit, QStackedWidget, QGridLayout)
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
from PySide6.QtGui import QIcon, QPixmap

from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.colors import (
    BG_SIDEBAR, ACCENT, ACCENT_HOVER, ACCENT_LIGHT,
    SIDEBAR_BTN_HOVER, SIDEBAR_BTN_ACTIVE, SIDEBAR_TEXT,
    SIDEBAR_TEXT_DIM, SIDEBAR_BORDER, TEXT_PRIMARY, TEXT_DIM, BG_ROOT, resolve
)
from dizel_ui.theme.fonts import LOGO, NAV_ITEM, NAV_ITEM_SM, BTN_LABEL, LABEL_SM, LABEL_DIM
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style, get_scrollbar_style, get_input_style


class HistoryItem(QFrame):
    def __init__(self, session: Dict, on_click: Callable[[str], None], 
                 on_delete: Callable[[str], None], on_pin: Callable[[str], None] = None, parent=None):
        super().__init__(parent)
        self._session_id = session["id"]
        self._on_click = on_click
        self._on_delete = on_delete
        self._on_pin = on_pin
        self._is_pinned = session.get("pinned", False)
        
        self.setStyleSheet("background: transparent; border: none;")
        self.setFixedHeight(36)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 6, 1)  # Shifting left
        layout.setSpacing(0)
        
        self._row = QFrame(self)
        self._row.setStyleSheet(get_frame_style("transparent", radius=8))
        self._row.setCursor(Qt.PointingHandCursor)
        
        row_layout = QHBoxLayout(self._row)
        row_layout.setContentsMargins(6, 6, 6, 6)
        
        title = session.get("title", "Untitled")[:24]
        
        # Pinned Indicator
        if self._is_pinned:
            self._pin_indicator = QLabel(self._row)
            pin_ico = get_icon("pin", size=(12, 12), color=ACCENT)
            if pin_ico:
                self._pin_indicator.setPixmap(pin_ico.pixmap(12, 12))
            self._pin_indicator.setStyleSheet("background: transparent; border: none;")
            row_layout.addWidget(self._pin_indicator)
            title_text = title + ("..." if len(session.get("title", "")) > 24 else "")
        else:
            title_text = title + ("..." if len(session.get("title", "")) > 24 else "")

        self._lbl = QLabel(title_text, self._row)
        self._lbl.setFont(NAV_ITEM)
        self._lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT)}; background: transparent;")
        self._lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row_layout.addWidget(self._lbl, stretch=1)
        
        self._pin_btn = QPushButton(self._row)
        self._pin_btn.setFixedSize(24, 24)
        btn_color = "#BBBBBB" if self._is_pinned else "#888888"
        btn_ico = get_icon("pin-off" if self._is_pinned else "pin", size=(14, 14), color=btn_color)
        if btn_ico:
            self._pin_btn.setIcon(btn_ico)
        self._pin_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none; padding-bottom: 2px; }}
            QPushButton:hover {{ background: {resolve(SIDEBAR_BTN_ACTIVE)}; border-radius: 4px; }}
        """)
        self._pin_btn.setCursor(Qt.PointingHandCursor)
        if self._on_pin:
            self._pin_btn.clicked.connect(self._pin)
        self._pin_btn.hide()
        row_layout.addWidget(self._pin_btn)

        self._del_btn = QPushButton("✕", self._row)
        self._del_btn.setFixedSize(24, 24)
        self._del_btn.setStyleSheet(f"""
            QPushButton {{ background: transparent; border: none; color: #888888; font-size: 14px; padding-bottom: 2px; }}
            QPushButton:hover {{ background: #ff4444; color: #ffffff; border-radius: 4px; }}
        """)
        self._del_btn.setCursor(Qt.PointingHandCursor)
        self._del_btn.clicked.connect(self._delete)
        self._del_btn.hide()
        row_layout.addWidget(self._del_btn)
        
        layout.addWidget(self._row)

    def _pin(self):
        if self._on_pin:
            self._on_pin(self._session_id)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_click(self._session_id)
        super().mousePressEvent(event)
        
    def enterEvent(self, event):
        self._row.setStyleSheet(get_frame_style(SIDEBAR_BTN_HOVER, radius=8))
        self._del_btn.show()
        if self._on_pin: self._pin_btn.show()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self._row.setStyleSheet(get_frame_style("transparent", radius=8))
        self._del_btn.hide()
        if self._on_pin: self._pin_btn.hide()
        super().leaveEvent(event)

    def _delete(self):
        self._on_delete(self._session_id)


class FileItem(QFrame):
    """A row representing an attached file in the archive view."""
    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        self.setFixedHeight(48)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 6, 4)
        
        row = QFrame(self)
        row.setStyleSheet(get_frame_style("transparent", radius=8))
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 8, 8, 8)
        
        icon = QLabel(row)
        # Using archive as a safe fallback
        ico = get_icon("archive", size=(18, 18), color=TEXT_DIM)
        if ico:
            icon.setPixmap(ico.pixmap(18, 18))
        row_layout.addWidget(icon)
        
        lbl = QLabel(os.path.basename(filepath), row)
        lbl.setFont(NAV_ITEM)
        lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT)}; background: transparent;")
        lbl.setToolTip(filepath)
        row_layout.addWidget(lbl, stretch=1)
        
        layout.addWidget(row)


class ImageThumbnail(QFrame):
    """A thumbnail card representing an image attachment."""
    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setStyleSheet(f"background: {resolve(BG_SIDEBAR)}; border-radius: 8px; border: 1px solid {resolve(SIDEBAR_BORDER)};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        lbl = QLabel(self)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("background: transparent; border: none;")
        pix = QPixmap(filepath)
        if not pix.isNull():
            lbl.setPixmap(pix.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl.setText("Broken")
            
        layout.addWidget(lbl)
        self.setToolTip(filepath)


class SecondarySidebar(QFrame):
    def __init__(self, on_session_select: Callable[[str], None], 
                 on_session_delete: Callable[[str], None], 
                 on_pin: Callable[[str], None] = None, 
                 on_import: Callable[[str], None] = None,
                 parent=None):
        super().__init__(parent)
        self._on_session_select = on_session_select
        self._on_session_delete = on_session_delete
        self._on_pin = on_pin
        self._on_import = on_import
        
        self._is_open = False
        self._original_width = 280
        self.setFixedWidth(0)
        
        # Animations
        self.anim_group = QParallelAnimationGroup(self)
        self.anim_min = QPropertyAnimation(self, b"minimumWidth")
        self.anim_max = QPropertyAnimation(self, b"maximumWidth")
        self.anim_min.setEasingCurve(QEasingCurve.InOutQuart)
        self.anim_max.setEasingCurve(QEasingCurve.InOutQuart)
        self.anim_min.setDuration(250)
        self.anim_max.setDuration(250)
        self.anim_group.addAnimation(self.anim_min)
        self.anim_group.addAnimation(self.anim_max)
        
        self.setStyleSheet(f"QFrame {{ background-color: {resolve(BG_ROOT)}; border-right: 1px solid {resolve(SIDEBAR_BORDER)}; }}")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 24, 16, 16)
        main_layout.setSpacing(16)
        
        # Header Label (Dynamic)
        self._header_lbl = QLabel("Chats", self)
        self._header_lbl.setFont(LOGO)
        self._header_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; border: none; background: transparent;")
        main_layout.addWidget(self._header_lbl)
        
        self.stack = QStackedWidget(self)
        self.stack.setStyleSheet("background: transparent; border: none;")
        
        self._init_chats_view()
        self._init_image_view()
        self._init_presentation_view()
        self._init_riset_view()
        self._init_archive_view()
        self._init_library_view()
        
        main_layout.addWidget(self.stack)

    # 1. CHATS VIEW
    def _init_chats_view(self):
        page = QWidget()
        lyt = QVBoxLayout(page)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(16)
        
        top_lyt = QHBoxLayout()
        top_lyt.setContentsMargins(0, 0, 0, 0)
        top_lyt.setSpacing(8)
        
        self._search_bar = QLineEdit(page)
        self._search_bar.setPlaceholderText("Search chats...")
        self._search_bar.setFont(NAV_ITEM)
        self._search_bar.setFixedHeight(36)
        search_style = get_input_style(BG_SIDEBAR, SIDEBAR_BORDER, TEXT_PRIMARY, ACCENT_LIGHT)
        self._search_bar.setStyleSheet(search_style + "QLineEdit { padding-left: 36px; border-radius: 8px; }")
        
        icon_lbl = QLabel(self._search_bar)
        icon_lbl.setFixedSize(36, 36)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("background: transparent; border: none;")
        s_ico = get_icon("message-search", size=(18, 18), color=TEXT_DIM)
        if s_ico: icon_lbl.setPixmap(s_ico.pixmap(18, 18))
        
        top_lyt.addWidget(self._search_bar, stretch=1)
        
        self.import_btn = QPushButton(page)
        self.import_btn.setFixedSize(36, 36)
        self.import_btn.setCursor(Qt.PointingHandCursor)
        self.import_btn.setStyleSheet(get_button_style(BG_SIDEBAR, SIDEBAR_BORDER, TEXT_PRIMARY, radius=8))
        import_ico = get_icon("download", size=(18, 18), color=TEXT_DIM)
        if import_ico: self.import_btn.setIcon(import_ico)
        self.import_btn.setToolTip("Import Chat (.json, .md)")
        self.import_btn.clicked.connect(self._handle_import)
        top_lyt.addWidget(self.import_btn)
        
        lyt.addLayout(top_lyt)
        
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(get_scrollbar_style(BG_ROOT, SIDEBAR_BORDER, ACCENT) + "QScrollArea { background: transparent; border: none; }")
        
        self.chats_content = QWidget()
        self.chats_content.setStyleSheet("background: transparent; border: none;")
        self.chats_layout = QVBoxLayout(self.chats_content)
        self.chats_layout.setContentsMargins(0, 0, 0, 16)
        self.chats_layout.setSpacing(0)
        self.chats_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.chats_content)
        lyt.addWidget(scroll)
        self.stack.addWidget(page) # Index 0

    # 2. IMAGE VIEW
    def _init_image_view(self):
        page = QWidget()
        lyt = QVBoxLayout(page)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(16)
        
        gen_btn = QPushButton("  Generate Image", page)
        gen_btn.setFixedHeight(36)
        gen_btn.setFont(BTN_LABEL)
        gen_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, TEXT_PRIMARY, radius=8))
        img_ico = get_icon("image", size=(20, 20), color=TEXT_PRIMARY)
        if img_ico: gen_btn.setIcon(img_ico)
        gen_btn.setCursor(Qt.PointingHandCursor)
        lyt.addWidget(gen_btn)
        
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(get_scrollbar_style(BG_ROOT, SIDEBAR_BORDER, ACCENT) + "QScrollArea { background: transparent; border: none; }")
        
        self.img_content = QWidget()
        self.img_content.setStyleSheet("background: transparent; border: none;")
        self.img_layout = QGridLayout(self.img_content)
        self.img_layout.setContentsMargins(0, 0, 0, 16)
        self.img_layout.setSpacing(8)
        self.img_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.img_content)
        lyt.addWidget(scroll)
        self.stack.addWidget(page) # Index 1

    # 3. PRESENTATION VIEW
    def _init_presentation_view(self):
        page = QWidget()
        lyt = QVBoxLayout(page)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(12)
        
        btn = QPushButton("  New Presentation", page)
        btn.setFixedHeight(36)
        btn.setFont(BTN_LABEL)
        btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, TEXT_PRIMARY, radius=8))
        btn_ico = get_icon("layout", (20,20), TEXT_PRIMARY)
        if btn_ico: btn.setIcon(btn_ico)
        btn.setCursor(Qt.PointingHandCursor)
        lyt.addWidget(btn)
        
        imp_btn = QPushButton("   Import Slides", page)
        imp_btn.setFixedHeight(36)
        imp_btn.setFont(BTN_LABEL)
        imp_btn.setStyleSheet(get_button_style(BG_SIDEBAR, SIDEBAR_BORDER, TEXT_PRIMARY, radius=8))
        imp_ico = get_icon("upload", (18,18), TEXT_DIM)
        if imp_ico: imp_btn.setIcon(imp_ico)
        imp_btn.setCursor(Qt.PointingHandCursor)
        lyt.addWidget(imp_btn)

        lbl = QLabel("No active or recent presentations.", page)
        lbl.setFont(LABEL_DIM)
        lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; background: transparent; border: none;")
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignTop)
        
        lyt.addWidget(lbl)
        lyt.addStretch(1)
        self.stack.addWidget(page) # Index 2

    # 4. RISET (SEARCH) VIEW
    def _init_riset_view(self):
        page = QWidget()
        lyt = QVBoxLayout(page)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(16)
        
        self._global_search = QLineEdit(page)
        self._global_search.setPlaceholderText("Search everywhere...")
        self._global_search.setFont(NAV_ITEM)
        self._global_search.setFixedHeight(36)
        self._global_search.setStyleSheet(get_input_style(BG_SIDEBAR, SIDEBAR_BORDER, TEXT_PRIMARY, ACCENT_LIGHT) + "QLineEdit { padding-left: 36px; border-radius: 8px; }")
        
        icon_lbl = QLabel(self._global_search)
        icon_lbl.setFixedSize(36, 36)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet("background: transparent; border: none;")
        search_ico = get_icon("search", size=(18, 18), color=TEXT_DIM)
        if search_ico: icon_lbl.setPixmap(search_ico.pixmap(18, 18))
        
        lbl = QLabel("Global Search: Try searching across all messages and contexts simultaneously.", page)
        lbl.setFont(LABEL_DIM)
        lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; background: transparent; border: none;")
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignTop)
        
        from PySide6.QtWidgets import QComboBox
        filter_combo = QComboBox(page)
        filter_combo.addItems(["Search all projects", "Current workspace only", "Include archived", "Only starred content"])
        filter_combo.setFixedHeight(36)
        filter_combo.setFont(NAV_ITEM)
        filter_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {resolve(BG_SIDEBAR)};
                color: {resolve(TEXT_PRIMARY)};
                border: 1px solid {resolve(SIDEBAR_BORDER)};
                border-radius: 8px;
                padding-left: 12px;
            }}
            QComboBox::drop-down {{ border: none; width: 30px; }}
            QComboBox QAbstractItemView {{
                background-color: {resolve(BG_SIDEBAR)};
                color: {resolve(TEXT_PRIMARY)};
                selection-background-color: {resolve(ACCENT)};
                selection-color: white;
                border: 1px solid {resolve(SIDEBAR_BORDER)};
                border-radius: 4px;
            }}
        """)
        
        lyt.addWidget(self._global_search)
        lyt.addWidget(filter_combo)
        lyt.addWidget(lbl)
        lyt.addStretch(1)
        self.stack.addWidget(page) # Index 3

    # 5. ARCHIVED VIEW (FILES)
    def _init_archive_view(self):
        page = QWidget()
        lyt = QVBoxLayout(page)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(16)
        
        upload_btn = QPushButton("  Upload Document", page)
        upload_btn.setFixedHeight(36)
        upload_btn.setFont(BTN_LABEL)
        upload_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, TEXT_PRIMARY, radius=8))
        up_ico = get_icon("upload-cloud", size=(20, 20), color=TEXT_PRIMARY)
        if up_ico: upload_btn.setIcon(up_ico)
        upload_btn.setCursor(Qt.PointingHandCursor)
        lyt.addWidget(upload_btn)
        
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(get_scrollbar_style(BG_ROOT, SIDEBAR_BORDER, ACCENT) + "QScrollArea { background: transparent; border: none; }")
        
        self.arc_content = QWidget()
        self.arc_content.setStyleSheet("background: transparent; border: none;")
        self.arc_layout = QVBoxLayout(self.arc_content)
        self.arc_layout.setContentsMargins(0, 0, 0, 16)
        self.arc_layout.setSpacing(0)
        self.arc_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.arc_content)
        lyt.addWidget(scroll)
        self.stack.addWidget(page) # Index 4

    # 6. LIBRARY VIEW
    def _init_library_view(self):
        page = QWidget()
        lyt = QVBoxLayout(page)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(16)
        
        lbl = QLabel("Prompt & Agent Library", page)
        lbl.setFont(NAV_ITEM)
        lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent; border: none;")
        
        desc = QLabel("Save your best prompts, instructions, and persona structures here.", page)
        desc.setFont(LABEL_DIM)
        desc.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; background: transparent; border: none;")
        desc.setWordWrap(True)
        
        btn = QPushButton("  New Prompt", page)
        btn.setFixedHeight(36)
        btn.setFont(BTN_LABEL)
        btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, TEXT_PRIMARY, radius=8))
        btn_ico = get_icon("plus-circle", size=(20,20), color=TEXT_PRIMARY)
        if btn_ico: btn.setIcon(btn_ico)
        btn.setCursor(Qt.PointingHandCursor)
        
        fav_lbl = QLabel("Featured Prompts", page)
        fav_lbl.setFont(NAV_ITEM)
        fav_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; background: transparent; border: none; margin-top: 10px;")
        
        fav_desc = QLabel("None saved yet.", page)
        fav_desc.setFont(LABEL_DIM)
        fav_desc.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; background: transparent; border: none;")
        
        lyt.addWidget(lbl)
        lyt.addWidget(desc)
        lyt.addWidget(btn)
        lyt.addWidget(fav_lbl)
        lyt.addWidget(fav_desc)
        lyt.addStretch(1)
        self.stack.addWidget(page) # Index 5


    def refresh_data(self, sessions: list):
        """Scans ALL sessions to build History, Image Gallery, and Archive files dynamically."""
        # 1. Update Chats
        while self.chats_layout.count():
            child = self.chats_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        for s in sessions:
            item = HistoryItem(
                s, 
                self._on_session_select, 
                self._on_session_delete, 
                on_pin=self._on_pin, 
                parent=self.chats_content
            )
            self.chats_layout.addWidget(item)

        # Build scanners
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        all_images = []
        all_files = []

        for s in sessions:
            for message in s.get("messages", []):
                atts = message.get("attachments", [])
                for att in atts:
                    ext = os.path.splitext(att)[1].lower()
                    if ext in image_exts:
                        all_images.append(att)
                    else:
                        all_files.append(att)

        # 2. Update Image Gallery
        while self.img_layout.count():
            child = self.img_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        if not all_images:
            lbl = QLabel("No images have been attached in any chats.", self.img_content)
            lbl.setFont(LABEL_DIM)
            lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; background: transparent;")
            lbl.setWordWrap(True)
            self.img_layout.addWidget(lbl, 0, 0)
        else:
            COLUMNS = 2
            for i, img_path in enumerate(all_images):
                thumb = ImageThumbnail(img_path, self.img_content)
                self.img_layout.addWidget(thumb, i // COLUMNS, i % COLUMNS)

        # 3. Update Archive (Files)
        while self.arc_layout.count():
            child = self.arc_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        if not all_files:
            lbl = QLabel("No documents have been attached in any chats.", self.arc_content)
            lbl.setFont(LABEL_DIM)
            lbl.setStyleSheet(f"color: {resolve(SIDEBAR_TEXT_DIM)}; background: transparent;")
            lbl.setWordWrap(True)
            self.arc_layout.addWidget(lbl)
        else:
            for f_path in all_files:
                f_item = FileItem(f_path, self.arc_content)
                self.arc_layout.addWidget(f_item)


    def switch_view(self, view_name: str):
        mapping = {
            "Chats": 0,
            "Image": 1,
            "Presentation": 2,
            "Riset": 3,
            "Archived": 4,
            "Library": 5
        }
        idx = mapping.get(view_name, 0)
        
        # If open and user clicks the active view button again, close it
        if self._is_open and self.stack.currentIndex() == idx and self._header_lbl.text() == view_name:
            self.toggle()
            return
            
        self.stack.setCurrentIndex(idx)
        self._header_lbl.setText(view_name)
        
        if not self._is_open:
            self.toggle()

    def _handle_import(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Import Chat", "", "Chat Data (*.json *.md)")
        if path and self._on_import:
            self._on_import(path)

    def toggle(self):
        self._is_open = not self._is_open
        target_w = self._original_width if self._is_open else 0
        
        self.anim_min.setStartValue(self.minimumWidth())
        self.anim_min.setEndValue(target_w)
        self.anim_max.setStartValue(self.maximumWidth())
        self.anim_max.setEndValue(target_w)
        
        self.anim_group.start()
