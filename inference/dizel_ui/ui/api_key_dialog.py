"""
dizel_ui/ui/api_key_dialog.py — API Key configuration modal.

Provides UI for entering API keys, testing connections,
and configuring provider-specific settings (Ollama URL, Azure fields).
"""

import os
import threading

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QComboBox, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPainterPath

from dizel_ui.theme.colors import (
    BG_ROOT, BG_CHAT, BG_CARD, BG_INPUT, ACCENT, ACCENT_HOVER,
    BORDER, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, resolve,
)
from dizel_ui.theme.fonts import LOGO, LABEL, BTN_LABEL
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style
from dizel_ui.logic.config_manager import ConfigManager, encrypt_key, decrypt_key


_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(os.path.dirname(_HERE), "assets")


def _rounded_pixmap(pixmap: QPixmap, size: int) -> QPixmap:
    """Clip a pixmap to a circle."""
    scaled = pixmap.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    result = QPixmap(size, size)
    result.fill(Qt.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addEllipse(0, 0, size, size)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, scaled)
    painter.end()
    return result


class APIKeyDialog(QDialog):
    """Modal for configuring an API provider's key and connection."""

    connection_validated = Signal(str, list)  # (provider_slug, [ProviderModel])

    def __init__(self, provider_info, parent=None):
        super().__init__(parent)
        self._provider_info = provider_info
        self.setWindowTitle(f"Configure {provider_info.name}")
        # Azure has extra fields, needs more height
        height = 520 if provider_info.slug == "azure" else 420
        self.setFixedSize(500, height)
        self.setStyleSheet(f"QDialog {{ background-color: {resolve(BG_ROOT)}; }}")
        self._validated_models = []
        self._build()
        self._load_saved_key()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(20)

        # Header: avatar + name
        header = QHBoxLayout()
        avatar_lbl = QLabel()
        avatar_path = os.path.join(_ASSETS, "avatars", "providers", self._provider_info.avatar_file)
        if os.path.exists(avatar_path):
            pix = QPixmap(avatar_path)
            avatar_lbl.setPixmap(_rounded_pixmap(pix, 48))
        avatar_lbl.setFixedSize(48, 48)
        header.addWidget(avatar_lbl)

        title = QLabel(self._provider_info.name)
        title.setFont(LOGO)
        title.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 22px;")
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)

        desc = QLabel(self._provider_info.description)
        desc.setStyleSheet(f"color: {resolve(TEXT_SECONDARY)}; font-size: 13px;")
        layout.addWidget(desc)

        # Card
        card = QFrame()
        card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(20, 20, 20, 20)
        card_l.setSpacing(16)

        input_style = (
            f"background: {resolve(BG_INPUT)}; color: {resolve(TEXT_PRIMARY)}; "
            f"border: 1px solid {resolve(BORDER)}; padding: 0px 14px; border-radius: 8px; "
            f"font-size: 14px; min-height: 40px;"
        )

        if self._provider_info.slug == "ollama":
            # Ollama: URL field instead of key
            url_lbl = QLabel("Ollama Server URL")
            url_lbl.setFont(LABEL)
            url_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
            card_l.addWidget(url_lbl)

            self._url_edit = QLineEdit()
            self._url_edit.setPlaceholderText("http://localhost:11434")
            self._url_edit.setStyleSheet(input_style)
            card_l.addWidget(self._url_edit)
            self._key_edit = None
        else:
            # API Key field
            key_lbl = QLabel("API Key")
            key_lbl.setFont(LABEL)
            key_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
            card_l.addWidget(key_lbl)

            self._key_edit = QLineEdit()
            self._key_edit.setPlaceholderText("sk-... or paste your key here")
            self._key_edit.setEchoMode(QLineEdit.Password)
            self._key_edit.setStyleSheet(input_style)
            card_l.addWidget(self._key_edit)
            self._url_edit = None

        # Azure extra fields
        if self._provider_info.slug == "azure":
            res_lbl = QLabel("Azure Resource Name")
            res_lbl.setFont(LABEL)
            res_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
            card_l.addWidget(res_lbl)

            self._azure_resource = QLineEdit()
            self._azure_resource.setPlaceholderText("my-openai-resource")
            self._azure_resource.setStyleSheet(input_style)
            card_l.addWidget(self._azure_resource)

            dep_lbl = QLabel("Deployment Name")
            dep_lbl.setFont(LABEL)
            dep_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
            card_l.addWidget(dep_lbl)

            self._azure_deployment = QLineEdit()
            self._azure_deployment.setPlaceholderText("gpt-4o-deployment")
            self._azure_deployment.setStyleSheet(input_style)
            card_l.addWidget(self._azure_deployment)
        else:
            self._azure_resource = None
            self._azure_deployment = None

        layout.addWidget(card)

        # Status
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; font-size: 13px;")
        self._status_lbl.setWordWrap(True)
        layout.addWidget(self._status_lbl)

        # Buttons
        btn_row = QHBoxLayout()

        test_btn = QPushButton("Test Connection")
        test_btn.setStyleSheet(get_button_style("transparent", BORDER, TEXT_PRIMARY, border_color=BORDER))
        test_btn.setCursor(Qt.PointingHandCursor)
        test_btn.clicked.connect(self._test_connection)
        btn_row.addWidget(test_btn)

        btn_row.addStretch(1)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(get_button_style("transparent", BORDER, TEXT_SECONDARY, border_color=BORDER))
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._save_btn = QPushButton("Save/Connect")
        self._save_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, "#ffffff"))
        self._save_btn.setCursor(Qt.PointingHandCursor)
        self._save_btn.clicked.connect(self._save_and_close)
        btn_row.addWidget(self._save_btn)

        layout.addLayout(btn_row)

    def _load_saved_key(self):
        cfg = ConfigManager.load()
        api_cfg = cfg.get("api_router", {})
        if self._provider_info.slug == "ollama":
            url = api_cfg.get("ollama_url", "http://localhost:11434")
            if self._url_edit:
                self._url_edit.setText(url)
        else:
            if api_cfg.get("provider") == self._provider_info.slug:
                key = decrypt_key(api_cfg.get("api_key", ""))
                if self._key_edit and key:
                    self._key_edit.setText(key)
            if self._provider_info.slug == "azure":
                if self._azure_resource:
                    self._azure_resource.setText(api_cfg.get("azure_resource", ""))
                if self._azure_deployment:
                    self._azure_deployment.setText(api_cfg.get("azure_deployment", ""))

    def _get_kwargs(self) -> dict:
        kw = {}
        if self._url_edit:
            kw["ollama_url"] = self._url_edit.text().strip() or "http://localhost:11434"
        if self._azure_resource:
            kw["azure_resource"] = self._azure_resource.text().strip()
        if self._azure_deployment:
            kw["azure_deployment"] = self._azure_deployment.text().strip()
        return kw

    def _test_connection(self):
        self._status_lbl.setText("⏳ Testing connection...")
        self._status_lbl.setStyleSheet(f"color: {resolve(TEXT_SECONDARY)}; font-size: 13px;")

        key = self._key_edit.text().strip() if self._key_edit else ""
        slug = self._provider_info.slug
        kwargs = self._get_kwargs()

        def _worker():
            try:
                from dizel_ui.logic.providers import get_provider
                provider = get_provider(slug, **kwargs)
                provider.validate(key=key, **kwargs)
                models = provider.list_models(key=key, **kwargs)
                self._validated_models = models
                count = len(models)
                QTimer.singleShot(0, lambda: self._show_status(
                    f"🟢 Connected — {count} model{'s' if count != 1 else ''} available", "success"
                ))
            except (ValueError, ConnectionError, ImportError) as e:
                QTimer.singleShot(0, lambda: self._show_status(f"🔴 {e}", "error"))
            except Exception as e:
                QTimer.singleShot(0, lambda: self._show_status(f"🔴 Unexpected error: {e}", "error"))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_status(self, msg, kind="info"):
        colors = {
            "success": "#10b981",
            "error": "#ef4444",
            "info": resolve(TEXT_DIM),
        }
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color: {colors.get(kind, resolve(TEXT_DIM))}; font-size: 13px;")

    def _save_and_close(self):
        cfg = ConfigManager.load()
        slug = self._provider_info.slug

        api_cfg = cfg.get("api_router", {})
        api_cfg["provider"] = slug

        if slug == "ollama":
            url = self._url_edit.text().strip() if self._url_edit else "http://localhost:11434"
            api_cfg["ollama_url"] = url
            api_cfg["api_key"] = ""
        else:
            key = self._key_edit.text().strip() if self._key_edit else ""
            api_cfg["api_key"] = encrypt_key(key)

        if slug == "azure":
            api_cfg["azure_resource"] = self._azure_resource.text().strip() if self._azure_resource else ""
            api_cfg["azure_deployment"] = self._azure_deployment.text().strip() if self._azure_deployment else ""

        # Cache model list
        api_cfg["available_models"] = [
            {"id": m.id, "name": m.name}
            for m in self._validated_models
        ]

        cfg["api_router"] = api_cfg
        ConfigManager.save(cfg)

        self.connection_validated.emit(slug, self._validated_models)
        self.accept()
