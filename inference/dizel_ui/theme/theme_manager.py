# dizel_ui/theme/theme_manager.py

import sys
from PySide6.QtCore import QObject, Signal

class _ThemeManager(QObject):
    theme_changed = Signal(str)  # Emits "dark" or "light"

    def __init__(self):
        super().__init__()
        self._mode = "dark"

    @property
    def mode(self) -> str:
        return self._mode

    def toggle(self):
        self._mode = "light" if self._mode == "dark" else "dark"
        self.theme_changed.emit(self._mode)

    def set_mode(self, mode: str):
        if mode in ["system", "light", "dark", "dark_blue"] and mode != self._mode:
            self._mode = mode
            self._cached_system_mode = None
            self.theme_changed.emit(mode)

    def get(self, light_val, dark_val, dark_blue_val=None):
        mode = self._mode
        if mode == "system":
            if not getattr(self, "_cached_system_mode", None):
                try:
                    from PySide6.QtGui import QGuiApplication, QPalette
                    app = QGuiApplication.instance()
                    if app and app.palette().color(QPalette.Window).lightness() < 128:
                        self._cached_system_mode = "dark"
                    else:
                        self._cached_system_mode = "light"
                except Exception:
                    self._cached_system_mode = "dark"
            mode = self._cached_system_mode

        if mode == "dark_blue":
            return dark_blue_val if dark_blue_val is not None else dark_val
        elif mode == "dark":
            return dark_val
        return light_val

# Global singleton instance
Theme = _ThemeManager()
