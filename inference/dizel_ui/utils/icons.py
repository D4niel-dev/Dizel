import os
import io
from typing import Dict, Tuple, Optional
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import QByteArray, Qt

# Root path resolution
_HERE = os.path.dirname(os.path.abspath(__file__)) # this is inference/dizel_ui/utils
_UI_DIR = os.path.dirname(_HERE)                   # this is inference/dizel_ui
_FEATHER_DIR = os.path.join(_UI_DIR, "assets", "icons")

from dizel_ui.theme.colors import resolve

_ICON_CACHE: Dict[str, QIcon] = {}

def get_icon(icon_name: str, size: Tuple[int, int] = (18, 18), color = "white") -> Optional[QIcon]:
    """
    Loads an SVG feather icon, recolors it, and returns a PySide6 QIcon.
    """
    resolved_color = resolve(color)
    cache_key = f"{icon_name}_{size[0]}x{size[1]}_{resolved_color}"
    
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    svg_path_original = os.path.join(_FEATHER_DIR, f"{icon_name}.svg")
    if not os.path.exists(svg_path_original):
        print(f"Warning: Icon not found {svg_path_original}")
        return None

    try:
        with open(svg_path_original, 'r', encoding='utf-8') as f:
            svg_data = f.read()
            
        import re
        # Convert standard feather 'currentColor'
        svg_data = svg_data.replace('currentColor', resolved_color)
        
        # Aggressively replace hardcoded black colors typical in custom exported SVGs
        # This catches fill="#000000", stroke="#000", fill="black", etc.
        svg_data = re.sub(r'(fill|stroke)\s*=\s*["\']#(?:000000|000|111111)["\']', r'\1="' + resolved_color + '"', svg_data, flags=re.IGNORECASE)
        svg_data = re.sub(r'(fill|stroke)\s*=\s*["\'](?:black)["\']', r'\1="' + resolved_color + '"', svg_data, flags=re.IGNORECASE)
        
        # Render to QPixmap
        renderer = QSvgRenderer(QByteArray(svg_data.encode('utf-8')))
        
        # Scaling adjustment based on size tuple provided with High-DPI support
        from PySide6.QtGui import QGuiApplication
        ratio = QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0
        
        pixmap = QPixmap(int(size[0] * ratio), int(size[1] * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        # Better rendering quality
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        renderer.render(painter)
        painter.end()
        
        icon = QIcon(pixmap)
        _ICON_CACHE[cache_key] = icon
        return icon
        
    except Exception as e:
        print(f"Error loading icon '{icon_name}': {e}")
        return None
