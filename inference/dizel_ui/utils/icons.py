import os
import io
import tempfile
from typing import Dict, Tuple
from PIL import Image
import customtkinter as ctk

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

# Root path resolution
_HERE = os.path.dirname(os.path.abspath(__file__)) # this is inference/dizel_ui/utils
_UI_DIR = os.path.dirname(_HERE)                   # this is inference/dizel_ui
_FEATHER_DIR = os.path.join(_UI_DIR, "assets", "icons")

_ICON_CACHE: Dict[str, ctk.CTkImage] = {}

def _resolve_color(color) -> Tuple[str, str]:
    """
    Resolve a color value into (light_color, dark_color).
    Accepts a plain string or a (light, dark) tuple.
    """
    if isinstance(color, tuple) and len(color) == 2:
        return color
    return (color, color)


def get_icon(icon_name: str, size: Tuple[int, int] = (18, 18), color = "white") -> ctk.CTkImage:
    """
    Loads an SVG feather icon, recolors it, rasterizes to PNG in memory using svglib,
    and returns a fully compatible CTkImage scaled to the requested size.

    `color` can be a plain hex string or a (light_hex, dark_hex) tuple.
    """
    light_color, dark_color = _resolve_color(color)
    cache_key = f"{icon_name}_{size[0]}x{size[1]}_{light_color}_{dark_color}"
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]

    svg_path_original = os.path.join(_FEATHER_DIR, f"{icon_name}.svg")
    if not os.path.exists(svg_path_original):
        print(f"Warning: Icon not found {svg_path_original}")
        return None

    try:
        # Recolor the SVG to pure BLACK to create a perfect alpha mask
        with open(svg_path_original, 'r', encoding='utf-8') as f:
            svg_data = f.read()
            
        svg_data = svg_data.replace('currentColor', 'black')
        
        # Write temporary modified SVG so svg2rlg can parse it
        fd, temp_path = tempfile.mkstemp(suffix=".svg")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(svg_data)
            
        drawing = svg2rlg(temp_path)
        os.remove(temp_path)
        
        if drawing is None:
            return None
            
        # Rasterize strictly to a PNG byte buffer (ReportLab bg defaults to white)
        png_data = io.BytesIO()
        renderPM.drawToFile(drawing, png_data, fmt="PNG")
        png_data.seek(0)
        
        # Open as Grayscale (L). Black stroke becomes 0, white bg becomes 255.
        mask_img = Image.open(png_data).convert("L")
        
        import PIL.ImageOps
        # Invert so stroke is 255 (opaque) and bg is 0 (transparent)
        alpha_channel = PIL.ImageOps.invert(mask_img)
        
        # Create separate light and dark images for CTkImage
        light_img = Image.new("RGBA", mask_img.size, color=light_color)
        light_img.putalpha(alpha_channel)
        
        dark_img = Image.new("RGBA", mask_img.size, color=dark_color)
        dark_img.putalpha(alpha_channel)
        
        # CTkImage handles HighDPI scaling and light/dark switching natively
        ctk_img = ctk.CTkImage(light_image=light_img, dark_image=dark_img, size=size)
        _ICON_CACHE[cache_key] = ctk_img
        
        return ctk_img
        
    except Exception as e:
        print(f"Error loading icon '{icon_name}': {e}")
        return None
