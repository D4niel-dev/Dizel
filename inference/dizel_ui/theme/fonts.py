# dizel_ui/theme/fonts.py
# ──────────────────────────────────────────────────────────────────────────────
# Font objects used throughout the Dizel desktop UI.
# PySide6 uses QFont objects.
# ──────────────────────────────────────────────────────────────────────────────
import sys
from PySide6.QtGui import QFont

# ── Primary family (system sans-serif stack) ──────────────────────────────────
if sys.platform == "win32":
    _PRIMARY   = "Segoe UI"
    _MONO      = "Consolas"
elif sys.platform == "darwin":
    _PRIMARY   = "SF Pro Display"
    _MONO      = "Menlo"
else:
    _PRIMARY   = "Inter"          # fallback: "DejaVu Sans"
    _MONO      = "JetBrains Mono" # fallback: "DejaVu Sans Mono"

def _make_font(family: str, size: int, weight: str = "normal") -> QFont:
    font = QFont(family, size)
    if weight == "bold":
        font.setBold(True)
    return font

# ── UI chrome ─────────────────────────────────────────────────────────────────
LOGO          = _make_font(_PRIMARY, 18, "bold")
NAV_ITEM      = _make_font(_PRIMARY, 12, "normal")
NAV_ITEM_SM   = _make_font(_PRIMARY, 11, "normal")
BTN_LABEL     = _make_font(_PRIMARY, 12, "normal")
BTN_LABEL_SM  = _make_font(_PRIMARY, 11, "normal")
LABEL         = _make_font(_PRIMARY, 12, "normal")
LABEL_SM      = _make_font(_PRIMARY, 11, "normal")
LABEL_DIM     = _make_font(_PRIMARY, 10, "normal")

# ── Chat ──────────────────────────────────────────────────────────────────────
MSG_TEXT      = _make_font(_PRIMARY, 13, "normal")   # message bubble body
MSG_META      = _make_font(_PRIMARY, 10, "normal")   # timestamp / token count
INPUT_TEXT    = _make_font(_PRIMARY, 13, "normal")   # input field
TYPING_TEXT   = _make_font(_PRIMARY, 12, "normal")   # "Dizel is thinking…"
CODE_TEXT     = _make_font(_MONO,    12, "normal")   # inline code / code blocks

# ── Welcome ───────────────────────────────────────────────────────────────────
WELCOME_TITLE = _make_font(_PRIMARY, 28, "bold")
WELCOME_SUB   = _make_font(_PRIMARY, 14, "normal")
CARD_TITLE    = _make_font(_PRIMARY, 13, "bold")
CARD_BODY     = _make_font(_PRIMARY, 12, "normal")
ACTION_PILL   = _make_font(_PRIMARY, 12, "bold")
