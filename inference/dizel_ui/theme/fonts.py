# dizel_ui/theme/fonts.py
# ──────────────────────────────────────────────────────────────────────────────
# Font tuples used throughout the Dizel desktop UI.
# CustomTkinter accepts ("Family", size, "weight") tuples.
# ──────────────────────────────────────────────────────────────────────────────
import sys

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

# ── UI chrome ─────────────────────────────────────────────────────────────────
LOGO          = (_PRIMARY, 18, "bold")
NAV_ITEM      = (_PRIMARY, 12, "normal")
NAV_ITEM_SM   = (_PRIMARY, 11, "normal")
BTN_LABEL     = (_PRIMARY, 12, "normal")
BTN_LABEL_SM  = (_PRIMARY, 11, "normal")
LABEL         = (_PRIMARY, 12, "normal")
LABEL_SM      = (_PRIMARY, 11, "normal")
LABEL_DIM     = (_PRIMARY, 10, "normal")

# ── Chat ──────────────────────────────────────────────────────────────────────
MSG_TEXT      = (_PRIMARY, 13, "normal")   # message bubble body
MSG_META      = (_PRIMARY, 10, "normal")   # timestamp / token count
INPUT_TEXT    = (_PRIMARY, 13, "normal")   # input field
TYPING_TEXT   = (_PRIMARY, 12, "normal")   # "Dizel is thinking…"
CODE_TEXT     = (_MONO,    12, "normal")   # inline code / code blocks

# ── Welcome ───────────────────────────────────────────────────────────────────
WELCOME_TITLE = (_PRIMARY, 28, "bold")
WELCOME_SUB   = (_PRIMARY, 14, "normal")
CARD_TITLE    = (_PRIMARY, 13, "bold")
CARD_BODY     = (_PRIMARY, 12, "normal")
ACTION_PILL   = (_PRIMARY, 12, "bold")
