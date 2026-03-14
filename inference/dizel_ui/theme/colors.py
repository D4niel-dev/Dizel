# dizel_ui/theme/colors.py
# ──────────────────────────────────────────────────────────────────────────────
# All colour constants for the Dizel desktop UI (Zyricon Theme).
# Import this module anywhere instead of hard-coding hex values.
# ──────────────────────────────────────────────────────────────────────────────

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_ROOT         = "#140f1a"   # deepest space background (window, root)
BG_SIDEBAR      = "#100a14"   # solid dark sidebar panel
BG_CHAT         = "#1b112c"   # gradient-like deep purple main area
BG_INPUT        = "#22163b"   # floating input field panel
BG_INPUT_FIELD  = "#1b112c"   # actual CTkTextbox inside input panel

# ── Message bubbles ───────────────────────────────────────────────────────────
BUBBLE_USER     = "#281845"   # user message — solid purple
BUBBLE_ASST     = "transparent" # assistant message blends into BG
BUBBLE_USER_TXT = "#ffffff"
BUBBLE_ASST_TXT = "#e2e2f0"

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR_BTN_HOVER   = "#1f152b"
SIDEBAR_BTN_ACTIVE  = "#2a1c3d"
SIDEBAR_TEXT        = "#e0cce8"
SIDEBAR_TEXT_DIM    = "#806d8a"
SIDEBAR_BORDER      = "#241830"
SIDEBAR_PREMIUM_BG  = "#1a1024"

# ── Accent / interactive ──────────────────────────────────────────────────────
ACCENT          = "#a366ff"   # vibrant purple — primary accent
ACCENT_HOVER    = "#8f4dfa"   # darker purple on hover
ACCENT_LIGHT    = "#c8a1ff"   # light purple for highlights
SEND_BTN        = "#7030a0"   # send button 
SEND_BTN_HOVER  = "#8c4bc0"

# ── Typography ────────────────────────────────────────────────────────────────
TEXT_PRIMARY    = "#f6f0fa"
TEXT_SECONDARY  = "#a693b5"
TEXT_DIM        = "#756285"
TEXT_ERROR      = "#ff6b8b"
TEXT_SUCCESS    = "#4ade80"

# ── Borders / separators ─────────────────────────────────────────────────────
BORDER          = "#322345"
BORDER_FOCUS    = "#a366ff"

# ── Typing indicator ─────────────────────────────────────────────────────────
TYPING_DOT      = "#a366ff"

# ── Welcome screen ───────────────────────────────────────────────────────────
WELCOME_CARD    = "#22163b"
WELCOME_CARD_HOVER = "#2d1e4f"
ACTION_PILL     = "#100a14"

# ── Scrollbar ────────────────────────────────────────────────────────────────
SCROLLBAR       = "#241830"
SCROLLBAR_HOVER = "#352347"
