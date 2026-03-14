# dizel_ui/theme/colors.py
# ──────────────────────────────────────────────────────────────────────────────
# All colour constants for the Dizel desktop UI.
# Import this module anywhere instead of hard-coding hex values.
# ──────────────────────────────────────────────────────────────────────────────

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_ROOT         = "#0f0f14"   # deepest background (window, root)
BG_SIDEBAR      = "#111118"   # sidebar panel
BG_CHAT         = "#151520"   # main chat area
BG_INPUT        = "#1a1a28"   # input field panel
BG_INPUT_FIELD  = "#1e1e2e"   # actual CTkTextbox inside input panel

# ── Message bubbles ───────────────────────────────────────────────────────────
BUBBLE_USER     = "#3a3af0"   # user message — vibrant indigo
BUBBLE_ASST     = "#242430"   # assistant message — dark slate
BUBBLE_USER_TXT = "#ffffff"
BUBBLE_ASST_TXT = "#e2e2f0"

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR_BTN_HOVER   = "#1e1e2e"
SIDEBAR_BTN_ACTIVE  = "#2a2a40"
SIDEBAR_TEXT        = "#c0c0d8"
SIDEBAR_TEXT_DIM    = "#606078"
SIDEBAR_BORDER      = "#2a2a3a"

# ── Accent / interactive ──────────────────────────────────────────────────────
ACCENT          = "#7c3aed"   # purple — primary accent
ACCENT_HOVER    = "#6d28d9"   # darker purple on hover
ACCENT_LIGHT    = "#a78bfa"   # light purple for text highlights
SEND_BTN        = "#3a3af0"   # send button (matches user bubble)
SEND_BTN_HOVER  = "#4a4aff"

# ── Typography ────────────────────────────────────────────────────────────────
TEXT_PRIMARY    = "#e8e8f8"
TEXT_SECONDARY  = "#9090aa"
TEXT_DIM        = "#606080"
TEXT_ERROR      = "#f87171"
TEXT_SUCCESS    = "#4ade80"

# ── Borders / separators ─────────────────────────────────────────────────────
BORDER          = "#2a2a3a"
BORDER_FOCUS    = "#7c3aed"

# ── Typing indicator ─────────────────────────────────────────────────────────
TYPING_DOT      = "#7c3aed"

# ── Welcome screen ───────────────────────────────────────────────────────────
WELCOME_CARD    = "#1a1a28"
WELCOME_CARD_HOVER = "#222236"

# ── Scrollbar ────────────────────────────────────────────────────────────────
SCROLLBAR       = "#2a2a3a"
SCROLLBAR_HOVER = "#3a3a50"
