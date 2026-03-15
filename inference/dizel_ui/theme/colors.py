# dizel_ui/theme/colors.py
# ──────────────────────────────────────────────────────────────────────────────
# All colour constants for the Dizel desktop UI (Sky Blue Theme).
# Import this module anywhere instead of hard-coding hex values.
# ──────────────────────────────────────────────────────────────────────────────

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_ROOT         = ("#f8fafc", "#0a0f18")   # deepest space background (window, root)
BG_SIDEBAR      = ("#f1f5f9", "#06090f")   # solid dark sidebar panel
BG_CHAT         = ("#ffffff", "#0f1725")   # deep blue-gray main area
BG_INPUT        = ("#f8fafc", "#182438")   # floating input field panel
BG_INPUT_FIELD  = ("#ffffff", "#0f1725")   # actual CTkTextbox inside input panel
BG_CARD         = ("#f1f5f9", "#121a28")   # premium settings card 

# ── Tabs ──────────────────────────────────────────────────────────────────────
TAB_BG          = ("#f8fafc", "#0a0f18")
TAB_UNSELECTED  = ("#f1f5f9", "#121a28")

# ── Message bubbles ───────────────────────────────────────────────────────────
BUBBLE_USER     = ("#3b82f6", "#1e40af")   # user message — solid blue
BUBBLE_ASST     = "transparent"  # assistant message blends into BG (must be plain string)
BUBBLE_USER_TXT = ("#ffffff", "#ffffff")
BUBBLE_ASST_TXT = ("#0f172a", "#e2e8f0")

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR_BTN_HOVER   = ("#e2e8f0", "#162032")
SIDEBAR_BTN_ACTIVE  = ("#cbd5e1", "#1e293b")
SIDEBAR_TEXT        = ("#334155", "#cbd5e1")
SIDEBAR_TEXT_DIM    = ("#64748b", "#64748b")
SIDEBAR_BORDER      = ("#e2e8f0", "#1e293b")
SIDEBAR_PREMIUM_BG  = ("#ffffff", "#0f172a")

# ── Accent / interactive ──────────────────────────────────────────────────────
ACCENT          = ("#0ea5e9", "#38bdf8")   # vibrant sky blue — primary accent
ACCENT_HOVER    = ("#0284c7", "#0ea5e9")   # darker sky blue on hover
ACCENT_LIGHT    = ("#bae6fd", "#7dd3fc")   # light sky blue for highlights
SEND_BTN        = ("#0284c7", "#0284c7")   # send button 
SEND_BTN_HOVER  = ("#0369a1", "#0369a1")

# ── Typography ────────────────────────────────────────────────────────────────
TEXT_PRIMARY    = ("#0f172a", "#f1f5f9")
TEXT_SECONDARY  = ("#475569", "#94a3b8")
TEXT_DIM        = ("#64748b", "#475569")
TEXT_ERROR      = ("#e11d48", "#f43f5e")
TEXT_SUCCESS    = ("#059669", "#10b981")

# ── Borders / separators ─────────────────────────────────────────────────────
BORDER          = ("#e2e8f0", "#1e293b")
BORDER_FOCUS    = ("#0ea5e9", "#38bdf8")

# ── Typing indicator ─────────────────────────────────────────────────────────
TYPING_DOT      = ("#0ea5e9", "#38bdf8")

# ── Welcome screen ───────────────────────────────────────────────────────────
WELCOME_CARD    = ("#f8fafc", "#1e293b")
WELCOME_CARD_HOVER = ("#f1f5f9", "#334155")
ACTION_PILL     = ("#f1f5f9", "#0f172a")

# ── Scrollbar ────────────────────────────────────────────────────────────────
SCROLLBAR       = ("#cbd5e1", "#1e293b")
SCROLLBAR_HOVER = ("#94a3b8", "#334155")
