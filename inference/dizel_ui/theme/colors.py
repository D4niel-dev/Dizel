# dizel_ui/theme/colors.py
# ──────────────────────────────────────────────────────────────────────────────
# All colour constants for the Dizel desktop UI (Sky Blue Theme).
# Import this module anywhere instead of hard-coding hex values.
# ──────────────────────────────────────────────────────────────────────────────

# ── Backgrounds ───────────────────────────────────────────────────────────────
BG_ROOT         = ("#f8fafc", "#212121", "#0a0f18")   # deepest space background (window, root)
BG_SIDEBAR      = ("#f1f5f9", "#171717", "#06090f")   # solid dark sidebar panel
BG_CHAT         = ("#ffffff", "#212121", "#0f1725")   # deep blue-gray main area
BG_INPUT        = ("#f8fafc", "#212121", "#182438")   # floating input field panel
BG_INPUT_FIELD  = ("#ffffff", "#2F2F2F", "#0f1725")   # actual CTkTextbox inside input panel
BG_CARD         = ("#f1f5f9", "#171717", "#121a28")   # premium settings card 

# ── Tabs ──────────────────────────────────────────────────────────────────────
TAB_BG          = ("#f8fafc", "#212121", "#0a0f18")
TAB_UNSELECTED  = ("#f1f5f9", "#2F2F2F", "#121a28")

# ── Message bubbles ───────────────────────────────────────────────────────────
BUBBLE_USER     = ("#3b82f6", "#2F2F2F", "#1e40af")   # user message — solid blue
BUBBLE_ASST     = "transparent"  # assistant message blends into BG (must be plain string)
BUBBLE_USER_TXT = ("#ffffff", "#ececec", "#ffffff")
BUBBLE_ASST_TXT = ("#0f172a", "#ececec", "#e2e8f0")

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR_BTN_HOVER   = ("#e2e8f0", "#212121", "#162032")
SIDEBAR_BTN_ACTIVE  = ("#cbd5e1", "#2F2F2F", "#1e293b")
SIDEBAR_TEXT        = ("#334155", "#ececec", "#cbd5e1")
SIDEBAR_TEXT_DIM    = ("#64748b", "#9B9B9B", "#64748b")
SIDEBAR_BORDER      = ("#e2e8f0", "#2F2F2F", "#1e293b")
SIDEBAR_PREMIUM_BG  = ("#ffffff", "#212121", "#0f172a")

# ── Accent / interactive ──────────────────────────────────────────────────────
ACCENT          = ("#0ea5e9", "#10a37f", "#38bdf8")   # vibrant sky blue / GPT green
ACCENT_HOVER    = ("#0284c7", "#1a7f64", "#0ea5e9")   # darker hover
ACCENT_LIGHT    = ("#bae6fd", "#2F2F2F", "#7dd3fc")   # light highlights
SEND_BTN        = ("#0284c7", "#10a37f", "#0284c7")   # send button 
SEND_BTN_HOVER  = ("#0369a1", "#1a7f64", "#0369a1")

# ── Typography ────────────────────────────────────────────────────────────────
TEXT_PRIMARY    = ("#0f172a", "#ececec", "#f1f5f9")
TEXT_SECONDARY  = ("#475569", "#B4B4B4", "#94a3b8")
TEXT_DIM        = ("#64748b", "#9B9B9B", "#475569")
TEXT_ERROR      = ("#e11d48", "#f87171", "#f43f5e")
TEXT_SUCCESS    = ("#059669", "#34d399", "#10b981")

# ── Borders / separators ─────────────────────────────────────────────────────
BORDER          = ("#e2e8f0", "#303030", "#1e293b")
BORDER_FOCUS    = ("#0ea5e9", "#ececec", "#38bdf8")

# ── Typing indicator ─────────────────────────────────────────────────────────
TYPING_DOT      = ("#0ea5e9", "#ececec", "#38bdf8")

# ── Welcome screen ───────────────────────────────────────────────────────────
WELCOME_CARD    = ("#f8fafc", "#2F2F2F", "#1e293b")
WELCOME_CARD_HOVER = ("#f1f5f9", "#383838", "#334155")
ACTION_PILL     = ("#f1f5f9", "#171717", "#0f172a")

# ── Scrollbar ────────────────────────────────────────────────────────────────
SCROLLBAR       = ("#cbd5e1", "#424242", "#1e293b")
SCROLLBAR_HOVER = ("#94a3b8", "#565656", "#334155")

# ── Resolver ─────────────────────────────────────────────────────────────────
from dizel_ui.theme.theme_manager import Theme

def resolve(color) -> str:
    """
    Resolve a color tuple (light, dark, dark_blue) or string hex to a target hex
    based on the current Theme setting.
    """
    if isinstance(color, tuple):
        if len(color) >= 3:
            return Theme.get(color[0], color[1], color[2])
        elif len(color) == 2:
            return Theme.get(color[0], color[1])
    return color
