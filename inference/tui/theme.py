"""Terminal palette and Rich styles.

The same ladder as the desktop dark theme in ``inference/gui/src/styles/tokens.css``:
chat ``#212121``, chrome ``#171717``, panels ``#1c1c1c``, chips ``#2f2f2f``,
one blue accent ``#0ea5e9``. Signal colours are only used for state: blue for
live, amber for work in progress, red for failure.

``cmd_ui.tcss`` mirrors these as CSS variables; keep the two in sync.
"""

# ── Surfaces ──────────────────────────────────────────────────────────────
# OpenCode's palette: a warm near-black with a red undertone (#201d1d), warm
# off-white text, and Apple system blue as the only accent. Deliberately not
# the desktop ladder — the TUI is its own surface.
BG = "#201d1d"          # screen / transcript
BG_DEEP = "#191616"     # header, sidebar, status line
SURFACE = "#302c2c"     # composer, cards
SURFACE_HI = "#3b3636"  # chips, hover

# ── Lines and type ────────────────────────────────────────────────────────
BORDER = "#332e2e"
BORDER_HI = "#4a4444"
TEXT = "#fdfcfc"
TEXT_2 = "#b9b5b5"
TEXT_3 = "#9a9898"

# ── Signals ───────────────────────────────────────────────────────────────
ACCENT = "#007aff"
ACCENT_HI = "#4d9bff"
# Text placed on the accent needs to stay above 4.5:1; the warm near-black ink
# is not quite dark enough for the small keycaps.
ACCENT_INK = "#0b1220"
WARN = "#ff9f0a"
DANGER = "#ff3b30"
SUCCESS = "#30d158"

# Section labels, matching the stylesheet tokens.
VIOLET = "#a78bfa"

# A selection is a cool tint of the accent, not a full blue bar: visible at a
# glance without turning a list into a stack of highlights.
SELECT_BG = "#143048"

# Patch colours from OpenCode's v2 theme: cool green/red against the warm
# canvas, with the line background carrying the emphasis.
DIFF_ADD = "#4fd6be"
DIFF_ADD_BG = "#20303b"
DIFF_DEL = "#ef6c6c"
DIFF_DEL_BG = "#37222c"
DIFF_HUNK = "#8ab4f8"

# One-cell activity spinner, same rhythm as OpenCode's.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# ── Role styles (plain Rich colour strings, safe in f-strings) ────────────
STYLE_USER = TEXT_2
STYLE_ASSISTANT = ACCENT
STYLE_SYSTEM = TEXT_3
STYLE_TOOL = WARN

RAIL = "│"

# One syntax theme for the transcript and the artifact panel. Two themes in
# one window read as two products.
CODE_BG = "#262222"
CODE_THEME = "native"
