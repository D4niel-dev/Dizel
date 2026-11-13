"""Terminal-only preferences, applied to the running app.

These have no desktop equivalent: a window always has a mouse and no bell, so
there is nothing for the GUI to store. Keeping them here rather than inline in
the modal means startup and save share one implementation.

The appearance block is shared with the desktop: the keys below mean the same
thing on both surfaces, with the same defaults, so a change made in either
settings dialog carries across.
"""

from __future__ import annotations

from typing import Any

# Mirrors DEFAULT_PREFS / prefsFromConfig in inference/gui/src/store/chatStore.ts.
# Only the keys the terminal can act on are listed.
CHAT_DEFAULTS: dict[str, Any] = {
    "typing_animation": "smooth",
    "show_timestamps": True,
    "clock_24h": True,
    "show_speed": True,
    "show_model_tag": False,
    "syntax_highlighting": True,
    "separators": False,
    "message_spacing": "comfortable",
    "message_width": "comfortable",
    "auto_scroll": True,
    "show_suggestions": True,
    "animations": True,
    "enter_to_send": True,
    "auto_focus_composer": True,
    "escape_clears_draft": False,
    "arrow_up_edits_last": True,
}


def apply_terminal_prefs(app: Any, cfg: dict | None = None) -> None:
    if cfg is None:
        try:
            from dizel_core.config_manager import ConfigManager

            cfg = ConfigManager.load()
        except Exception:
            cfg = {}

    prefs = (cfg or {}).get("tui", {}) or {}

    # Cache on the app so panels can read a preference without touching disk on
    # every status-bar repaint.
    try:
        app._tui_prefs = dict(prefs)
    except Exception:
        pass

    appearance = (cfg or {}).get("appearance", {}) or {}
    try:
        app._chat_prefs = {
            key: appearance.get(key, default) for key, default in CHAT_DEFAULTS.items()
        }
    except Exception:
        pass

    try:
        # Textual's own switch: off means the terminal handles the mouse, which
        # is what you want when selecting text to copy.
        app.mouse = bool(prefs.get("mouse", False))
    except Exception:
        pass

    _apply_chat_prefs(app)


def _apply_chat_prefs(app: Any) -> None:
    """The shared prefs that need a push, not just a cached read."""
    try:
        container = app.query_one("#workspace-container")
        width = chat_pref(app, "message_width")
        container.set_class(width == "wide", "-width-wide")
        container.set_class(width == "full", "-width-full")
    except Exception:
        pass
    try:
        from inference.tui.rendering.empty_state import EmptyStateBlock

        for block in app.query(EmptyStateBlock):
            block.apply_suggestions()
    except Exception:
        pass


def pref(app: Any, key: str, default: bool = True) -> bool:
    """Read one terminal preference from the copy cached on the app."""
    prefs = getattr(app, "_tui_prefs", None)
    if prefs is None:
        return default
    value = prefs.get(key, default)
    return bool(value)


def chat_pref(app: Any, key: str, default: Any = None) -> Any:
    """Read one shared appearance preference, falling back to the GUI default."""
    if default is None:
        default = CHAT_DEFAULTS.get(key)
    prefs = getattr(app, "_chat_prefs", None)
    if prefs is None:
        return default
    return prefs.get(key, default)


def bell_enabled(cfg: dict | None = None) -> bool:
    if cfg is None:
        try:
            from dizel_core.config_manager import ConfigManager

            cfg = ConfigManager.load()
        except Exception:
            return False
    return bool(((cfg or {}).get("tui", {}) or {}).get("bell", False))