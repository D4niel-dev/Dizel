"""
dizel_ui/ui/message_bubble.py
──────────────────────────────
Individual chat message bubble widget.

Each bubble is a self-contained CTkFrame that renders:
  • role label ("You" or "Dizel")
  • message text (wrapping CTkTextbox, read-only)
  • copy button (top-right corner)
  • optional token count / timestamp meta line

User bubbles are right-aligned with an indigo background.
Assistant bubbles are left-aligned with a dark slate background.
"""

import tkinter as tk
import customtkinter as ctk
from datetime import datetime

from ..theme.colors import (
    BUBBLE_USER, BUBBLE_ASST, BUBBLE_USER_TXT, BUBBLE_ASST_TXT,
    BG_CHAT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM, ACCENT_LIGHT,
)
from ..theme.fonts import MSG_TEXT, MSG_META, LABEL_SM
from ..logic.config_manager import ConfigManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_textbox(parent, text: str, fg_color: str, text_color: str,
                  width: int) -> ctk.CTkTextbox:
    """
    Create a read-only, auto-sized CTkTextbox for the message content.
    Height expands to fit content.
    """
    box = ctk.CTkTextbox(
        parent,
        fg_color=fg_color,
        text_color=text_color,
        font=MSG_TEXT,
        wrap="word",
        activate_scrollbars=False,
        border_width=0,
        width=width,
    )
    box.insert("0.0", text)
    box.configure(state="disabled")

    # Calculate natural height
    lines     = text.count("\n") + 1
    max_chars = max((len(ln) for ln in text.splitlines()), default=0)
    # Rough char-per-line estimate
    chars_per_line = max(width // 8, 40)
    wrapped_lines  = sum(
        max(1, (len(ln) + chars_per_line - 1) // chars_per_line)
        for ln in text.splitlines()
    ) if text.strip() else 1
    height = max(30, wrapped_lines * 22 + 10)
    box.configure(height=height)
    return box


# ── MessageBubble ──────────────────────────────────────────────────────────────

class MessageBubble(ctk.CTkFrame):
    """
    A single chat message rendered as a styled bubble.

    Parameters
    ----------
    parent      : parent widget (usually the chat scroll frame)
    role        : "user" | "assistant"
    content     : message text to display
    meta        : optional metadata string (e.g. "128 tokens • 2.3 s")
    bubble_width: max pixel width of the bubble itself (auto-set by ChatWindow)
    """

    # How wide the bubble can grow relative to the chat column
    _MAX_FRACTION = 0.78

    def __init__(
        self,
        parent,
        role:         str,
        content:      str,
        meta:         str  = "",
        bubble_width: int  = 500,
        **kwargs,
    ) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._role    = role
        self._content = content
        self._is_user = (role == "user")

        bg_color   = BUBBLE_USER  if self._is_user else BUBBLE_ASST
        txt_color  = BUBBLE_USER_TXT if self._is_user else BUBBLE_ASST_TXT
        align_side = "e" if self._is_user else "w"   # east = right, west = left
        role_label = "You" if self._is_user else "✦ Dizel"

        # ── Outer row frame (full width, anchored left or right) ──────────
        self.columnconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=0, pady=4)

        # ── Bubble container ──────────────────────────────────────────────
        bubble = ctk.CTkFrame(
            row,
            fg_color=bg_color,
            corner_radius=16,
        )

        # ── Role label ────────────────────────────────────────────────────
        role_lbl = ctk.CTkLabel(
            bubble,
            text=role_label,
            font=LABEL_SM,
            text_color=ACCENT_LIGHT if not self._is_user else TEXT_PRIMARY,
            anchor="w",
        )
        role_lbl.pack(anchor="w", padx=16, pady=(12, 2))

        # ── Message text ──────────────────────────────────────────────────
        self._textbox = _make_textbox(
            bubble, content, bg_color, txt_color, bubble_width - 24
        )
        self._textbox.pack(fill="x", padx=14, pady=(0, 4))

        # ── Copy button ───────────────────────────────────────────────────
        bottom_row = ctk.CTkFrame(bubble, fg_color="transparent")
        bottom_row.pack(fill="x", padx=10, pady=(0, 8))

        self._copy_btn = ctk.CTkButton(
            bottom_row,
            text="⎘ Copy",
            font=MSG_META,
            width=60,
            height=22,
            fg_color="transparent",
            hover_color=BUBBLE_USER if self._is_user else BG_CHAT,
            text_color=TEXT_SECONDARY,
            border_width=0,
            command=self._copy_text,
            anchor="w",
        )
        self._copy_btn.pack(side="left")

        # ── Meta line (token count / latency) ─────────────────────────────
        if meta:
            meta_lbl = ctk.CTkLabel(
                bottom_row,
                text=meta,
                font=MSG_META,
                text_color=TEXT_DIM,
                anchor="e",
            )
            meta_lbl.pack(side="right", padx=6)

        # ── Timestamp (hover tooltip substitute) ──────────────────────────
        app_cfg = ConfigManager.load().get("appearance", {})
        ts = datetime.now().strftime("%H:%M")
        ts_lbl = ctk.CTkLabel(
            bottom_row,
            text=ts,
            font=MSG_META,
            text_color=TEXT_DIM,
        )
        if app_cfg.get("show_timestamps", True):
            ts_lbl.pack(side="right")

        # Pack bubble aligned to correct side
        if self._is_user:
            bubble.pack(anchor="e", padx=(80, 12), pady=2)
        else:
            bubble.pack(anchor="w", padx=(12, 80), pady=2)

    def _copy_text(self) -> None:
        """Copy message content to clipboard."""
        self.clipboard_clear()
        self.clipboard_append(self._content)
        # Brief visual feedback
        self._copy_btn.configure(text="✓ Copied")
        self.after(1500, lambda: self._copy_btn.configure(text="⎘ Copy"))

    def append_text(self, piece: str) -> None:
        """
        Append a streaming token piece to this bubble's textbox.
        Used during live generation so text appears incrementally.
        """
        self._textbox.configure(state="normal")
        self._textbox.insert("end", piece)
        self._textbox.configure(state="disabled")
        self._textbox.see("end")

    def finalise(self, full_text: str, meta: str = "") -> None:
        """
        Replace the textbox content with the final full response and
        update the meta line.  Called when generation completes.
        """
        self._content = full_text
        self._textbox.configure(state="normal")
        self._textbox.delete("0.0", "end")
        self._textbox.insert("0.0", full_text)
        self._textbox.configure(state="disabled")
