"""
dizel_ui/ui/chat_window.py
───────────────────────────
Main chat area containing:
  • Welcome screen (shown before first message)
  • Scrollable bubble list
  • Typing / "Dizel is thinking…" indicator
  • Streaming token append to live bubble

This widget owns no model logic. It renders what ChatManager tells it
to render via the callbacks wired up in main.py.
"""

from __future__ import annotations

import os
import time
import tkinter as tk
import customtkinter as ctk
from typing import Optional
from PIL import Image

from .message_bubble import MessageBubble
from ..theme.colors import (
    BG_CHAT, BG_ROOT, ACCENT, ACCENT_LIGHT, ACCENT_HOVER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    WELCOME_CARD, WELCOME_CARD_HOVER, TYPING_DOT, BUBBLE_ASST,
    ACTION_PILL, SCROLLBAR
)
from dizel_ui.utils.icons import get_icon
from ..theme.fonts import (
    WELCOME_TITLE, WELCOME_SUB, CARD_TITLE, CARD_BODY,
    MSG_TEXT, TYPING_TEXT, BTN_LABEL,
)
from ..logic.config_manager import ConfigManager


# ── Action Pills ──────────────────────────────────────────────────────────────
ACTION_PILLS = [
    ("image", "Create Image", "Generate an image of a futuristic city."),
    ("zap", "Brainstorm", "Give me 5 ideas for a new web project."),
    ("file-text", "Make a plan", "Create a structured study plan for Python."),
]

# ── Feature Cards ─────────────────────────────────────────────────────────────
FEATURE_CARDS = [
    ("image", "Image Generator", "High-quality, dynamic image creation tool."),
    ("layout", "AI Presentation", "Generate professional slides in seconds."),
    ("code", "Dev Assistant", "Write, debug, and optimize your code."),
]


class TypingIndicator(ctk.CTkFrame):
    """
    Animated three-dot typing indicator shown while the model generates.
    Dots pulse via a simple tkinter after() loop.
    """

    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, fg_color=BUBBLE_ASST, corner_radius=14, **kwargs)
        self._running = False

        lbl = ctk.CTkLabel(
            self, text="Dizel  ", font=TYPING_TEXT,
            text_color=TEXT_SECONDARY, anchor="w",
        )
        lbl.pack(side="left", padx=(14, 2), pady=10)

        self._dot_lbl = ctk.CTkLabel(
            self, text="●", font=TYPING_TEXT,
            text_color=TYPING_DOT, anchor="w",
        )
        self._dot_lbl.pack(side="left", pady=10, padx=(0, 14))

    def start(self) -> None:
        self._running = True
        self._animate(0)

    def stop(self) -> None:
        self._running = False

    def _animate(self, frame: int) -> None:
        if not self._running:
            return
            
        app_cfg = ConfigManager.load().get("appearance", {})
        if not app_cfg.get("animations", True):
            # Static indicator if animations are disabled
            try:
                self._dot_lbl.configure(text="●  ●  ●")
            except tk.TclError:
                pass
            return

        dots = ["●  ○  ○", "○  ●  ○", "○  ○  ●"][frame % 3]
        try:
            self._dot_lbl.configure(text=dots)
            self.after(420, lambda: self._animate(frame + 1))
        except tk.TclError:
            pass   # widget was destroyed


class ChatWindow(ctk.CTkFrame):
    """
    The central scrollable chat area.

    Public API
    ----------
    show_welcome()                   — render welcome screen
    add_user_message(text)           — add user bubble, return it
    start_assistant_message()        — add empty assistant bubble + typing indicator
    append_token(piece)              — stream one token piece into active bubble
    finish_assistant_message(meta)   — finalise bubble, hide typing indicator
    clear()                          — remove all messages, show welcome again
    """

    def __init__(self, parent, on_quick_action: callable, **kwargs) -> None:
        super().__init__(parent, fg_color=BG_CHAT, corner_radius=0, **kwargs)
        self._on_quick_action   = on_quick_action
        self._active_bubble: Optional[MessageBubble] = None
        self._typing_indicator: Optional[TypingIndicator] = None
        self._welcome_frame: Optional[ctk.CTkFrame] = None
        self._welcome_shown = False
        self._gen_start_time: float = 0.0
        self._token_count: int = 0

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Scrollable area for all message bubbles
        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=BG_CHAT,
            scrollbar_button_color=SCROLLBAR,
            scrollbar_button_hover_color=ACCENT,
            corner_radius=0,
        )
        self._scroll.pack(fill="both", expand=True)
        self._scroll.columnconfigure(0, weight=1)
        # Initial state
        self.show_welcome()

    # ── Welcome screen ────────────────────────────────────────────────────

    def show_welcome(self) -> None:
        """Display the welcome / empty-state screen."""
        self._clear_messages()
        self._welcome_shown = True

        self._welcome_frame = ctk.CTkFrame(
            self._scroll, fg_color="transparent"
        )
        self._welcome_frame.pack(fill="both", expand=True, pady=(60, 20))

        # Avatar Image
        try:
            # Assuming main.py is in dizel_ui, so assets is at ./assets
            # Let's use the current file path to find the asset
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            avatar_path = os.path.join(base_dir, "assets", "avatars", "Diszi_beta2.png")
            img = Image.open(avatar_path)
            avatar_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(80, 80))
            lbl_avatar = ctk.CTkLabel(self._welcome_frame, text="", image=avatar_ctk)
            lbl_avatar.pack(pady=(0, 20))
        except Exception as e:
            # Fallback text if image missing
            ctk.CTkLabel(
                self._welcome_frame, text="⬡", font=("", 54), text_color=ACCENT_LIGHT,
            ).pack(pady=(0, 20))

        # Central Title
        ctk.CTkLabel(
            self._welcome_frame,
            text="Ready to Create Something New?",
            font=WELCOME_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack(pady=(0, 24))

        # Action Pills Row
        pills_row = ctk.CTkFrame(self._welcome_frame, fg_color="transparent")
        pills_row.pack(pady=(0, 40))

        for (icon_name, label, prompt) in ACTION_PILLS:
            ico = get_icon(icon_name, size=(16, 16), color=TEXT_PRIMARY)
            btn = ctk.CTkButton(
                pills_row, text=f"  {label}", image=ico, hover_color=WELCOME_CARD_HOVER, fg_color="transparent",
                border_color=TEXT_DIM, border_width=1, text_color=TEXT_PRIMARY, corner_radius=16,
                command=lambda p=prompt: self._on_quick_action(p)
            )
            btn.pack(side="left", padx=8)

        # Feature Cards (Bottom row)
        cards_row = ctk.CTkFrame(self._welcome_frame, fg_color="transparent")
        cards_row.pack(padx=20, fill="x")
        
        for c in range(3):
            cards_row.columnconfigure(c, weight=1)

        for i, (icon_name, title, desc) in enumerate(FEATURE_CARDS):
            card = self._make_feature_card(cards_row, icon_name, title, desc)
            card.grid(row=0, column=i, padx=12, sticky="ew")

    def _make_feature_card(self, parent, icon_name: str, title: str, desc: str) -> ctk.CTkFrame:
        """A display card for the features section."""
        card = ctk.CTkFrame(
            parent,
            fg_color=WELCOME_CARD,
            corner_radius=16,
            height=120,
        )
        card.pack_propagate(False)

        ctk.CTkLabel(
            card, text=title,
            font=CARD_TITLE, text_color=TEXT_PRIMARY, anchor="w",
        ).pack(padx=16, pady=(16, 4), anchor="w")

        ctk.CTkLabel(
            card, text=desc,
            font=CARD_BODY, text_color=TEXT_SECONDARY, anchor="w",
            wraplength=180, justify="left"
        ).pack(padx=16, anchor="nw")
        
        # Add visual sparkle / graphic hint
        img_placeholder = ctk.CTkFrame(card, fg_color=ACTION_PILL, width=36, height=36, corner_radius=8)
        img_placeholder.place(relx=0.85, rely=0.5, anchor="center")
        
        ico = get_icon(icon_name, size=(18, 18), color=ACCENT)
        ico_lbl = ctk.CTkLabel(img_placeholder, text="", image=ico)
        ico_lbl.place(relx=0.5, rely=0.5, anchor="center")

        def enter(_e): card.configure(fg_color=WELCOME_CARD_HOVER)
        def leave(_e): card.configure(fg_color=WELCOME_CARD)
        
        for w in card.winfo_children() + [card]:
            w.bind("<Enter>", enter)
            w.bind("<Leave>", leave)

        return card

    # ── Message management ────────────────────────────────────────────────

    def _dismiss_welcome(self) -> None:
        if self._welcome_frame:
            self._welcome_frame.destroy()
            self._welcome_frame = None
        self._welcome_shown = False

    def _clear_messages(self) -> None:
        for widget in self._scroll.winfo_children():
            widget.destroy()
        self._active_bubble     = None
        self._typing_indicator  = None
        self._welcome_frame     = None

    def clear(self) -> None:
        """Remove everything and show the welcome screen."""
        self._clear_messages()
        self.show_welcome()

    def _bubble_width(self) -> int:
        """Compute a sensible max bubble width from the current window size."""
        w = self.winfo_width()
        return max(340, int(w * 0.74))

    def add_user_message(self, text: str) -> MessageBubble:
        self._dismiss_welcome()
        bubble = MessageBubble(
            self._scroll,
            role="user",
            content=text,
            bubble_width=self._bubble_width(),
        )
        bubble.pack(fill="x", padx=0, pady=4)
        self._scroll_to_bottom()
        return bubble

    def start_assistant_message(self) -> None:
        """
        Show typing indicator and create an empty assistant bubble that
        will be populated token by token via append_token().
        """
        self._dismiss_welcome()
        self._gen_start_time = time.time()
        self._token_count    = 0

        # Typing indicator (left-aligned, same visual weight as assistant)
        indicator_wrap = ctk.CTkFrame(self._scroll, fg_color="transparent")
        indicator_wrap.pack(anchor="w", padx=(12, 80), pady=(4, 0))

        self._typing_indicator = TypingIndicator(indicator_wrap)
        self._typing_indicator.pack(anchor="w")
        self._typing_indicator.start()

        # Empty live bubble beneath the indicator
        self._active_bubble = MessageBubble(
            self._scroll,
            role="assistant",
            content="",
            bubble_width=self._bubble_width(),
        )
        self._active_bubble.pack(fill="x", padx=0, pady=(2, 4))
        self._scroll_to_bottom()

    def append_token(self, piece: str) -> None:
        """Stream one token piece into the live assistant bubble."""
        if self._typing_indicator:
            self._typing_indicator.stop()
            self._typing_indicator.destroy()
            self._typing_indicator = None

        if self._active_bubble:
            self._active_bubble.append_text(piece)
            self._token_count += 1
            self._scroll_to_bottom()

    def finish_assistant_message(self, full_text: str) -> None:
        """Finalise the live bubble with the complete decoded response."""
        if self._typing_indicator:
            self._typing_indicator.stop()
            self._typing_indicator.destroy()
            self._typing_indicator = None

        elapsed = time.time() - self._gen_start_time
        tps     = self._token_count / max(elapsed, 0.01)
        meta    = f"{self._token_count} tokens  •  {elapsed:.1f}s  •  {tps:.0f} tok/s"

        if self._active_bubble:
            self._active_bubble.finalise(full_text, meta)
            self._active_bubble = None

        self._scroll_to_bottom()

    def show_error(self, msg: str) -> None:
        """Display a red error message bubble."""
        if self._typing_indicator:
            self._typing_indicator.stop()
            self._typing_indicator.destroy()
            self._typing_indicator = None
        if self._active_bubble:
            self._active_bubble.finalise(f"⚠ Error: {msg}")
            self._active_bubble = None

    # ── Scroll helper ─────────────────────────────────────────────────────

    def _scroll_to_bottom(self) -> None:
        self.after(30, lambda: self._scroll._parent_canvas.yview_moveto(1.0))
