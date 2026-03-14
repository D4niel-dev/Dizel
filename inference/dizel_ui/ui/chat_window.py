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

import time
import tkinter as tk
import customtkinter as ctk
from typing import Optional

from .message_bubble import MessageBubble
from ..theme.colors import (
    BG_CHAT, BG_ROOT, ACCENT, ACCENT_LIGHT, ACCENT_HOVER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    WELCOME_CARD, WELCOME_CARD_HOVER, TYPING_DOT, BUBBLE_ASST,
)
from ..theme.fonts import (
    WELCOME_TITLE, WELCOME_SUB, CARD_TITLE, CARD_BODY,
    MSG_TEXT, TYPING_TEXT, BTN_LABEL,
)


# ── Quick-action cards shown on the welcome screen ────────────────────────────
WELCOME_CARDS = [
    ("💡 Ask about coding",      "Explain how a for loop works in Python."),
    ("⚡ Generate code",          "Write a Python function that reverses a string."),
    ("🔬 Explain a concept",     "Explain photosynthesis in simple terms."),
    ("📊 Structured output",     "List the planets in our solar system as JSON."),
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
            scrollbar_button_color="#2a2a3a",
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
        self._welcome_frame.pack(fill="both", expand=True, pady=40)

        # Logo / tagline
        ctk.CTkLabel(
            self._welcome_frame,
            text="⬡",
            font=("", 54),
            text_color=ACCENT_LIGHT,
        ).pack(pady=(30, 8))

        ctk.CTkLabel(
            self._welcome_frame,
            text="Dizel AI",
            font=WELCOME_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack()

        ctk.CTkLabel(
            self._welcome_frame,
            text="A structured analytical language model running locally.",
            font=WELCOME_SUB,
            text_color=TEXT_SECONDARY,
        ).pack(pady=(6, 30))

        # Quick-action cards
        cards_row = ctk.CTkFrame(self._welcome_frame, fg_color="transparent")
        cards_row.pack(padx=40)

        for i, (label, prompt) in enumerate(WELCOME_CARDS):
            col = i % 2
            row = i // 2
            card = self._make_card(cards_row, label, prompt)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

        for c in range(2):
            cards_row.columnconfigure(c, weight=1)

    def _make_card(self, parent, label: str, prompt: str) -> ctk.CTkFrame:
        """A clickable quick-action card."""
        card = ctk.CTkFrame(
            parent,
            fg_color=WELCOME_CARD,
            corner_radius=12,
            cursor="hand2",
            width=220,
            height=72,
        )
        card.pack_propagate(False)

        ctk.CTkLabel(
            card, text=label,
            font=CARD_TITLE, text_color=TEXT_PRIMARY, anchor="w",
        ).pack(padx=14, pady=(12, 2), anchor="w")

        ctk.CTkLabel(
            card, text=prompt[:48] + ("…" if len(prompt) > 48 else ""),
            font=CARD_BODY, text_color=TEXT_SECONDARY, anchor="w",
            wraplength=190,
        ).pack(padx=14, pady=(0, 10), anchor="w")

        # Hover + click
        def enter(_e): card.configure(fg_color=WELCOME_CARD_HOVER)
        def leave(_e): card.configure(fg_color=WELCOME_CARD)
        def click(_e): self._on_quick_action(prompt)

        for w in card.winfo_children() + [card]:
            w.bind("<Enter>", enter)
            w.bind("<Leave>", leave)
            w.bind("<Button-1>", click)

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
