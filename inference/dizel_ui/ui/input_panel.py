"""
dizel_ui/ui/input_panel.py
───────────────────────────
Bottom input panel containing:
  • Multiline CTkTextbox (Enter = send, Shift+Enter = newline)
  • Send button with icon
  • Stop button (shown while generating)
  • Character / token counter label
  • Typing placeholder text
"""

import tkinter as tk
import customtkinter as ctk
from typing import Callable

from ..theme.colors import (
    BG_INPUT, BG_INPUT_FIELD, SEND_BTN, SEND_BTN_HOVER,
    BORDER, BORDER_FOCUS, TEXT_PRIMARY, TEXT_DIM, TEXT_SECONDARY,
    ACCENT, ACCENT_HOVER,
)
from ..theme.fonts import INPUT_TEXT, BTN_LABEL, LABEL_DIM

PLACEHOLDER = "Message Dizel…"


class InputPanel(ctk.CTkFrame):
    """
    Bottom input bar.

    Callbacks
    ---------
    on_send(text: str)  : called when the user submits a message
    on_stop()           : called when the user clicks Stop Generation
    """

    def __init__(
        self,
        parent,
        on_send: Callable[[str], None],
        on_stop: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            fg_color=BG_INPUT,
            corner_radius=0,
            **kwargs,
        )
        self._on_send    = on_send
        self._on_stop    = on_stop
        self._generating = False
        self._placeholder_active = True

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Top thin border line
        border = ctk.CTkFrame(self, fg_color=BORDER, height=1)
        border.pack(fill="x", side="top")

        # Inner container with padding
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=10)

        # ── Text input field ──────────────────────────────────────────────
        self._input = ctk.CTkTextbox(
            inner,
            font=INPUT_TEXT,
            fg_color=BG_INPUT_FIELD,
            text_color=TEXT_DIM,        # dim = placeholder colour
            border_color=BORDER,
            border_width=1,
            corner_radius=12,
            wrap="word",
            height=48,
            activate_scrollbars=True,
        )
        self._input.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self._input.insert("0.0", PLACEHOLDER)

        # Placeholder binding
        self._input.bind("<FocusIn>",  self._on_focus_in)
        self._input.bind("<FocusOut>", self._on_focus_out)
        self._input.bind("<Return>",   self._on_return)
        self._input.bind("<KeyRelease>", self._on_key_release)

        # ── Button column ────────────────────────────────────────────────
        btn_col = ctk.CTkFrame(inner, fg_color="transparent")
        btn_col.pack(side="right", fill="y")

        self._send_btn = ctk.CTkButton(
            btn_col,
            text="Send ➤",
            font=BTN_LABEL,
            width=80,
            height=48,
            fg_color=SEND_BTN,
            hover_color=SEND_BTN_HOVER,
            text_color="#ffffff",
            corner_radius=12,
            command=self._submit,
        )
        self._send_btn.pack()

        self._stop_btn = ctk.CTkButton(
            btn_col,
            text="■ Stop",
            font=BTN_LABEL,
            width=80,
            height=48,
            fg_color="#3a1a1a",
            hover_color="#5a2a2a",
            text_color="#f87171",
            corner_radius=12,
            command=self._on_stop,
        )
        # Stop button is hidden initially (shown during generation)

        # ── Char counter (below input) ────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=20, pady=(0, 6))

        self._counter_lbl = ctk.CTkLabel(
            footer,
            text="",
            font=LABEL_DIM,
            text_color=TEXT_DIM,
            anchor="e",
        )
        self._counter_lbl.pack(side="right")

        self._hint_lbl = ctk.CTkLabel(
            footer,
            text="Enter to send  •  Shift+Enter for new line",
            font=LABEL_DIM,
            text_color=TEXT_DIM,
            anchor="w",
        )
        self._hint_lbl.pack(side="left")

    # ── Placeholder logic ─────────────────────────────────────────────────

    def _on_focus_in(self, _evt=None) -> None:
        if self._placeholder_active:
            self._input.delete("0.0", "end")
            self._input.configure(text_color=TEXT_PRIMARY)
            self._placeholder_active = False
            self._input.configure(border_color=BORDER_FOCUS)

    def _on_focus_out(self, _evt=None) -> None:
        text = self._input.get("0.0", "end").strip()
        if not text:
            self._input.insert("0.0", PLACEHOLDER)
            self._input.configure(text_color=TEXT_DIM)
            self._placeholder_active = True
        self._input.configure(border_color=BORDER)

    # ── Key handlers ─────────────────────────────────────────────────────

    def _on_return(self, evt) -> str:
        """Enter = send; Shift+Enter = newline."""
        if evt.state & 0x1:   # Shift held
            return              # allow default newline insertion
        self._submit()
        return "break"         # prevent default newline

    def _on_key_release(self, _evt=None) -> None:
        """Update character counter."""
        if self._placeholder_active:
            self._counter_lbl.configure(text="")
            return
        text = self._input.get("0.0", "end").rstrip("\n")
        n    = len(text)
        self._counter_lbl.configure(text=f"{n} chars")

        # Auto-expand height (max ~5 lines)
        lines = text.count("\n") + 1
        h = max(48, min(lines * 24 + 8, 120))
        self._input.configure(height=h)

    # ── Submit / state control ────────────────────────────────────────────

    def _submit(self) -> None:
        if self._generating or self._placeholder_active:
            return
        text = self._input.get("0.0", "end").strip()
        if not text:
            return
        self.clear()
        self._on_send(text)

    def clear(self) -> None:
        """Clear the input field and reset to placeholder."""
        self._input.delete("0.0", "end")
        self._input.configure(text_color=TEXT_DIM)
        self._placeholder_active = True
        self._input.insert("0.0", PLACEHOLDER)
        self._counter_lbl.configure(text="")
        self._input.configure(height=48)

    def set_generating(self, generating: bool) -> None:
        """Toggle between Send and Stop button."""
        self._generating = generating
        if generating:
            self._send_btn.pack_forget()
            self._stop_btn.pack()
            self._input.configure(state="disabled")
        else:
            self._stop_btn.pack_forget()
            self._send_btn.pack()
            self._input.configure(state="normal")
            self._input.focus_set()

    def focus_input(self) -> None:
        self._input.focus_set()
