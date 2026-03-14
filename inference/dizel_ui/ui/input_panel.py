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

from dizel_ui.utils.icons import get_icon
from ..theme.colors import (
    BG_INPUT, BG_INPUT_FIELD, SEND_BTN, SEND_BTN_HOVER,
    BORDER, BORDER_FOCUS, TEXT_PRIMARY, TEXT_DIM, TEXT_SECONDARY,
    ACCENT, ACCENT_HOVER, WELCOME_CARD_HOVER
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
        on_settings: Callable[[], None] = lambda: None,
        on_attach: Callable[[], None] = lambda: None,
        on_options: Callable[[], None] = lambda: None,
        on_voice: Callable[[], None] = lambda: None,
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
        self._on_settings = on_settings
        self._on_attach   = on_attach
        self._on_options  = on_options
        self._on_voice    = on_voice
        self._generating = False
        self._placeholder_active = True

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Wrap the whole panel in a padded transparent container
        self.configure(fg_color="transparent")
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        # Floating rounded box
        box = ctk.CTkFrame(outer, fg_color=BG_INPUT, corner_radius=18, border_color=BORDER, border_width=1)
        box.pack(fill="x", expand=True)

        # ── Text input field (Top half of box) ────────────────────────────
        self._input = ctk.CTkTextbox(
            box,
            font=INPUT_TEXT,
            fg_color="transparent",
            text_color=TEXT_DIM,
            border_width=0,
            wrap="word",
            height=60,
            activate_scrollbars=True,
        )
        self._input.pack(fill="both", expand=True, padx=16, pady=(16, 4))
        self._input.insert("0.0", PLACEHOLDER)

        self._input.bind("<FocusIn>",  self._on_focus_in)
        self._input.bind("<FocusOut>", self._on_focus_out)
        self._input.bind("<Return>",   self._on_return)
        self._input.bind("<KeyRelease>", self._on_key_release)

        # ── Inline Action Row (Bottom half of box) ────────────────────────
        action_row = ctk.CTkFrame(box, fg_color="transparent", height=40)
        action_row.pack(fill="x", padx=12, pady=(0, 12))
        
        # Left Actions
        left_actions = ctk.CTkFrame(action_row, fg_color="transparent")
        left_actions.pack(side="left")
        
        for ico_name, lbl, cmd in [
            ("link", "Attach", self._on_attach),
            ("settings", "Settings", self._on_settings),
            ("grid", "Options", self._on_options)
        ]:
            ico = get_icon(ico_name, size=(16, 16), color=TEXT_DIM)
            btn = ctk.CTkButton(
                left_actions, text=f"  {lbl}", image=ico, font=BTN_LABEL, fg_color="transparent", text_color=TEXT_DIM, 
                hover_color=WELCOME_CARD_HOVER, width=60, height=28, corner_radius=14,
                command=cmd
            )
            btn.pack(side="left", padx=4)

        # Right Actions
        right_actions = ctk.CTkFrame(action_row, fg_color="transparent")
        right_actions.pack(side="right")

        mic_ico = get_icon("mic", size=(18, 18), color=TEXT_DIM)
        voice_btn = ctk.CTkButton(
            right_actions, text="", image=mic_ico, font=BTN_LABEL, fg_color="transparent", text_color=TEXT_DIM,
            hover_color=WELCOME_CARD_HOVER, width=32, height=32, corner_radius=16,
            command=self._on_voice
        )
        voice_btn.pack(side="left", padx=4)

        send_ico = get_icon("arrow-up", size=(20, 20), color="#ffffff")
        self._send_btn = ctk.CTkButton(
            right_actions,
            text="",
            image=send_ico,
            font=BTN_LABEL,
            width=36,
            height=36,
            fg_color=SEND_BTN,
            hover_color=SEND_BTN_HOVER,
            text_color="#ffffff",
            corner_radius=18,
            command=self._submit,
        )
        self._send_btn.pack(side="left", padx=(4, 0))

        stop_ico = get_icon("square", size=(16, 16), color="#f87171")
        self._stop_btn = ctk.CTkButton(
            right_actions,
            text="",
            image=stop_ico,
            font=BTN_LABEL,
            width=36,
            height=36,
            fg_color="#3a1a1a",
            hover_color="#5a2a2a",
            text_color="#f87171",
            corner_radius=18,
            command=self._on_stop,
        )

        # ── Char counter (below floated box) ──────────────────────────────
        footer = ctk.CTkFrame(outer, fg_color="transparent")
        footer.pack(fill="x", pady=(8, 0), padx=8)

        self._counter_lbl = ctk.CTkLabel(
            footer, text="", font=LABEL_DIM, text_color=TEXT_DIM, anchor="e"
        )
        self._counter_lbl.pack(side="right")

        self._hint_lbl = ctk.CTkLabel(
            footer, text="Enter to send  •  Shift+Enter for new line", font=LABEL_DIM, text_color=TEXT_DIM, anchor="w"
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
