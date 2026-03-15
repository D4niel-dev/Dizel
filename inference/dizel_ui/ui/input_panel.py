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
        self._on_send_msg = on_send
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

        # ── Attachment Preview Area (Hidden by default) ───────────────────
        self._preview_area = ctk.CTkFrame(box, fg_color="transparent", height=96)
        self._preview_area.pack_propagate(False)
        # Don't pack it initially until an attachment is added
        
        # We need a scrollable row inside it in case of many attachments
        self._preview_scroll = ctk.CTkScrollableFrame(
            self._preview_area, fg_color="transparent", orientation="horizontal", height=76,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=ACCENT
        )
        self._preview_scroll.pack(fill="x", padx=16, pady=(8, 0))

        # Keep track of attached widgets to destroy them later
        self._attachment_widgets = []
        self._attachments = []

        # ── Text input field (Middle of box) ────────────────────────────
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
        self._input.pack(fill="both", expand=True, padx=16, pady=(12, 4))
        self._input.insert("0.0", PLACEHOLDER)

        self._input.bind("<FocusIn>",  self._on_focus_in)
        self._input.bind("<FocusOut>", self._on_focus_out)
        self._input.bind("<Button-1>", self._on_click)
        self._input.bind("<Return>",   self._on_return)
        self._input.bind("<Key>",       self._on_key_press)
        self._input.bind("<KeyRelease>", self._on_key_release)

        # ── Inline Action Row (Bottom half of box) ────────────────────────
        self._action_row = ctk.CTkFrame(box, fg_color="transparent", height=40)
        self._action_row.pack(fill="x", padx=12, pady=(0, 12))
        
        # Left Actions
        left_actions = ctk.CTkFrame(self._action_row, fg_color="transparent")
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
        right_actions = ctk.CTkFrame(self._action_row, fg_color="transparent")
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
            command=self._do_submit,
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

    def _clear_placeholder(self) -> None:
        """Clear the placeholder text if it's currently showing."""
        if self._placeholder_active:
            self._input.delete("0.0", "end")
            self._input.configure(text_color=TEXT_PRIMARY)
            self._placeholder_active = False

    def _on_focus_in(self, _evt=None) -> None:
        self._clear_placeholder()
        self._input.configure(border_color=BORDER_FOCUS)

    def _on_click(self, _evt=None) -> None:
        self._clear_placeholder()

    def _on_focus_out(self, _evt=None) -> None:
        text = self._input.get("0.0", "end").strip()
        if not text:
            self._input.insert("0.0", PLACEHOLDER)
            self._input.configure(text_color=TEXT_DIM)
            self._placeholder_active = True
        self._input.configure(border_color=BORDER)

    # ── Key handlers ─────────────────────────────────────────────────────

    def _on_key_press(self, evt) -> None:
        """Clear placeholder on any printable key press."""
        if evt.keysym in ('Shift_L', 'Shift_R', 'Control_L', 'Control_R',
                          'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab',
                          'Up', 'Down', 'Left', 'Right', 'Escape'):
            return
        self._clear_placeholder()

    def _on_return(self, evt) -> str:
        """Enter = send; Shift+Enter = newline."""
        if evt.state & 0x1:   # Shift held
            return              # allow default newline insertion
        self._do_submit()
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

    def _do_submit(self, _evt=None) -> None:
        if self._generating:
            return

        text = self._input.get("0.0", "end").strip()
        files = self.get_attachments()
        
        # We need either text or an attachment to send
        if not text and not files:
            return

        if self._placeholder_active and not files:
            return

        # Clear input box FIRST (before disabling it)
        self._input.delete("0.0", "end")
        self._input.insert("0.0", PLACEHOLDER)
        self._input.configure(text_color=TEXT_DIM)
        self._placeholder_active = True
        self._counter_lbl.configure(text="")
        self._input.configure(height=48)
        
        # Clear attachment previews
        self.clear_attachments()

        # Disable input while generating
        self.set_generating(True)

        # Notify ChatWindow (via main app callback)
        if self._on_send_msg:
            self._on_send_msg(text)

    def clear(self) -> None:
        """Clear the input field and reset to placeholder."""
        self._input.delete("0.0", "end")
        self._input.configure(text_color=TEXT_DIM)
        self._placeholder_active = True
        self._input.insert("0.0", PLACEHOLDER)
        self._counter_lbl.configure(text="")
        self._input.configure(height=48)

    def add_attachment(self, file_path: str) -> None:
        import os
        from PIL import Image, ImageOps
        
        # If this is the first attachment, repack everything to insert the preview area at the top
        if not self._attachments:
            self._input.pack_forget()
            self._action_row.pack_forget()
            
            self._preview_area.pack(fill="x", pady=(4, 0))
            self._input.pack(fill="both", expand=True, padx=16, pady=(4, 4))
            self._action_row.pack(fill="x", padx=12, pady=(0, 12))
            
        self._attachments.append(file_path)
        
        ext = os.path.splitext(file_path)[1].lower()
        
        # Determine icon and styling
        bg_color = BG_INPUT_FIELD if 'BG_INPUT_FIELD' in globals() else "#1a1a24"
        
        if ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            ico_name = "image"
            thumb_img = None
            try:
                from PIL import ImageDraw
                pil_img = Image.open(file_path).convert("RGBA")
                pil_img = ImageOps.fit(pil_img, (62, 62))
                
                # apply rounded mask
                mask = Image.new('L', (62, 62), 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle((0, 0, 62, 62), radius=8, fill=255)
                
                img_masked = Image.new("RGBA", (62, 62), (0,0,0,0))
                img_masked.paste(pil_img, (0,0), mask)
                
                thumb_img = ctk.CTkImage(light_image=img_masked, dark_image=img_masked, size=(62, 62))
            except Exception as e:
                pass
        elif ext in [".zip", ".tar", ".gz", ".rar", ".7z"]:
            ico_name = "archive"
            bg_color = "#2a1e1e" # slightly reddish/brown for zips
            thumb_img = None
        else:
            ico_name = "file"
            thumb_img = None

        pill = ctk.CTkFrame(self._preview_scroll, fg_color="transparent")
        pill.pack(side="left", padx=(0, 12), pady=0)
        
        content = ctk.CTkFrame(pill, width=64, height=64, fg_color=bg_color, corner_radius=8, border_width=1, border_color=BORDER)
        content.pack_propagate(False)
        content.pack(side="left")
        
        if thumb_img:
            img_lbl = ctk.CTkLabel(content, text="", image=thumb_img)
            img_lbl.pack(fill="both", expand=True)
        else:
            type_ico = get_icon(ico_name, size=(24, 24), color=ACCENT)
            ico_lbl = ctk.CTkLabel(content, text="", image=type_ico)
            ico_lbl.pack(pady=(8, 0))
            
            ext_str = ext[1:].upper() if ext else "FILE"
            if len(ext_str) > 5: ext_str = ext_str[:5]
            name_lbl = ctk.CTkLabel(content, text=ext_str, font=LABEL_DIM, text_color=TEXT_PRIMARY)
            name_lbl.pack(pady=(0, 0))
            
        # Top-right X button packed cleanly next to the tile
        x_ico = get_icon("x", size=(14, 14), color=TEXT_DIM)
        rm_btn = ctk.CTkButton(
            pill, text="", image=x_ico, width=20, height=20, corner_radius=0, 
            fg_color="transparent", hover_color=BORDER, border_width=0, cursor="hand2",
            command=lambda p=pill, f=file_path: self._remove_attachment(p, f)
        )
        rm_btn.pack(side="left", anchor="n", padx=(4, 0))
        
        self._attachment_widgets.append(pill)

    def _remove_attachment(self, pill: ctk.CTkFrame, file_path: str) -> None:
        pill.destroy()
        if pill in self._attachment_widgets:
            self._attachment_widgets.remove(pill)
        if file_path in self._attachments:
            self._attachments.remove(file_path)
            
        if not self._attachments:
            self._preview_area.pack_forget()

    def get_attachments(self) -> list[str]:
        return self._attachments.copy()

    def clear_attachments(self) -> None:
        for w in self._attachment_widgets:
            w.destroy()
        self._attachment_widgets.clear()
        self._attachments.clear()
        self._preview_area.pack_forget()

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
            
            # Restore placeholder if empty
            content = self._input.get("0.0", "end").strip()
            if not content or content == PLACEHOLDER:
                self._input.delete("0.0", "end")
                self._input.insert("0.0", PLACEHOLDER)
                self._input.configure(text_color=TEXT_DIM)
                self._placeholder_active = True

    def focus_input(self) -> None:
        self._input.focus_set()
