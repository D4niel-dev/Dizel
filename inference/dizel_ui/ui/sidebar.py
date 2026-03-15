"""
dizel_ui/ui/sidebar.py
───────────────────────
Collapsible left sidebar containing:
  • Dizel logo / app name
  • "New Chat" button
  • Scrollable chat history list
  • Collapse / expand toggle
  • Settings button (bottom)

The sidebar fires callbacks into ChatWindow; it owns no model state itself.
"""

import tkinter as tk
import customtkinter as ctk
import os
from PIL import Image
from typing import Callable, List, Dict, Optional

from dizel_ui.utils.icons import get_icon
from ..theme.colors import (
    BG_SIDEBAR, ACCENT, ACCENT_HOVER, ACCENT_LIGHT,
    SIDEBAR_BTN_HOVER, SIDEBAR_BTN_ACTIVE, SIDEBAR_TEXT,
    SIDEBAR_TEXT_DIM, SIDEBAR_BORDER, TEXT_PRIMARY, TEXT_DIM, BG_ROOT, SIDEBAR_PREMIUM_BG
)
from ..theme.fonts import LOGO, NAV_ITEM, NAV_ITEM_SM, BTN_LABEL, LABEL_SM, LABEL_DIM


# ── Dimensions ────────────────────────────────────────────────────────────────
SIDEBAR_W_OPEN   = 240
SIDEBAR_W_CLOSED = 56


class HistoryItem(ctk.CTkFrame):
    """A single clickable row in the history list."""

    def __init__(
        self,
        parent,
        session:    Dict,
        on_click:   Callable[[str], None],
        on_delete:  Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._session_id = session["id"]
        self._on_click   = on_click
        self._on_delete  = on_delete

        self.configure(cursor="hand2")

        # Main row
        self._row = ctk.CTkFrame(self, fg_color="transparent", corner_radius=8)
        self._row.pack(fill="x", padx=6, pady=1)

        title = session.get("title", "Untitled")[:32]
        self._lbl = ctk.CTkLabel(
            self._row,
            text=title,
            font=NAV_ITEM,
            text_color=SIDEBAR_TEXT,
            anchor="w",
        )
        self._lbl.pack(side="left", padx=(10, 4), pady=6, fill="x", expand=True)

        # Delete button (hidden until hover)
        self._del_btn = ctk.CTkButton(
            self._row,
            text="✕",
            width=20,
            height=20,
            fg_color="transparent",
            hover_color=SIDEBAR_BORDER,
            text_color="#888",
            font=LABEL_SM,
            command=self._delete,
        )

        # Hover binding
        for w in (self, self._row, self._lbl):
            w.bind("<Enter>",  self._on_enter)
            w.bind("<Leave>",  self._on_leave)
            w.bind("<Button-1>", self._on_click_evt)
        self._del_btn.bind("<Enter>", self._on_enter)
        self._del_btn.bind("<Leave>", self._on_leave)

    def _on_enter(self, _evt=None) -> None:
        self._row.configure(fg_color=SIDEBAR_BTN_HOVER)
        self._del_btn.pack(side="right", padx=(0, 6))

    def _on_leave(self, _evt=None) -> None:
        self._row.configure(fg_color="transparent")
        self._del_btn.pack_forget()

    def _on_click_evt(self, _evt=None) -> None:
        self._on_click(self._session_id)

    def _delete(self) -> None:
        self._on_delete(self._session_id)


class Sidebar(ctk.CTkFrame):
    """
    Left sidebar with collapsible animation.

    Callbacks
    ---------
    on_new_chat()            : user clicked "New Chat"
    on_session_select(id)    : user clicked a history item
    on_session_delete(id)    : user deleted a history item
    on_settings()            : user clicked Settings
    """

    def __init__(
        self,
        parent,
        on_new_chat:       Callable[[], None],
        on_session_select: Callable[[str], None],
        on_session_delete: Callable[[str], None],
        on_settings:       Callable[[], None],
        on_action:         Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            parent,
            fg_color=BG_SIDEBAR,
            width=SIDEBAR_W_OPEN,
            corner_radius=0,
            **kwargs,
        )
        self.pack_propagate(False)

        self._on_new_chat       = on_new_chat
        self._on_session_select = on_session_select
        self._on_session_delete = on_session_delete
        self._on_settings       = on_settings
        self._on_action         = on_action
        self._is_open           = True
        self._anim_target       = SIDEBAR_W_OPEN
        self._hist_open         = True  # Toggled by the Chat button
        
        self._feature_btns      = []    # Keep references to update text later
        self._workspace_btns    = []

        self._build()

    # ── Build layout ──────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Top bar: logo + collapse button ──────────────────────────────
        top_bar = ctk.CTkFrame(self, fg_color="transparent", height=64)
        top_bar.pack(fill="x", padx=0, pady=0)
        top_bar.pack_propagate(False)

        # We assume the script is run from project root, but we can resolve relative to this file
        _UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(_UI_DIR, "assets", "app", "Dizel.png")
        
        try:
            pil_img = Image.open(logo_path)
            self._logo_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(24, 24))
            
            self._logo_lbl = ctk.CTkLabel(
                top_bar,
                text="  Dizel",
                image=self._logo_img,
                compound="left",
                font=LOGO,
                text_color=TEXT_PRIMARY,
                anchor="w",
            )
        except Exception as e:
            # Fallback if image fails to load
            print(f"Could not load Dizel.png for sidebar: {e}")
            self._logo_lbl = ctk.CTkLabel(
                top_bar,
                text="⬡ Dizel",
                font=LOGO,
                text_color=TEXT_PRIMARY,
                anchor="w",
            )
            
        self._logo_lbl.pack(side="left", padx=16, pady=20)

        self._icon_open = get_icon("sidebar", size=(18, 18), color=SIDEBAR_TEXT_DIM)
        self._icon_closed = get_icon("menu", size=(18, 18), color=SIDEBAR_TEXT_DIM)

        self._toggle_btn = ctk.CTkButton(
            top_bar,
            text="",
            image=self._icon_open,
            width=28,
            height=28,
            fg_color="transparent",
            hover_color=SIDEBAR_BTN_HOVER,
            command=self.toggle,
        )
        self._toggle_btn.pack(side="right", padx=12, pady=18)

        # ── New Chat button (Zyricon Pill Style) ─────────────────────────
        plus_ico = get_icon("plus", size=(18, 18), color=TEXT_PRIMARY)
        self._new_btn = ctk.CTkButton(
            self,
            text="New Chat",
            image=plus_ico,
            font=BTN_LABEL,
            fg_color=SIDEBAR_BTN_HOVER,
            hover_color=SIDEBAR_BTN_ACTIVE,
            text_color=TEXT_PRIMARY,
            corner_radius=8,
            height=36,
            anchor="w",
            command=self._on_new_chat,
        )
        self._new_btn.pack(fill="x", padx=14, pady=(12, 24))

        # ── Features Container (Now just Chat) ────────────────────────────
        self._chat_container = ctk.CTkFrame(self, fg_color="transparent")
        self._chat_container.pack(fill="x", pady=(8, 2))

        # ── Chat Button & History Accordion ──
        chat_ico = get_icon("message-square", size=(18, 18), color=SIDEBAR_TEXT)
        self._chat_btn = ctk.CTkButton(
            self._chat_container, text="  Chats", image=chat_ico, font=NAV_ITEM, fg_color="transparent", 
            hover_color=SIDEBAR_BTN_HOVER, text_color=SIDEBAR_TEXT, anchor="w", height=32, corner_radius=6,
            command=self._toggle_history
        )
        self._chat_btn.pack(fill="x", padx=12, pady=2)
        
        # ── Scrollable history list (For Saved Chats) ──────────────
        self._hist_frame = ctk.CTkScrollableFrame(
            self._chat_container, fg_color="transparent",
            scrollbar_button_color=SIDEBAR_BORDER,
            scrollbar_button_hover_color=ACCENT,
            height=180
        )
        self._hist_frame.pack(fill="x", padx=(28, 12), pady=0)

        # ── Separator ─────────────────────────────────────────────────────
        self._sep1 = ctk.CTkFrame(self, fg_color=SIDEBAR_BORDER, height=1)
        self._sep1.pack(fill="x", padx=16, pady=16)

        # ── Workspaces Section ────────────────────────────────────────────
        self._work_lbl = ctk.CTkLabel(
            self, text="Workspaces", font=LABEL_DIM, text_color=SIDEBAR_TEXT_DIM, anchor="w"
        )
        self._work_lbl.pack(fill="x", padx=16, pady=(0, 6))

        self._work_container = ctk.CTkFrame(self, fg_color="transparent")
        self._work_container.pack(fill="x", pady=2)

        workspaces = [
            ("folder-plus", "New Project"),
            ("image", "Image"),
            ("layout", "Presentation"),
            ("search", "Riset"),
            ("archive", "Archived"),
            ("book", "Library")
        ]

        for icon_name, text in workspaces:
            ico = get_icon(icon_name, size=(18, 18), color=SIDEBAR_TEXT)
            cmd = (lambda t=text: self._on_action(f"Opening {t}...")) if self._on_action else None
            btn = ctk.CTkButton(
                self._work_container, text=f"  {text}", image=ico, font=NAV_ITEM, fg_color="transparent", 
                hover_color=SIDEBAR_BTN_HOVER, text_color=SIDEBAR_TEXT, anchor="w", height=32, corner_radius=6,
                command=cmd
            )
            btn.pack(fill="x", padx=12, pady=2)
            self._workspace_btns.append((btn, text))

        # ── History frame is now created under Chat ───────────────────────

        # ── Premium Upgrade Card ──────────────────────────────────────────
        self._premium_card = ctk.CTkFrame(self, fg_color=SIDEBAR_PREMIUM_BG, corner_radius=12)
        self._premium_card.pack(fill="x", padx=16, pady=(8, 16))
        
        star_ico = get_icon("star", size=(24, 24), color="#ffd700")  # Gold star
        crown_lbl = ctk.CTkLabel(self._premium_card, text="", image=star_ico)
        crown_lbl.pack(pady=(12, 4))
        
        title_lbl = ctk.CTkLabel(self._premium_card, text="Upgrade to premium", font=LABEL_SM, text_color=TEXT_PRIMARY)
        title_lbl.pack()
        
        desc_lbl = ctk.CTkLabel(
            self._premium_card, text="Boost productivity with seamless automation and responsive AI, built to adapt to your needs.",
            font=("", 9), text_color=TEXT_DIM, wraplength=170
        )
        desc_lbl.pack(pady=(2, 12), padx=12)

        upg_btn = ctk.CTkButton(
            self._premium_card, text="Upgrade", font=BTN_LABEL, fg_color=SIDEBAR_BTN_HOVER,
            hover_color=SIDEBAR_BTN_ACTIVE, text_color=TEXT_PRIMARY, height=28, corner_radius=6
        )
        upg_btn.pack(fill="x", padx=12, pady=(0, 16))

    # ── History management ────────────────────────────────────────────────

    def refresh_history(self, sessions: List[Dict]) -> None:
        """Re-render the history list with updated session data."""
        for widget in self._hist_frame.winfo_children():
            widget.destroy()

        if not sessions:
            empty = ctk.CTkLabel(
                self._hist_frame,
                text="No saved chats yet.",
                font=NAV_ITEM_SM,
                text_color=SIDEBAR_TEXT_DIM,
                anchor="w",
            )
            empty.pack(padx=12, pady=8, anchor="w")
            return

        for session in sessions:
            item = HistoryItem(
                self._hist_frame,
                session    = session,
                on_click   = self._on_session_select,
                on_delete  = self._on_session_delete,
            )
            item.pack(fill="x")

    def _toggle_history(self) -> None:
        """Toggle the accordion state of the saved chats history."""
        # Only allow toggle if the sidebar itself is open
        if not self._is_open:
            self.toggle()  # Open sidebar first
            self._hist_open = True
            return
            
        self._hist_open = not self._hist_open
        if self._hist_open:
            self._hist_frame.pack(fill="x", padx=(28, 12), pady=0)
        else:
            self._hist_frame.pack_forget()

    # ── Collapse / expand animation ───────────────────────────────────────

    def toggle(self) -> None:
        self._is_open    = not self._is_open
        self._anim_target = SIDEBAR_W_OPEN if self._is_open else SIDEBAR_W_CLOSED
        self._animate()
        # Flip toggle icon
        self._toggle_btn.configure(image=self._icon_open if self._is_open else self._icon_closed)
        # Hide / show text elements
        if self._is_open:
            # Temporarily un-pack following containers to maintain vertical order when re-packing _work_lbl
            self._work_container.pack_forget()
            self._premium_card.pack_forget()

            self._logo_lbl.configure(text="  Dizel")
            self._logo_lbl.pack(side="left", padx=16, pady=20)
            
            self._toggle_btn.pack_forget()
            self._toggle_btn.pack(side="right", padx=12, pady=18)
            
            self._new_btn.configure(text="  New Chat", anchor="w")
            self._new_btn.pack_configure(padx=16)
            
            # Show texts
            self._chat_btn.configure(text="  Chats")
            if self._hist_open:
                self._hist_frame.pack(fill="x", padx=(28, 12), pady=0)
                
            self._sep1.pack(fill="x", padx=16, pady=16)
            self._work_lbl.pack(fill="x", padx=16, pady=(0, 6))
            
            # Re-pack the containers
            self._work_container.pack(fill="x", pady=2)
            for btn, text in self._workspace_btns:
                btn.configure(text=f"  {text}")
                
            self._premium_card.pack(fill="x", padx=16, pady=(8, 16))

        else:
            self._logo_lbl.pack_forget()
            
            self._toggle_btn.pack_forget()
            self._toggle_btn.pack(side="top", pady=18)
            
            self._new_btn.configure(text="", anchor="center")
            self._new_btn.pack_configure(padx=(10, 10))
            
            # Hide texts but keep buttons (icons) visible
            self._work_lbl.pack_forget()
            self._sep1.pack_forget()
            self._premium_card.pack_forget()
            self._hist_frame.pack_forget()
            
            self._chat_btn.configure(text="")
            for btn, _ in self._workspace_btns:
                btn.configure(text="")

    def _animate(self) -> None:
        """Instant toggle instead of animated."""
        self.configure(width=self._anim_target)

    @property
    def is_open(self) -> bool:
        return self._is_open
