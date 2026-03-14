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
ANIM_STEPS       = 8
ANIM_DELAY_MS    = 14


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
            hover_color="#3a1a1a",
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
        self._is_open           = True
        self._anim_target       = SIDEBAR_W_OPEN

        self._build()

    # ── Build layout ──────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Top bar: logo + collapse button ──────────────────────────────
        top_bar = ctk.CTkFrame(self, fg_color="transparent", height=64)
        top_bar.pack(fill="x", padx=0, pady=0)
        top_bar.pack_propagate(False)

        self._logo_lbl = ctk.CTkLabel(
            top_bar,
            text="⬡ Dizel",
            font=LOGO,
            text_color=TEXT_PRIMARY,
            anchor="w",
        )
        self._logo_lbl.pack(side="left", padx=16, pady=20)

        self._toggle_btn = ctk.CTkButton(
            top_bar,
            text="◫",
            width=28,
            height=28,
            fg_color="transparent",
            hover_color=SIDEBAR_BTN_HOVER,
            text_color=SIDEBAR_TEXT_DIM,
            font=LOGO,
            command=self.toggle,
        )
        self._toggle_btn.pack(side="right", padx=12, pady=18)

        # ── New Chat button (Zyricon Pill Style) ─────────────────────────
        plus_ico = get_icon("plus", size=(18, 18), color=TEXT_PRIMARY)
        self._new_btn = ctk.CTkButton(
            self,
            text="  New Chat",
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
        self._new_btn.pack(fill="x", padx=16, pady=(12, 24))

        # ── Features Section ──────────────────────────────────────────────
        self._feat_lbl = ctk.CTkLabel(
            self, text="Features", font=LABEL_DIM, text_color=SIDEBAR_TEXT_DIM, anchor="w"
        )
        self._feat_lbl.pack(fill="x", padx=16, pady=(0, 6))

        self._feat_container = ctk.CTkFrame(self, fg_color="transparent")
        self._feat_container.pack(fill="x", pady=2)

        features = [
            ("message-square", "Chat"),
            ("archive", "Archived"),
            ("book", "Library")
        ]
        
        for icon_name, text in features:
            ico = get_icon(icon_name, size=(18, 18), color=SIDEBAR_TEXT)
            btn = ctk.CTkButton(
                self._feat_container, text=f"  {text}", image=ico, font=NAV_ITEM, fg_color="transparent", 
                hover_color=SIDEBAR_BTN_HOVER, text_color=SIDEBAR_TEXT, anchor="w", height=32, corner_radius=6
            )
            btn.pack(fill="x", padx=12, pady=2)

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
        ]

        for icon_name, text in workspaces:
            ico = get_icon(icon_name, size=(18, 18), color=SIDEBAR_TEXT)
            btn = ctk.CTkButton(
                self._work_container, text=f"  {text}", image=ico, font=NAV_ITEM, fg_color="transparent", 
                hover_color=SIDEBAR_BTN_HOVER, text_color=SIDEBAR_TEXT, anchor="w", height=32, corner_radius=6
            )
            btn.pack(fill="x", padx=12, pady=2)

        # ── Scrollable history list (For Saved Chats if any) ──────────────
        self._hist_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=SIDEBAR_BORDER,
            scrollbar_button_hover_color=ACCENT,
        )
        self._hist_frame.pack(fill="both", expand=True, padx=0, pady=4)

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

    # ── Collapse / expand animation ───────────────────────────────────────

    def toggle(self) -> None:
        self._is_open    = not self._is_open
        self._anim_target = SIDEBAR_W_OPEN if self._is_open else SIDEBAR_W_CLOSED
        self._animate()
        # Flip toggle arrow
        self._toggle_btn.configure(text="◫" if self._is_open else "◨")
        # Hide / show text elements
        if self._is_open:
            self._logo_lbl.pack(side="left", padx=16, pady=20)
            self._new_btn.configure(text="  New Chat")
            
            # Repack sections
            self._feat_lbl.pack(fill="x", padx=16, pady=(0, 6))
            self._feat_container.pack(fill="x", pady=2)
            self._sep1.pack(fill="x", padx=16, pady=16)
            self._work_lbl.pack(fill="x", padx=16, pady=(0, 6))
            self._work_container.pack(fill="x", pady=2)
            self._premium_card.pack(fill="x", padx=16, pady=(8, 16))
            self._hist_frame.pack(fill="both", expand=True, padx=0, pady=4)

        else:
            self._logo_lbl.pack_forget()
            self._new_btn.configure(text="")
            
            # Hide sections
            self._feat_lbl.pack_forget()
            self._feat_container.pack_forget()
            self._sep1.pack_forget()
            self._work_lbl.pack_forget()
            self._work_container.pack_forget()
            self._premium_card.pack_forget()
            self._hist_frame.pack_forget()

    def _animate(self) -> None:
        current = self.winfo_width()
        target  = self._anim_target
        if current == target:
            return
        step = (target - current) // ANIM_STEPS
        if step == 0:
            step = 1 if target > current else -1
        new_w = current + step
        # Clamp
        if (step > 0 and new_w >= target) or (step < 0 and new_w <= target):
            new_w = target
        self.configure(width=new_w)
        if new_w != target:
            self.after(ANIM_DELAY_MS, self._animate)

    @property
    def is_open(self) -> bool:
        return self._is_open
