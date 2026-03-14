"""
dizel_ui/ui/settings_dialog.py
────────────────────────────────
Modal settings dialog for configuring:
  • Checkpoint path (file-browser)
  • Device selection (CPU / CUDA)
  • Sampling parameters (temperature, top-k, top-p, rep. penalty, max tokens)
  • System prompt

Changes take effect immediately on Save.
"""

import os
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from typing import Callable

from ..theme.colors import (
    BG_ROOT, BG_CHAT, ACCENT, ACCENT_HOVER, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
)
from ..theme.fonts import LOGO, BTN_LABEL, LABEL, LABEL_SM


class SettingsDialog(ctk.CTkToplevel):
    """
    Settings modal.

    Parameters
    ----------
    parent      : root window
    chat_mgr    : ChatManager instance (settings applied to it on save)
    on_reload   : callback() called when the user requests a model reload
    """

    def __init__(self, parent, chat_mgr, on_reload: Callable[[], None]) -> None:
        super().__init__(parent)
        self._mgr      = chat_mgr
        self._on_reload = on_reload

        self.title("Settings — Dizel AI")
        self.geometry("540x620")
        self.resizable(False, False)
        self.configure(fg_color=BG_ROOT)
        self.grab_set()          # modal behaviour
        self.lift()
        self.focus_force()

        self._build()
        self._load_current()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # Title
        ctk.CTkLabel(
            self, text="Settings",
            font=LOGO, text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(20, 4))

        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1)
        sep.pack(fill="x", padx=24)

        scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=BORDER,
        )
        scroll.pack(fill="both", expand=True, padx=24, pady=12)

        # ── Checkpoint ───────────────────────────────────────────────────
        self._section(scroll, "Model Checkpoint")

        ckpt_row = ctk.CTkFrame(scroll, fg_color="transparent")
        ckpt_row.pack(fill="x", pady=(2, 8))

        self._ckpt_var = tk.StringVar()
        ckpt_entry = ctk.CTkEntry(
            ckpt_row,
            textvariable=self._ckpt_var,
            placeholder_text="Path to .pt checkpoint…",
            font=LABEL,
            fg_color=BG_CHAT,
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
        )
        ckpt_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        ctk.CTkButton(
            ckpt_row,
            text="Browse…",
            font=BTN_LABEL,
            width=80,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._browse_checkpoint,
        ).pack(side="right")

        # ── Device ───────────────────────────────────────────────────────
        self._section(scroll, "Device")
        self._device_var = tk.StringVar(value="cpu")
        device_row = ctk.CTkFrame(scroll, fg_color="transparent")
        device_row.pack(fill="x", pady=(2, 8))

        for dev in ("cpu", "cuda"):
            ctk.CTkRadioButton(
                device_row,
                text=dev.upper(),
                variable=self._device_var,
                value=dev,
                font=LABEL,
                text_color=TEXT_PRIMARY,
                fg_color=ACCENT,
            ).pack(side="left", padx=(0, 20))

        # ── System prompt ────────────────────────────────────────────────
        self._section(scroll, "System Prompt")
        self._sys_box = ctk.CTkTextbox(
            scroll,
            height=80,
            font=LABEL,
            fg_color=BG_CHAT,
            text_color=TEXT_PRIMARY,
            border_color=BORDER,
            border_width=1,
            corner_radius=8,
            wrap="word",
        )
        self._sys_box.pack(fill="x", pady=(2, 8))

        # ── Sampling sliders ────────────────────────────────────────────
        self._section(scroll, "Sampling Parameters")

        self._temp_var   = tk.DoubleVar(value=0.8)
        self._topk_var   = tk.IntVar(value=50)
        self._topp_var   = tk.DoubleVar(value=0.92)
        self._rep_var    = tk.DoubleVar(value=1.15)
        self._maxt_var   = tk.IntVar(value=200)

        sliders = [
            ("Temperature",        self._temp_var,  0.0,   2.0,  2),
            ("Top-K",              self._topk_var,  1,     200,  0),
            ("Top-P",              self._topp_var,  0.1,   1.0,  2),
            ("Repetition Penalty", self._rep_var,   1.0,   2.0,  2),
            ("Max New Tokens",     self._maxt_var,  32,    512,  0),
        ]
        self._slider_widgets = {}
        for label, var, lo, hi, decimals in sliders:
            self._make_slider(scroll, label, var, lo, hi, decimals)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=(0, 20))

        ctk.CTkButton(
            btn_row,
            text="Cancel",
            font=BTN_LABEL,
            width=100,
            fg_color="transparent",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_SECONDARY,
            hover_color="#1a1a28",
            command=self.destroy,
        ).pack(side="left")

        ctk.CTkButton(
            btn_row,
            text="Load & Apply",
            font=BTN_LABEL,
            width=130,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._save_and_reload,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_row,
            text="Apply",
            font=BTN_LABEL,
            width=90,
            fg_color="#2a2a40",
            hover_color="#3a3a50",
            text_color=TEXT_PRIMARY,
            command=self._save_only,
        ).pack(side="right")

    def _section(self, parent, title: str) -> None:
        ctk.CTkLabel(
            parent, text=title,
            font=LABEL_SM, text_color=TEXT_SECONDARY,
            anchor="w",
        ).pack(anchor="w", pady=(10, 2))

    def _make_slider(self, parent, label: str, var, lo, hi, decimals: int) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)

        fmt = f".{decimals}f"
        val_lbl = ctk.CTkLabel(
            row,
            text=f"{var.get():{fmt}}",
            font=LABEL,
            text_color=TEXT_PRIMARY,
            width=46,
            anchor="e",
        )

        def _update(_v=None):
            val_lbl.configure(text=f"{var.get():{fmt}}")

        ctk.CTkLabel(
            row, text=label, font=LABEL,
            text_color=TEXT_PRIMARY, anchor="w", width=160,
        ).pack(side="left")

        ctk.CTkSlider(
            row,
            variable=var,
            from_=lo, to=hi,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            progress_color=ACCENT,
            command=_update,
        ).pack(side="left", fill="x", expand=True, padx=8)

        val_lbl.pack(side="right")

    # ── Data helpers ─────────────────────────────────────────────────────

    def _load_current(self) -> None:
        """Pre-populate fields from current ChatManager state."""
        self._ckpt_var.set("")
        self._device_var.set(self._mgr._device or "cpu")
        self._temp_var.set(self._mgr.temperature)
        self._topk_var.set(self._mgr.top_k)
        self._topp_var.set(self._mgr.top_p)
        self._rep_var.set(self._mgr.repetition_penalty)
        self._maxt_var.set(self._mgr.max_new_tokens)

        self._sys_box.delete("0.0", "end")
        self._sys_box.insert("0.0", self._mgr.system_prompt)

    def _apply_to_manager(self) -> None:
        """Write UI values into the ChatManager."""
        self._mgr.temperature        = round(self._temp_var.get(), 2)
        self._mgr.top_k              = int(self._topk_var.get())
        self._mgr.top_p              = round(self._topp_var.get(), 2)
        self._mgr.repetition_penalty = round(self._rep_var.get(), 2)
        self._mgr.max_new_tokens     = int(self._maxt_var.get())
        self._mgr.system_prompt      = self._sys_box.get("0.0", "end").strip()
        self._mgr._device            = self._device_var.get()

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Dizel checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._ckpt_var.set(path)

    def _save_only(self) -> None:
        self._apply_to_manager()
        ckpt = self._ckpt_var.get().strip()
        if ckpt:
            self._mgr._device = self._device_var.get()
        self.destroy()

    def _save_and_reload(self) -> None:
        self._apply_to_manager()
        ckpt = self._ckpt_var.get().strip()
        if ckpt:
            # Store new checkpoint path so on_reload can pick it up
            self._mgr._pending_checkpoint = ckpt
            self._mgr._device             = self._device_var.get()
        self.destroy()
        self._on_reload()
