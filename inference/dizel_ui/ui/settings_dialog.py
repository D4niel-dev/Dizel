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

# ── Make project root importable ──────────────────────────────────────────────
import os
import sys

_HERE          = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_DIR = os.path.dirname(_HERE)
_PROJ_ROOT     = os.path.dirname(_INFERENCE_DIR)

if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)


import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from typing import Callable

from ..theme.colors import (
    BG_ROOT, BG_CHAT, ACCENT, ACCENT_HOVER, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    BG_CARD, TAB_BG, TAB_UNSELECTED, BG_INPUT
)
from ..theme.fonts import LOGO, BTN_LABEL, LABEL, LABEL_SM
from ..logic.config_manager import ConfigManager


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
        self.update()
        self.grab_set()          # modal behaviour
        self.lift()
        self.focus_force()

        self._build()
        self._load_current()

        # Set Window Icon (deferred — CTkToplevel needs to be mapped first)
        self.after(200, self._set_icon)

    def _set_icon(self) -> None:
        try:
            ico_path = os.path.join(_INFERENCE_DIR, "assets", "app", "Dizel.ico")
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
                self.after(10, lambda: self.iconbitmap(ico_path))  # re-apply after CTk override
            else:
                png_path = os.path.join(_INFERENCE_DIR, "assets", "app", "Dizel.png")
                if os.path.exists(png_path):
                    img = tk.PhotoImage(file=png_path)
                    self.iconphoto(False, img)
        except Exception as e:
            print(f"Failed to load settings window icon: {e}")

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        from dizel_ui.utils.icons import get_icon

        # ── Title Row ──
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.pack(fill="x", padx=24, pady=(20, 10))
        
        ico = get_icon("sliders", size=(20, 20), color=TEXT_PRIMARY)
        ctk.CTkLabel(
            title_row, text="  Settings", image=ico, compound="left",
            font=LOGO, text_color=TEXT_PRIMARY,
        ).pack(side="left")

        # ── Tabview ──
        self.tabs = ctk.CTkTabview(
            self, fg_color=TAB_BG,
            segmented_button_fg_color=TAB_UNSELECTED,
            segmented_button_selected_color=BG_CARD,
            segmented_button_unselected_color=TAB_UNSELECTED,
            text_color=TEXT_SECONDARY,
            segmented_button_selected_hover_color=BG_CARD,
            segmented_button_unselected_hover_color=BORDER
        )
        self.tabs.pack(fill="both", expand=True, padx=24, pady=(0, 10))
        
        t_chat  = self.tabs.add("Chat")
        t_model = self.tabs.add("Model")
        t_app   = self.tabs.add("Appearance")
        t_abt   = self.tabs.add("About")

        scroll = ctk.CTkScrollableFrame(
            t_chat, fg_color="transparent",
            scrollbar_button_color=BORDER,
        )
        scroll.pack(fill="both", expand=True, pady=4)

        # ── Base Model Card ──
        self._section(scroll, "Checkpoint Loader")
        model_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        model_card.pack(fill="x", pady=(0, 12))

        ckpt_row = ctk.CTkFrame(model_card, fg_color="transparent")
        ckpt_row.pack(fill="x", padx=12, pady=(12, 6))

        self._ckpt_var = tk.StringVar()
        ckpt_box = ctk.CTkFrame(ckpt_row, fg_color=BG_INPUT, corner_radius=6, border_color=BORDER, border_width=1)
        ckpt_box.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        hard_ico = get_icon("hard-drive", size=(14, 14), color=TEXT_PRIMARY)
        ctk.CTkLabel(ckpt_box, text="", image=hard_ico).pack(side="left", padx=(10, 0))
        
        ckpt_entry = ctk.CTkEntry(
            ckpt_box, textvariable=self._ckpt_var, placeholder_text="Select checkpoint...",
            font=LABEL, fg_color="transparent", border_width=0, text_color=TEXT_PRIMARY,
        )
        ckpt_entry.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        
        val_btn = ctk.CTkButton(
            ckpt_box, text="↑ Val", width=50, height=24, fg_color="transparent",
            hover_color=BORDER, font=LABEL_SM, text_color=TEXT_SECONDARY, command=self._browse_checkpoint
        )
        val_btn.pack(side="right", padx=4)

        device_row = ctk.CTkFrame(model_card, fg_color="transparent")
        device_row.pack(fill="x", padx=16, pady=(6, 12))
        ctk.CTkLabel(device_row, text="Device", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        
        self._device_var = ctk.StringVar(value="cpu")
        dev_opt = ctk.CTkOptionMenu(
            device_row, variable=self._device_var, values=["cpu", "cuda"],
            font=LABEL, fg_color=BG_CHAT, button_color=BG_CHAT,
            button_hover_color=BORDER, text_color=TEXT_PRIMARY, width=80, anchor="e"
        )
        dev_opt.pack(side="right")

        # ── System Card ──
        sys_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        sys_card.pack(fill="x", pady=(0, 12))
        
        sys_hdr = ctk.CTkFrame(sys_card, fg_color="transparent")
        sys_hdr.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(sys_hdr, text="System Prompt", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        
        chv_ico = get_icon("chevron-down", size=(14, 14), color=TEXT_DIM)
        ctk.CTkLabel(sys_hdr, text="", image=chv_ico).pack(side="right")

        self._sys_box = ctk.CTkTextbox(
            sys_card, height=120, font=LABEL, fg_color=BG_CHAT,
            text_color=TEXT_PRIMARY, border_width=0, corner_radius=6, wrap="word",
        )
        self._sys_box.pack(fill="x", padx=12, pady=(0, 8))

        seed_row = ctk.CTkFrame(sys_card, fg_color="transparent")
        seed_row.pack(fill="x", padx=16, pady=(4, 12))
        ctk.CTkLabel(seed_row, text="Random Seed", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(seed_row, text="Auto >", font=LABEL, text_color=TEXT_DIM).pack(side="right")

        # ── Sampling Card ──
        self._section(scroll, "Sampling Settings")
        samp_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        samp_card.pack(fill="x", pady=(0, 12))

        self._temp_var   = tk.DoubleVar(value=0.7)
        self._topk_var   = tk.IntVar(value=40)
        self._topp_var   = tk.DoubleVar(value=0.90)
        self._rep_var    = tk.DoubleVar(value=1.10)
        self._maxt_var   = tk.IntVar(value=400)

        # Derive max-tokens ceiling from model context length (dynamic)
        _info = self._mgr.model_info if self._mgr.is_ready else {}
        _ctx  = _info.get("ctx_len", 512)
        _max_tok_ceil = max(_ctx - 50, 64)   # always leave 50 tokens for the prompt

        sliders = [
            ("Temperature",        self._temp_var,  0.0,   2.0,           2),
            ("Top-K",              self._topk_var,  1,     200,           0),
            ("Top-P",              self._topp_var,  0.1,   1.0,           2),
            ("Repetition penalty", self._rep_var,   1.0,   2.0,           2),
            ("Max new tokens",     self._maxt_var,  32,    _max_tok_ceil, 0),
        ]
        
        slider_pad = ctk.CTkFrame(samp_card, fg_color="transparent")
        slider_pad.pack(fill="x", padx=16, pady=16)
        
        for label, var, lo, hi, decimals in sliders:
            self._make_slider(slider_pad, label, var, lo, hi, decimals)

        # ── Footer Stats ──
        ctk.CTkLabel(
            scroll, text="Context: 512 / 512 tokens",
            font=LABEL_SM, text_color=TEXT_DIM, anchor="w"
        ).pack(anchor="w", padx=4, pady=(8, 4))
        
        stat_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        stat_card.pack(fill="x", pady=(0, 10))
        
        s1 = ctk.CTkFrame(stat_card, fg_color="transparent")
        s1.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(s1, text="●  Tokens/sec", font=LABEL_SM, text_color=TEXT_SECONDARY).pack(side="left")
        ctk.CTkLabel(s1, text="314 ms  〰", font=LABEL_SM, text_color=TEXT_DIM).pack(side="right")
        
        s2 = ctk.CTkFrame(stat_card, fg_color="transparent")
        s2.pack(fill="x", padx=12, pady=(2, 10))
        ctk.CTkLabel(s2, text="●  Memory usage", font=LABEL_SM, text_color=TEXT_SECONDARY).pack(side="left")
        ctk.CTkLabel(s2, text="1.4 GB RAM", font=LABEL_SM, text_color=TEXT_DIM).pack(side="right")

        # ── Model Tab ──
        m_scroll = ctk.CTkScrollableFrame(t_model, fg_color="transparent", scrollbar_button_color=BORDER)
        m_scroll.pack(fill="both", expand=True, pady=4)
        
        self._section(m_scroll, "Model Architecture")
        arch_card = ctk.CTkFrame(m_scroll, fg_color=BG_CARD, corner_radius=12)
        arch_card.pack(fill="x", pady=(0, 12))
        
        a1 = ctk.CTkFrame(arch_card, fg_color="transparent")
        a1.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(a1, text="Family", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(a1, text="Llama 3 (Transformer)", font=LABEL, text_color=TEXT_DIM).pack(side="right")
        
        a2 = ctk.CTkFrame(arch_card, fg_color="transparent")
        a2.pack(fill="x", padx=16, pady=(4, 12))
        ctk.CTkLabel(a2, text="Parameters", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(a2, text="8B", font=LABEL, text_color=TEXT_DIM).pack(side="right")
        
        self._section(m_scroll, "Execution")
        exec_card = ctk.CTkFrame(m_scroll, fg_color=BG_CARD, corner_radius=12)
        exec_card.pack(fill="x", pady=(0, 12))
        
        o1 = ctk.CTkFrame(exec_card, fg_color="transparent")
        o1.pack(fill="x", padx=16, pady=(12, 4))
        ctk.CTkLabel(o1, text="GPU Offload Layers", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        
        self._offload_var = ctk.IntVar(value=32)
        ctk.CTkSlider(exec_card, variable=self._offload_var, from_=0, to=40, button_color=TEXT_PRIMARY, button_hover_color="#ffffff", progress_color=ACCENT, height=12).pack(fill="x", padx=16, pady=4)
        
        o2 = ctk.CTkFrame(exec_card, fg_color="transparent")
        o2.pack(fill="x", padx=16, pady=(4, 12))
        ctk.CTkSwitch(o2, text="Auto-detect max VRAM", font=LABEL, text_color=TEXT_PRIMARY, progress_color=ACCENT, button_color=TEXT_PRIMARY).pack(side="left")

        # ── Appearance Tab ──
        a_scroll = ctk.CTkScrollableFrame(t_app, fg_color="transparent", scrollbar_button_color=BORDER)
        a_scroll.pack(fill="both", expand=True, pady=4)
        
        self._section(a_scroll, "Theme Preferences")
        theme_card = ctk.CTkFrame(a_scroll, fg_color=BG_CARD, corner_radius=12)
        theme_card.pack(fill="x", pady=(0, 12))
        
        t1 = ctk.CTkFrame(theme_card, fg_color="transparent")
        t1.pack(fill="x", padx=16, pady=(12, 8))
        ctk.CTkLabel(t1, text="Color Mode", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        self._mode_var = ctk.StringVar(value="Dark")
        ctk.CTkSegmentedButton(t1, variable=self._mode_var, values=["System", "Light", "Dark"], selected_color=ACCENT, selected_hover_color=ACCENT_HOVER, unselected_color=BG_CHAT, unselected_hover_color=BORDER).pack(side="right")
        
        t2 = ctk.CTkFrame(theme_card, fg_color="transparent")
        t2.pack(fill="x", padx=16, pady=(8, 12))
        ctk.CTkLabel(t2, text="UI Scale", font=LABEL, text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(t2, text="100%", font=LABEL, text_color=TEXT_DIM).pack(side="right")
        
        self._section(a_scroll, "Chat Display")
        disp_card = ctk.CTkFrame(a_scroll, fg_color=BG_CARD, corner_radius=12)
        disp_card.pack(fill="x", pady=(0, 12))
        
        d1 = ctk.CTkFrame(disp_card, fg_color="transparent")
        d1.pack(fill="x", padx=16, pady=(12, 8))
        self._ts_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(d1, text="Show message timestamps", variable=self._ts_var, font=LABEL, text_color=TEXT_PRIMARY, progress_color=ACCENT).pack(side="left")

        d2 = ctk.CTkFrame(disp_card, fg_color="transparent")
        d2.pack(fill="x", padx=16, pady=(4, 12))
        self._anim_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(d2, text="Enable UI animations", variable=self._anim_var, font=LABEL, text_color=TEXT_PRIMARY, progress_color=ACCENT).pack(side="left")

        # ── About Tab ──
        v_card = ctk.CTkFrame(t_abt, fg_color="transparent")
        v_card.pack(fill="both", expand=True, pady=30)
        
        # Load app logo for About tab
        from PIL import Image
        import os
        logo_path = os.path.join(_INFERENCE_DIR, "assets", "app", "Dizel.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(_INFERENCE_DIR, "assets", "app", "Dizel.ico")
            
        try:
            pil_img = Image.open(logo_path)
            logo_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(64, 64))
            ctk.CTkLabel(v_card, text="", image=logo_img).pack(pady=(0, 16))
        except Exception:
            # Fallback if image missing/corrupt
            logo_ico = get_icon("box", size=(48, 48), color=ACCENT)
            ctk.CTkLabel(v_card, text="", image=logo_ico).pack(pady=(0, 16))
        
        ctk.CTkLabel(v_card, text="Dizel AI", font=LOGO, text_color=TEXT_PRIMARY).pack(pady=4)
        ctk.CTkLabel(v_card, text="Version 1.0.0-beta", font=LABEL, text_color=TEXT_SECONDARY).pack()
        
        ctk.CTkLabel(v_card, text="A lightweight, localized LLM desktop interface.\nPowered by CustomTkinter.", font=LABEL, text_color=TEXT_DIM, justify="center").pack(pady=24)
        
        def open_url(url="https://github.com/D4niel-dev/Dizel"): import webbrowser; webbrowser.open(url)

        links_row = ctk.CTkFrame(v_card, fg_color="transparent")
        links_row.pack(pady=8)
        
        ctk.CTkButton(links_row, text="GitHub Repo", fg_color=BG_CARD, border_width=1, border_color=BORDER, text_color=TEXT_PRIMARY, hover_color=BORDER, width=120, command=lambda: open_url("https://github.com/D4niel-dev/Dizel")).pack(side="left", padx=6)
        ctk.CTkButton(links_row, text="Documentation", fg_color=BG_CARD, border_width=1, border_color=BORDER, text_color=TEXT_PRIMARY, hover_color=BORDER, width=120, command=lambda: open_url("https://github.com/D4niel-dev/Dizel/blob/main/README.md")).pack(side="left", padx=6)
        
        ctk.CTkButton(v_card, text="Report an Issue", fg_color="transparent", text_color=ACCENT, hover_color=BG_CARD, width=200, command=lambda: open_url("https://github.com/D4niel-dev/Dizel/issues")).pack(pady=8)

        # ── Buttons ──
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=24, pady=16)

        ctk.CTkButton(
            btn_row, text="< Back", font=BTN_LABEL, width=80, fg_color="transparent",
            border_width=1, border_color=BORDER, text_color=TEXT_PRIMARY, corner_radius=16,
            hover_color=BORDER, command=self.destroy,
        ).pack(side="left")

        ctk.CTkButton(
            btn_row, text="Close", font=BTN_LABEL, width=100, fg_color=ACCENT,
            text_color="#ffffff", hover_color=ACCENT_HOVER, corner_radius=16,
            command=self._save_and_reload,
        ).pack(side="right")

    def _section(self, parent, title: str) -> None:
        ctk.CTkLabel(
            parent, text=title, font=LABEL, text_color=TEXT_PRIMARY, anchor="w",
        ).pack(anchor="w", padx=4, pady=(10, 6))

    def _make_slider(self, parent, label: str, var, lo, hi, decimals: int) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=6)

        fmt = f".{decimals}f"
        val_lbl = ctk.CTkLabel(
            row, text=f"{var.get():{fmt}}", font=LABEL, text_color=TEXT_PRIMARY, width=40, anchor="e",
        )

        def _update(*args):
            try:
                val_lbl.configure(text=f"{var.get():{fmt}}")
            except Exception:
                pass

        var.trace_add("write", _update)

        ctk.CTkLabel(
            row, text=label, font=LABEL, text_color=TEXT_PRIMARY, anchor="w", width=140,
        ).pack(side="left")

        ctk.CTkSlider(
            row, variable=var, from_=lo, to=hi, button_color=TEXT_PRIMARY,
            button_hover_color="#ffffff", progress_color=ACCENT, command=lambda _: _update(),
            height=12,
        ).pack(side="left", fill="x", expand=True, padx=16)

        val_lbl.pack(side="right")

    # ── Data helpers ─────────────────────────────────────────────────────

    def _load_current(self) -> None:
        """Pre-populate fields from current ChatManager state & persistent config."""
        cfg = ConfigManager.load()
        
        # Checkpoint is an app-level thing, so we load it from config
        self._ckpt_var.set(cfg.get("checkpoint", ""))
        self._device_var.set(self._mgr._device or cfg.get("device", "cpu"))
        
        self._temp_var.set(self._mgr.temperature)
        self._topk_var.set(self._mgr.top_k)
        self._topp_var.set(self._mgr.top_p)
        self._rep_var.set(self._mgr.repetition_penalty)

        # Clamp max_new_tokens to the model-derived ceiling
        _info = self._mgr.model_info if self._mgr.is_ready else {}
        _ctx  = _info.get("ctx_len", 512)
        _ceil = max(_ctx - 50, 64)
        self._maxt_var.set(min(self._mgr.max_new_tokens, _ceil))

        self._sys_box.delete("0.0", "end")
        self._sys_box.insert("0.0", self._mgr.system_prompt)

        # Load appearance defaults
        app_cfg = cfg.get("appearance", {})
        self._mode_var.set(app_cfg.get("color_mode", "Dark"))
        self._ts_var.set(app_cfg.get("show_timestamps", True))
        self._anim_var.set(app_cfg.get("animations", True))

    def _apply_to_manager(self) -> None:
        """Write UI values into the ChatManager and persist to disk via ConfigManager."""
        self._mgr.temperature        = round(self._temp_var.get(), 2)
        self._mgr.top_k              = int(self._topk_var.get())
        self._mgr.top_p              = round(self._topp_var.get(), 2)
        self._mgr.repetition_penalty = round(self._rep_var.get(), 2)
        self._mgr.max_new_tokens     = int(self._maxt_var.get())
        self._mgr.system_prompt      = self._sys_box.get("0.0", "end").strip()
        self._mgr._device            = self._device_var.get()
        
        # Persist to disk
        cfg = ConfigManager.load()
        cfg["device"] = self._device_var.get()
        
        ckpt = self._ckpt_var.get().strip()
        if ckpt:
            cfg["checkpoint"] = ckpt
            
        cfg["system_prompt"] = self._mgr.system_prompt
        cfg["sampling"] = {
            "temperature": self._mgr.temperature,
            "top_k": self._mgr.top_k,
            "top_p": self._mgr.top_p,
            "repetition_penalty": self._mgr.repetition_penalty,
            "max_new_tokens": self._mgr.max_new_tokens,
        }
        cfg["appearance"] = {
            "color_mode": self._mode_var.get(),
            "show_timestamps": self._ts_var.get(),
            "animations": self._anim_var.get(),
        }
        ConfigManager.save(cfg)
        
        # Apply appearance mode instantly
        ctk.set_appearance_mode(self._mode_var.get())

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
