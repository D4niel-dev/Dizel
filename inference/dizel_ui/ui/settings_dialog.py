# dizel_ui/ui/settings_dialog.py

import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, 
                               QWidget, QLabel, QPushButton, QLineEdit, QComboBox, 
                               QTextEdit, QSlider, QCheckBox, QFileDialog, QScrollArea, QFrame, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap

from dizel_ui.utils.icons import get_icon
from dizel_ui.theme.colors import (
    BG_ROOT, BG_CHAT, ACCENT, ACCENT_HOVER, BORDER,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    BG_CARD, TAB_BG, TAB_UNSELECTED, BG_INPUT, resolve
)
from dizel_ui.theme.fonts import LOGO, BTN_LABEL, LABEL, LABEL_SM
from dizel_ui.logic.config_manager import ConfigManager
from dizel_ui.theme.stylesheets import get_button_style, get_frame_style, get_scrollbar_style
from dizel_ui.theme.theme_manager import Theme

try:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    import model.dizel_info as info
except ImportError:
    info = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_DIR = os.path.dirname(_HERE)

class SettingsDialog(QDialog):
    def __init__(self, parent, chat_mgr, on_reload):
        super().__init__(parent)
        self._mgr = chat_mgr
        self._on_reload = on_reload

        self.setWindowTitle("Settings — Dizel AI")
        self.setFixedSize(850, 700)
        self.setStyleSheet(f"QDialog {{ background-color: {resolve(BG_ROOT)}; }}")
        
        self._build()
        self._load_current()

    def _build(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 16)
        
        # Title Row
        title_row = QHBoxLayout()
        title_lbl = QLabel("  Settings")
        title_lbl.setFont(LOGO)
        title_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
        ico = get_icon("sliders", size=(20, 20), color=TEXT_PRIMARY)
        if hasattr(title_lbl, 'setPixmap') and not ico:
            pass # fallbacks
        if ico:
            # Add icon label
            ico_lbl = QLabel()
            ico_lbl.setPixmap(ico.pixmap(20, 20))
            title_row.addWidget(ico_lbl)
            
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)
        main_layout.addLayout(title_row)

        c_bg_tab = resolve(TAB_BG)
        c_unsel = resolve(TAB_UNSELECTED)
        c_card = resolve(BG_CARD)
        c_text = resolve(TEXT_SECONDARY)
        c_border = resolve(BORDER)

        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 0; background: transparent; }}
            QTabBar::tab {{
                background: transparent;
                color: {c_text};
                padding: 10px 24px;
                border-radius: 18px;
                margin: 4px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{ background: {resolve(ACCENT)}; color: #ffffff; }}
            QTabBar::tab:hover:!selected {{ background: {c_border}; color: {resolve(TEXT_PRIMARY)}; }}
        """)
        main_layout.addWidget(self.tabs)
        
        self._build_chat_tab()
        self._build_model_tab()
        self._build_app_tab()
        self._build_about_tab()

        # Buttons Row
        btn_layout = QHBoxLayout()
        back_btn = QPushButton("< Back")
        back_btn.setStyleSheet(get_button_style("transparent", BORDER, TEXT_PRIMARY, border_color=BORDER))
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self.reject)
        
        close_btn = QPushButton("Save & Close")
        close_btn.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, "#ffffff"))
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self._save_and_reload)
        
        btn_layout.addWidget(back_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(close_btn)
        main_layout.addLayout(btn_layout)

    def _create_scroll_tab(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 4, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.NoFrame)
        scroll.setStyleSheet(get_scrollbar_style("transparent", BORDER, ACCENT) + "QScrollArea { background: transparent; }")
        
        cont = QWidget()
        cont.setStyleSheet("background: transparent;")
        cont.setLayout(QVBoxLayout())
        cont.layout().setAlignment(Qt.AlignTop)
        
        scroll.setWidget(cont)
        l.addWidget(scroll)
        return w, cont.layout()

    def _section(self, layout, title):
        lbl = QLabel(title)
        lbl.setFont(LABEL)
        lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-weight: 600; margin-top: 14px; margin-bottom: 8px; letter-spacing: 0.5px;")
        layout.addWidget(lbl)

    def _build_chat_tab(self):
        tab, layout = self._create_scroll_tab()
        self.tabs.addTab(tab, "Chat")

        # Base Model
        self._section(layout, "CHECKPOINT LOADER")
        card = QFrame()
        card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(16, 16, 16, 16)
        card_l.setSpacing(12)
        
        row1 = QHBoxLayout()
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("Select checkpoint...")
        self._ckpt_edit.setStyleSheet(f"background: {resolve(BG_INPUT)}; color: {resolve(TEXT_PRIMARY)}; border: 1px solid {resolve(BORDER)}; padding: 10px 12px; border-radius: 6px;")
        
        browse_btn = QPushButton("↑ Val")
        browse_btn.setStyleSheet(get_button_style("transparent", BORDER, TEXT_SECONDARY))
        browse_btn.clicked.connect(self._browse_checkpoint)
        
        row1.addWidget(self._ckpt_edit)
        row1.addWidget(browse_btn)
        card_l.addLayout(row1)

        row2 = QHBoxLayout()
        d_lbl = QLabel("Device")
        d_lbl.setFont(LABEL)
        d_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cpu", "cuda"])
        self._device_combo.setStyleSheet(f"background: {resolve(BG_CHAT)}; color: {resolve(TEXT_PRIMARY)}; border: 1px solid {resolve(BORDER)}; padding: 8px 12px; border-radius: 6px;")
        
        row2.addWidget(d_lbl)
        row2.addStretch(1)
        row2.addWidget(self._device_combo)
        card_l.addLayout(row2)
        
        layout.addWidget(card)

        # System Card
        self._section(layout, "SYSTEM PROMPT")
        sys_card = QFrame()
        sys_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        sys_l = QVBoxLayout(sys_card)
        sys_l.setContentsMargins(16, 16, 16, 16)
        
        self._sys_box = QTextEdit()
        self._sys_box.setFixedHeight(120)
        self._sys_box.setStyleSheet(f"background: {resolve(BG_CHAT)}; color: {resolve(TEXT_PRIMARY)}; border: none; padding: 12px; border-radius: 6px; font-size: 13px;")
        sys_l.addWidget(self._sys_box)
        layout.addWidget(sys_card)

        # Sampling Card
        self._section(layout, "SAMPLING SETTINGS")
        samp_card = QFrame()
        samp_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        samp_l = QVBoxLayout(samp_card)
        samp_l.setContentsMargins(16, 16, 16, 16)
        samp_l.setSpacing(14)
        
        self._sliders = {}
        _info = self._mgr.model_info if getattr(self._mgr, "is_ready", False) else {}
        _ctx = _info.get("ctx_len", 512)
        _max_tok_ceil = max(_ctx - 50, 64)

        sliders = [
            ("Temperature", "temp", 0, 200, 70, 100),       # 0.0-2.0
            ("Top-K", "topk", 1, 200, 40, 1),               # int
            ("Top-P", "topp", 10, 100, 90, 100),            # 0.1-1.0
            ("Repe/Penalty", "rep", 100, 200, 110, 100),    # 1.0-2.0
            ("Max new tokens", "maxt", 32, _max_tok_ceil, min(400, _max_tok_ceil), 1)
        ]

        for label, key, min_v, max_v, def_v, div in sliders:
            row = QHBoxLayout()
            l1 = QLabel(label)
            l1.setFixedWidth(120)
            l1.setFont(LABEL)
            l1.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)};")
            
            sl = QSlider(Qt.Horizontal)
            sl.setRange(min_v, max_v)
            sl.setValue(def_v)
            sl.setStyleSheet(f"""
                QSlider::groove:horizontal {{
                    border-radius: 2px;
                    height: 4px;
                    background: {resolve(BORDER)};
                }}
                QSlider::handle:horizontal {{
                    background: {resolve(ACCENT)};
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                }}
            """)
            
            val_lbl = QLabel(f"{def_v/div:.2f}" if div > 1 else str(def_v))
            val_lbl.setFixedWidth(40)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setFont(LABEL)
            val_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-weight: 500;")
            
            def make_updater(lbl, d):
                return lambda v: lbl.setText(f"{v/d:.2f}" if d > 1 else str(v))
            sl.valueChanged.connect(make_updater(val_lbl, div))
            
            row.addWidget(l1)
            row.addWidget(sl)
            row.addWidget(val_lbl)
            samp_l.addLayout(row)
            
            self._sliders[key] = (sl, div)
            
        layout.addWidget(samp_card)
        layout.addStretch(1)

    def _build_model_tab(self):
        tab, layout = self._create_scroll_tab()
        self.tabs.addTab(tab, "Model")
        
        def add_row(parent_layout, label_text, val_text):
            row = QHBoxLayout()
            l1 = QLabel(label_text)
            l1.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
            l2 = QLabel(val_text)
            l2.setStyleSheet(f"color: {resolve(TEXT_DIM)};")
            row.addWidget(l1)
            row.addStretch(1)
            row.addWidget(l2)
            parent_layout.addLayout(row)
        
        self._section(layout, "MODEL ARCHITECTURE")
        card = QFrame()
        card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(16, 20, 16, 20)
        
        add_row(card_l, "Family", getattr(info, "ARCHITECTURE", "Transformer") if info else "Llama 3 (Transformer)")
        add_row(card_l, "Parameters", getattr(info, "MODEL_SIZE", "Unknown") if info else "Unknown")
        add_row(card_l, "Context Length", getattr(info, "CONTEXT_LENGTH", "Unknown") if info else "Unknown")
        add_row(card_l, "Vocab Size", getattr(info, "VOCAB_SIZE", "Unknown") if info else "Unknown")
        
        row_stats = QHBoxLayout()
        l_stat = QLabel("Layers / Heads / Dim")
        l_stat.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
        v_stat = QLabel(f"{getattr(info, 'NUM_LAYERS', '?')} / {getattr(info, 'NUM_HEADS', '?')} / {getattr(info, 'HIDDEN_DIM', '?')}" if info else "?")
        v_stat.setStyleSheet(f"color: {resolve(TEXT_DIM)};")
        row_stats.addWidget(l_stat)
        row_stats.addStretch(1)
        row_stats.addWidget(v_stat)
        card_l.addLayout(row_stats)
        
        layout.addWidget(card)
        
        self._section(layout, "TRAINING DATA")
        t_card = QFrame()
        t_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        t_l = QVBoxLayout(t_card)
        t_l.setContentsMargins(16, 20, 16, 20)
        
        add_row(t_l, "Dataset Composition", getattr(info, "TRAINING_DATA", "Unknown") if info else "Unknown")
        add_row(t_l, "Corpus Size", getattr(info, "TRAIN_TOKENS", "Unknown") if info else "Unknown")
        add_row(t_l, "Training Steps", getattr(info, "TRAIN_STEPS", "Unknown") if info else "Unknown")
        
        layout.addWidget(t_card)
        
        self._section(layout, "CAPABILITIES")
        c_card = QFrame()
        c_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        c_l = QVBoxLayout(c_card)
        c_l.setContentsMargins(16, 20, 16, 20)
        c_l.setSpacing(12)
        
        caps = getattr(info, "CAPABILITIES", []) if info else []
        if not caps:
            caps = ["Reasoning", "Structured Output"]
            
        for c in caps:
            lbl = QLabel("• " + c)
            lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; font-size: 14px;")
            c_l.addWidget(lbl)
            
        layout.addWidget(c_card)
        layout.addStretch(1)

    def _build_app_tab(self):
        tab, layout = self._create_scroll_tab()
        self.tabs.addTab(tab, "Appearance")
        
        toggle_qss = f"""
            QCheckBox {{
                color: {resolve(TEXT_PRIMARY)};
                spacing: 12px;
                font-size: 14px;
            }}
            QCheckBox::indicator {{
                width: 36px;
                height: 20px;
                border-radius: 10px;
                background-color: {resolve(BORDER)};
            }}
            QCheckBox::indicator:checked {{
                background-color: {resolve(ACCENT)};
            }}
        """

        self._section(layout, "THEME PREFERENCES")
        card = QFrame()
        card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        card_l = QVBoxLayout(card)
        card_l.setContentsMargins(16, 20, 16, 20)
        card_l.setSpacing(16)
        
        row1 = QHBoxLayout()
        ico1 = QLabel()
        pm_ico = get_icon("moon", (18,18), TEXT_DIM)
        if pm_ico: ico1.setPixmap(pm_ico.pixmap(18,18))
        row1.addWidget(ico1)
        
        l1 = QLabel("  Color Mode")
        l1.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
        row1.addWidget(l1)
        row1.addStretch(1)
        
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["System", "Light", "Dark", "Dark Blue"])
        
        combo_style = f"""
            QComboBox {{
                background: {resolve(BG_CHAT)}; 
                color: {resolve(TEXT_PRIMARY)}; 
                padding: 8px 12px; 
                border-radius: 6px;
                border: 1px solid {resolve(BORDER)};
            }}
            QComboBox::drop-down {{ border: none; width: 30px; }}
            QComboBox QAbstractItemView {{
                background-color: {resolve(BG_CHAT)};
                color: {resolve(TEXT_PRIMARY)};
                selection-background-color: {resolve(ACCENT)};
                selection-color: white;
                border: 1px solid {resolve(BORDER)};
                border-radius: 4px;
            }}
        """
        self._theme_combo.setStyleSheet(combo_style)
        
        def _theme_changed(idx):
            mode = self._theme_combo.currentText().lower().replace(" ", "_")
            Theme.set_mode(mode)
        
        self._theme_combo.currentIndexChanged.connect(_theme_changed)
        row1.addWidget(self._theme_combo)
        card_l.addLayout(row1)
        
        row2 = QHBoxLayout()
        ico2 = QLabel()
        fs_ico = get_icon("type", (18,18), TEXT_DIM)
        if fs_ico: ico2.setPixmap(fs_ico.pixmap(18,18))
        row2.addWidget(ico2)
        
        l2 = QLabel("  Font Scale")
        l2.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
        row2.addWidget(l2)
        row2.addStretch(1)
        
        self._font_combo = QComboBox()
        self._font_combo.addItems(["Small", "Medium", "Large"])
        self._font_combo.setStyleSheet(combo_style)
        row2.addWidget(self._font_combo)
        card_l.addLayout(row2)
        layout.addWidget(card)
        
        self._section(layout, "CHAT DISPLAY")
        d_card = QFrame()
        d_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        d_l = QVBoxLayout(d_card)
        d_l.setContentsMargins(16, 20, 16, 20)
        d_l.setSpacing(16)
        
        self._ts_chk = QCheckBox("Show message timestamps")
        self._ts_chk.setStyleSheet(toggle_qss)
        
        self._anim_chk = QCheckBox("Enable UI animations")
        self._anim_chk.setStyleSheet(toggle_qss)
        
        self._md_chk = QCheckBox("Syntax highlighting for code blocks")
        self._md_chk.setStyleSheet(toggle_qss)
        
        self._send_chk = QCheckBox("Press Enter to send message")
        self._send_chk.setStyleSheet(toggle_qss)
        
        d_l.addWidget(self._ts_chk)
        d_l.addWidget(self._anim_chk)
        d_l.addWidget(self._md_chk)
        d_l.addWidget(self._send_chk)
        layout.addWidget(d_card)
        
        self._section(layout, "MESSAGE BUBBLES")
        b_card = QFrame()
        b_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        b_l = QVBoxLayout(b_card)
        b_l.setContentsMargins(16, 20, 16, 20)
        
        b_row = QHBoxLayout()
        b_lbl = QLabel("Avatar Display")
        b_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
        b_row.addWidget(b_lbl)
        b_row.addStretch(1)
        
        self._avatar_combo = QComboBox()
        self._avatar_combo.addItems(["Show avatars", "Compact (names only)"])
        self._avatar_combo.setStyleSheet(combo_style)
        b_row.addWidget(self._avatar_combo)
        b_l.addLayout(b_row)
        layout.addWidget(b_card)
        
        layout.addStretch(1)

    def _build_about_tab(self):
        tab, layout = self._create_scroll_tab()
        self.tabs.addTab(tab, "About")
        
        # Logo & App Info Section
        header = QFrame()
        header.setStyleSheet("background: transparent; border: none;")
        h_layout = QVBoxLayout(header)
        h_layout.setAlignment(Qt.AlignCenter)
        h_layout.setSpacing(12)
        h_layout.setContentsMargins(0, 24, 0, 16)
        
        logo_path = os.path.join(_INFERENCE_DIR, "assets", "app", "Dizel.png")
        if os.path.exists(logo_path):
            lbl = QLabel()
            lbl.setPixmap(QPixmap(logo_path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            h_layout.addWidget(lbl, alignment=Qt.AlignCenter)
            
        t1 = QLabel(getattr(info, "MODEL_NAME", "Dizel AI") + " CLI" if info else "Dizel AI")
        t1.setFont(LOGO)
        t1.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 28px;")
        h_layout.addWidget(t1, alignment=Qt.AlignCenter)
        
        v_str = getattr(info, 'VERSION', '1.0.0-beta')
        b_str = getattr(info, 'BUILD', 'YYYY-MM-DD')
        t2 = QLabel(f"Version {v_str} • Build {b_str}")
        t2.setStyleSheet(f"color: {resolve(TEXT_SECONDARY)}; font-size: 14px; font-weight: 500;")
        h_layout.addWidget(t2, alignment=Qt.AlignCenter)
        
        layout.addWidget(header)

        # Buttons (Updates, Debug info)
        updates_row = QHBoxLayout()
        updates_row.setAlignment(Qt.AlignCenter)
        btn_update = QPushButton(" Check for Updates")
        u_ico = get_icon("refresh-cw", (16,16), TEXT_PRIMARY)
        if u_ico: btn_update.setIcon(u_ico)
        btn_update.setFixedHeight(36)
        btn_update.setFont(BTN_LABEL)
        btn_update.setStyleSheet(get_button_style(ACCENT, ACCENT_HOVER, TEXT_PRIMARY, radius=8))
        btn_update.setCursor(Qt.PointingHandCursor)
        updates_row.addWidget(btn_update)
        layout.addLayout(updates_row)
        layout.addSpacing(24)

        self._section(layout, "LINKS & COMMUNITY")
        links_card = QFrame()
        links_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        lc_lyt = QVBoxLayout(links_card)
        lc_lyt.setContentsMargins(16, 20, 16, 20)
        
        def add_link_row(text, icon_name, link):
            row = QHBoxLayout()
            ico_lbl = QLabel()
            ico = get_icon(icon_name, (18,18), TEXT_DIM)
            if ico: ico_lbl.setPixmap(ico.pixmap(18,18))
            row.addWidget(ico_lbl)
            
            lbl = QLabel("  " + text)
            lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
            row.addWidget(lbl)
            row.addStretch(1)
            
            link_lbl = QLabel(link)
            link_lbl.setStyleSheet(f"color: {resolve(TEXT_DIM)}; font-size: 13px;")
            row.addWidget(link_lbl)
            lc_lyt.addLayout(row)

        add_link_row("Official Website", "globe", "https://dizel.ai")
        lc_lyt.addSpacing(16)
        add_link_row("GitHub Repository", "github", getattr(info, 'REPOSITORY', 'github.com/d4niel-dev/dizel'))
        lc_lyt.addSpacing(16)
        add_link_row("Discord Community", "discord", "discord.gg/dizel")
        layout.addWidget(links_card)

        self._section(layout, "DATA & PRIVACY")
        data_card = QFrame()
        data_card.setStyleSheet(get_frame_style(BG_CARD, radius=12))
        dc_lyt = QVBoxLayout(data_card)
        dc_lyt.setContentsMargins(16, 20, 16, 20)
        
        dc_row1 = QHBoxLayout()
        d_lbl = QLabel("Clear Local Data")
        d_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
        dc_row1.addWidget(d_lbl)
        dc_row1.addStretch(1)
        clr_btn = QPushButton("Clear Cache")
        clr_btn.setStyleSheet(get_button_style("transparent", BORDER, "#FF4D4D", border_color=BORDER))
        clr_btn.setFixedHeight(32)
        clr_btn.setCursor(Qt.PointingHandCursor)
        dc_row1.addWidget(clr_btn)
        dc_lyt.addLayout(dc_row1)
        
        dc_lyt.addSpacing(16)
        
        dc_row2 = QHBoxLayout()
        ack_lbl = QLabel("Acknowledgments")
        ack_lbl.setStyleSheet(f"color: {resolve(TEXT_PRIMARY)}; font-size: 14px;")
        dc_row2.addWidget(ack_lbl)
        dc_row2.addStretch(1)
        ack_val = QLabel("PySide6 • PyTorch • HuggingFace")
        ack_val.setStyleSheet(f"color: {resolve(TEXT_DIM)}; font-size: 13px;")
        dc_row2.addWidget(ack_val)
        dc_lyt.addLayout(dc_row2)
        
        layout.addWidget(data_card)
        layout.addStretch(1)

    def _load_current(self):
        cfg = ConfigManager.load()
        self._ckpt_edit.setText(cfg.get("checkpoint", ""))
        
        # Load from chat_mgr if available
        mgr_dev = getattr(self._mgr, "_device", cfg.get("device", "cpu"))
        if mgr_dev == "cuda": self._device_combo.setCurrentIndex(1)
        else: self._device_combo.setCurrentIndex(0)
        
        self._sys_box.setPlainText(getattr(self._mgr, "system_prompt", cfg.get("system_prompt", "")))
        
        # Load sliders
        def set_val(key, val):
            sl, div = self._sliders[key]
            sl.setValue(int(val * div))
            
        set_val("temp", getattr(self._mgr, "temperature", 0.7))
        set_val("topk", getattr(self._mgr, "top_k", 40))
        set_val("topp", getattr(self._mgr, "top_p", 0.90))
        set_val("rep", getattr(self._mgr, "repetition_penalty", 1.10))
        
        maxt = getattr(self._mgr, "max_new_tokens", 400)
        maxt_sl, maxt_div = self._sliders["maxt"]
        maxt_sl.setValue(min(maxt, maxt_sl.maximum()))
        
        app_cfg = cfg.get("appearance", {})
        cm = app_cfg.get("color_mode", "system").lower().replace(" ", "_")
        if cm == "system":
            self._theme_combo.setCurrentIndex(0)
        elif cm == "light":
            self._theme_combo.setCurrentIndex(1)
        elif cm == "dark_blue":
            self._theme_combo.setCurrentIndex(3)
        else:
            self._theme_combo.setCurrentIndex(2)
            
        self._ts_chk.setChecked(app_cfg.get("show_timestamps", True))
        self._anim_chk.setChecked(app_cfg.get("animations", True))
        self._md_chk.setChecked(app_cfg.get("syntax_highlighting", True))
        self._send_chk.setChecked(app_cfg.get("enter_to_send", True))
        
        font_idx = self._font_combo.findText(app_cfg.get("font_scale", "Medium"))
        if font_idx >= 0: self._font_combo.setCurrentIndex(font_idx)
        
        av_idx = self._avatar_combo.findText(app_cfg.get("avatar_display", "Show avatars"))
        if av_idx >= 0: self._avatar_combo.setCurrentIndex(av_idx)

    def _apply_to_manager(self):
        # Update settings to memory object if needed
        self._mgr.temperature = self._sliders["temp"][0].value() / self._sliders["temp"][1]
        self._mgr.top_k = self._sliders["topk"][0].value()
        self._mgr.top_p = self._sliders["topp"][0].value() / self._sliders["topp"][1]
        self._mgr.repetition_penalty = self._sliders["rep"][0].value() / self._sliders["rep"][1]
        self._mgr.max_new_tokens = self._sliders["maxt"][0].value()
        self._mgr.system_prompt = self._sys_box.toPlainText().strip()
        self._mgr._device = self._device_combo.currentText()
        
        cfg = ConfigManager.load()
        cfg["device"] = self._device_combo.currentText()
        ckpt = self._ckpt_edit.text().strip()
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
            "color_mode": self._theme_combo.currentText(),
            "show_timestamps": self._ts_chk.isChecked(),
            "animations": self._anim_chk.isChecked(),
            "syntax_highlighting": self._md_chk.isChecked(),
            "enter_to_send": self._send_chk.isChecked(),
            "font_scale": self._font_combo.currentText(),
            "avatar_display": self._avatar_combo.currentText(),
        }
        ConfigManager.save(cfg)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Dizel checkpoint", "", "PyTorch checkpoint (*.pt);;All files (*.*)")
        if path:
            self._ckpt_edit.setText(path)

    def _save_and_reload(self):
        self._apply_to_manager()
        ckpt = self._ckpt_edit.text().strip()
        if ckpt:
            self._mgr._pending_checkpoint = ckpt
            self._mgr._device = self._device_combo.currentText()
        self.accept()
        self._on_reload()
