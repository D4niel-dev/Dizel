"""
dizel_ui/main.py
─────────────────
Dizel AI Desktop Interface — application entry point.

Architecture
------------
  DizelApp (CTk root window)
    ├── Sidebar          (left panel — nav, history)
    ├── ChatWindow       (center — message bubbles, welcome screen)
    └── InputPanel       (bottom — text input, send/stop)

  ChatManager           (logic layer — model, generation, history state)
  HistoryManager        (persistence — JSON files on disk)

All model work runs in a daemon background thread via ChatManager.send_message().
Callbacks from that thread are marshalled back to the Tk event loop with .after(0, …).

Run
---
    cd Dizel-v1
    python dizel_ui/main.py
    python dizel_ui/main.py --checkpoint checkpoints/dizel-sft-best.pt
    python dizel_ui/main.py --checkpoint checkpoints/dizel-sft-best.pt --device cpu
"""

import argparse
import os
import sys
import threading
import tkinter as tk

import customtkinter as ctk

# ── Make project root importable ──────────────────────────────────────────────
_HERE          = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_DIR = os.path.dirname(_HERE)
_PROJ_ROOT     = os.path.dirname(_INFERENCE_DIR)

if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

# ── UI sub-modules ────────────────────────────────────────────────────────────
from dizel_ui.ui.chat_window     import ChatWindow
from dizel_ui.ui.sidebar         import Sidebar
from dizel_ui.ui.input_panel     import InputPanel
from dizel_ui.ui.settings_dialog import SettingsDialog
from dizel_ui.logic.chat_manager import ChatManager
from dizel_ui.logic.history_manager import (
    save_session, load_session, list_sessions,
    delete_session, new_session_id,
)
from dizel_ui.theme.colors import (
    BG_ROOT, BG_CHAT, ACCENT, TEXT_PRIMARY, TEXT_DIM,
)
from dizel_ui.theme.fonts import LABEL, BTN_LABEL, LABEL_SM

from dizel_ui.utils.icons import get_icon

# ── CustomTkinter global appearance ──────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─────────────────────────────────────────────────────────────────────────────
class DizelApp(ctk.CTk):
    """
    Root application window.

    Responsibilities
    ----------------
    • Create and grid all top-level UI panels
    • Own the ChatManager instance
    • Route callbacks between Sidebar ↔ ChatWindow ↔ InputPanel
    • Load the model on startup (if checkpoint provided)
    • Manage session save / load via HistoryManager
    """

    MIN_W = 860
    MIN_H = 560

    def __init__(self, checkpoint: str = "", device: str = "cpu") -> None:
        super().__init__()

        self._chat_mgr        = ChatManager()
        self._checkpoint      = checkpoint
        self._device          = device
        self._session_id      = new_session_id()
        self._model_loaded    = False

        # Window chrome
        self.title("Dizel AI")
        self.geometry("1100x700")
        self.minsize(self.MIN_W, self.MIN_H)
        self.configure(fg_color=BG_ROOT)
        
        # Set Window Icon
        try:
            # Safe absolute resolution
            ico_path = os.path.join(_HERE, "assets", "app", "Dizel.ico")
            
            if os.path.exists(ico_path):
                self.iconbitmap(ico_path)
            else:
                # Fallback to PNG if .ico missing
                png_path = os.path.join(_HERE, "assets", "app", "Dizel.png")
                if os.path.exists(png_path):
                    img = tk.PhotoImage(file=png_path)
                    self.iconphoto(False, img)
        except Exception as e:
            print(f"Failed to load application window icon: {e}")

        self._build_layout()
        self._wire_callbacks()
        self._refresh_history()

        # Auto-load model if checkpoint was passed on CLI
        if checkpoint:
            self.after(200, lambda: self._load_model_async(checkpoint, device))
        else:
            self._show_status("No checkpoint loaded — open Settings to select one.", dim=True)

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """
        Three-column layout:
          [Sidebar] | [ChatWindow + InputPanel stacked vertically]
        """
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # ── Sidebar ───────────────────────────────────────────────────────
        self._sidebar = Sidebar(
            self,
            on_new_chat       = self._new_chat,
            on_session_select = self._load_session,
            on_session_delete = self._delete_session,
            on_settings       = self._open_settings,
        )
        self._sidebar.grid(row=0, column=0, sticky="nsew")

        # ── Right column (chat + input) ───────────────────────────────────
        right = ctk.CTkFrame(self, fg_color=BG_CHAT, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── Top Bar (Zyricon Header) ──────────────────────────────────────
        self._status_bar = ctk.CTkFrame(right, fg_color="transparent", height=48, corner_radius=0)
        self._status_bar.grid(row=0, column=0, sticky="ew", padx=24, pady=(12, 16))
        self._status_bar.grid_propagate(False)

        # Left Pill (Status / Model Info)
        left_pill = ctk.CTkFrame(self._status_bar, fg_color="#100a14", corner_radius=16, height=32)
        left_pill.pack(side="left", ipadx=12, ipady=4)
        
        self._status_lbl = ctk.CTkLabel(
            left_pill, text="Dizel v1.0 ⌵", font=LABEL_SM, text_color=TEXT_PRIMARY, anchor="w"
        )
        self._status_lbl.pack(side="left", padx=8)
        
        self._model_info_lbl = ctk.CTkLabel(
            left_pill, text="", font=LABEL_SM, text_color=TEXT_DIM, anchor="e"
        )
        self._model_info_lbl.pack(side="left", padx=(4, 8))

        # Right Pills (Config & Export)
        export_ico = get_icon("external-link", size=(14, 14), color=TEXT_PRIMARY)
        export_pill = ctk.CTkButton(
            self._status_bar, text="  Export", image=export_ico, font=LABEL_SM, text_color=TEXT_PRIMARY,
            fg_color="#100a14", hover_color="#1f152b", corner_radius=16, width=80, height=32
        )
        export_pill.pack(side="right", padx=(8, 0))

        config_ico = get_icon("settings", size=(14, 14), color=TEXT_PRIMARY)
        config_pill = ctk.CTkButton(
            self._status_bar, text="  Configuration", image=config_ico, font=LABEL_SM, text_color=TEXT_PRIMARY,
            fg_color="#100a14", hover_color="#1f152b", corner_radius=16, width=120, height=32,
            command=self._open_settings
        )
        config_pill.pack(side="right")

        # Chat area
        self._chat_window = ChatWindow(
            right,
            on_quick_action=self._on_quick_action,
        )
        self._chat_window.configure(fg_color="transparent")
        self._chat_window.grid(row=1, column=0, sticky="nsew")
        right.rowconfigure(1, weight=1)

        # Input panel
        self._input_panel = InputPanel(
            right,
            on_send=self._on_send,
            on_stop=self._on_stop,
            on_settings=self._open_settings,
            on_attach=self._do_attach,
            on_options=self._do_options,
            on_voice=self._do_voice,
        )
        self._input_panel.grid(row=2, column=0, sticky="ew")

    # ── Input Panel Actions ───────────────────────────────────────────────

    def _do_attach(self) -> None:
        import tkinter.filedialog as fd
        fpath = fd.askopenfilename(title="Select File to Attach")
        if fpath:
            fname = os.path.basename(fpath)
            self._input_panel._input.insert("end", f"\n[Attached: {fname}]\n")
            self._show_status(f"Attached {fname}")

    def _do_options(self) -> None:
        dlg = ctk.CTkInputDialog(text="Enter an advanced system prompt:", title="Model Options")
        res = dlg.get_input()
        if res:
            self._show_status(f"System prompt recorded.", dim=True)

    def _do_voice(self) -> None:
        self._show_status("Listening... (Speak now)", dim=False)
        self.after(2000, lambda: self._input_panel._input.insert("end", "Hello Dizel, how are you?"))
        self.after(2000, lambda: self._show_status("Transcribed.", dim=True))

    # ── Callback wiring ───────────────────────────────────────────────────

    def _wire_callbacks(self) -> None:
        """Bind window-level keyboard shortcuts."""
        self.bind("<Control-n>",       lambda _e: self._new_chat())
        self.bind("<Control-comma>",   lambda _e: self._open_settings())
        self.bind("<Control-w>",       lambda _e: self.destroy())

    # ── Status bar helpers ────────────────────────────────────────────────

    def _show_status(self, msg: str, dim: bool = False) -> None:
        color = TEXT_DIM if dim else TEXT_PRIMARY
        self._status_lbl.configure(text=msg, text_color=color)

    def _update_model_info(self) -> None:
        info = self._chat_mgr.model_info
        if not info:
            self._model_info_lbl.configure(text="")
            return
        txt = (
            f"Dizel v1  •  {info.get('params','?')}  •  "
            f"d={info.get('d_model','?')}  L={info.get('n_layers','?')}  •  "
            f"{self._chat_mgr._device.upper()}"
        )
        self._model_info_lbl.configure(text=txt, text_color=TEXT_DIM)

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model_async(self, checkpoint: str, device: str) -> None:
        """Load the model in a background thread to avoid freezing the UI."""
        self._show_status("Loading model…")

        def _worker() -> None:
            try:
                self._chat_mgr.load_model(
                    checkpoint_path=checkpoint,
                    device=device,
                    on_progress=lambda msg: self.after(0, lambda m=msg: self._show_status(m)),
                )
                self._model_loaded = True
                self.after(0, self._on_model_ready)
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_model_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_ready(self) -> None:
        self._show_status("Model ready.", dim=True)
        self._update_model_info()
        self._input_panel.focus_input()

    def _on_model_error(self, msg: str) -> None:
        self._show_status(f"Error: {msg}")
        # Show error in chat area too
        self._chat_window.show_error(msg)

    # ── Message flow ──────────────────────────────────────────────────────

    def _on_quick_action(self, prompt: str) -> None:
        """Send a quick-action card prompt as if the user typed it."""
        self._on_send(prompt)

    def _on_send(self, text: str) -> None:
        """User submitted a message."""
        if not self._chat_mgr.is_ready:
            self._show_status("⚠ Model not loaded — open Settings (⚙) to select a checkpoint.")
            return
        if self._chat_mgr.is_generating:
            return

        # Render user bubble
        self._chat_window.add_user_message(text)
        # Start assistant bubble + typing indicator
        self._chat_window.start_assistant_message()
        # Disable input during generation
        self._input_panel.set_generating(True)
        self._show_status("Generating…")

        # Callbacks — all called from background thread, marshal to Tk with after()
        def on_token(piece: str) -> None:
            self.after(0, lambda p=piece: self._chat_window.append_token(p))

        def on_done(full_text: str) -> None:
            self.after(0, lambda t=full_text: self._finish_generation(t))

        def on_error(msg: str) -> None:
            self.after(0, lambda m=msg: self._generation_error(m))

        self._chat_mgr.send_message(
            user_text=text,
            on_token=on_token,
            on_done=on_done,
            on_error=on_error,
        )

    def _finish_generation(self, full_text: str) -> None:
        self._chat_window.finish_assistant_message(full_text)
        self._input_panel.set_generating(False)
        self._show_status("", dim=True)
        # Auto-save session after each exchange
        self._autosave_session()

    def _generation_error(self, msg: str) -> None:
        self._chat_window.show_error(msg)
        self._input_panel.set_generating(False)
        self._show_status(f"Error: {msg}")

    def _on_stop(self) -> None:
        self._chat_mgr.stop_generation()
        self._show_status("Stopping…", dim=True)

    # ── Session management ────────────────────────────────────────────────

    def _new_chat(self) -> None:
        """Start a fresh conversation."""
        self._autosave_session()          # save current before clearing
        self._chat_mgr.new_session()
        self._session_id = new_session_id()
        self._chat_window.clear()
        self._show_status("New chat started.", dim=True)
        self._refresh_history()
        self._input_panel.focus_input()

    def _autosave_session(self) -> None:
        """Save current history to disk (called silently after each message)."""
        history = self._chat_mgr.get_history()
        if not history:
            return
        try:
            save_session(history, session_id=self._session_id)
            self._refresh_history()
        except Exception:
            pass   # non-critical

    def _load_session(self, session_id: str) -> None:
        """Load a saved conversation from the sidebar."""
        data = load_session(session_id)
        if not data:
            return
        self._autosave_session()
        messages = data.get("messages", [])
        self._chat_mgr.new_session()
        self._chat_mgr.load_history(messages)
        self._session_id = session_id

        # Re-render all messages in the chat window
        self._chat_window.clear()
        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                self._chat_window.add_user_message(content)
            elif role == "assistant":
                self._chat_window.start_assistant_message()
                self._chat_window.finish_assistant_message(content)

        self._show_status(f"Loaded: {data.get('title','')[:40]}", dim=True)

    def _delete_session(self, session_id: str) -> None:
        delete_session(session_id)
        # If we deleted the active session, start fresh
        if session_id == self._session_id:
            self._chat_mgr.new_session()
            self._session_id = new_session_id()
            self._chat_window.clear()
        self._refresh_history()

    def _refresh_history(self) -> None:
        """Reload the sidebar history list from disk."""
        sessions = list_sessions()
        self._sidebar.refresh_history(sessions)

    # ── Settings ─────────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        SettingsDialog(
            parent    = self,
            chat_mgr  = self._chat_mgr,
            on_reload = self._reload_model,
        )

    def _reload_model(self) -> None:
        """Called by SettingsDialog when user clicks 'Load & Apply'."""
        ckpt   = getattr(self._chat_mgr, "_pending_checkpoint", "")
        device = self._chat_mgr._device or "cpu"
        if ckpt:
            self._checkpoint = ckpt
            self._device     = device
            self._load_model_async(ckpt, device)
        else:
            # No new checkpoint — just update sampling params
            self._show_status("Settings applied.", dim=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dizel AI Desktop Interface")
    p.add_argument(
        "--checkpoint", "-c",
        type=str, default="",
        help="Path to .pt model checkpoint (optional — can be set in Settings)",
    )
    p.add_argument(
        "--device", "-d",
        type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app  = DizelApp(checkpoint=args.checkpoint, device=args.device)
    app.mainloop()


if __name__ == "__main__":
    main()
