"""Settings editor with a section sidebar, validation, and checkpoint browser."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from inference.tui.constants import MODES
from inference.tui.settings_apply import CHAT_DEFAULTS
from inference.tui.panels.modals.common import ConfirmModal, plain_text, set_error
from inference.tui.panels.modals.file_browser import FileBrowserModal
from inference.tui.runtime_actions import (
    project_root,
    provider_catalog,
    provider_requires_key,
    validate_provider_async,
)


@dataclass(frozen=True)
class FieldSpec:
    section: str
    key: str
    label: str
    kind: str = "text"
    hint: str = ""
    options: tuple[tuple[str, str], ...] = ()
    placeholder: str = ""
    minimum: float | None = None
    maximum: float | None = None
    full_width: bool = False
    pane: str = "runtime"
    # "both" | "tui" | "gui". The TUI was showing desktop-only switches — theme,
    # font scale, avatars, the tutorial, the whole Nova voice section — as if they
    # did something here. They did not.
    surface: str = "both"
    # A labelled cluster inside a pane ("Replies", "Transcript", …). Empty for
    # panes short enough to read as one list.
    group: str = ""


@dataclass
class FieldWidgets:
    spec: FieldSpec
    control: Any
    hint: Static
    hint_text: str = ""


def _with_current(
    options: Iterable[tuple[str, str]], current: str, label: str | None = None
) -> tuple[tuple[str, str], ...]:
    pairs = tuple(options)
    if any(value == current for _, value in pairs):
        return pairs
    return pairs + ((label or current or "None", current),)


def _float_text(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value or "")


class SettingsModal(ModalScreen[bool]):
    """General, chat, behaviour, runtime, sampling, budget and API settings."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fields: dict[str, FieldWidgets] = {}
        self._browse_targets: dict[str, Input] = {}
        self._section_order: list[str] = []
        self._section_info: dict[str, tuple[str, str]] = {}
        self._active_section = "general"
        self._dirty = False
        self._ready = False

    # ── Layout ────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        from dizel_core.config_manager import ConfigManager

        cfg = ConfigManager.load()
        sections = self._sections(cfg)
        self._section_order = [sid for sid, _, _, _ in sections]
        self._section_info = {
            sid: (title, description) for sid, title, description, _ in sections
        }

        with Vertical(classes="modal-card settings-card"):
            with Horizontal(classes="modal-head"):
                yield Static("Settings", classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static(
                "Saved to settings.json · changes apply when you save",
                classes="modal-sub",
            )

            with Horizontal(classes="settings-body"):
                with Vertical(classes="settings-nav"):
                    yield OptionList(id="settings-nav", classes="settings-nav-list")
                    yield Static(
                        plain_text(self._summary_text()),
                        classes="settings-summary",
                    )
                with Vertical(classes="settings-content"):
                    # The section name lives in the nav; only the one-line summary
                    # is worth repeating here.
                    yield Static(
                        "", id="settings-section-desc", classes="settings-section-desc"
                    )
                    with VerticalScroll(classes="settings-scroll"):
                        for sid, _title, _description, fields in sections:
                            with Vertical(
                                classes="settings-pane", id=f"pane-{sid}"
                            ):
                                yield from self._render_fields(fields, cfg)

            yield Static("", classes="modal-error")
            with Horizontal(classes="settings-actions"):
                yield Static(
                    "ctrl+s save · esc cancel",
                    id="settings-status",
                    classes="settings-status",
                )
                yield Button("Cancel", id="settings-cancel")
                yield Button("Save changes", id="settings-save", variant="primary")

    def on_mount(self) -> None:
        self.query_one(".modal-error", Static).styles.display = "none"
        nav = self.query_one("#settings-nav", OptionList)
        for sid in self._section_order:
            nav.add_option(Option(self._section_info[sid][0], id=sid))
        nav.highlighted = 0
        self._show_section(self._section_order[0])
        nav.focus()
        for widgets in self._fields.values():
            if widgets.spec.kind == "path" and widgets.control.value:
                widgets.control.cursor_position = len(widgets.control.value)
        self._apply_density()
        self.call_after_refresh(self._set_ready)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_density()

    def _set_ready(self) -> None:
        self._ready = True

    def _apply_density(self) -> None:
        narrow = self.size.width < 104
        # Nine sections plus the ACTIVE summary no longer fit side by side on
        # a 30-row terminal; drop the supporting copy before they start
        # scrolling out of the nav.
        short = self.size.height < 34
        try:
            card = self.query_one(".modal-card")
            card.set_class(narrow, "-narrow")
            card.set_class(short, "-short")
        except Exception:
            pass

    def _summary_text(self) -> str:
        app = self.app
        return (
            "ACTIVE\n"
            f"{app.active_mode} · {app.active_model or 'no model'}\n"
            f"{app.active_provider} · {getattr(app, '_device', 'cpu')}"
        )

    # ── Section definitions ───────────────────────────────────────────
    def _sections(self, cfg: dict) -> list[tuple[str, str, str, list[FieldSpec]]]:
        api = cfg.get("api_router", {})
        nova = cfg.get("nova", {})
        appearance = cfg.get("appearance", {})

        providers = tuple((meta["name"], meta["slug"]) for meta in provider_catalog())
        providers = _with_current(
            providers, api.get("provider", "local"), "Configured provider"
        )
        devices = _with_current(
            (("CPU", "cpu"), ("CUDA", "cuda")), cfg.get("device", "cpu"), "Other"
        )
        languages = _with_current(
            (
                ("Auto", "auto"),
                ("English", "en"),
                ("Indonesian", "id"),
                ("Japanese", "ja"),
            ),
            nova.get("language", "auto"),
        )
        sizes = _with_current(
            (("Tiny", "tiny"), ("Base", "base"), ("Small", "small")),
            nova.get("model_size", "base"),
        )
        themes = _with_current(
            (
                ("System", "System"),
                ("Light", "Light"),
                ("Dark", "Dark"),
                ("Dark Blue", "Dark Blue"),
            ),
            appearance.get("color_mode", "System"),
        )
        font_scales = _with_current(
            (("Small", "Small"), ("Medium", "Medium"), ("Large", "Large")),
            appearance.get("font_scale", "Medium"),
        )
        avatars = _with_current(
            (("Show avatars", "Show avatars"), ("Hide avatars", "Hide avatars")),
            appearance.get("avatar_display", "Show avatars"),
        )

        general = [
            FieldSpec(
                "", "system_prompt", "System prompt", "multiline",
                "Used until a mode profile replaces it.",
                full_width=True, pane="general", group="Prompt",
            ),
            FieldSpec(
                "user_profile", "username", "User name", "text",
                "Used in the greeting on both surfaces.",
                pane="general", group="Identity",
            ),
            FieldSpec(
                "appearance", "color_mode", "Desktop theme", "select",
                "Desktop app only; applies next launch.", themes, pane="general", surface="gui",
            ),
            FieldSpec(
                "appearance", "show_title_bar", "Title bar", "bool",
                "Desktop app only.", pane="general", surface="gui",
            ),
            FieldSpec(
                "appearance", "font_scale", "Font scale", "select",
                "Desktop app only.", font_scales, pane="general", surface="gui",
            ),
            FieldSpec(
                "appearance", "avatar_display", "Avatars", "select",
                "Desktop app only.", avatars, pane="general", surface="gui",
            ),
            FieldSpec(
                "@ui", "reset_tutorial", "Replay tutorial", "bool",
                "Show the first-run tutorial again.", pane="general", surface="gui",
            ),
        ]

        # The desktop's Chat and Shortcuts tabs, ported to the transcript and
        # composer this terminal actually draws.
        chat = [
            FieldSpec(
                "appearance", "typing_animation", "Typing animation", "select",
                "How quickly streamed text catches up to the model.",
                (("Off", "off"), ("Fast", "fast"), ("Smooth", "smooth"),
                 ("Cinematic", "cinematic")),
                pane="chat", group="Replies",
            ),
            FieldSpec(
                "appearance", "show_speed", "Show reply speed", "bool",
                "Estimated tokens per second beside each reply.",
                pane="chat", group="Replies",
            ),
            FieldSpec(
                "appearance", "show_model_tag", "Show model tag", "bool",
                "Which model answered, in each reply's meta line.",
                pane="chat", group="Replies",
            ),
            FieldSpec(
                "appearance", "syntax_highlighting", "Syntax highlighting", "bool",
                "Colour code blocks.", pane="chat", group="Replies",
            ),
            FieldSpec(
                "appearance", "show_timestamps", "Show timestamps", "bool",
                "Time beside each reply's token count.", pane="chat", group="Transcript",
            ),
            FieldSpec(
                "appearance", "clock_24h", "24-hour clock", "bool",
                "Use 24h instead of am/pm.", pane="chat", group="Transcript",
            ),
            FieldSpec(
                "appearance", "separators", "Separators", "bool",
                "A hairline between messages.", pane="chat", group="Transcript",
            ),
            FieldSpec(
                "appearance", "message_spacing", "Message spacing", "select",
                "Gap between turns.",
                (("Compact", "compact"), ("Comfortable", "comfortable")),
                pane="chat", group="Transcript",
            ),
            FieldSpec(
                "appearance", "message_width", "Message width", "select",
                "How wide the conversation runs.",
                (("Comfortable", "comfortable"), ("Wide", "wide"), ("Full", "full")),
                pane="chat", group="Transcript",
            ),
            FieldSpec(
                "appearance", "auto_scroll", "Follow new messages", "bool",
                "Keep the transcript pinned to the newest reply while it streams.",
                pane="chat", group="Transcript",
            ),
            FieldSpec(
                "appearance", "show_suggestions", "Starter suggestions", "bool",
                "Show prompt pills on an empty chat.", pane="chat", group="Welcome",
            ),
            FieldSpec(
                "appearance", "animations", "UI animations", "bool",
                "Welcome shimmer, working spinner and rotating prompts.",
                pane="chat", group="Welcome",
            ),
        ]

        behaviour = [
            FieldSpec(
                "appearance", "enter_to_send", "Press Enter to send", "bool",
                "Off: Enter opens the multi-line editor and ctrl+s sends from there.",
                pane="behaviour", group="Composer",
            ),
            FieldSpec(
                "appearance", "auto_focus_composer", "Auto-focus the composer", "bool",
                "Jump to the input when a chat opens or the app is ready.",
                pane="behaviour", group="Composer",
            ),
            FieldSpec(
                "appearance", "escape_clears_draft", "Escape clears the draft", "bool",
                "Esc wipes the composer while no reply is running.",
                pane="behaviour", group="Editing keys",
            ),
            FieldSpec(
                "appearance", "arrow_up_edits_last", "Arrow Up edits your last message",
                "bool",
                "↑ in an empty composer reopens your previous prompt.",
                pane="behaviour", group="Editing keys",
            ),
        ]

        runtime = [
            FieldSpec(
                "@runtime", "mode", "Mode", "select",
                "Sampling and prompt profile.",
                tuple((mode, mode) for mode in MODES), pane="runtime", group="Model",
            ),
            FieldSpec(
                "", "device", "Device", "select",
                "GGUF CUDA needs a GPU-enabled llama.cpp build.",
                devices, pane="runtime", group="Model",
            ),
            FieldSpec(
                "", "checkpoint", "Checkpoint", "path",
                "Loaded immediately after saving (local provider).",
                placeholder="Auto-discover in checkpoints/",
                full_width=True, pane="runtime", group="Checkpoint",
            ),
        ]

        sampling = [
            FieldSpec(
                "sampling", "temperature", "Temperature", "float",
                "0 deterministic · 2 chaotic", minimum=0, maximum=2, pane="sampling",
            ),
            FieldSpec(
                "sampling", "top_k", "Top-k", "int",
                "Candidate tokens kept per step.", minimum=1, maximum=500,
                pane="sampling",
            ),
            FieldSpec(
                "sampling", "top_p", "Top-p", "float",
                "Nucleus sampling mass.", minimum=0, maximum=1, pane="sampling",
            ),
            FieldSpec(
                "sampling", "repetition_penalty", "Repetition", "float",
                "Higher discourages repeated phrases.", minimum=0.5, maximum=3,
                pane="sampling",
            ),
            FieldSpec(
                "sampling", "max_new_tokens", "Max new tokens", "int",
                "Per-reply cap; budgets may lower it.",
                minimum=16, maximum=4096, pane="sampling",
            ),
        ]

        budget_fields = [
            FieldSpec(
                "token_budget", "chat_budget", "Chat budget", "int",
                "Token allowance for casual chat.", minimum=1, pane="budget",
                group="Per-task allowances",
            ),
            FieldSpec(
                "token_budget", "coding_budget", "Coding budget", "int",
                "Allowance when code is detected.", minimum=1, pane="budget",
                group="Per-task allowances",
            ),
            FieldSpec(
                "token_budget", "complex_budget", "Complex budget", "int",
                "Analysis and long-form answers.", minimum=1, pane="budget",
                group="Per-task allowances",
            ),
            FieldSpec(
                "token_budget", "factual_budget", "Factual budget", "int",
                "Short factual questions.", minimum=1, pane="budget",
                group="Per-task allowances",
            ),
            FieldSpec(
                "token_budget", "tool_budget", "Tool budget", "int",
                "Answers after web search or deep think.", minimum=1, pane="budget",
                group="Per-task allowances",
            ),
            FieldSpec(
                "token_budget", "max_context_tokens", "Max context", "int",
                "Capacity shown by the context meter.", minimum=256, pane="budget",
                group="Limits",
            ),
            FieldSpec(
                "token_budget", "hard_output_limit", "Hard output limit", "int",
                "0 keeps the adaptive budget.", minimum=0, maximum=32768,
                pane="budget", group="Limits",
            ),
            FieldSpec(
                "token_budget", "verbosity", "Verbosity", "select",
                "Scales the base allowance.",
                (("Low", "low"), ("Normal", "normal"), ("High", "high")),
                pane="budget", group="Limits",
            ),
        ]

        stored_key = bool(api.get("api_key"))
        api_fields = [
            FieldSpec(
                "api_router", "provider", "Provider", "select",
                "Switching providers may require a key.", providers, pane="api",
                group="Provider",
            ),
            FieldSpec(
                "api_router", "model", "Model id", "text",
                "Provider model id, e.g. gpt-4o.", placeholder="e.g. gpt-4o",
                pane="api", group="Provider",
            ),
            FieldSpec(
                "api_router", "api_key", "API key", "secret",
                "Stored encrypted. Leave blank to keep the current key."
                if stored_key
                else "Not set. Stored encrypted in settings.json.",
                full_width=True, pane="api", group="Provider",
            ),
            FieldSpec(
                "api_router", "custom_name", "Custom name", "text",
                "Display name for the custom provider.",
                placeholder="My endpoint", pane="api", group="Endpoints",
            ),
            FieldSpec(
                "api_router", "custom_url", "Custom base URL", "text",
                "Required when provider is “custom”.",
                placeholder="https://api.example.com/v1", full_width=True,
                pane="api", group="Endpoints",
            ),
            FieldSpec(
                "api_router", "ollama_url", "Ollama URL", "text",
                "Only used by the ollama provider.",
                placeholder="http://localhost:11434", full_width=True, pane="api",
                group="Endpoints",
            ),
            FieldSpec(
                "api_router", "azure_resource", "Azure resource", "text",
                "Only used by the azure provider.", placeholder="resource-name",
                pane="api", group="Azure",
            ),
            FieldSpec(
                "api_router", "azure_deployment", "Azure deployment", "text",
                "Only used by the azure provider.", placeholder="deployment-name",
                pane="api", group="Azure",
            ),
        ]

        nova_fields = [
            FieldSpec(
                "nova", "model_size", "Whisper model", "select",
                "Larger models are slower but more accurate.", sizes, pane="nova",
            ),
            FieldSpec(
                "nova", "language", "Language", "select",
                "Auto detects the spoken language.", languages, pane="nova",
            ),
            FieldSpec(
                "nova", "silence_timeout", "Silence timeout", "int",
                "Seconds of silence before recording stops.",
                minimum=1, maximum=60, pane="nova",
            ),
            FieldSpec(
                "nova", "show_waveform", "Waveform", "bool",
                "Draw the live audio waveform while recording.", pane="nova",
            ),
        ]

        # A terminal has a mouse and a bell; a window does not have either in
        # the same sense. These are the TUI's own settings.
        terminal = [
            FieldSpec(
                "tui", "mouse", "Mouse support", "bool",
                "Click to focus, scroll the transcript. Off keeps text selection working.",
                pane="terminal",
            ),
            FieldSpec(
                "tui", "bell", "Bell on reply", "bool",
                "Ring the terminal bell when a reply finishes.", pane="terminal",
            ),
            FieldSpec(
                "tui", "confirm_quit", "Confirm before quitting", "bool",
                "Ctrl+C arms; a second press within three seconds exits.", pane="terminal",
            ),
            FieldSpec(
                "tui", "show_tokens", "Show token count", "bool",
                "The running context count in the status line.", pane="terminal",
            ),
            FieldSpec(
                "tui", "show_hints", "Show key hints", "bool",
                "ctrl+k / ctrl+t reminders on the right of the status line.",
                pane="terminal",
            ),
        ]

        nova_fields = [spec.__class__(
            **{**spec.__dict__, "surface": "gui"}
        ) for spec in nova_fields]

        tool_fields = [
            FieldSpec(
                "@tools", "enabled", "Model can call tools", "bool",
                "Read files, search, calculate. Costs ~330 prompt tokens per request.",
                pane="tools",
            ),
        ]

        sections = [
            ("general", "General", "Prompt and identity.", general),
            ("chat", "Chat", "How replies read, stream and settle into the transcript.", chat),
            ("behaviour", "Behaviour", "What the composer's keys do.", behaviour),
            ("tools", "Tools", "What the model may call.", tool_fields),
            ("terminal", "Terminal", "This client only — no desktop equivalent.", terminal),
            ("runtime", "Runtime", "Mode, compute device and checkpoint.", runtime),
            ("sampling", "Sampling", "How the local model picks each next token.", sampling),
            ("budget", "Budget", "Adaptive output limits by task type.", budget_fields),
            ("api", "API", "Cloud or self-hosted inference.", api_fields),
            ("nova", "Nova", "Speech-to-text for the desktop app.", nova_fields),
        ]

        # Drop anything this surface cannot act on, then drop sections left empty.
        visible = []
        for sid, title, description, fields in sections:
            kept = [f for f in fields if f.surface in ("both", "tui")]
            if kept:
                visible.append((sid, title, description, kept))
        return visible

    # ── Field construction ────────────────────────────────────────────
    def _render_fields(self, specs: list[FieldSpec], cfg: dict) -> ComposeResult:
        batch: list[FieldSpec] = []
        active_group = ""
        for spec in specs:
            if spec.group != active_group:
                yield from self._render_grid(batch, cfg)
                batch = []
                active_group = spec.group
                if active_group:
                    yield Static(
                        f"◆  {plain_text(active_group)}", classes="form-group"
                    )
            if spec.kind in ("bool", "select"):
                # Toggles and pickers read best as a switch on the right of
                # their own line, not as another stacked cell in the grid.
                yield from self._render_grid(batch, cfg)
                batch = []
                yield from self._form_row(spec, cfg)
            elif spec.full_width:
                yield from self._render_grid(batch, cfg)
                batch = []
                yield from self._form_item(spec, cfg)
            else:
                batch.append(spec)
        yield from self._render_grid(batch, cfg)

    def _render_grid(
        self, batch: list[FieldSpec], cfg: dict
    ) -> ComposeResult:
        if not batch:
            return
        with Vertical(classes="form-grid"):
            for item in batch:
                yield from self._form_item(item, cfg)

    def _register_field(self, spec: FieldSpec, cfg: dict) -> tuple[Any, Static]:
        control = self._make_control(spec, cfg)
        hint = Static(plain_text(spec.hint), classes="form-hint")
        self._fields[self._field_key(spec)] = FieldWidgets(
            spec, control, hint, spec.hint
        )
        return control, hint

    def _form_item(self, spec: FieldSpec, cfg: dict) -> ComposeResult:
        control, hint = self._register_field(spec, cfg)
        classes = "form-item" + (" -wide" if spec.full_width else "")
        with Vertical(classes=classes):
            yield Static(plain_text(spec.label), classes="form-label")
            if spec.kind == "path":
                with Horizontal(classes="form-path"):
                    yield control
                    button = Button(
                        "Browse…",
                        id=f"browse-{control.id}",
                        classes="form-browse",
                    )
                    self._browse_targets[button.id] = control
                    yield button
            else:
                yield control
            yield hint

    def _form_row(self, spec: FieldSpec, cfg: dict) -> ComposeResult:
        control, hint = self._register_field(spec, cfg)
        row_class = "form-row -check" if spec.kind == "bool" else "form-row -select"
        with Horizontal(classes=row_class):
            with Vertical(classes="form-row-text"):
                yield Static(plain_text(spec.label), classes="form-row-label")
                yield hint
            yield control

    def _make_control(self, spec: FieldSpec, cfg: dict):
        section_id = spec.section.lstrip("@") or "root"
        control_id = "settings-" + f"{section_id}-{spec.key}".replace("_", "-")
        current = self._current_value(spec, cfg)

        if spec.kind == "select":
            return Select(
                spec.options,
                value=current,
                allow_blank=False,
                compact=True,
                id=control_id,
                classes="form-select",
            )
        if spec.kind == "bool":
            return Checkbox(
                "On" if current else "Off",
                value=bool(current),
                compact=True,
                id=control_id,
                classes="form-check",
            )
        if spec.kind == "secret":
            stored = bool(cfg.get("api_router", {}).get("api_key"))
            return Input(
                "",
                password=True,
                placeholder="•••••••• stored" if stored else "No key set",
                id=control_id,
                classes="form-input",
            )
        if spec.kind == "multiline":
            return TextArea(
                str(current or ""),
                id=control_id,
                classes="form-textarea",
            )
        if spec.kind == "int":
            return Input(
                str(current if current is not None else ""),
                type="integer",
                id=control_id,
                classes="form-input",
            )
        if spec.kind == "float":
            return Input(
                _float_text(current),
                type="number",
                id=control_id,
                classes="form-input",
            )
        return Input(
            str(current if current is not None else ""),
            placeholder=spec.placeholder,
            id=control_id,
            classes="form-input",
        )

    def _current_value(self, spec: FieldSpec, cfg: dict):
        if spec.section == "@runtime":
            return self.app.active_mode
        if spec.section == "@ui":
            return False
        if spec.section == "@tools":
            try:
                from dizel_core import tools as tool_registry

                return tool_registry.is_enabled()
            except Exception:
                return False
        if spec.section == "":
            return cfg.get(spec.key, "")
        if spec.section == "api_router" and spec.key == "api_key":
            return ""
        section = cfg.get(spec.section, {})
        if not isinstance(section, dict):
            return ""
        if spec.section == "appearance":
            defaults = {
                "color_mode": "System",
                "font_scale": "Medium",
                "avatar_display": "Show avatars",
                "show_title_bar": False,
                **CHAT_DEFAULTS,
            }
            return section.get(spec.key, defaults.get(spec.key, ""))
        return section.get(spec.key, "")

    @staticmethod
    def _field_key(spec: FieldSpec) -> str:
        return f"{spec.section or 'root'}.{spec.key}"

    # ── Navigation and dirty state ────────────────────────────────────
    @on(OptionList.OptionHighlighted, "#settings-nav")
    def _nav_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id:
            self._show_section(event.option.id)

    @on(OptionList.OptionSelected, "#settings-nav")
    def _nav_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id:
            self._show_section(event.option.id)

    def _show_section(self, section_id: str) -> None:
        if section_id not in self._section_info:
            return
        self._active_section = section_id
        nav = self.query_one("#settings-nav", OptionList)
        target = self._section_order.index(section_id)
        if nav.highlighted != target:
            nav.highlighted = target
        for pane in self.query(".settings-pane"):
            pane.set_class(pane.id == f"pane-{section_id}", "-active")
        _title, description = self._section_info[section_id]
        self.query_one("#settings-section-desc", Static).update(
            plain_text(description)
        )

    def _dirty_now(self) -> None:
        if not self._ready or self._dirty:
            return
        self._dirty = True
        status = self.query_one("#settings-status", Static)
        status.update(plain_text("Unsaved changes · ctrl+s to save"))
        status.add_class("-dirty")

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        self._dirty_now()

    @on(Select.Changed)
    def _select_changed(self, event: Select.Changed) -> None:
        self._dirty_now()

    @on(TextArea.Changed)
    def _text_changed(self, event: TextArea.Changed) -> None:
        self._dirty_now()

    @on(Checkbox.Changed)
    def _checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.checkbox.label = "On" if event.value else "Off"
        self._dirty_now()

    # ── Checkpoint browser ────────────────────────────────────────────
    @on(Button.Pressed, ".form-browse")
    def _browse_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        target = self._browse_targets.get(event.button.id or "")
        if target is None:
            return

        def _chosen(path: str | None) -> None:
            if not path:
                return
            target.value = path
            target.cursor_position = len(path)
            target.focus()
            self._dirty_now()

        self.app.push_screen(FileBrowserModal(target.value), _chosen)

    # ── Validation and save ───────────────────────────────────────────
    def _read(self, spec: FieldSpec, control) -> Any:
        if spec.kind == "bool":
            return bool(control.value)
        if spec.kind == "select":
            if control.value is Select.NULL:
                raise ValueError("Choose a value.")
            return control.value
        if spec.kind == "multiline":
            return str(control.text).strip()

        raw = str(control.value).strip()
        if spec.kind in ("text", "secret", "path"):
            return raw
        if raw == "":
            raise ValueError("Required.")

        try:
            number = int(raw) if spec.kind == "int" else float(raw)
        except ValueError:
            raise ValueError(
                "Whole number required." if spec.kind == "int" else "Number required."
            )

        if spec.minimum is not None and number < spec.minimum:
            raise ValueError(f"Must be at least {spec.minimum:g}.")
        if spec.maximum is not None and number > spec.maximum:
            raise ValueError(f"Must be at most {spec.maximum:g}.")
        return number

    def _field_for(self, key: str) -> FieldSpec | None:
        widgets = self._fields.get(key)
        return widgets.spec if widgets else None

    def _mark_error(self, spec: FieldSpec, message: str) -> None:
        widgets = self._fields.get(self._field_key(spec))
        if not widgets:
            return
        widgets.hint.update(plain_text(message))
        widgets.hint.add_class("-error")

    def _clear_errors(self) -> None:
        for widgets in self._fields.values():
            widgets.hint.update(plain_text(widgets.hint_text))
            widgets.hint.remove_class("-error")

    def _focus(self, spec: FieldSpec) -> None:
        widgets = self._fields.get(self._field_key(spec))
        if not widgets:
            return
        self._show_section(spec.pane)
        self.call_after_refresh(widgets.control.focus)

    def _report_errors(self, errors: list[tuple[FieldSpec, str]]) -> None:
        for spec, message in errors:
            self._mark_error(spec, message)
        set_error(
            self,
            errors[0][1] if len(errors) == 1 else f"{len(errors)} fields need attention.",
        )
        self._focus(errors[0][0])

    def action_save(self) -> None:
        self._save()

    @on(Button.Pressed, "#settings-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._save()

    @on(Button.Pressed, "#settings-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel()

    def action_cancel(self) -> None:
        self._cancel()

    def _cancel(self) -> None:
        if not self._dirty:
            self.dismiss(False)
            return

        def _confirmed(discard: bool | None) -> None:
            if discard:
                self.dismiss(False)

        self.app.push_screen(
            ConfirmModal(
                "Discard changes?",
                "Your edits have not been saved.",
                confirm_label="Discard",
            ),
            _confirmed,
        )

    def _save(self) -> None:
        from dizel_core.config_manager import ConfigManager, encrypt_key

        self._clear_errors()
        set_error(self, "")

        cfg = ConfigManager.load()
        original = deepcopy(cfg)
        values: dict[str, Any] = {}
        errors: list[tuple[FieldSpec, str]] = []

        for widgets in self._fields.values():
            try:
                values[self._field_key(widgets.spec)] = self._read(
                    widgets.spec, widgets.control
                )
            except ValueError as error:
                errors.append((widgets.spec, str(error)))

        if errors:
            self._report_errors(errors)
            return

        mode = values.get("@runtime.mode", self.app.active_mode)
        provider = values.get("api_router.provider", "local")
        model = values.get("api_router.model", "")
        custom_url = values.get("api_router.custom_url", "")
        new_key = values.get("api_router.api_key", "")
        manager = self.app.chat_bridge.manager
        previous_provider = manager._provider_slug
        previous_checkpoint = getattr(self.app, "_checkpoint", "")
        previous_device = getattr(self.app, "_device", "cpu")

        if provider == "custom":
            if not custom_url:
                url_spec = self._field_for("api_router.custom_url")
                if url_spec:
                    errors.append((url_spec, "A base URL is required for custom."))
            if not model:
                model_spec = self._field_for("api_router.model")
                if model_spec:
                    errors.append((model_spec, "A model id is required for custom."))

        if provider != "local" and provider_requires_key(provider):
            same_provider = provider == manager._provider_slug
            stored = bool(cfg.get("api_router", {}).get("api_key"))
            key_spec = self._field_for("api_router.api_key")
            if not new_key and not (same_provider and stored):
                if key_spec:
                    errors.append((key_spec, "This provider needs an API key."))
            elif new_key and len(new_key) < 8 and key_spec:
                errors.append((key_spec, "That key looks too short."))

        checkpoint_raw = str(values.get("root.checkpoint", "") or "").strip()
        checkpoint = checkpoint_raw
        if checkpoint:
            path = Path(os.path.expandvars(checkpoint)).expanduser()
            if not path.is_absolute():
                path = project_root() / path
            if not path.is_file():
                checkpoint_spec = self._field_for("root.checkpoint")
                if checkpoint_spec:
                    errors.append((checkpoint_spec, "Choose a checkpoint file."))
            else:
                checkpoint = str(path.resolve())

        device = str(values.get("root.device", cfg.get("device", "cpu")))
        original_checkpoint = str(original.get("checkpoint", "") or "").strip()
        checkpoint_edited = checkpoint_raw != original_checkpoint
        device_changed = device != previous_device
        provider_becomes_local = provider == "local" and previous_provider != "local"

        pending_checkpoint = ""
        if checkpoint_edited:
            pending_checkpoint = checkpoint
        elif device_changed:
            pending_checkpoint = previous_checkpoint or checkpoint
        elif provider_becomes_local:
            pending_checkpoint = (
                checkpoint or previous_checkpoint or self.app._discover_checkpoint()
            )
        elif checkpoint and checkpoint != previous_checkpoint:
            pending_checkpoint = checkpoint

        wants_local_load = provider == "local" and bool(pending_checkpoint)

        if wants_local_load and (
            manager.is_generating
            or self.app.generation_state in ("STREAMING", "THINKING", "LOADING")
        ):
            target_spec = None
            if checkpoint_edited or not device_changed:
                target_spec = self._field_for("root.checkpoint")
            else:
                target_spec = self._field_for("root.device")
            if target_spec:
                busy = self.app.generation_state == "LOADING"
                errors.append(
                    (
                        target_spec,
                        "Wait for the current model load."
                        if busy
                        else "Stop the current generation before switching models.",
                    )
                )

        if errors:
            self._report_errors(errors)
            return

        for key, value in values.items():
            if key in ("@runtime.mode", "@ui.reset_tutorial", "@tools.enabled", "api_router.api_key"):
                continue
            spec = self._fields[key].spec
            if spec.section == "":
                cfg[spec.key] = value
            elif spec.section:
                section = cfg.setdefault(spec.section, {})
                if isinstance(section, dict):
                    section[spec.key] = value

        # Terminal-only switches take effect now, not on the next launch.
        try:
            from inference.tui.settings_apply import apply_terminal_prefs

            apply_terminal_prefs(self.app, cfg)
        except Exception:
            pass

        cfg["checkpoint"] = checkpoint
        cfg["device"] = values.get("root.device", cfg.get("device", "cpu"))
        system_prompt = values.get("root.system_prompt", cfg.get("system_prompt", ""))

        api_cfg = cfg.setdefault("api_router", {})
        api_cfg["provider"] = provider
        api_cfg["model"] = model
        if new_key:
            api_cfg["api_key"] = encrypt_key(new_key)

        if "@tools.enabled" in values:
            try:
                from dizel_core import tools as tool_registry

                tool_registry.set_enabled(bool(values["@tools.enabled"]))
            except Exception:
                pass

        if values.get("@ui.reset_tutorial"):
            tutorial = cfg.setdefault("tutorial", {})
            tutorial["completed"] = False
            tutorial["skipped"] = False

        ConfigManager.save(cfg)

        notes = ["Settings saved"]
        try:
            manager.reload_sampling()
            notes.append("sampling applied")
        except Exception as error:
            notes.append(f"sampling apply failed: {error}")

        try:
            manager.reload_provider()
            notes.append("provider reloaded")
        except Exception as error:
            notes.append(f"provider apply failed: {error}")

        self.app.active_provider = manager.active_provider_slug
        if self.app.active_provider == "local":
            from model.naming import model_display_name

            self.app.active_model = model_display_name(checkpoint) if checkpoint else ""
        else:
            self.app.active_model = manager._api_model or model

        previous_mode = self.app.active_mode
        if mode != previous_mode:
            self.app.set_mode(mode, announce=False)
            notes.append(f"mode {mode}")
        if system_prompt != original.get("system_prompt", ""):
            manager.system_prompt = system_prompt
            notes.append("prompt applied")

        self.app._checkpoint = checkpoint
        self.app._device = device

        load_path = ""
        if provider == "local" and self.app.active_provider == "local":
            # Only load when the checkpoint actually changed. Saving an unrelated
            # setting used to re-run the whole load and freeze the interface.
            already = getattr(manager, "checkpoint_path", "") or ""
            same = bool(pending_checkpoint) and bool(already) and (
                os.path.abspath(pending_checkpoint) == os.path.abspath(already)
            )
            if pending_checkpoint and not same:
                load_path = pending_checkpoint
                notes.append(f"loading {os.path.basename(load_path)}")
            elif same and manager.is_ready:
                notes.append("model already loaded")
            elif provider_becomes_local:
                notes.append("no checkpoint selected")

        if provider != "local" and self.app.active_provider == "local":
            notes.append("provider failed to load")
        elif provider != "local" and provider == self.app.active_provider:
            validate_provider_async(self.app, provider)

        self.app._log_system(" · ".join(notes))
        if load_path:
            self.app._load_model_async(load_path, device)
        self.dismiss(True)
