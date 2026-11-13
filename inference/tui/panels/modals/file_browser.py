"""Filesystem browser for picking checkpoint files from anywhere on disk."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, OptionList, Static
from textual.widgets.option_list import Option

from inference.tui.panels.modals.common import clip, plain_text
from inference.tui.runtime_actions import project_root

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".gguf", ".safetensors", ".bin", ".ckpt"}


def _size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _start_dir(start: str) -> Path:
    if start:
        candidate = Path(start).expanduser()
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.is_dir():
            return candidate

    checkpoints = project_root() / "checkpoints"
    if checkpoints.is_dir():
        return checkpoints
    return Path.home()


class FileBrowserModal(ModalScreen[str | None]):
    """Browse directories and pick a checkpoint, or type an absolute path."""

    BINDINGS = [Binding("escape", "cancel", "Close")]

    def __init__(self, start: str = "", title: str = "Choose checkpoint", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._current = _start_dir(start)
        self._show_all = False
        self._by_id: dict[str, Path] = {}

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card file-card"):
            with Horizontal(classes="modal-head"):
                yield Static(plain_text(self._title), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static("", id="file-sub", classes="modal-sub")
            with Horizontal(classes="file-path-row"):
                yield Input(
                    value=str(self._current),
                    placeholder="Type a folder or checkpoint path…",
                    classes="modal-input",
                    id="file-path",
                )
                yield Button("Go", id="file-go")
            yield OptionList(classes="modal-list file-list", id="file-list")
            with Horizontal(classes="file-options"):
                yield Checkbox(
                    "All files", value=False, compact=True, id="file-all"
                )
                yield Static("", id="file-status")
            yield Static(
                "enter open · tab to path · esc close",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        self._load(self._current)
        self.query_one("#file-list", OptionList).focus()
        self._apply_density()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_density()

    def _apply_density(self) -> None:
        try:
            self.query_one(".modal-card").set_class(
                self.size.height < 30, "-short"
            )
        except Exception:
            pass

    # ── Navigation ────────────────────────────────────────────────────
    def _load(self, directory: Path) -> None:
        try:
            directory = directory.expanduser().resolve()
        except OSError:
            self._status("That path cannot be opened.")
            return
        if not directory.is_dir():
            self._status(f"Not a folder: {directory}")
            return

        try:
            entries = list(directory.iterdir())
        except OSError as error:
            self._status(f"Cannot read folder: {error}")
            return

        self._current = directory
        path_input = self.query_one("#file-path", Input)
        path_input.value = str(directory)
        path_input.cursor_position = len(path_input.value)
        self._status("")

        folders = sorted(
            (entry for entry in entries if entry.is_dir()),
            key=lambda entry: entry.name.lower(),
        )
        files = sorted(
            (
                entry
                for entry in entries
                if entry.is_file()
                and (self._show_all or entry.suffix.lower() in CHECKPOINT_EXTENSIONS)
            ),
            key=lambda entry: entry.name.lower(),
        )

        option_list = self.query_one("#file-list", OptionList)
        option_list.clear_options()
        self._by_id.clear()

        if directory.parent != directory:
            option_list.add_option(Option(plain_text("↑  Parent folder"), id="up"))
        home = Path.home()
        if home != directory:
            option_list.add_option(Option(plain_text("⌂  Home"), id="home"))
        checkpoints = project_root() / "checkpoints"
        if checkpoints.is_dir() and checkpoints != directory:
            option_list.add_option(
                Option(plain_text("◆  Project checkpoints"), id="project")
            )
        if os.name == "nt":
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                drive = Path(f"{letter}:\\")
                try:
                    if drive.exists() and drive != directory:
                        option_list.add_option(
                            Option(Content.assemble((f"{letter}:\\", "bold")), id=f"dir:{drive}")
                        )
                        self._by_id[f"dir:{drive}"] = drive
                except OSError:
                    continue

        first_folder_index = option_list.option_count if folders else None
        for folder in folders:
            option_list.add_option(
                Option(
                    Content.assemble(("▸ ", ""), (folder.name, "bold"), ("/", "")),
                    id=f"dir:{folder}",
                )
            )
            self._by_id[f"dir:{folder}"] = folder

        first_file_index = None
        for item in files:
            if first_file_index is None:
                first_file_index = option_list.option_count
            try:
                stat = item.stat()
                when = datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %H:%M")
                detail = f"{_size(stat.st_size)} · {when}"
            except OSError:
                detail = "checkpoint"
            option_list.add_option(
                Option(
                    Content.assemble(
                        (item.name, "bold"), (f"\n  {clip(detail, 44)}", "")
                    ),
                    id=f"file:{item}",
                )
            )
            self._by_id[f"file:{item}"] = item

        total = len(self._by_id)
        if total == 0:
            option_list.add_option(
                Option(plain_text("This folder is empty."), id="__empty__", disabled=True)
            )
            option_list.highlighted = None
        else:
            if first_file_index is not None:
                option_list.highlighted = first_file_index
            elif first_folder_index is not None:
                option_list.highlighted = first_folder_index
            else:
                option_list.highlighted = 0
            option_list.scroll_to_highlight()

        folder_label = "folder" if len(folders) == 1 else "folders"
        file_label = "checkpoint" if len(files) == 1 else "checkpoints"
        self.query_one("#file-sub", Static).update(
            plain_text(
                f"{len(folders)} {folder_label} · {len(files)} {file_label} · "
                f"{'all files' if self._show_all else 'checkpoints only'}"
            )
        )

    def _status(self, message: str) -> None:
        status = self.query_one("#file-status", Static)
        status.update(plain_text(message))
        status.set_class(bool(message), "-error")

    def _choose(self, raw: str) -> None:
        target = Path(os.path.expandvars(raw.strip())).expanduser()
        if not target.is_absolute():
            target = self._current / target
        try:
            target = target.resolve()
        except OSError:
            self._status("That path cannot be resolved.")
            return

        if target.is_dir():
            self._load(target)
            return
        if not target.is_file():
            self._status("That file does not exist.")
            return
        if not self._show_all and target.suffix.lower() not in CHECKPOINT_EXTENSIONS:
            self._status("Not a checkpoint file. Enable “All files” to use it anyway.")
            return
        self.dismiss(str(target))

    def _activate_selected(self) -> None:
        options = self.query_one("#file-list", OptionList)
        if options.highlighted is None:
            return
        option = options.get_option_at_index(options.highlighted)
        if option.id == "up":
            self._load(self._current.parent)
        elif option.id == "home":
            self._load(Path.home())
        elif option.id == "project":
            self._load(project_root() / "checkpoints")
        elif option.id and option.id.startswith("file:"):
            target = self._by_id.get(option.id)
            if target:
                self.dismiss(str(target))
        elif option.id and option.id.startswith("dir:"):
            target = self._by_id.get(option.id)
            if target:
                self._load(target)

    # ── Events ────────────────────────────────────────────────────────
    @on(OptionList.OptionSelected, "#file-list")
    def _option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._activate_selected()

    @on(Input.Submitted, "#file-path")
    def _path_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._choose(event.value)

    @on(Button.Pressed, "#file-go")
    def _go_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._choose(self.query_one("#file-path", Input).value)

    @on(Checkbox.Changed, "#file-all")
    def _show_all_changed(self, event: Checkbox.Changed) -> None:
        self._show_all = event.value
        self._load(self._current)

    def action_cancel(self) -> None:
        self.dismiss(None)
