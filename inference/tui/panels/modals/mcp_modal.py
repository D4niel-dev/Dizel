"""List, add, edit, test and toggle MCP servers from the TUI."""

from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from inference.tui.panels.modals.common import ConfirmModal, plain_text, set_error

_GLYPHS = {"ready": "●", "starting": "◐", "error": "✕", "disabled": "○"}


def row_text(server: dict, state: dict) -> str:
    glyph = _GLYPHS.get(str(state.get("state", "disabled")), "○")
    count = len(state.get("tools", []) or [])
    tools = f"{count} tool" + ("" if count == 1 else "s")
    return f"{glyph}  {server['name']}  ·  {server['transport']}  ·  {tools}"


def parse_kv_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        key, sep, value = trimmed.partition("=")
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip() if sep else ""
    return values


def format_kv_lines(mapping: dict) -> str:
    return "\n".join(f"{key}={value}" for key, value in (mapping or {}).items())


def merge_form_payload(
    servers: list[dict], payload: dict, edit_id: str | None = None
) -> tuple[list[dict] | None, str]:
    """Merge a validated form payload into the server list.

    ``edit_id`` identifies the row an edit started from; ``None`` means the
    payload came from the add form.  An add whose normalized id already exists
    is refused instead of silently overwriting the stored server.
    """
    updated = [dict(entry) for entry in servers]
    if edit_id is None:
        existing = next(
            (entry for entry in updated if entry["id"] == payload.get("id")), None
        )
        if existing is not None:
            return None, (
                f"A server with id “{payload.get('id')}” already exists "
                f"(“{existing.get('name', '')}”). Edit that server, or choose "
                "a different name."
            )
        updated.append(payload)
        return updated, ""

    index = next(
        (position for position, entry in enumerate(updated) if entry["id"] == edit_id),
        None,
    )
    if index is None:
        return None, "That server is no longer in the list."
    updated[index] = {**payload, "id": edit_id}
    return updated, ""


class McpFormModal(ModalScreen[dict | None]):
    """Add or edit one server; dismisses with a raw payload."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self, server: dict | None = None, *, edit: bool | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._server = dict(server) if server else {}
        self._edit = bool(self._server) if edit is None else edit

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card mcp-form-card"):
            with Horizontal(classes="modal-head"):
                title = "Edit MCP server" if self._edit else "Add MCP server"
                yield Static(plain_text(title), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static(plain_text("name"), classes="field-label")
            yield Input(value=self._server.get("name", ""), id="mcp-name", classes="modal-input")
            yield Static(plain_text("transport: stdio or http"), classes="field-label")
            yield Input(value=self._server.get("transport", "stdio"), id="mcp-transport", classes="modal-input")
            yield Static(plain_text("stdio command"), classes="field-label")
            yield Input(value=self._server.get("command", ""), id="mcp-command", classes="modal-input")
            yield Static(plain_text("stdio args, one per line"), classes="field-label")
            yield TextArea("\n".join(self._server.get("args", [])), id="mcp-args")
            yield Static(plain_text("stdio env, KEY=VALUE per line"), classes="field-label")
            yield TextArea(format_kv_lines(self._server.get("env", {})), id="mcp-env")
            yield Static(plain_text("http url"), classes="field-label")
            yield Input(value=self._server.get("url", ""), id="mcp-url", classes="modal-input")
            yield Static(plain_text("http headers, KEY=VALUE per line"), classes="field-label")
            yield TextArea(format_kv_lines(self._server.get("headers", {})), id="mcp-headers")
            yield Static("", classes="modal-error")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="mcp-cancel")
                yield Button("Save", id="mcp-save", variant="primary")

    def on_mount(self) -> None:
        # The form is taller than a typical terminal, and a plain Vertical clips
        # instead of scrolling, which would leave Save unreachable.
        self.query_one(".modal-card").styles.overflow_y = "auto"
        self.query_one(".modal-error", Static).styles.display = "none"
        self.query_one("#mcp-name", Input).focus()

    @on(Button.Pressed, "#mcp-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#mcp-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        from dizel_core.mcp import config as mcp_config

        transport = self.query_one("#mcp-transport", Input).value.strip().lower()
        payload: dict[str, Any] = {
            # An add form must not carry an id: the id is derived from the
            # name, including when the add form is re-opened after a clash.
            "id": self._server.get("id") if self._edit else None,
            "name": self.query_one("#mcp-name", Input).value.strip(),
            "transport": transport,
            "enabled": self._server.get("enabled", True),
        }
        if transport == "stdio":
            payload["command"] = self.query_one("#mcp-command", Input).value.strip()
            payload["args"] = [
                line.strip()
                for line in self.query_one("#mcp-args", TextArea).text.splitlines()
                if line.strip()
            ]
            payload["env"] = parse_kv_lines(self.query_one("#mcp-env", TextArea).text)
        else:
            payload["url"] = self.query_one("#mcp-url", Input).value.strip()
            payload["headers"] = parse_kv_lines(self.query_one("#mcp-headers", TextArea).text)

        try:
            normalized = mcp_config.validate_servers([payload])[0]
        except mcp_config.MCPConfigError as error:
            set_error(self, str(error))
            return
        self.dismiss(normalized)

    def action_cancel(self) -> None:
        self.dismiss(None)


class McpModal(ModalScreen[None]):
    """The /mcp surface: list servers, toggle, test, add, edit, delete."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close"),
        Binding("space", "toggle", "Toggle"),
        Binding("t", "test", "Test"),
        Binding("a", "add", "Add"),
        Binding("e", "edit", "Edit"),
        Binding("d", "delete", "Delete"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card mcp-card"):
            with Horizontal(classes="modal-head"):
                yield Static(plain_text("MCP servers"), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static(
                plain_text(
                    "Enabled servers start with the app. Tools appear when the model's tools switch is on."
                ),
                classes="modal-sub",
            )
            yield OptionList(classes="modal-list")
            yield Static(
                "space toggle · t test · a add · e edit · d delete · esc close",
                classes="modal-footer",
            )

    def on_mount(self) -> None:
        self._servers: list[dict] = []
        self._states: dict[str, dict] = {}
        self._reload()
        self.set_interval(1.5, self._reload)

    def _reload(self) -> None:
        from dizel_core.config_manager import ConfigManager
        from dizel_core.mcp import config as mcp_config
        from dizel_core.mcp.manager import get_manager

        manager = get_manager()
        self._servers = mcp_config.load(ConfigManager.load())
        if self._servers != manager.configured():
            # The sidecar (or another shell) may have edited settings.json;
            # reconcile so this modal and the live connections agree.
            manager.apply_async(self._servers)
        self._states = {row["id"]: row for row in manager.status()}
        options = self.query_one(OptionList)
        previous = None
        if options.highlighted is not None and options.option_count:
            previous = options.get_option_at_index(options.highlighted).id
        options.clear_options()
        for server in self._servers:
            state = self._states.get(
                server["id"],
                {"state": "starting" if server.get("enabled") else "disabled", "tools": []},
            )
            options.add_option(Option(plain_text(row_text(server, state)), id=server["id"]))
        if previous:
            for index in range(options.option_count):
                if options.get_option_at_index(index).id == previous:
                    options.highlighted = index
                    break
        # Textual's OptionList does not pick a highlight when options are added
        # after mount, so keyboard actions would have no target without this.
        if options.option_count and options.highlighted is None:
            options.highlighted = 0

    def _selected(self) -> dict | None:
        options = self.query_one(OptionList)
        if options.highlighted is None:
            return None
        option = options.get_option_at_index(options.highlighted)
        return next((s for s in self._servers if s["id"] == option.id), None)

    def _persist(self, servers: list[dict]) -> None:
        from dizel_core.mcp import config as mcp_config
        from dizel_core.mcp.manager import get_manager

        try:
            saved = mcp_config.save(servers)
        except mcp_config.MCPConfigError as error:
            self.notify(str(error), severity="error")
            return
        get_manager().apply_async(saved)
        self._servers = saved
        self._reload()

    def action_toggle(self) -> None:
        server = self._selected()
        if server is None:
            return
        servers = [dict(entry) for entry in self._servers]
        for entry in servers:
            if entry["id"] == server["id"]:
                entry["enabled"] = not entry.get("enabled")
        self._persist(servers)

    def action_test(self) -> None:
        server = self._selected()
        if server is None:
            return
        self.notify(
            f"Testing {server['name']}…", severity="information", markup=False
        )
        self._test_server(server["id"], server["name"])

    @work(thread=True, exclusive=True, group="mcp-test", exit_on_error=False)
    def _test_server(self, server_id: str, server_name: str) -> None:
        """Run ``manager.test`` off the UI thread; the TUI stays responsive."""
        from dizel_core.mcp.manager import get_manager

        state: dict | None = None
        error_message = ""
        severity = "warning"
        try:
            state = get_manager().test(server_id)
        except KeyError:
            error_message = "Save the server first, then test it."
        except RuntimeError as error:
            error_message = f"Could not test {server_name}: {error}"
            severity = "error"
        except Exception as error:  # pragma: no cover - defensive
            error_message = f"Could not test {server_name}: {error}"
            severity = "error"

        if get_current_worker().is_cancelled:
            return
        try:
            self.app.call_from_thread(
                self._show_test_result, server_name, state, error_message, severity
            )
        except RuntimeError:
            # The app (and with it this screen) went away while testing.
            pass

    def _show_test_result(
        self,
        server_name: str,
        state: dict | None,
        error_message: str,
        severity: str,
    ) -> None:
        # Runs on the UI thread.  The modal may have been dismissed while the
        # worker was running, in which case its widgets must not be touched.
        if not self.is_attached:
            return
        if error_message:
            self.notify(error_message, severity=severity, markup=False)
            return
        state = state or {}
        message = f"{server_name}: {state.get('state')}"
        if state.get("error"):
            message += f" — {state['error']}"
        self.notify(
            message,
            severity="information" if state.get("state") == "ready" else "warning",
            markup=False,
        )
        self._reload()

    def action_add(self) -> None:
        self.app.push_screen(
            McpFormModal(edit=False),
            lambda payload: self._on_form(payload, edit_id=None),
        )

    def action_edit(self) -> None:
        server = self._selected()
        if server is not None:
            edit_id = server["id"]
            self.app.push_screen(
                McpFormModal(server, edit=True),
                lambda payload: self._on_form(payload, edit_id=edit_id),
            )

    def _on_form(self, payload: dict | None, edit_id: str | None = None) -> None:
        if not payload:
            return
        servers, error = merge_form_payload(self._servers, payload, edit_id)
        if servers is None:
            self.notify(error, severity="error", markup=False)
            if edit_id is None:
                # The form's dismiss callback runs before the screen is popped,
                # so pushing a replacement here would be popped instead.  Wait
                # one loop turn, then reopen the add form with the attempted
                # values so nothing the user typed is lost.
                self.call_later(self._reopen_add_form, payload)
            else:
                self._reload()
            return
        self._persist(servers)

    def _reopen_add_form(self, payload: dict) -> None:
        if not self.is_attached:
            return
        self.app.push_screen(
            McpFormModal(payload, edit=False),
            lambda again: self._on_form(again, edit_id=None),
        )

    def action_delete(self) -> None:
        server = self._selected()
        if server is None:
            return

        def _confirm(ok: bool) -> None:
            if ok:
                self._persist([s for s in self._servers if s["id"] != server["id"]])

        self.app.push_screen(
            ConfirmModal(
                "Remove server",
                f"Remove “{server['name']}” and its tools?",
                confirm_label="Remove",
            ),
            _confirm,
        )

    def action_close_modal(self) -> None:
        self.dismiss(None)
