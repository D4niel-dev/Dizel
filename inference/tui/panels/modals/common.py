"""Shared modal building blocks: pickers, confirmations and text prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option


@dataclass
class PickerItem:
    id: str
    label: str
    detail: str = ""
    trailing: str = ""
    active: bool = False
    disabled: bool = False
    keywords: str = ""
    value: str | None = None


def plain_text(value: object) -> Content:
    """Literal content that never interprets brackets as markup."""
    return Content.from_text(str(value), markup=False)


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def set_error(screen: ModalScreen, message: str) -> None:
    try:
        error = screen.query_one(".modal-error", Static)
    except Exception:
        return
    error.update(plain_text(message))
    error.styles.display = "block" if message else "none"


def _row(item: PickerItem) -> Content:
    marker = "●" if item.active else " "
    trailing = f"  ({item.trailing})" if item.trailing else ""
    label_width = max(20, 58 - len(trailing))
    parts: list[tuple[str, str]] = [
        (f"{marker} ", ""),
        (clip(item.label, label_width), "bold"),
    ]
    if trailing:
        parts.append((trailing, ""))
    if item.detail:
        parts.append((f"\n  {clip(item.detail, 58)}", ""))
    return Content.assemble(*parts)


class NavInput(Input):
    """Input that hands arrow keys back to the owning picker."""

    BINDINGS = [
        Binding("up", "nav_up", "Previous", show=False),
        Binding("down", "nav_down", "Next", show=False),
    ]

    class Navigate(Message):
        def __init__(self, direction: int) -> None:
            super().__init__()
            self.direction = direction

    def action_nav_up(self) -> None:
        self.post_message(self.Navigate(-1))

    def action_nav_down(self) -> None:
        self.post_message(self.Navigate(1))


class PickerModal(ModalScreen[str | None]):
    """Searchable, keyboard-first list. Dismisses with the chosen item id."""

    BINDINGS = [Binding("escape", "cancel", "Close")]

    def __init__(
        self,
        title: str,
        items: Iterable[PickerItem],
        subtitle: str = "",
        placeholder: str = "Search…",
        footer: str = "↑↓ move · enter select · esc close",
        empty_text: str = "Nothing to show",
        allow_custom: bool = False,
        wide: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._subtitle = subtitle
        self._placeholder = placeholder
        self._footer = footer
        self._empty_text = empty_text
        self._allow_custom = allow_custom
        self._wide = wide
        self._items: list[PickerItem] = list(items)
        self._by_id: dict[str, PickerItem] = {}
        self._last_query: str | None = None

    def compose(self) -> ComposeResult:
        card_classes = "modal-card picker-card"
        if self._wide:
            card_classes += " -wide"
        with Vertical(classes=card_classes):
            with Horizontal(classes="modal-head"):
                yield Static(plain_text(self._title), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            if self._subtitle:
                yield Static(plain_text(self._subtitle), classes="modal-sub")
            yield NavInput(placeholder=self._placeholder, classes="modal-input")
            yield OptionList(classes="modal-list")
            yield Static(self._footer, classes="modal-footer")

    def on_mount(self) -> None:
        self._populate("")
        self._input.focus()
        self._apply_density()
        self._after_mount()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_density()

    def _apply_density(self) -> None:
        compact = self.size.height < 30
        try:
            self.query_one(".modal-card").set_class(compact, "-dense")
        except Exception:
            pass

    @property
    def _input(self) -> NavInput:
        return self.query_one(NavInput)

    @property
    def _list(self) -> OptionList:
        return self.query_one(".modal-list", OptionList)

    def _after_mount(self) -> None:
        """Hook for subclasses that need to start work after the first paint."""

    def set_items(
        self,
        items: Iterable[PickerItem],
        subtitle: str | None = None,
        empty_text: str | None = None,
    ) -> None:
        self._items = list(items)
        if subtitle is not None:
            self._subtitle = subtitle
            try:
                self.query_one(".modal-sub", Static).update(plain_text(subtitle))
            except Exception:
                pass
        if empty_text is not None:
            self._empty_text = empty_text
        self._populate(self._input.value)

    def _populate(self, query: str) -> None:
        options = self._list
        preserve = query == self._last_query
        previous = None
        if preserve and options.highlighted is not None and options.option_count:
            option = options.get_option_at_index(options.highlighted)
            previous = option.id
        self._last_query = query

        options.clear_options()
        self._by_id.clear()

        matches = [
            item
            for _, _, item in sorted(
                (
                    (self._score(item, query), index, item)
                    for index, item in enumerate(self._items)
                    if self._matches(item, query)
                ),
                key=lambda entry: (entry[0], entry[1]),
            )
        ]
        for item in matches:
            self._by_id[item.id] = item
            options.add_option(
                Option(_row(item), id=item.id, disabled=item.disabled)
            )

        if not matches:
            hint = self._empty_text
            if self._allow_custom and query.strip():
                hint = f"Press enter to use “{query.strip()}”"
            options.add_option(Option(plain_text(f"  {hint}"), id="__empty__", disabled=True))
            options.highlighted = None
            return

        target = None
        if previous:
            for index, item in enumerate(matches):
                if item.id == previous and not item.disabled:
                    target = index
                    break
        if target is None:
            for index, item in enumerate(matches):
                if not item.disabled:
                    target = index
                    break
        options.highlighted = target
        if target is not None:
            options.scroll_to_highlight()

    @staticmethod
    def _matches(item: PickerItem, query: str) -> bool:
        if not query:
            return True
        haystack = (
            f"{item.id} {item.label} {item.detail} {item.trailing} {item.keywords}"
        ).lower()
        return query.lower() in haystack

    @staticmethod
    def _score(item: PickerItem, query: str) -> int:
        """Prefer label matches over matches found only in help text or aliases."""
        if not query:
            return 0
        q = query.lower().lstrip("/")
        label = item.label.lower().lstrip("/")
        ident = item.id.lower().lstrip("/")
        if label.startswith(q):
            return 0
        if q in label:
            return 1
        if ident.startswith(q):
            return 2
        if q in ident:
            return 3
        if q in item.detail.lower():
            return 4
        if q in item.trailing.lower():
            return 5
        return 6

    def _value_for(self, option_id: str) -> str:
        item = self._by_id.get(option_id)
        if item is not None and item.value is not None:
            return item.value
        return option_id

    def _move(self, direction: int) -> None:
        options = self._list
        if options.option_count == 0:
            return
        current = options.highlighted
        index = 0 if current is None else current + direction
        while 0 <= index < options.option_count:
            option = options.get_option_at_index(index)
            if not option.disabled:
                options.highlighted = index
                options.scroll_to_highlight()
                return
            index += direction

    def _choose(self) -> None:
        options = self._list
        if options.highlighted is None:
            query = self._input.value.strip()
            if self._allow_custom and query:
                self.dismiss(query)
            return
        option = options.get_option_at_index(options.highlighted)
        if option.id and option.id != "__empty__" and option.id in self._by_id:
            self.dismiss(self._value_for(option.id))

    @on(NavInput.Navigate)
    def _on_navigate(self, event: NavInput.Navigate) -> None:
        event.stop()
        self._move(event.direction)

    @on(Input.Changed)
    def _on_query(self, event: Input.Changed) -> None:
        self._populate(event.value)

    @on(Input.Submitted)
    def _on_submit(self, event: Input.Submitted) -> None:
        event.stop()
        self._choose()

    @on(OptionList.OptionSelected)
    def _on_option(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id and event.option.id in self._by_id:
            self.dismiss(self._value_for(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmModal(ModalScreen[bool]):
    """A small yes/no gate for destructive actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("n", "cancel", "No", show=False),
        Binding("y", "confirm", "Yes", show=False),
    ]

    def __init__(
        self,
        title: str,
        message: str,
        confirm_label: str = "Confirm",
        danger: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._danger = danger

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card -compact"):
            with Horizontal(classes="modal-head"):
                yield Static(plain_text(self._title), classes="modal-title")
                yield Static("y / n", classes="modal-kbd")
            yield Static(plain_text(self._message), classes="modal-body")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="confirm-cancel")
                yield Button(
                    self._confirm_label,
                    id="confirm-ok",
                    variant="error" if self._danger else "primary",
                )

    def on_mount(self) -> None:
        self.query_one("#confirm-cancel", Button).focus()

    @on(Button.Pressed, "#confirm-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-ok")
    def _confirm_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)


class TextPromptModal(ModalScreen[str | None]):
    """One-line text entry with inline validation."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        title: str,
        label: str,
        placeholder: str = "",
        value: str = "",
        password: bool = False,
        help_text: str = "",
        submit_label: str = "Save",
        validate: Callable[[str], str | None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._label = label
        self._placeholder = placeholder
        self._value = value
        self._password = password
        self._help = help_text
        self._submit_label = submit_label
        self._validate = validate

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-card -compact"):
            with Horizontal(classes="modal-head"):
                yield Static(plain_text(self._title), classes="modal-title")
                yield Static("esc", classes="modal-kbd")
            yield Static(plain_text(self._label), classes="field-label")
            yield Input(
                value=self._value,
                placeholder=self._placeholder,
                password=self._password,
                classes="modal-input",
                id="prompt-value",
            )
            if self._help:
                yield Static(plain_text(self._help), classes="field-help")
            yield Static("", classes="modal-error")
            with Horizontal(classes="modal-actions"):
                yield Button("Cancel", id="prompt-cancel")
                yield Button(self._submit_label, id="prompt-save", variant="primary")

    def on_mount(self) -> None:
        self.query_one(".modal-error", Static).styles.display = "none"
        entry = self.query_one("#prompt-value", Input)
        entry.focus()
        entry.cursor_position = len(entry.value)

    @on(Button.Pressed, "#prompt-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#prompt-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#prompt-value")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        value = self.query_one("#prompt-value", Input).value.strip()
        if not value:
            set_error(self, "A value is required.")
            return
        if self._validate:
            message = self._validate(value)
            if message:
                set_error(self, message)
                return
        set_error(self, "")
        self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)
