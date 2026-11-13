from rich.markup import escape
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from inference.tui.theme import ACCENT, TEXT, TEXT_2, TEXT_3


class SessionPanel(Container):
    """Left sidebar: chat history with search, pin state and tags.

    Visible by default on wide terminals; the app hides it below 100 columns
    and ctrl+t overrides that either way.
    """

    DEFAULT_CLASSES = "-hidden"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._signature = None

    def compose(self) -> ComposeResult:
        with Vertical(id="session-inner"):
            with Horizontal(id="session-head"):
                yield Static("SESSIONS", id="session-header")
                yield Static(id="session-count")
            yield Static(f"[{ACCENT}]＋[/] [b {TEXT}]New chat[/]", id="session-new")
            yield Input(placeholder="⌕  Search…", id="session-search")
            yield OptionList(id="session-list")
            yield Static(id="session-footer")

    def on_mount(self):
        self._update_footer()
        self.set_interval(5.0, self.update_sessions)

    # ── Visibility ───────────────────────────────────────────────────
    def set_sidebar_visible(self, visible: bool) -> None:
        self.set_class(not visible, "-hidden")
        if visible:
            self.update_sessions(force=True)

    def toggle_panel(self) -> None:
        self.set_sidebar_visible(self.has_class("-hidden"))

    def open_panel(self) -> None:
        self.set_sidebar_visible(True)
        self.app.call_after_refresh(self.query_one("#session-search", Input).focus)

    def close_panel(self) -> None:
        self.set_sidebar_visible(False)
        try:
            self.app.query_one("#prompt-input", Input).focus()
        except Exception:
            pass

    # ── Data ─────────────────────────────────────────────────────────
    def update_sessions(self, query: str = "", force: bool = False) -> None:
        # The panel refreshes on a timer; rebuilding a hidden list or an
        # unchanged list would reset the highlight and scroll position.
        if self.has_class("-hidden") and not force:
            return

        sessions = self.app.session_bridge.get_all(query=query or None)
        current_id = self.app.session_id
        signature = tuple(
            (s["id"], s.get("title"), bool(s.get("pinned")), tuple(s.get("tags", [])))
            for s in sessions
        ) + (current_id, query)
        if signature == self._signature and not force:
            return
        self._signature = signature

        option_list = self.query_one("#session-list", OptionList)
        previous = option_list.highlighted
        previous_id = None
        if previous is not None and option_list.option_count > previous:
            previous_id = option_list.get_option_at_index(previous).id

        option_list.clear_options()
        if not sessions:
            option_list.add_option(Option("No sessions yet", id="__empty__", disabled=True))
            self._update_footer(count=0)
            self.query_one("#session-count").update("")
            return

        pinned = [s for s in sessions if s.get("pinned")]
        unpinned = [s for s in sessions if not s.get("pinned")]

        if pinned:
            option_list.add_option(
                Option(Text("PINNED", style=TEXT_3), id="__label_pinned__", disabled=True)
            )
            for session in pinned:
                option_list.add_option(Option(self._format_session(session, current_id), id=session["id"]))

        if unpinned:
            option_list.add_option(
                Option(Text("RECENT", style=TEXT_3), id="__label_recent__", disabled=True)
            )

        for session in unpinned[:20]:
            option_list.add_option(Option(self._format_session(session, current_id), id=session["id"]))

        restored = False
        if previous_id:
            for index in range(option_list.option_count):
                if option_list.get_option_at_index(index).id == previous_id:
                    option_list.highlighted = index
                    restored = True
                    break
        if not restored and option_list.option_count:
            option_list.highlighted = min(1 if pinned else 0, option_list.option_count - 1)

        self._update_footer(count=len(sessions))
        self.query_one("#session-count").update(f"[{TEXT_3}]{len(sessions)}[/]")

    ROW_CELLS = 24

    def _format_session(self, session: dict, current_id: str) -> str:
        is_current = session["id"] == current_id
        is_pinned = bool(session.get("pinned"))

        tags = session.get("tags") or []
        tag_str = f" #{tags[0]}" if tags else ""
        title = self._clip_text(session.get("title") or "Untitled",
                                max(8, self.ROW_CELLS - 2 - len(tag_str)))

        # A rail reads as "you are here" without filling the whole row with
        # colour, which is what made the list look like a stack of blocks.
        if is_current:
            marker = f"[{ACCENT}]▌[/]"
        elif is_pinned:
            marker = f"[{TEXT_3}]•[/]"
        else:
            marker = " "
        tag_markup = f"[{TEXT_3}] {escape(tag_str)}[/]" if tag_str else ""

        # One line per session. Two-line rows read well at ten sessions and turn
        # into an unscannable wall at a hundred — the preview is what you read
        # *after* you have picked a chat, not instead of picking one.
        # No explicit colour: the row must inherit, otherwise a highlighted row
        # keeps its dark-mode grey and becomes unreadable on the orange fill.
        return f"{marker} {escape(title)}{tag_markup}"

    def _clip_text(self, value: str, max_length: int) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= max_length else f"{text[: max_length - 1].rstrip()}…"

    def _update_footer(self, count: int = None) -> None:
        footer = self.query_one("#session-footer", Static)
        parts = []
        if count is not None:
            label = "session" if count == 1 else "sessions"
            parts.append(f"[{TEXT_3}]{count} {label}[/]")
        parts.append(f"[{TEXT_3}]ctrl+t  close[/]")
        footer.update("   ".join(parts))

    # ── Events ───────────────────────────────────────────────────────
    @on(events.Click, "#session-new")
    def on_new_chat(self, event: events.Click) -> None:
        self.app.action_new_session()

    @on(Input.Changed, "#session-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        self.update_sessions(query=event.value.strip(), force=True)

    @on(OptionList.OptionSelected, "#session-list")
    def on_session_selected(self, event: OptionList.OptionSelected) -> None:
        session_id = event.option.id
        if not session_id or session_id.startswith("__"):
            return
        if session_id != self.app.session_id:
            self.app.load_session_to_workspace(session_id)
            self.update_sessions(force=True)
