# Dizel TUI — design spec

The terminal client is its own surface, and as of 2026-10-29 it no longer mirrors
the desktop palette: it follows **OpenCode** — a warm near-black with a red
undertone, warm off-white type, Apple system blue as the only accent, and Apple HIG
signal colours. A terminal has different tools from a GUI (no pixels, no shadows,
no font sizes), so everything below is about what replaces those.

Reference points: **Warp** (a terminal that markets like a design product — warm,
near-monochrome, accent only for state), **Linear** (density with hierarchy;
"darkness as space"), **Ollama** (terminal-first restraint).

---

## 1. The grid is the design system

There is one unit: **the cell**. Vertical rhythm is measured in rows, never in
ad-hoc margins.

| Token | Rows | Use |
|---|---|---|
| `gap-0` | 0 | inside a component (title → preview) |
| `gap-1` | 1 | between related items (session rows) |
| `gap-2` | 2 | between groups (label → list, greeting → starters) |
| `gap-3` | 3 | between major regions (workspace → composer) |

**Rule:** no element gets a margin that isn't one of these four values. The
current stylesheet mixes `margin-top: 1`, `margin: 0 3`, `margin: 1 0 0 0`,
`padding: 0 2` and `padding: 0 1` with no system behind them — that is why the
screen reads as flat rather than calm.

Horizontal: the sidebar is a fixed `32` cells. The transcript column is capped at
`96` and centred, so long lines don't run the full width of a wide terminal.

## 2. Type hierarchy without font sizes

A terminal gives you one size. Hierarchy comes from **case, weight and tone** —
three tools, used in a fixed order:

| Role | Treatment | Example |
|---|---|---|
| Display | `bold` + `$text` | `Good afternoon.` |
| Section label | `UPPERCASE` + `$text-3` | `SESSIONS` |
| Body | `$text-2` | message text |
| Meta | `$text-3` | `/help for commands`, timestamps |
| Emphasis | `$text` (never bold inside body) | the active session title |

**Rule:** two adjacent lines never share the same role. A greeting and its
subtitle are display + meta, never display + display.

## 3. Surfaces: three levels, no more

| Level | Colour | Holds |
|---|---|---|
| Chrome | `$bg-deep` `#191616` | top bar, sidebar, status line |
| Canvas | `$bg` `#201d1d` | transcript, workspace |
| Object | `$surface` `#302c2c` | composer, cards, hover |
| Raised | `$surface-hi` `#3b3636` | **only** for transient hover/press |

Type: `$text` `#fdfcfc`, `$text-2` `#b9b5b5`, `$text-3` `#9a9898`.
Signals: `$accent` `#007aff`, `$warn` `#ff9f0a`, `$danger` `#ff3b30`,
`SUCCESS` `#30d158`.

**Rule:** selection is *not* `$surface-hi`. A selected row uses `$surface` plus an
accent rail `▌` in the first cell. Filling a whole row with the raised colour
makes a list look like a stack of blocks — the sidebar used to do exactly this.

**On the warm shift:** the desktop stays cool grey, so the two surfaces no longer
match. That is a deliberate trade — OpenCode's warmth is the point of the
reference, and a terminal is allowed to feel different from a GUI.

## 4. One accent, three jobs

`$accent` `#0ea5e9` appears only where something is **live, focused, or current**:

1. the rail on the open session
2. the composer border when focused
3. streaming/ready state in the status line

Everything else is monochrome. Amber is *work in progress*, red is *failure* —
never decoration. A screen with five blue things has no focus.

## 5. Components

**Top bar** — one row, `$bg-deep`, with a hairline `$border` rule beneath. Wordmark
left, model chip right. The version moves to the status line: it is reference
data, not identity.

**Sidebar** — 32 cells. `SESSIONS` label (uppercase, `$text-3`) with the count
right-aligned; then `+ New chat`; then search. Session rows are one line: rail,
title, optional tag. Titles `$text-2`, the open one `$text`. Pinned sessions get
`•` instead of the rail. Section labels (`PINNED`, `RECENT`) follow the same
uppercase treatment with `gap-2` above.

**Empty state** — the only screen with room to breathe, so it earns a real
composition: greeting (display), one line of meta, then three starters separated
by `gap-1`, then the hint at `gap-2`. Starters are the same row as a session:
accent glyph, body text, and they fill the composer on click.

**Composer** — three rows, `$surface`, `$border`, and `$accent` on `:focus-within`.
Prompt glyph in accent. No placeholder that repeats what the border already says.

**Launcher (pre-start)** — a control deck, not a centred menu. At wide sizes the
pyfiglet wordmark, local-first kicker, and a session ledger sit in the left
column; the surface chooser with keycap rows, selected rail, and a LOCAL footer
sits in the right. Rows read icon → title/hint → shortcut keycap → open arrow;
the icons are `>_`, `▤`, and `↺`, not emoji. At 80×24 it stacks: wordmark,
tagline, three choices, a keycap status bar, and no ledger/footer. The row
keycap uses `$accent-ink` on `$accent` so small text stays above 4.5:1. The
exit-code contract stays `1 TUI`, `2 GUI`, `3 cancel`, `4 resume`.

**Status line** — one row, `$bg-deep`, `$text-3`, separated by `·`. Order: state,
model, device, context, message count. Nothing here is interactive.

## 6. Do / don't

- **Do** let empty space do the separating. Two regions with `gap-3` and no
  divider read cleaner than two regions with a rule and `gap-1`.
- **Do** align everything to the left edge of its column. Centred text is for the
  empty state only.
- **Don't** use `$surface-hi` for anything that persists.
- **Don't** put two uppercase labels within three rows of each other.
- **Don't** add a second accent colour "just for tool calls". Use `$warn`.
- **Don't** centre the transcript. Chat is a reading surface; it reads left.

## 7. Decisions (2026-10-29)

1. **Two-line session rows — shipped.** Title at `$text`/`$text-2`, preview at
   `$text-3` clipped to the row width. Roughly ten sessions visible instead of
   thirty-one; the sidebar now reads as a list rather than a wall.
   *Note:* `margin-bottom` on `.option-list--option` has no effect (the list
   renders options on a fixed virtual grid), so separation comes from the tone
   difference between title and preview rather than a gap row.
2. **Panels are overlays — shipped.** `ctrl+r` (context) and `ctrl+a` (artifacts)
   push a `PanelOverlay`: a right-docked panel over a dimmed workspace, closed with
   `escape` or the same key. Panels are built fresh and filled after mount, so the
   layout copies stay inert and `ArtifactPanel._seen` no longer has to be shared —
   it is rebuilt from `session_bridge.current_messages` each time.
   *Gotcha:* `ContextPanel { display: none }` applies to **every** instance, so the
   copy inside the overlay was invisible until `#overlay-dock ContextPanel` was
   given `display: block`. Same class, same rule, different parent.
3. **Cool palette confirmed.** No warm shift; parity with the desktop wins.

### Correction to an earlier claim
The "6 dead columns" in the first review were **container padding**, not the
hidden panels — measured `workspace-container w=120` against `WorkspacePanel w=114`.
The panels take zero width because they are closed, which is correct.

### A rule costs a row (learned the hard way)
Textual borders are **inside** the widget's box. A `height: 1` widget with
`border-bottom: solid $border` has zero rows left for its content — the wordmark,
the `SESSIONS` label and the whole status line all silently disappeared when I
added hairlines under them. Either give the widget `height: 2`, or separate with
background contrast instead.

Since `$bg-deep` (chrome) against `$bg` (canvas) already reads as a boundary at
1-row scale, the rules were dropped rather than paid for in rows.

### Verification loop
`scripts/snap_tui.py` + `scripts/svg2png.cjs` render the real app headlessly at
150×42. Every change in this pass was checked that way, which is how the vanished
header was caught in one step instead of after a round of screenshots.

---

## 8. Structural pass: closer to the real OpenCode TUI (2026-10-29)

The palette alone was the website, not the product. The actual TUI (from
`packages/web/src/assets/lander/screenshot.png`) is **single-column**: no sidebar,
a title block at the top, tool lines prefixed `*` and `→`, and — the distinctive
bit — an **input box that carries its own footer** (`Build  Claude Opus 4.5  OpenCode Zen`)
with a dim hint bar underneath.

Applied here:

1. **No sidebar by default.** The chat is the screen; `ctrl+t` raises the session
   list. `_sessions_auto = False` at mount so a resize doesn't bring it back.
2. **Input footer.** `InputBar` grew from 3 rows to 4: prompt row, then a footer
   row with the tier in accent and the model in `$text-3`. This is why the header
   chips are now redundant — same information, twice.
3. **Warm palette** (section 3) to match.

Still to do for a true match: replace the top-bar chips with OpenCode's title block
(`# <chat title>` + right-aligned context stats), and render tool activity with the
`*` / `→` prefixes.

### Capture limitation
`app.export_screenshot()` renders only the app's viewport, and the SVG canvas is
taller than the content — so the composer and status line fall outside the image
at 150×42. Verified the footer by geometry instead
(`#input-agent` at x=37 w=5, `#input-model` at x=44 w=17, both on row 38).

---

## 9. OpenCode v2 element pass (2026-11-06)

The current OpenCode TUI reports reasoning as a compact line, uses a two-cell
gutter for tool activity, and puts context + shortcuts in the composer footer.
This pass adopts those elements without changing the warm palette or the
single-column layout:

1. **Thought.** Live turns use a left-railed `Collapsible`: open while the model
   is thinking, titled `Thinking for Ns`/`Nm Ns`/`Nh Nm`, and switched to
   `Thought for …` when the answer starts, then folded on finalize. Reloaded
   history keeps the `Thought · N chars` fallback because older turns carry no
   timing data.
2. **Tool rows.** `│ ✻ tool  running  detail`, `│ ✓ tool  done`, `│ ✗ tool
   failed` — a two-cell icon gutter with a one-line detail preview. Unified
   `diff` / `patch` fences render green additions, red deletions, and blue hunk
   headers instead of a plain syntax block.
3. **Status line.** One animated cell (`⠋⠙⠹…`) for generating/thinking/loading,
   a steady `●` at rest, and token count. The percentage lives in the composer
   footer so the top bar, status line, and footer do not say the same thing
   three times.
4. **Composer footer.** Agent/tier, model/provider, `ctx N%`, and contextual
   hints (`esc stop`, `ctrl+k commands`). Placeholders rotate every 4.5s while
   the composer is empty and idle.
5. **Sessions.** The New chat row is filled and bold; `RECENT` is always shown
   above unpinned chats, even when nothing is pinned.

---

## 10. Session tabs and resume (2026-11-06)

The top of the terminal now carries the working set, not just the current title.

1. **Tab strip.** `SessionTabs` now owns the top edge of the window; the old
   `TopBar` title/stats block is gone, and the status line/composer footer carry
   the live numbers. It uses Textual's `Tabs` widget for click/left-right
   switching, with a raised active tab and an accent underline. A fixed `×` and
   `+` at the right close the active tab and start a new one. Tab widget IDs
   carry a `session-` prefix because raw session ids begin with a timestamp
   digit, which Textual rejects as an identifier.
2. **Lifecycle.** `DizelCMDApp.open_tabs` is the source of truth. New chats and
   loaded sessions open or reuse a tab; the session panel, session manager and
   slash commands all call the same `load_session_to_workspace` path. Closing a
   tab removes only the tab; the saved session stays in the session list. A new
   chat has no file until its first message, so its messages are also cached in
   `_tab_messages`; switching away and back does not lose it. Any tab with no
   messages restores the welcome screen; the transcript takes over only after a
   prompt is sent or a saved chat with messages is opened.
3. **Safety.** Switching or closing while a reply is streaming is refused with
   a notice. Closing the last tab creates a fresh chat. Tabs last for the run;
   they are not persisted across restarts.
4. **Launcher resume.** When saved history exists, the pre-start chooser gains a
   third `Resume last chat` row with the latest title. It exits with code 4;
   `dizel.cmd` routes that to `inference.tui.main --resume`, and the app opens
   the latest session as its first tab.

---

## 11. Launcher information rail (2026-11-06)

The pre-start chooser now uses the width it has instead of leaving the extra
terminal empty:

1. **Rail below choices.** At 96+ columns, a three-block rail sits below the
   `Terminal UI` / `Desktop app` / `Resume last chat` rows. The chooser is 72
   cells wide; the rail spans it with a hairline top rule, `TABS`, `RESUME` and
   `SESSIONS` labels, and one short line each. The session count is read from
   `list_sessions()`.
2. **Responsive collapse.** Below 96 columns the rail hides, the 72-cell chooser
   stays centered, and the saved-chat count remains in the status bar. The 80×24
   layout keeps all three choices and the footer visible.
3. **Full-width status bar.** A one-row `$surface` strip anchors the screen:
   `● ready to start · N saved chats` on the left, key hints on the right. Full
   hints are used at 90+ columns; below that the compact `↑↓ move · esc` form
   avoids clipping.
4. **Color discipline.** The accent color remains reserved for the selected
   choice; the rail uses `$text-2` labels and `$text-3` body copy. Its top rule
   uses `$border`.

---

## 12. Main-window visual pass (2026-11-08)

The chat window now speaks the same control-deck language as the launcher while
keeping the warm near-black ladder and the single Apple-blue accent. This pass
supersedes the composer, selection, and welcome details in sections 3–5; the
shell, palette, and OpenCode structure stay.

1. **Tab rail.** The strip begins directly with the working set; the in-app
   `◆ DIZEL` cell was removed because the window title already carries the mark
   and the cell read as a second tab. A one-cell left inset keeps the active tab
   off the window edge. `+` is accent, `×` turns red on hover.
2. **Welcome as a composition.** A `LOCAL WORKSPACE` kicker, the pyfiglet
   wordmark, time-aware greeting, one context line, the subline, three starter
   shelves, and a keycap hint row. Three of 24 starters are sampled for each new
   welcome; hover lifts the whole row to `$surface-hi`, bolds the label, and
   brightens the glyph and `↵`. The column is 68 cells on the transcript measure.
   Full-height rhythm is `gap-1` from the kicker to the wordmark, `gap-2` into
   the greeting and the first starter, and `gap-3` before the keycap hint;
   compact keeps `gap-0/1` so 80×24 still fits.
3. **Density.** Below 34 rows the wordmark yields (`-compact`); below 26 rows the
   margins tighten (`-tiny`). At 80×24 the full onboarding still fits — kicker,
   greeting, context, subline, starters, keycaps — with `max_scroll_y = 0`.
4. **Composer deck.** A filled mode/tier chip, the model with a trailing profile
   stripped so `Dizel 1.6 Flash` never repeats `Flash`, a five-cell context
   gauge (three at compact widths), and a hint that reacts to state: `↵ send`
   while there is text, `esc stop` while generating, `shift+tab mode` when idle.
   The chip carries two cells of padding, footer groups sit three cells apart,
   and the gauge and percentage keep two spaces between them. Accent and warning
   colors on the gauge encode capacity. The prompt glyph brightens with focus.
5. **Transcript signals.** `▌ you` stays quiet; `◆ dizel` carries the mark. The
   reply meta line is `◷ <time> · <chars>`, so a finished turn is visibly
   finished. Code fences get an accent language label; thought blocks use `✻`.
6. **Selection.** Lists select with a deep-blue accent tint (`$select-bg`
   `#143048`) and bold text instead of the old orange fill or a full accent bar.
   The open session keeps its accent rail; the slash drawer and command palette
   use the same tint.
7. **Status line.** State, provider, and tokens on the left; context hints on the
   right. At 110+ columns the version joins the right side as reference data; at
   narrow widths hints switch to `^K`/`^T`.

### Verification

`scripts/svg2png.cjs` now reads the SVG `viewBox` and forces device scale 1, so
headless captures include the composer and status line instead of cropping them
on Windows at 125%. The pass was checked at 80×24, 100×30, 120×36, and 150×42,
with the empty state, a streaming turn with thought + code, hover, the command
palette, the slash drawer, and the session sidebar. Suggestion clicks, palette
open/close, drawer selection, tiny/compact classes, and welcome fit were all
asserted under `run_test`.
