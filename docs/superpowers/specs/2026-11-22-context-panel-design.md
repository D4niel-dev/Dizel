# Context panel — design spec

**Status:** Draft (2026-11-22). Awaiting spec review before the implementation plan.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** Close the recorded `context.rail` gap on the desktop app and fix the system-token under-count introduced by the prompt split. `api/` plus one new GUI panel.

---

## 1. Summary

`capabilities.json` records `context.rail` as `{"gui": null}` — the TUI has `Ctrl+R`, the desktop app has nothing. The status bar shows `used / total ctx` with a two-segment bar and a tooltip, but there is no way to see *what* is in the window.

That matters more here than in a hosted product: the window is 2048 tokens, the system prompt is a fixed ~330-token tax, and the trimmer silently drops the oldest turns. "Why did it forget?" is currently unanswerable from the UI.

This adds a **context panel** opened from the status-bar meter. It also fixes a bug this work exposed: `context_usage` counts `manager.system_prompt` only, which after the prompt split no longer includes the mode clause or project context.

---

## 2. Goals and non-goals

### Goals

1. `used` counts the composed system prompt: base + mode clause + project context.
2. The panel breaks that system block into its three parts, so a large project-context file is visible rather than hidden inside one number.
3. The panel shows conversation, reserved reply budget, trimmed tokens, and free space against the real total.
4. The panel shows the tool and plugin counts the TUI rail shows, so the two surfaces agree.
5. Clicking the status-bar meter opens it; Escape and click-outside close it.
6. `capabilities.json` stops recording this as a GUI gap.

### Non-goals

- No CPU/RAM/host section. The TUI rail has one because a terminal has no other place for it; the desktop app does not need a second monitor.
- No per-message token table. Counting each turn's tokens on open would cost more than the panel is worth; the conversation stays one number.
- No editing anything from the panel. Settings owns configuration.
- No change to the trimmer, the budget calculation, or the usage window (the 12-hour policy number is a different concept and stays in the status bar).
- No new keyboard shortcut. `Ctrl+R` belongs to the TUI; the desktop opens by click.

---

## 3. Current state

| Piece | Behaviour today |
|---|---|
| `ChatService.context_usage()` | `system = count_tokens(manager.system_prompt)` — **misses** `mode_prompt` and `project_context`. |
| Payload | `system, conversation, reserved, trimmed, used, total, percent, exact, task` |
| `StatusBar.tsx` | `used / total ctx`, `.ctx-bar` with `.ctx-fill` and `.ctx-reserved`, tooltip `System N · Conversation M · Reserved R of T`. |
| TUI `context_panel.py` | CONTEXT (tokens, meter, budget), TOOLS, PLUGINS, HOST. |
| `capabilities.json` | `context.rail` → `gui: null`. |

---

## 4. Design

### 4.1 API

`context_usage()` splits the system prompt and keeps `system` as the sum, so existing callers stay correct:

```python
base = count_tokens(manager.system_prompt)
mode = count_tokens(manager.mode_prompt)
project = count_tokens(manager.project_context)
system = base + mode + project
```

Payload gains one object:

```json
"system_blocks": { "base": 210, "mode": 118, "project": 0 }
```

Everything else is unchanged. `used` now includes the mode clause and project context, which is the fix.

### 4.2 Panel

A popover anchored above the status-bar meter, not a permanent rail. A rail would compete with the artifacts rail for the same edge and would be empty most of the time; the meter is where the user already looks.

**Layout**, top to bottom:

1. **Stacked bar** — one row, segments in the order they enter the prompt: base, mode, project, conversation, reserved. Free space is the unfilled remainder.
2. **Rows** — `System prompt`, `Mode instructions`, `Project context`, `Conversation`, `Reply reserve`, each with a token count and a share of the total. Rows under the system block are indented one level.
3. **Footer** — `N of M tokens · K free`, plus `trimmed: N` only when the trimmer has dropped something, and the classifier note when `task` is set.
4. **Counts** — `Tools: 6 enabled`, `Plugins: N`. Read from state the app already has.

Zero-token blocks are still listed, greyed, so the shape of the prompt does not change as the user edits settings.

### 4.3 Trigger

The `.ctx-bar` and the `used / total` text become one button. It gains `aria-haspopup`, `aria-expanded`, and a title that no longer has to carry the whole breakdown (the tooltip stays for hover, the panel is for detail).

### 4.4 Empty and loading states

- Model not loaded: the panel shows `Estimated` with the local fallback numbers the status bar already uses, and says so.
- No session: conversation is 0, and the row reads `0` rather than disappearing.

---

## 5. Files

| File | Change |
|---|---|
| `api/services/chat_service.py` | Split the system block; add `system_blocks`. |
| `api/routes/chat.py` | No change — `/chat/context` already returns the dict. |
| `inference/gui/src/components/ContextPanel.tsx` (new) | The panel. |
| `inference/gui/src/components/StatusBar.tsx` | Meter becomes the trigger; passes the payload down. |
| `inference/gui/src/styles/app.css` | Panel and stacked-bar styles. |
| `capabilities.json` | `context.rail` → gui label + control check. |
| `tests/test_context_usage.py` (new) | The split and the sum. |
| `inference/gui/scripts/smoke.js` | Opens the panel, asserts the blocks sum to `used`. |

---

## 6. Data flow

```
manager.system_prompt / mode_prompt / project_context
        │  count_tokens() each
        ▼
GET /chat/context  →  { system, system_blocks, conversation, reserved, trimmed, used, total }
        │
        ▼
StatusBar (meter)  ──click──►  ContextPanel
        │                          │
        └── polled every few s ────┘ (panel reads the same store value)
```

The panel does not poll on its own; it renders the value the status bar already polls.

---

## 7. Error handling

- `system_blocks` missing (older sidecar): the panel falls back to a single `System prompt` row using `system`. No crash, no empty panel.
- `total` of 0 (model not loaded): percentages become 0 and the footer says `Estimated`.
- Token counting can be slow on a large project context; it already runs per status poll, and this change does not add a call — it splits one count into three.

---

## 8. Testing

**Unit** (`tests/test_context_usage.py`):

1. `system == base + mode + project` for a manager with all three set.
2. A manager with only `system_prompt` reports `mode: 0`, `project: 0` and an unchanged `system`.
3. `used == system + conversation` still holds.

**Smoke:** open the panel from the meter; assert `system_blocks.base + mode + project === system`, that the panel lists five rows, and that Escape closes it.

**Manual:** with a project `DIZEL.md` present, confirm the project row is non-zero and the total moves when the file changes.

---

## 9. Decisions

- **Popover, not a rail.** The artifacts rail already owns the right edge; a second permanent rail would halve both.
- **`system` stays the sum.** Changing its meaning would silently break the status bar and the TUI, which both read it.
- **No per-message table.** Cost without a decision to make.
- **No host section.** Terminal-only concern.
