# Response length: chip + Settings, with a prompt clause — design spec

**Status:** Draft (2026-11-22). Direction chosen: composer-bar chip for quick changes plus a Settings → Chat control, both bound to one persisted value. Awaiting spec review.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** Move verbosity out of the `+` popover, add a system-prompt clause for Short/Detailed, and keep both controls in sync.

---

## 1. Summary

Response length currently lives at the bottom of the `+` menu and only scales the token budget (`VERBOSITY_MULTIPLIERS`). It never reaches the prompt, so "Detailed" asks for more tokens without telling the model to use them.

This moves the control to a composer-bar chip beside the mode chip, adds a matching Settings → Chat control, and gives Short and Detailed explicit clauses in the composed system prompt. **Normal stays silent** — no clause, so default behaviour is unchanged.

Both controls write one persisted value (`token_budget.verbosity`), so changing either updates the other on the next render.

---

## 2. Goals and non-goals

### Goals

1. A composer-bar chip shows the current length (`Short` / `Normal` / `Detailed`) and opens a three-item popover.
2. Settings → Chat has the same three options; changing either surface updates the other immediately.
3. Short and Detailed add a clause to `system_content()`, after the mode clause and before project rules.
4. The clause is counted by `/chat/context`, so the context panel reflects its token cost.
5. The clause follows the **effective** verbosity of a turn — a per-request override must not leave budget and prompt disagreeing.
6. No control remains in the `+` menu.

### Non-goals

- No new verbosity levels, no numeric token entry.
- No per-session-only override. Verbosity is already persisted config; keep one source.
- No change to `VERBOSITY_MULTIPLIERS` or the usage-window policy number.
- No TUI UI change beyond what already exists; `/verbosity` keeps writing the same config key.

---

## 3. Current state

| Piece | Behaviour today |
|---|---|
| `token_budget.py` | `VERBOSITY_MULTIPLIERS = {short, normal, detailed}`; affects budget only. |
| `ChatRequest.verbosity` | Optional per-request override; otherwise config is used. |
| `ChatManager.system_content()` | `base + mode_prompt + project_context`. No verbosity. |
| `ChatService.context_usage()` | Counts `system_blocks = {base, mode, project}`. |
| GUI | `+` popover section with `data-capability="chat.verbosity"`; store persists via `saveConfig({token_budget:{verbosity}})`. |
| Settings → Chat | Has a System prompt section; no response-length control. |
| `/chat/reload_settings` | Re-reads config into the live manager via `reload_sampling()` + `reload_provider()`. |

---

## 4. Design

### 4.1 Prompt clauses

In `token_budget.py`:

```python
VERBOSITY_CLAUSES = {
    "short": "Response length: Short — answer in the fewest words that cover the "
             "question; no elaboration, examples or recap unless asked.",
    "normal": "",
    "detailed": "Response length: Detailed — explain the reasoning and relevant "
                "edge cases, include examples where useful, and structure the answer.",
}
```

`normal` is deliberately empty. Both clauses must fit the 240-character collapsed cap after composition.

### 4.2 Manager state and composition

`ChatManager` gains `turn_verbosity: str = "normal"` (in-memory, not persisted separately).

- `system_content()` inserts `VERBOSITY_CLAUSES.get(self.turn_verbosity, "")` between the mode block and project rules. Empty clauses drop out, so Normal produces a byte-identical prompt to today.
- `ChatService` sets `manager.turn_verbosity` from the **effective** verbosity before composing each turn: `request.verbosity or configured value`.
- `reload_settings()` sets it from config too, so an immediately following `/chat/context` is accurate even before a message is sent.

### 4.3 Context accounting

`ChatService._system_blocks()` gains a fourth key, `verbosity`, counted from the same clause. `system` remains the sum, so nothing else breaks.

`ContextPanel` gains a `Response length` row when the count is non-zero. The row is greyed when it is zero, matching the other zero blocks.

### 4.4 GUI

**Composer chip:** a `Picker` beside the mode chip, `title="Response length"`, showing the current label with a small chevron. Three options with their existing hints. `data-capability="chat.verbosity"` moves from the `+` menu section to this control, so the manifest check still resolves. The `+` menu section is deleted entirely.

**Settings → Chat:** a `Response length` section with the same three options, bound to `useChat(s => s.verbosity)` and `setVerbosity`. No separate store action — both surfaces call the existing one.

**Store:** `setVerbosity` keeps saving config, then calls `api.reloadSettings()` so the backend clause updates without a restart. The chip updates from the store before the request resolves, so there is no waiting state.

### 4.5 TUI

No new UI. `/verbosity` already writes the same config key and the manager now derives a clause from it.

---

## 5. Files

| File | Change |
|---|---|
| `dizel_core/token_budget.py` | `VERBOSITY_CLAUSES`. |
| `dizel_core/chat_manager.py` | `turn_verbosity`; fourth block in `system_content()`. |
| `api/services/chat_service.py` | Set effective verbosity per turn and on reload; `_system_blocks` gains `verbosity`. |
| `inference/gui/src/store/chatStore.ts` | `setVerbosity` also calls `reloadSettings()`. |
| `inference/gui/src/components/Composer.tsx` | New chip; remove the `+` menu section. |
| `inference/gui/src/components/SettingsDialog.tsx` | Response length section in Chat. |
| `inference/gui/src/components/ContextPanel.tsx` | Fourth block row. |
| `inference/gui/src/services/api.ts` | `system_blocks.verbosity` type field. |
| `inference/gui/src/styles/app.css` | Chip styles (reuse `bar-chip`). |
| `tests/test_verbosity_prompt.py` (new) | Clauses, composition, effective override, cap. |
| `inference/gui/scripts/smoke.js` | Chip and Settings control stay in sync. |

---

## 6. Data flow

```
chip option / Settings option
   → store.setVerbosity(value)
   → saveConfig({token_budget:{verbosity:value}}) + POST /chat/reload_settings
   → manager.turn_verbosity = value
   → system_content() includes VERBOSITY_CLAUSES[value]
   → /chat/context system_blocks.verbosity
   → both controls re-render from the same store value
```

Per-request overrides set `turn_verbosity` for that turn only, before the prompt is built.

---

## 7. Error handling

- Unknown verbosity values normalise to `normal`; no clause, budget multiplier 1.0.
- A failed `reloadSettings()` leaves the store value changed but the backend unchanged until the next reload; the error is swallowed the same way `saveConfig` failures are today, so it cannot block typing.
- An older sidecar without `system_blocks.verbosity` renders no extra row; the panel already handles missing block fields.

---

## 8. Testing

**Unit** (`tests/test_verbosity_prompt.py`):

1. `normal` adds no clause and produces a byte-identical `system_content()`.
2. `short` and `detailed` add their clauses in the right position.
3. Collapsed prompt stays within 240 characters for every mode × verbosity.
4. A per-request override wins over config for that turn.
5. `system_blocks.verbosity` is non-zero for short/detailed and zero for normal.

**Smoke:** the chip shows `Normal`; choose `Detailed`, assert the chip reads `Detailed`, open Settings → Chat and assert the same option is selected; change it there and assert the chip follows; assert no `Response length` label remains in the `+` popover.

**Manual:** with Detailed active, confirm the context panel shows a non-zero Response length row and the Settings mode-prompt preview contains the clause.

---

## 9. Decisions

- **Two controls, one value.** Both call `setVerbosity`; neither owns state.
- **Normal emits nothing.** Keeps existing behaviour and existing prompt tests intact.
- **Clause follows the effective turn verbosity.** Prevents a Short budget with a Detailed prompt.
- **No `+` menu entry.** Removing it rather than duplicating it avoids two places drifting again.
- **Chip over mode-panel.** Length is not a mode; it applies across all four.
