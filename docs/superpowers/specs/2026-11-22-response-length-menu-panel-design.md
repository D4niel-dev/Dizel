# Response length in the + menu side panel — design spec

**Status:** Draft (2026-11-22). Awaiting spec review.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** Replace the composer-bar length chip with a row under Images… in the `+` menu, opening a side flyout of three explained options. No backend or store changes.

---

## 1. Summary

Response length currently lives in a composer-bar chip and in Settings → Chat. The chip takes permanent bar space for a control that is used occasionally, and its popover shows only names.

This moves the quick control into the `+` menu, directly under **Images…** in the Attach group. The row opens a side flyout — the same portal pattern as Thinking effort — listing the three lengths with a short description and their budget multiplier each. The prompt clause, context-panel row and Settings control stay exactly as they are.

---

## 2. Goals and non-goals

### Goals

1. The `+` menu has a `Response length` row immediately below Images…, showing the current choice as its hint.
2. Hovering or clicking the row's arrow opens a flyout to the side, like Thinking effort and Fast speed.
3. The flyout lists Short / Normal / Detailed with a description and budget multiplier per option; no slider.
4. The composer-bar chip is removed, freeing that space.
5. Settings → Chat keeps its control; both surfaces still share `setVerbosity`.
6. `data-capability="chat.verbosity"` moves to the new menu row.

### Non-goals

- No change to clauses, budget multipliers, `turn_verbosity`, or context accounting.
- No draggable thumb — these are three discrete options with prose, not a range.
- No changes to the Settings control.
- No TUI or demo UI change.

---

## 3. Current state

| Piece | Behaviour today |
|---|---|
| `Composer.tsx` | `+` menu: Attach → Files…, Folder…, Images…, then Recent → Insert… |
| Composer chip | A second `Picker` after `ModePicker`, carrying `data-capability="chat.verbosity"`. |
| ModePicker | Nested effort/speed panels portalled to `document.body`, placed by hand from the trigger rect. |
| Store | `verbosity` + `setVerbosity` (saves config, reloads settings). |
| Settings → Chat | Three `persona-chip` buttons bound to the same action. |

---

## 4. Design

### 4.1 Menu row

Inserted after the Images… button, before the Recent group:

```
Response length                   Detailed ›
```

- A small icon (same family as the other rows) plus the label.
- Hint: the current label, matching the chip's old behaviour.
- An arrow button with `aria-label="Response length"` and `aria-expanded`.
- `data-capability="chat.verbosity"` on the row.
- Hovering the row or clicking the arrow opens the flyout, with the same 150 ms close delay and pinned/unpinned behaviour as `ModePicker`.

### 4.2 Side flyout

A portal on `document.body`, using the existing `.effort-panel` shell so its position, shadow and animation match the other flyouts. Three option buttons, each a full row:

```
┌──────────────────────────────────────┐
│ RESPONSE LENGTH                      │
│ ○ Short       0.6× budget            │
│   Answer only what was asked         │
│ ● Normal      1.0× budget            │
│   The default length                 │
│ ○ Detailed    1.5× budget            │
│   Reasoning, examples and caveats    │
│                                      │
│ Also in Settings → Chat.             │
└──────────────────────────────────────┘
```

- The active option is marked with a check and the mode colour convention.
- The multiplier is the real value from `VERBOSITY_MULTIPLIERS` — no invented token counts, which vary by task and input.
- Escape and click-outside close it; clicking an option applies `setVerbosity(id)` and closes the flyout but leaves the `+` menu open, so a user can keep working in it.

### 4.3 Placement logic

Reuse the measured-rect approach from `ModePicker`: read the arrow's bounding rect, prefer the right side, flip left when there is not enough room, and clamp vertically. The flyout is portalled because the `+` popover clips its own overflow.

### 4.4 What does not change

The store action, the prompt clause, `system_blocks.verbosity`, the context panel row, and the Settings section are untouched. This is a pure control relocation.

---

## 5. Files

| File | Change |
|---|---|
| `inference/gui/src/components/VerbosityPicker.tsx` (new) | Menu row + portalled option flyout. |
| `inference/gui/src/components/Composer.tsx` | Remove the chip and its state; render `VerbosityPicker` after Images…; restore the `+` trigger's plain `armed` behaviour. |
| `inference/gui/src/styles/app.css` | Option-row styles; reuse `.effort-panel` shell. |
| `inference/gui/scripts/smoke.js` | `data-capability` still resolves; new row and flyout checks. |

No backend files.

---

## 6. Data flow

```
row arrow → flyout → option click
   → store.setVerbosity(id)
   → saveConfig + /chat/reload_settings   (unchanged)
   → menu hint and Settings selection update from the same store value
```

---

## 7. Error handling

- A failed save keeps the local selection, as today; no flyout error state.
- A narrow window flips the flyout to the left of the arrow, matching `ModePicker`.
- The `+` menu closing while the flyout is open closes both through the existing `open` effect.

---

## 8. Testing

**Smoke:**

1. `data-capability="chat.verbosity"` resolves on a row inside the `+` menu, not the composer bar.
2. The row sits after Images… and before the Insert group.
3. The row's hint shows the current length.
4. Opening the flyout shows three options with `0.6×`, `1.0×`, `1.5×` and their descriptions; choosing Detailed updates the store and the row hint.
5. Settings still mirrors the choice.

**Manual:** open `+`, hover the new row, confirm the flyout opens to the side without closing the menu, and that Escape closes only the flyout.

---

## 9. Decisions

- **Row under Images…**, inside Attach, matching the requested position.
- **Options with prose, not a slider.** There are three discrete values; a thumb would imply a continuum.
- **Real multipliers in the UI.** Token counts vary by task and input; the multiplier is the honest number.
- **Settings stays.** It was added deliberately as the durable home; the menu row is the quick path.
- **No backend work.** The clause and accounting landed in the previous change and need no adjustment.
