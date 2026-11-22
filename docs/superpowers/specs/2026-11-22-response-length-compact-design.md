# Response length: compact flyout in a Reply section — design spec

**Status:** Approved (2026-11-22). Direction chosen: option C from review.
**Date:** 2026-11-22
**Scope:** Visual refinement of the response-length control only.

---

## 1. What changes

The screenshot review found two problems: the menu row wrapped "Response length" onto two lines, and the flyout inherited the tall `.effort-panel` shell, leaving a large empty band under three small options.

- The trigger row moves out of the **Attach** group into its own **Reply** section, placed after Attach/Recent and before **Insert**.
- The row label shortens to **Length**, with the current value as its hint, so it stays on one line.
- The flyout becomes a standalone compact panel: ~236px wide, padding 6px, no header and no footer, three tight option rows (label + multiplier on one line, note underneath). No inherited `.effort-panel` height or empty space.
- Behaviour is unchanged: hover/arrow opens it to the side, Escape closes it, choosing an option calls the same `setVerbosity`, and Settings stays in sync.

---

## 2. Non-goals

No change to the store, prompt clause, context accounting, multiplier values, Settings control or backend.

---

## 3. Files

| File | Change |
|---|---|
| `inference/gui/src/components/Composer.tsx` | Move the picker into a Reply section before Insert. |
| `inference/gui/src/components/VerbosityPicker.tsx` | Shorter label; compact panel markup; new positioning width. |
| `inference/gui/src/styles/app.css` | Replace the `.effort-panel`-derived styles with a compact panel. |
| `inference/gui/scripts/smoke.js` | Assert the Reply section position and the compact flyout. |

---

## 4. Testing

Smoke: the row follows a `Reply` label and precedes `Insert`; the flyout shows three options with `0.6×` / `1.0×` / `1.5×` and their notes, with no header or "Also in Settings" footer; choosing Detailed still updates the store and Settings mirrors it.
