# Fast reply speeds — design spec

**Status:** Draft (2026-11-22). Direction revised: cost and prompt grow with speed. Awaiting spec review.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** Add Fast → Turbo → Ultrafast as a nested control under the Fast mode, mirroring Thinking effort. Backend, GUI, TUI and the public demo.

---

## 1. Summary

Thinking has five efforts that change a prompt clause, temperature and budget. Fast has no such control. This adds three reply speeds: **Fast** (today's behaviour), **Turbo**, and **Ultrafast**.

Each higher speed adds a longer, more insistent speed clause and **raises** the output budget, so token cost and context usage increase with speed, as requested. Lower speeds keep replies short.

**This does not change the model's tokens/sec.** A local GGUF runs at a rate set by the model and hardware. What changes is how much of the window and how many tokens a reply is allowed, plus how strongly the prompt asks for a prompt answer. The UI says this plainly.

---

## 2. Goals and non-goals

### Goals

1. Fast gains three speeds; the default is today's Fast, so existing users see no change until they touch the control.
2. Each speed changes the mode clause and the reply budget. **Higher speed = longer clause and larger budget.**
3. Token cost and context usage increase monotonically: `fast < turbo < ultrafast`.
4. Temperature and top-p are **unchanged** across speeds.
5. The GUI chip shows **only the tier name** — `Fast`, `Turbo` or `Ultrafast` — and each tier has its own colour.
6. TUI `/mode fast turbo` and the demo's mode control expose the same three values.
7. Prompt length stays under the 1.5 tier's 240-character system-prompt cap at every speed.

### Non-goals

- No provider routing and no monetary price increase. "Cost" here means token budget and context usage, not dollars.
- No change to tokens/sec, no speculative decoding, no quantisation switch.
- No speed control under Planning, Coding or Thinking.
- No persistence of speed beyond the session (effort is in-memory too).
- No new capability entry. This is a nested control, not a new surface action.

---

## 3. Current state

| Piece | Behaviour today |
|---|---|
| `token_budget.py` | `THINKING_EFFORTS` + labels, temp delta, budget factor, clauses; `_thinking_profile()` derives five profiles per family. |
| `get_model_profile()` | `(model_name, mode_name, thinking_effort="medium")`. |
| `ChatManager.apply_profile()` | Takes model, mode, effort; sets profile, mode clause and sampling. |
| `ProfileRequest` | `model_name`, `mode_name`, `thinking_effort`. |
| GUI | `ModePicker` opens a nested slider for Thinking; store holds `thinkingEffort`; `modeLabel(mode, effort)` and `modeColor(mode, effort)` drive the chip. |
| Fast | `MODE_COLORS.Fast = "#38bdf8"`; clause `Mode: Fast — lead with the result; skip preamble, filler, and closing summaries.` |

---

## 4. Design

### 4.1 Backend

Mirror the effort constants in `token_budget.py`:

```python
FAST_SPEEDS = ("fast", "turbo", "ultrafast")
FAST_SPEED_LABELS = {"fast": "Fast", "turbo": "Turbo", "ultrafast": "Ultrafast"}
_FAST_SPEED_BUDGET_FACTOR = {"fast": 1.0, "turbo": 1.15, "ultrafast": 1.30}
_FAST_SPEED_CLAUSES = {...}
```

Clauses get longer and more insistent with speed. The `fast` clause is the existing text verbatim, so default behaviour is unchanged. Drafts, each to be verified against the 240-character cap with `_v15_system_prompt()`:

```
turbo      Mode: Fast (Turbo) — answer promptly: lead with the result, keep sentences
           short, and skip preamble, restating and closing summaries.
ultrafast  Mode: Fast (Ultrafast) — answer immediately: only the direct result, in as
           few words as correctness allows; no preamble, no recap, no bullet scaffolding.
```

`normalize_fast_speed()` accepts `fast`, `turbo`, `ultrafast`, plus `normal`/`default` → `fast` and `ultra` → `ultrafast`. Unknown values → `fast`.

`_fast_profile(base, prompt, speed, family_label)` returns a `ModelProfile` with the speed clause and `base.budget_multiplier * factor`. It copies `base.sampling` **unchanged** and labels itself `Dizel · Fast · Turbo`.

`get_model_profile()` gains `speed: str = "fast"`. Speed is ignored when `mode_name != "Fast"`.

`ChatManager.apply_profile()` gains `speed: str = "fast"` and forwards it. The mode clause still comes from `split_profile_prompt(profile.system_prompt)`, so the Settings mode block and context panel pick up the new clause without extra work.

### 4.2 API and store

`ProfileRequest` gains `speed: Optional[str] = None`; `ChatService.apply_profile` forwards it. Callers omitting the field keep today's behaviour.

The GUI store gains `fastSpeed: "fast" | "turbo" | "ultrafast"` and `setFastSpeed()`, which calls `applyProfile(..., speed)` and updates `modePrompt` from the response — the path `setThinkingEffort` already uses.

### 4.3 GUI chip and control

`services/modes.ts` gains `FAST_SPEED_COLORS`:

| Tier | Colour |
|---|---|
| Fast | `#38bdf8` (existing) |
| Turbo | `#4fd6be` |
| Ultrafast | `#fb923c` |

`modeLabel(mode, effort, speed)` returns **only** the tier name when the mode is Fast: `Turbo` rather than `Fast · Turbo`. Thinking keeps `Thinking · High`.

`modeColor(mode, effort, speed)` returns the tier colour for Fast, otherwise its existing behaviour. The chip's icon stays the Fast bolt; only text and colour change.

`ModePicker` gets a second nested panel, placed like `.effort-panel`. The Fast row opens it; the three stops preview and then call `onSpeed`.

**Copy under the control:** *"Higher speeds allow a larger reply budget and cost more tokens. Tokens per second is unchanged."*

### 4.4 TUI and demo

TUI: `/mode fast turbo`, `/mode fast ultrafast`; `/mode fast` resets. The footer badge shows only the tier name in the tier colour.

Demo: the chip shows the tier name in its colour, with a three-button speed row mirroring the effort control.

---

## 5. Files

| File | Change |
|---|---|
| `dizel_core/token_budget.py` | Speed constants, normaliser, `_fast_profile`, `_FAST_PROFILES`, extended `get_model_profile`. |
| `dizel_core/chat_manager.py` | `apply_profile(..., speed)` pass-through. |
| `api/routes/chat.py` | `ProfileRequest.speed`. |
| `api/services/chat_service.py` | Forward speed. |
| `inference/gui/src/services/modes.ts` | Tier colours, `modeLabel`/`modeColor` speed parameter. |
| `inference/gui/src/store/chatStore.ts` | `fastSpeed`, `setFastSpeed`. |
| `inference/gui/src/components/ModePicker.tsx` | Nested speed panel, `onSpeed` prop. |
| `inference/gui/src/styles/app.css` | Speed panel styles. |
| `inference/tui/` (mode command + footer) | Chained speed parsing and coloured badge. |
| `docs/js/demo.js` + demo CSS | Tier chip and three-stop row. |
| `tests/test_fast_speeds.py` (new) | Profiles, cap, aliases, monotonic cost, pass-through. |
| `inference/gui/scripts/smoke.js` | Speed picker changes the chip text and colour. |

---

## 6. Data flow

```
ModePicker speed stop
   → store.setFastSpeed(speed)
   → POST /chat/apply_profile { mode: "Fast", speed }
   → ChatManager.apply_profile → get_model_profile(..., speed)
   → profile { clause, budget }
   → response { mode_prompt, system_prompt }
   → store.modePrompt  → Settings mode block + context panel
   → chip shows tier name in tier colour
```

---

## 7. Error handling

- An unknown speed normalises to `fast`; never raises.
- Speed on a non-Fast mode is ignored, not rejected.
- An older sidecar ignoring the field still works: the GUI keeps its local speed and replies stay at default brevity.
- A failed `applyProfile` leaves the previous speed selected, matching effort behaviour.

---

## 8. Testing

**Unit** (`tests/test_fast_speeds.py`):

1. Three speeds exist; `fast` reproduces today's clause and budget exactly.
2. Budget factors increase strictly: `fast < turbo < ultrafast`.
3. Mode clauses are non-decreasing in length across speeds.
4. Temperature and top-p are identical to the base Fast profile at every speed.
5. Every composed prompt stays within 240 characters for each family.
6. Aliases: `normal`/`default` → `fast`, `ultra` → `ultrafast`, garbage → `fast`.
7. Speed is ignored for Thinking, Planning and Coding.

**Smoke:** open the Fast row, select Turbo, assert the chip text is exactly `Turbo` and its computed colour is the Turbo colour; assert the Settings mode-prompt block contains `(Turbo)`.

**Manual:** compare an identical reply at Fast and Ultrafast; confirm the tier chip colour, the larger token count in the context panel and the smaller used-vs-free split.

---

## 9. Decisions

- **Cost rises with speed.** Revision after review: higher tiers get a larger budget and a longer clause. Tokens/sec is still fixed; the UI says so.
- **The chip is the tier.** Only `Fast` / `Turbo` / `Ultrafast`, each with its own colour — no `Fast · Turbo` compound label.
- **No sampling changes.** Temperature does not control wall-clock time.
- **Fast is the default.** No behaviour change for anyone who does not open the control.
- **GUI, TUI and demo together.** All three already carry the mode control.
