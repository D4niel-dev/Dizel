# Thinking effort, mode colors, and auto-Planning — design spec

**Status:** Direction approved (2026-11-19): effort changes prompt + sampling + budget; default Medium; `/plan` keeps Planning; effort UI is a nested panel. Awaiting spec review before the implementation plan.
**Date:** 2026-11-19
**Owner:** D4niel-dev
**Scope:** Replace the Deep Think mode with five Thinking effort levels driven from the GUI mode popover and TUI `/mode`, give each mode a color and icon, and make `/plan` select Planning automatically. GUI, TUI, and the public demo.

---

## 1. Summary

Deep Think becomes the top level of Thinking instead of a separate mode. Modes become **Fast, Planning, Coding, Thinking**. Thinking gains **Low, Medium, High, X-High, Deep Think** effort levels, default **Medium**; each level changes the system-prompt clause, the temperature, and the output budget. The GUI mode popover shows a colored icon per mode; the Thinking row opens a nested five-stop slider in its own panel, and the chip shows `Thinking · <effort>` in the effort color. `/plan <task>` switches to Planning before sending and leaves it active. The TUI mirrors the model with colored glyphs, a chained effort picker, direct `/mode thinking <effort>` parsing, and an effort-colored footer badge. The demo mirrors the mode-only control.

---

## 2. Goals and non-goals

### Goals

1. One Thinking mode whose five effort levels reproduce the old Thinking→Deep Think gradient in prompt, temperature, and budget.
2. No Deep Think mode anywhere: profiles, pickers, commands, demo, tests.
3. Every mode has a distinct color; Thinking displays the color of its active effort.
4. Every mode has an icon in the pickers (GUI SVG, TUI glyph).
5. The GUI Thinking row opens a nested effort panel with a five-stop draggable slider; clicking the row applies Thinking at the current effort.
6. `/plan` selects Planning first, on both surfaces, and keeps it.
7. Effort survives mode switches and model-family changes within the session (in-memory); the demo persists it.

### Non-goals

- No per-reply effort history, no server-side reasoning-budget enforcement beyond prompt/sampling/budget.
- No change to provider models (modes remain local-model concepts).
- No settings-page duplicate of the effort control.
- No keyboard shortcut dedicated to effort (the slider and `/mode` cover it).
- No retention of the "Deep Think" mode name; `/mode deep` / `/mode deep think` are command aliases only.

---

## 3. Current state (grounded)

- **Profiles:** `dizel_core/token_budget.py` defines `_PROFILES[family][mode]` for Dizel/Mila/Generic plus the Flash/Pro copies; modes are Fast, Planning, Coding, Thinking, Deep Think. `get_model_profile(model_name, mode_name)` (`:450`) returns the profile. Dizel prompts are compact (`_DIZEL_BASE` + clause, all ≤240 chars).
- **Apply path:** `ChatManager.apply_profile(model_name, mode_name)` (`dizel_core/chat_manager.py:246`) selects the profile, stores `_active_profile`, and sets `system_prompt`; `_generate_worker` uses `_active_profile.sampling` and `budget_multiplier` (`:978-1035`). The sidecar route `POST /chat/apply_profile` (`api/routes/chat.py:194`) takes `{model_name, mode_name}`; `api.applyProfile` (`inference/gui/src/services/api.ts:480`) posts it.
- **GUI:** the mode picker is the popover in `Composer.tsx` (`:1021` region), rendered by `Picker` (`Picker.tsx`), with `.popover` styles at `app.css:2438`. Store `mode` + `setMode` live in `chatStore.ts` (`:601`, `:2751`); entries carry no effort. The chip shows the mode only. Icons available: `code`, `sparkles`, `bolt`/`listChecks` do not exist yet.
- **TUI:** `MODES` and `MODE_DETAILS` in `inference/tui/constants.py`; `ModePickerModal` (`panels/modals/pickers.py:135`); `ModeCommand` (`commands/builtins/mode.py`) parses mode names only; `DizelCMDApp.set_mode` applies the family profile (`app.py:649`); the footer badge `#input-agent` (`panels/input_bar.py:228`) is an accent badge fed by `response_label`.
- **Pickers:** `PickerItem` (`panels/modals/common.py:19`) has id/label/detail/trailing/active/disabled; `_row` renders a marker + bold label + detail, no colors/glyphs.
- **Plan:** GUI `runSlash("plan")` sends the drafting prompt (`chatStore.ts:1299` region); TUI `PlanCommand` (`commands/builtins/plan.py`) uses `send_composed`; neither changes the mode.
- **Demo:** `docs/js/demo.js` and `docs/demo.html` mirror the app's mode chip/popover and persist `mode` in localStorage; saved "Deep Think" is currently valid.

---

## 4. Design

### 4.1 Effort levels

Keys: `low`, `medium`, `high`, `xhigh`, `deep`. Display labels: **Low, Medium, High, X-High, Deep Think**. Default `medium`.

| Key | Label | Temp Δ | Budget × | Color |
|---|---|---|---|---|
| `low` | Low | −0.05 | 0.75 | `#60a5fa` |
| `medium` | Medium | 0 | 1.00 | `#a78bfa` |
| `high` | High | +0.03 | 1.15 | `#f472b6` |
| `xhigh` | X-High | +0.06 | 1.30 | `#fb923c` |
| `deep` | Deep Think | +0.10 | 1.40 | `#f43f5e` |

Temperature = family Thinking-Medium temperature + Δ, clamped to `[0.0, 1.0]`, rounded to 2 decimals. Budget = family Thinking-Medium `budget_multiplier` × factor, rounded half-up to 2 decimals. `top_k`, `top_p`, and `repetition_penalty` stay at the family Thinking-Medium values. Dizel Medium is today's Thinking profile (`temp 0.40`, budget `1.35`), so Deep lands at `temp 0.50`, budget `1.89` — today's Deep Think behavior. Examples: High `0.43 / 1.55`; X-High `0.46 / 1.76`.

### 4.2 Effort prompts

One clause per level, appended to the family Thinking base:
- Dizel: `_DIZEL_BASE + " " + clause` (collapsed lengths below; all ≤ the 1.5 240-char cap).
- Mila/Generic: `<family base prompt> + "\n" + clause` (uncapped).

Clauses (exact):

```
low:    Mode: Thinking (low) — think briefly inside <think>…</think>, then answer directly; skip extended analysis.
medium: Mode: Thinking — reason inside <think>…</think>, check assumptions and edge cases, then give a clean answer outside it.
high:   Mode: Thinking (high) — inside <think>…</think>, compare approaches and check edge cases; then answer cleanly outside it.
xhigh:  Mode: Thinking (x-high) — inside <think>…</think>, decompose, weigh alternatives, and check failure modes; then answer in a clear structure.
deep:   Mode: Thinking (deep) — inside <think>…</think>, decompose, test assumptions, compare alternatives, and check failure modes; answer in sections.
```

Collapsed Dizel prompt lengths: Low 199, Medium 211, High 213, X-High 232, Deep 236. A test enforces ≤240 and confirms `_v15_system_prompt` passes each through unchanged.

### 4.3 Profiles and apply API

- New `THINKING_EFFORTS` tuple, label/delta/factor tables, and `_THINKING_EFFORT_CLAUSES` in `dizel_core/token_budget.py`.
- Build effort profiles per family after `_PROFILES` is defined: `_THINKING[family][effort] -> ModelProfile`. Assign `_PROFILES[family]["Thinking"] = _THINKING[family]["medium"]`. Labels: `"Dizel · Thinking · High"`, `"Mila · Thinking · Deep Think"`, etc.
- Remove every `"Deep Think"` entry from `_PROFILES` (all families); the mode key set becomes `{Fast, Planning, Coding, Thinking}`. Remove `_DEEP_THINK_MODE_SUFFIX` (its intent lives in the deep clause).
- `get_model_profile(model_name, mode_name, thinking_effort="medium")`: for `mode_name == "Thinking"`, select the effort profile (unknown effort → `medium`); other modes ignore the argument. Missing family/mode still returns `_DEFAULT_PROFILE`.
- `ChatManager.apply_profile(model_name, mode_name, thinking_effort="")` selects via the new argument, stores `_active_thinking_effort` (normalized key), and keeps `_active_profile`.
- `ProfileRequest` gains `thinking_effort: Optional[str] = None`; the route forwards it; `ChatService.apply_profile` accepts it; `api.applyProfile(variant, mode, effort = "medium")` sends it.
- `/models/profiles` is unchanged and now naturally lists four modes.

### 4.4 TUI

- `inference/tui/constants.py`: `MODES = ("Fast", "Planning", "Coding", "Thinking")`; `THINKING_EFFORTS` (keys), `THINKING_EFFORT_LABELS`, `THINKING_EFFORT_COLORS`, `MODE_COLORS` (Fast `#38bdf8`, Planning `#f59e0b`, Coding `#34d399`), and `MODE_GLYPHS` (`Fast "↯"`, `Planning "☰"`, `Coding "{}"`, `Thinking "✦"`). `MODE_DETAILS` loses Deep Think; Thinking notes the current effort.
- `DizelCMDApp`: `active_thinking_effort = reactive("medium")`; `set_mode(mode, thinking_effort=None, announce=True)` applies the effort for Thinking (defaulting to the current one); `set_thinking_effort(effort, announce=True)` normalizes, stores, and re-applies when Thinking; `response_label` returns `f"Thinking · {label}"` for Thinking and the mode otherwise; `response_color` returns the mode color (effort color for Thinking); `_apply_profile` passes the effort.
- `ModeCommand`: help/usage/examples updated. No-args opens `ModePickerModal`; choosing Thinking applies Thinking at the current effort and then opens `ThinkingEffortPickerModal` to refine (cancelling leaves Thinking active). Direct forms: `/mode thinking` (current effort), `/mode thinking high`, `/mode thinking x-high`, `/mode deep` and `/mode deep think` (aliases for Thinking + `deep`). Unknown mode/effort returns the available values.
- `ThinkingEffortPickerModal` (pickers.py): five rows, each with a colored `●` glyph, label, short detail (e.g. `temp 0.43 · budget ×1.15`), current marker; dismisses with the effort key.
- `ModePickerModal`: rows get mode glyphs and colors; Thinking's trailing text shows the current effort label.
- `PickerItem` gains `glyph: str = ""` and `color: str = ""`; `_row` renders the glyph before the label in `color` and styles the label bold in `color` when set (other pickers are unchanged with empty fields).
- Footer: `#input-agent` uses `response_color` as its background with `ACCENT_INK` text, and labels `Thinking · High`; `active_thinking_effort` joins the watchers in `input_bar.py` and `status_bar.py`. The debug status line shows `mode` and `effort`.

### 4.5 GUI

- `src/services/modes.ts` (new): `ThinkingEffort` union, `THINKING_EFFORTS` (id/label/color/detail), `MODE_COLORS`, `modeColor(mode, effort)`, `modeLabel(mode, effort)` (`Thinking · High` / `Deep Think` case), and `normalizeEffort(value)`.
- `Icon.tsx`: add `bolt` (lightning) and `listChecks` (three check rows); reuse `code` and `sparkles` for Coding/Thinking.
- Store: `thinkingEffort: ThinkingEffort` initial `"medium"`; `setThinkingEffort(effort)` sets it and re-applies the profile when the mode is Thinking; `setMode(mode)` applies with the current effort; the family-change effect re-applies with the effort. `api.applyProfile(family, mode, effort)`.
- `/plan`: `runSlash("plan", arg)` awaits `setMode("Planning")` before `send(...)`, and leaves Planning active. Same for the TUI `PlanCommand` (`app.set_mode("Planning", announce=False)` first).
- Chip: label `modeLabel(mode, thinkingEffort)`; color = `modeColor(mode, thinkingEffort)` via an inline `--mode-color`/color style; the icon uses the mode icon and color; `is-armed` still applies when not Fast. Trigger title remains "Response mode".
- Popover: each row is `icon + name` with the mode color; Thinking shows `Thinking + effort tag + ›`. The Thinking row's main button applies Thinking at the current effort and closes; the `›` button, hovering the row, or focusing the tag opens the **nested effort panel**.
- Nested effort panel: a `.effort-panel` rendered inside the Picker's relative container (not inside the scrolling `.popover`), positioned beside the Thinking row (measured `top`, right edge; flips to the left when it would overflow the window). It contains a header "Thinking effort", a live label in the effort color, a five-stop track (`role="slider"`, `aria-valuemin=0`, `aria-valuemax=4`, `aria-valuenow`, `aria-valuetext`), five accessible stop buttons with their colors, and a short detail line. Pointer drag and click map the x position to the nearest stop; ArrowLeft/Right/Home/End move it; changes preview the label/color. Applying a stop calls `setThinkingEffort(effort)` and, when the mode is not Thinking, `setMode("Thinking")` with that effort, so one interaction selects the mode and the level. Hover-close uses a 150 ms delay so the pointer can enter the panel. Escape closes only the panel (stops propagation); outside click closes both.
- The nested panel is not portal-based, so the Picker's outside-click logic keeps working; the mode popover gets `overflow: visible` while the panel is open (its list is four rows).

### 4.6 Colors and icons

| Mode | Color | GUI icon | TUI glyph |
|---|---|---|---|
| Fast | `#38bdf8` sky | `bolt` (new) | `↯` |
| Planning | `#f59e0b` amber | `listChecks` (new) | `☰` |
| Coding | `#34d399` emerald | `code` | `{}` |
| Thinking | effort color | `sparkles` | `✦` |

Effort colors are the table in §4.1. The chip, popover icons, TUI footer badge, and effort rows use these values.

### 4.7 Demo

- `docs/js/demo.js`: `MODES` becomes four; add `THINKING_EFFORTS` (+labels/colors), `state.thinkingEffort` (default `medium`), and a `normalize` migration mapping a saved `mode === "Deep Think"` to `mode = "Thinking"`, `thinkingEffort = "deep"`; invalid efforts reset to `medium`.
- Mode popover markup gains colored mode glyphs/icons and the Thinking effort tag. The Thinking row opens the same nested five-stop slider (plain DOM, no build step); dragging/clicking applies immediately.
- Chip shows `modeLabel` in the mode/effort color. Settings → Model keeps only Mode, Checkpoint, and Context; the effort control lives solely in the popover (no settings duplicate).
- `docs/demo.html`: mode chip label and popover container unchanged structurally; add the effort panel container.

### 4.8 Deep Think removal

Deleted from: `_PROFILES` (all families), `MODES`, `MODE_DETAILS`, TUI `/mode` help/parser, GUI mode lists, smoke checks, tests, demo, and DESIGN.md. `/mode deep` and `/mode deep think` are accepted aliases that select Thinking + `deep`. Saved demo state migrates. Other surfaces already ignore unknown modes via `normalize`.

---

## 5. Error handling and edge cases

- Unknown effort (API, command, demo) → `medium`, no error dialog; TUI command names the valid values.
- The 1.5 cap test fails fast if any Dizel effort prompt exceeds 240 chars.
- Dragging the slider while the popover closes (Escape/outside) cancels the pending change; no orphan state.
- Switching family re-applies the current mode + effort; the effort persists in memory.
- `/plan` while streaming: the existing guard returns before the mode change (no half-applied mode). If `setMode` fails (sidecar error), the send still proceeds with the previous profile and the store surfaces the error.
- Thinking effort applies only to Thinking; other modes ignore it but remember the value.
- Window-edge panel: flips left; if neither side fits, it opens below the row rather than clipping.
- The footer badge keeps readable contrast by using the dark ink on every mode color.

---

## 6. Testing strategy

All tests use temporary config/data roots; no real settings or sessions.

- **Prompts (`tests/test_mode_prompts.py`):** every Dizel Thinking effort ≤240 collapsed, contains "Dizel", passes `_v15_system_prompt` unchanged, five distinct prompts; Flash/Pro inherit all five.
- **Profiles (`tests/test_modes.py`):** four modes per family (no Deep Think); `get_model_profile("Dizel","Thinking","high")` matches `temp 0.43`, budget `1.55`; default medium; unknown effort → medium; Deep budget > Thinking medium; Mila/Generic keep identity and per-effort prompts.
- **Manager/API:** `ChatManager.apply_profile("Dizel","Thinking","high")` stores the high profile; the route forwards `thinking_effort`.
- **TUI:** `set_mode("Thinking","high")` applies the effort; `set_thinking_effort` re-applies only when Thinking; `/mode thinking high`, `/mode deep`, `/mode deep think`; `/plan` sets Planning and keeps it; `response_label`/`response_color`; picker glyph/color rendering; effort picker dismissal; footer watcher.
- **GUI smoke:** popover lists exactly Fast/Planning/Coding/Thinking with icons; Thinking row opens the nested panel; five stops with distinct colors; clicking High sets `mode="Thinking"`, `thinkingEffort="high"`, chip text `Thinking · High`, chip color class/style; `/plan` sets `mode="Planning"` before the send; no "Deep Think" row.
- **Demo:** `node --check docs/js/demo.js`; a screenshot of the demo page shows the four colored mode rows and the effort panel.
- **Full verification:** pytest on `tests/`, `npm run build` + `npm run smoke`, docs rebuild/tests, and captures of the Thinking popover/panel and the Planning chip after `/plan`.

---

## 7. Files touched

- `dizel_core/token_budget.py` · `dizel_core/chat_manager.py` · `api/routes/chat.py` · `api/services/chat_service.py`
- `inference/tui/constants.py` · `app.py` · `panels/input_bar.py` · `panels/status_bar.py` · `panels/modals/common.py` · `panels/modals/pickers.py` · `commands/builtins/mode.py` · `commands/builtins/plan.py` · `commands/builtins/debug.py` · `DESIGN.md`
- `inference/gui/src/services/modes.ts` (new) · `services/api.ts` · `store/chatStore.ts` · `components/Composer.tsx` · `components/Icon.tsx` · `styles/app.css`
- `docs/js/demo.js` · `docs/demo.html` · `docs/styles/demo.css`
- Tests: `tests/test_mode_prompts.py`, `tests/test_modes.py`, TUI picker/command tests, `inference/gui/scripts/smoke.js`

---

## 8. Resolved decisions

- Effort changes prompt + sampling (temperature) + output budget (user, 2026-11-19).
- Default effort: Medium (user, 2026-11-19).
- `/plan` keeps Planning active (user, 2026-11-19).
- Effort UI: nested small panel beside the Thinking row (user, 2026-11-19).
- Modes: Fast, Planning, Coding, Thinking; Deep Think merged as the max effort (user, 2026-11-19).
- No commit without explicit request; this spec is written but uncommitted.
