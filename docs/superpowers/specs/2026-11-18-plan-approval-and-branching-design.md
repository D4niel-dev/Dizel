# Plan approval and conversation branching — design spec

**Status:** Direction approved (2026-11-18): approval-gated plan then tool loop; destructive TUI edit with confirm; GUI side-by-side compare. Awaiting spec review before the implementation plan.
**Date:** 2026-11-18
**Owner:** D4niel-dev
**Scope:** A `plan` tool whose approval gates a longer tool loop, plan approval UI on both surfaces, a side-by-side compare view for GUI sibling replies, and a destructive `/edit` for the TUI. App-only; no published baseline.

---

## 1. Summary

Phase 3 adds two kinds of *deliberate work* to Dizel:

1. **Plan → approve → execute.** For multi-step tasks the model can call a `plan` tool with a short task line and up to eight numbered steps. The call blocks on a dedicated **plan approval** — a separate object from file writes, so no trust setting can ever auto-approve it. On approval the tool loop's round budget rises from 3 to 16 for that turn; the model then executes the steps through the normal tools, and every write/edit still hits the diff gate. Discarding a plan tells the model to stop. Both surfaces get a plan checklist UI, and `/plan <task>` asks for a plan without executing.
2. **Conversation branching.** GUI replies that already keep sibling versions gain a **Compare** view showing every version side by side. The TUI gains `/edit`, which reopens the last user message in the multiline editor with a two-step destructive confirm, then truncates that turn and everything after it and resends.

This is architectural: it adds a new approval class shared by two processes, changes the tool-loop budget contract, and adds destructive transcript editing. It follows this spec's review and a written plan.

---

## 2. Goals and non-goals

### Goals

1. A plan can never execute without an explicit, per-plan approval on the surface that runs it.
2. Plan approval is structurally outside write trust: it is never auto-approved, offers no standing-allow choice, and records no trust.
3. Plan approval raises only the current turn's round budget; the next turn returns to the normal budget.
4. Plan checklist UI on both surfaces shows the steps and two outcomes: execute or discard.
5. `/plan <task>` exists on both surfaces with a compact transcript line; with tools off it degrades to a prose plan.
6. GUI sibling versions can be compared side by side from the reply's action row.
7. TUI `/edit` replaces the last user turn and everything after it only after a two-step confirm, and resyncs the model's history so the removed turn is gone.
8. The capability manifest records the new parity honestly.

### Non-goals

- No persisted per-step execution state machine, no step-by-step re-approval, no plan editing inside the approval UI. Execution is model-driven inside the raised round budget.
- No shell/execute tool. Writes and edits keep their existing per-call or trust-gated approvals; deletes stay per-call.
- No fork switcher in the TUI. TUI editing is destructive-with-confirm; branch preservation stays GUI-only (the existing fork/sibling system is unchanged).
- No new model prompts beyond the tool description, the plan instructions, and the `/plan` request text.
- No changes to the existing write-approval decisions, trust tiers, or delete invariants.

---

## 3. Current state (grounded)

- **Tool loop:** `ChatManager.send_with_tools` (`dizel_core/chat_manager.py:1081`) chains `send_message` callbacks; `state = {"round": 0, "done": False}`; when a reply parses to calls and `state["round"] >= max_rounds` (default 3, `:1089`, `:1156`) it stops with `_[stopped after N tool rounds]_`. `execute(call, approve)` (`dizel_core/tools.py:888`) passes `approve` only to tools with `requires_approval=True`; when `approve is None` it returns a refusal. `prompt_block(allow_writes)` lists approval tools only when writes are allowed (`:1130`).
- **Approval objects:** file tools build a frozen `FileChange` (`dizel_core/tools.py:72`) and call `_approved(change, approve, verb)` (`:314`). Trust is `dizel_core/tool_trust.py`: `should_auto_approve` returns False unless `change.tool in WRITE_TOOLS` — so a non-write object cannot be auto-approved, and `resolve_tool_approval` downgrades standing decisions for non-write tools to `allow_once`.
- **Sidecar approvals:** `ChatService._request_tool_approval(change, emit, tool_id, session_id)` (`api/services/chat_service.py:212`) consults trust, emits `{"type":"approval", ...}` and blocks on `threading.Event` (300 s); `resolve_tool_approval` (`:271`) answers it; route `POST /chat/tools/approve` (`api/routes/chat.py:58`). Pending entries live in `ChatService._approvals`; `cancel_tool_approvals` (`:297`) denies them on stop. The SSE stream drains the queue while the tool thread blocks (`api/services/chat_service.py:550`).
- **TUI approvals:** `ChatBridge._approve_tool(change)` (`inference/tui/bridge/chat_bridge.py:80`) auto-approves via trust or pushes `ToolApprovalModal` (`inference/tui/panels/modals/tool_approval.py`) and waits 180 s. `ChatBridge.send` wires `approve_tool=self._approve_tool` (`:282`).
- **GUI approvals:** `StreamEvent` includes the `approval` variant (`inference/gui/src/types.ts:417`); the store sets `approval` (`chatStore.ts:2313`) and `resolveApproval` posts to `/chat/tools/approve` (`:1019`); `ToolApprovalDialog` mounts in `App.tsx:457`. Debug fixture `/tooltest` appends a synthetic approval.
- **GUI branching:** `Message` supports `siblings` / `siblingIndex`; `showSibling` swaps versions (`chatStore.ts:2576`); the reply action row renders the `‹ n/m ›` switcher (`inference/gui/src/components/MessageBubble.tsx:673`). `beginEdit` already truncates a turn server-side. Smoke has a sibling fixture (`inference/gui/scripts/smoke.js:393`).
- **TUI transcript:** `SessionBridge.current_messages` holds role/content/ts/display (`inference/tui/bridge/session_bridge.py:12`); `save()` no-ops on an empty list; `App.load_session_to_workspace` (`inference/tui/app.py:890`) reloads manager history and replays the transcript. `App.send_composed(display_text, content)` (`:1080`) adds the user message, saves, and starts the tool run. `MultilineInputModal(initial_text, send_on_save)` (`inference/tui/panels/multiline_input.py:8`) is a `ModalScreen[str]` with TextArea and ctrl+s, used by the composer's ctrl+y path.
- **Manifest:** `capabilities.json` schema v2; `chat.edit` currently has `"tui": null` and a label-only GUI surface (`capabilities.json:43`). Rows with a `check` are enforced by pytest and GUI smoke; label-only/dashed surfaces are recorded gaps.
- **Docs:** `docs/tools/build_site.py` generates the capabilities page from the manifest; `tests/test_docs_site.py` enforces page completeness and version stamps.

---

## 4. Design

### 4.1 Plan object and tool — `dizel_core/tools.py`

New frozen approval object:

```python
@dataclass(frozen=True)
class PlanChange:
    task: str
    steps: tuple[str, ...]
    tool: str = "plan"
```

Because `tool` is `"plan"`, `tool_trust.should_auto_approve` returns False even with persistent trust on (it requires `change.tool in WRITE_TOOLS`). No trust module change is needed; a test pins this.

`Tool` gains `approval_verb: str = "Write"`. In `execute`, the `approve is None` refusal becomes
`f"[{name}] {tool.approval_verb} refused: this surface cannot approve it."` (existing tests assert only the `cannot approve` substring).

New tool, registered with `requires_approval=True` so it is listed only when the surface can approve (same rule as writes):

```python
Tool(
    "plan",
    "propose a short numbered plan for a multi-step task; the user approves it before you act",
    {"task": "string?", "steps": "string"},
    _plan,
    requires_approval=True,
    approval_verb="Plan",
)
```

`_plan(task="", steps="", approve=None) -> str`:

- Parse `steps` line by line; trim; drop empties; strip a leading list marker (`^\s*(?:\d+[.)]|[-*•])\s*`); truncate each step to 200 characters; keep at most **8** steps. No steps → `"[plan] A plan needs at least one step."`
- `task = " ".join(str(task or "").split())[:160]`; fall back to `"Untitled plan"` when empty.
- Build `PlanChange(task, tuple(parsed))`; require approval via the same callback contract. Outcomes (exact):
  - no approver → `"[plan] Plan refused: this surface cannot approve it."`
  - denied → `"[plan] Plan discarded. Do not execute it; reply with a short note that the plan was not run."`
  - approved → `"[plan] Approved. Execute the plan now, one step at a time, using the available tools. Writes and edits still need approval.\n"` followed by `1. …` numbered steps.
- `display_detail` returns `f"Plan · {n} steps"`; `ui_payload` for `plan` returns `{"steps": [...], "task": "..."}` so the inline drawer can list them.

System prompt: one added line in `TOOL_PROMPT` — `For a task with several steps, call plan first with the steps; do not start acting until it is approved.` It appears only when the tool list does (writes allowed).

### 4.2 Loop budget — `dizel_core/chat_manager.py`

- `state` gains `"plan_approved": False`.
- Module-level helper (unit-testable):
  `def round_limit(max_rounds: int, plan_approved: bool) -> int: return max(max_rounds, PLAN_ROUND_BUDGET) if plan_approved else max_rounds`, with `PLAN_ROUND_BUDGET = 16`.
- In `round_done`, after running a round's calls, set `state["plan_approved"] = True` when any outcome belongs to a `plan` call and starts with `"[plan] Approved"`. Detection lives in a module function `plan_approved(calls, outcomes) -> bool` for direct testing.
- The limit check uses `limit = round_limit(max_rounds, state["plan_approved"])` and the stop note uses the same `limit`. The plan call itself is executed before the raised limit matters; approval therefore lifts rounds 2…16. A denied plan leaves the budget at `max_rounds` and the model receives the discard text.

### 4.3 Sidecar plan approval — `api/services/chat_service.py`, `api/routes/chat.py`

- `_request_tool_approval` starts with `if isinstance(change, tools.PlanChange): return cls._request_plan_approval(change, emit)`. Plans never consult trust, never emit an auto-approve notice, and never record a grant.
- `_request_plan_approval(change, emit) -> bool` reuses the existing `_approvals` map and lock:
  - entry `{"event": threading.Event(), "approved": False, "kind": "plan", "claimed": False}`
  - emits exactly `{"type": "plan", "id": <id>, "task": change.task, "steps": list(change.steps)}`
  - waits 300 s; timeout or unclaimed wake → deny, same semantics as writes.
- `resolve_plan_approval(approval_id: str, approved: bool) -> bool`: unknown/claimed id → False; otherwise claim, store `bool(approved)`, set the event, return True. No `always_*` semantics exist.
- `cancel_tool_approvals` already denies every pending entry and now covers plans too.
- Route `POST /chat/tools/plan` with body `{"approval_id": str, "approved": bool}` → `{"status": "ok", "approved": bool}`; unknown id → 404 (`Approval request is no longer pending.`).

### 4.4 GUI — plan

- `types.ts`: `StreamEvent` adds `{ type: "plan"; id: string; task: string; steps: string[] }`. New:
  ```ts
  export interface PlanApproval {
    id: string;
    task: string;
    steps: string[];
    /** Debug fixture: resolve locally instead of answering a server request. */
    synthetic?: boolean;
  }
  ```
- `api.ts`: `resolvePlanApproval(approvalId, approved)` → POST `/chat/tools/plan` body `{ approval_id, approved }`.
- Store: state `planApproval: PlanApproval | null`; action `resolvePlan(approved: boolean): Promise<void>`.
  - SSE `case "plan"` sets `planApproval`. The plan's tool row settles through the ordinary tool events when the call returns.
  - `done`, `error`, and the stop/abort path clear `planApproval` alongside `approval`.
  - `resolvePlan`: synthetic → clear locally; otherwise POST then clear. On failure keep the dialog open and set `error` (mirrors `resolveApproval`'s busy-reset behavior).
- `PlanApprovalDialog` (new component, mounted in `App.tsx` beside `ToolApprovalDialog`): overlay classes `.plan-approval` / `.plan-approval-card`; heading "Plan approval"; kbd hint "y / n"; task line when present; an ordered list of steps; warning "The model runs these steps with its tools. Every write still needs your approval."; buttons **Discard** and **Execute plan**. Escape = discard; busy guard identical to `ToolApprovalDialog` (ref + disabled buttons while resolving).
- Slash command: palette entry `name: "plan"`, description "Draft a numbered plan before executing", category Chat. `runSlash("plan", arg)`: empty arg → hint `Usage: /plan <task>`; else send with a compact display line:
  - display text `` `Plan · ${task.length > 60 ? task.slice(0, 59) + "…" : task}` ``
  - content (exact):
    ```
    Draft a short numbered plan for this task. Call the plan tool with a short task line and the numbered steps; do not execute anything yet. If tools are unavailable, reply with the numbered plan only.

    Task: <task>
    ```
- Debug fixture: palette command `plantest` (debug-only, like `tooltest`) appends a synthetic approval with task "Ship the release checklist" and three steps ("Tag the release", "Publish the changelog", "Smoke-test the installer"), so the dialog is testable without a model.

### 4.5 TUI — plan

- New `inference/tui/panels/modals/plan_approval.py`: `PlanApprovalModal(ModalScreen[bool])`; header "Plan approval" + "y / n"; optional task line; numbered steps in a `VerticalScroll`; warning "The model runs these steps with its tools; every write still needs approval."; buttons **Discard** / **Execute plan**. Bindings: `escape`/`n` → dismiss False, `y` → dismiss True. No standing-trust options; it never calls `tool_trust`.
- `ChatBridge._approve_tool` starts with `if isinstance(change, PlanChange): return self._approve_plan(change)`. `_approve_plan` pushes the modal through `call_from_thread` and blocks on an Event (180 s, same timeout/modal-dismissal pattern as writes); timeout → False. It does not consult `should_auto_approve` or record grants.
- New `inference/tui/commands/builtins/plan.py`: `PlanCommand`, name `plan`, category `Chat`, usage `/plan <task>`; guards streaming (`Still generating — press esc to stop first.`); empty arg → `Usage: /plan <task>`; else `app.send_composed(display, content)` with the same display and content strings as §4.4. Registered in `builtins/__init__.py`.

### 4.6 GUI — compare sibling versions

- `MessageBubble` action row: when `siblings.length > 1`, add a **Compare** button (class `act sib-compare`) beside the `‹ n/m ›` switcher; clicking opens a local `CompareDialog` (new component) with `versions={siblings}` and `currentIndex={message.siblingIndex ?? 0}`.
- `CompareDialog`: fixed overlay `.compare-overlay` + panel `.compare-panel`; heading "Compare versions"; horizontally scrollable row of columns, one per version (`.compare-col`, current one marked `.is-current` with a "Current" tag), each headed `Version k` and rendering the reply content with the same markdown renderer the bubble uses; a `· N tool calls` line when that version has tools. Close via an icon button, Escape, and backdrop click. No store or session changes; versions come from the existing `siblings` data.
- Smoke: the existing sibling fixture opens Compare, asserts two columns with both texts and the current marker, then closes with Escape.

### 4.7 TUI — `/edit` (destructive, two-step)

- `MultilineInputModal` gains `title="Compose"`, `subtitle="Draft a longer prompt before sending it."`, `notice=""`. `notice` renders under the subtitle (class `modal-notice`). When `notice` is non-empty, ctrl+s is two-step: the first press replaces the notice with "Press ctrl+s again to replace this turn and every message after it — esc goes back." and shows footer `ctrl+s replace · esc cancel`; a second ctrl+s dismisses with the text; escape first disarms, then cancels. Existing ctrl+y/compose behavior is unchanged.
- New `inference/tui/commands/builtins/edit.py`: `EditCommand`, name `edit`, category `Chat`, usage `/edit`; guards streaming; no user message → `Nothing to edit yet.`; otherwise finds the last `role == "user"` index and prefills `(display or {}).get("text") or content`; opens `MultilineInputModal(initial_text=prefill, send_on_save=True, title="Edit message", subtitle="Change the prompt and send it again.", notice="Sending replaces this turn and every message after it.")` via `app.push_screen`; the callback calls `app.edit_and_resend(index, text)` unless the result is empty. Registered in `builtins/__init__.py`.
- `App.edit_and_resend(index: int, text: str) -> str`:
  1. Guard streaming; guard empty text and out-of-range index.
  2. Truncate `session_bridge.current_messages` to `[:index]`; update `_tab_messages[self.session_id]`.
  3. `manager.load_history([{role, content} …])` from the truncated list.
  4. Rebuild the transcript through a new `_render_transcript(messages)` extracted from `load_session_to_workspace` (clear workspace with `show_empty=not messages`, replay `HistoryMessage` / `add_user_message`, refresh artifacts and `context_tokens`). `load_session_to_workspace` calls the same helper; its behavior is unchanged.
  5. `send_composed(text, text)` — the edited prompt is both content and display line (skills are not reconstructed). The subsequent save persists the truncated transcript plus the new turn; no separate save of the empty/truncated state is needed.
- The removed assistant reply is gone from the session file after the resend; the model never sees the truncated turn.

### 4.8 Capability manifest

Three rows change (`capabilities.json`), enforced by `tests/test_capabilities.py` and GUI smoke:

```json
{ "id": "app.plan", "name": "Plan before executing",
  "tui": { "label": "/plan", "check": { "kind": "command", "value": "plan" } },
  "gui": { "label": "/plan", "check": { "kind": "palette", "value": "plan" } } }
{ "id": "chat.compare", "name": "Compare reply versions",
  "tui": null, "gui": { "label": "Compare button" } }
```

`chat.edit` becomes `"tui": { "label": "/edit", "check": { "kind": "command", "value": "edit" } }` with its GUI surface unchanged. The docs capabilities page and summary regenerate from the manifest (`docs/tools/build_site.py`); no manual docs edit.

---

## 5. Threat model and safety

- **Asset:** the transcript, the project tree, and the user's attention.
- **Plan approval is a human gate for autonomous multi-step behavior.** A plan under approval cannot be auto-approved: it is a distinct type, trust policy excludes it structurally, and no standing decision exists. The dialog offers only execute/discard.
- **Execution still hits every existing bound:** the write sandbox, per-call or trusted write approvals, the never-auto-delete rule, the 60 KB diff cap, and the 900 s stream ceiling. Raising rounds to 16 cannot bypass any of them; only the number of model→tool exchanges grows.
- **Timeout fails closed:** an unanswered plan approval denies (300 s sidecar, 180 s TUI).
- **Stop denies pending plans:** `cancel_tool_approvals` claims and denies them, so a stopped turn can never execute a plan afterwards.
- **Plan approval grants no trust:** approving a plan does not pre-approve any write; each write is decided on its own gate.
- **TUI edit is destructive by design but confirmed twice:** the prefill is visible, the first ctrl+s only arms the warning, the second replaces, and escape backs out. Nothing is deleted from disk until the resend persists the shortened session.
- **No new secrets, network, or process boundaries**; the plan text travels only in existing local channels (SSE / in-process callbacks) and the model prompt.

---

## 6. Data flow

**Plan turn (GUI):** `/plan` or a model-initiated call → `plan` tool → `PlanChange` → sidecar emits `plan` event and blocks → dialog → Execute → `[plan] Approved…` outcome → loop sets `plan_approved` → limit becomes 16 → model runs read tools and approval-gated writes → final reply. Discard → `[plan] Discarded…` → model stops.

**Plan turn (TUI):** same, in-process: `_plan` → `_approve_plan` modal → bool → outcome → loop budget.

**GUI compare:** reply with siblings → Compare → overlay renders all versions → close. No persistence change.

**TUI edit:** `/edit` → prefilled modal → ctrl+s arms → ctrl+s replaces → truncate → history reload → transcript rebuild → resend → session saved with the new turn.

---

## 7. Error handling and edge cases

- Empty/one-word plan arguments (`steps` empty after normalization) → the tool error string; no approval is requested.
- More than 8 steps → the first 8 are used; the outcome's numbered list matches the approved steps exactly.
- Model never calls `plan` on `/plan` → plain prose plan; acceptable degradation, no UI.
- Tools disabled → `plan` is absent from the prompt; `/plan` still sends, and the prose fallback applies.
- Plan approval raced with stop → the pending entry is claimed denied; resolve afterwards returns 404 and the GUI surfaces the error without wedging (same recovery as write approvals).
- Duplicate resolve on a plan id → second call 404 / False.
- Compare on a version whose reply failed or is streaming → content as stored; no tools animation; the column simply shows the stored text.
- `/edit` while streaming → refused before any modal opens.
- `/edit` with a display-only message (skills) → prefill is the display text; the resend text is exactly the edited editor content, skills are not reattached.
- Editing the first turn (`index 0`) → transcript empties, then the resend repopulates it; `send_composed` performs the save.
- `edit_and_resend` with an invalid index (session changed under the modal) → no-op message, modal result discarded.
- Modal dismissed (escape/empty text) → no truncation, no send.

---

## 8. Testing strategy

All tests use temporary config/data/write roots — never real settings, sessions, or project files.

- **`tests/test_tools_plan.py`:** normalization (markers stripped, empties dropped, 200-char cap, 8-step cap), empty-steps error, exact approved/discarded/no-approver outcomes, `PlanChange` fields, `display_detail`/`ui_payload`, `execute` passes the approver, and `tool_trust.should_auto_approve` is False for a `PlanChange` with persistent ON and a session grant present.
- **`tests/test_chat_manager_plan.py`:** `round_limit` matrix, `plan_approved` detection (approved plan outcome raises it; denied/other outcomes do not).
- **`tests/test_api_plan_approval.py`:** `_request_plan_approval` emits the exact event while blocked and resolves True/False; no auto-approval notice even with persistent trust on; unknown/duplicate ids 404; `cancel_tool_approvals` denies pending plans.
- **`tests/test_tui_plan.py`:** modal decisions (y/n/escape/buttons); `ChatBridge._approve_plan` pushes the modal and maps decisions without touching trust (fake app whose `call_from_thread` runs inline); timeout path dismisses.
- **`tests/test_tui_edit.py`:** command guards and prefill (display text wins); `edit_and_resend` truncates, reloads history, rebuilds the transcript, and sends the edited text; invalid index no-ops; empty text no-ops; the two-step arming behavior of `MultilineInputModal` (first ctrl+s arms, second dismisses, escape disarms) where the Textual test harness allows, otherwise unit-level on the modal state.
- **`tests/test_capabilities.py`:** row count updated for the three manifest changes; TUI `plan`/`edit` commands resolve; GUI `plan` palette check passes in smoke.
- **GUI smoke:** plan palette entry, `/plantest` fixture → dialog lists three steps → Discard clears; a second fixture → Execute clears; Compare on the existing sibling fixture opens two columns, marks the current version, closes on Escape. Unknown plan id resolve keeps the dialog open and re-enables buttons (mirrors the approval recovery check).
- **Docs:** `tests/test_docs_site.py` capability-page completeness passes with the new rows; version chrome unchanged.
- **Build/verification:** `pytest tests/` (fresh `--basetemp`), `npm run build` + `npm run smoke` in `inference/gui`, docs site rebuild, and DIZEL_SHOT captures of the plan card and the compare view. Existing counts (480 tests / 81 smoke checks) are expected to grow, not regress.

---

## 9. Files touched (implementation, after review + plan)

- `dizel_core/tools.py` · `dizel_core/chat_manager.py` · `capabilities.json`
- `api/services/chat_service.py` · `api/routes/chat.py`
- `inference/gui/src/types.ts` · `services/api.ts` · `store/chatStore.ts` · `components/PlanApprovalDialog.tsx` (new) · `components/CompareDialog.tsx` (new) · `components/MessageBubble.tsx` · `components/commands.ts` · `App.tsx` · stylesheet
- `inference/tui/panels/modals/plan_approval.py` (new) · `panels/multiline_input.py` · `bridge/chat_bridge.py` · `commands/builtins/plan.py` (new) · `commands/builtins/edit.py` (new) · `commands/builtins/__init__.py` · `app.py`
- Tests: `tests/test_tools_plan.py`, `tests/test_chat_manager_plan.py`, `tests/test_api_plan_approval.py`, `tests/test_tui_plan.py`, `tests/test_tui_edit.py`, `tests/test_capabilities.py`, `inference/gui/scripts/smoke.js`
- Generated: `docs/_site/pages/guide-capabilities.html`, `docs/guide/capabilities.html` (via `docs/tools/build_site.py`)

---

## 10. Resolved decisions

- Execution depth: **approval-gated plan, then the normal tool loop with a raised budget** (user, 2026-11-18). No persisted step state machine.
- TUI branching: **destructive edit with confirm** (user, 2026-11-18). GUI comparison: **side-by-side compare** (user, 2026-11-18).
- Plan approvals never create standing trust; writes keep their existing tiers.
- No commit without explicit request; this spec is written but uncommitted.
