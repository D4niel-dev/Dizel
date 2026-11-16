# Write-approval trust — design spec

**Status:** Direction approved (2026-11-16); persistent grant scope = root-bounded, all writes. Awaiting spec review before implementation. Revised 2026-11-16 to account for the separately-added `delete_file` tool, which must **never** be auto-approvable (§4.7).
**Date:** 2026-11-16
**Owner:** D4niel-dev
**Scope:** Let the user grant standing approval for the model's `write_file` / `edit_file` calls, per-session and persistently, across the GUI (sidecar) and TUI (in-process) surfaces.

---

## 1. Summary

Today every `write_file` / `edit_file` call blocks on a per-call diff approval. This adds two standing-trust tiers on top of the existing one-shot approval, so a user who trusts the model for a stretch of work is not interrupted on every file:

- **Allow once** — current behavior, one change.
- **Always allow this session** — auto-approve later writes in the *current session only*; in-memory, gone on restart or new session.
- **Always allow** — persistent grant stored in config; auto-approve writes across sessions and restarts. **Root-bounded, all writes** (the user's chosen scope).
- **Deny** — current reject.

Both tiers are revocable, show a visible "auto-approving" indicator, and never silently write: every auto-approved change still appears in the transcript and is logged. The grant only ever applies to changes that already passed the tool sandbox.

This is architectural: it changes the approval interface on both surfaces, adds a shared policy module and a config key, and is security-sensitive. Implementation follows this spec's review and a written plan.

---

## 2. Goals and non-goals

### Goals

1. Offer four explicit choices in both approval gates: allow once, always-this-session, always-persistent, deny.
2. Honor trust identically on the GUI (sidecar) and TUI (in-process) surfaces via one shared policy.
3. Keep the persistent grant root-bounded by inheriting the existing write sandbox; never auto-approve an out-of-root, binary, VCS-internal, or non-regular-file change.
4. Make trust visible (indicator) and reversible (Settings revoke on both surfaces), and auditable (transcript + log line per auto-approval).
5. Session grants are in-memory and per-process; the persistent grant is config-backed and shared by both processes.

### Non-goals

- No new tool capabilities added by *this* feature. `delete_file` now exists (added separately, bounded) but is **never** auto-approvable — see §4.7. No shell or execute tool exists or is added.
- No per-path or per-tool persistent granularity (the user chose root-bounded all-writes; per-tool was offered and declined).
- No change to the read-only tools (`read_file`, `grep`, `web_search`, …) — they never prompt.
- No change to the completion bell or tab badge (separate features).
- No backward-compatibility shim for the old `approved: bool` resolve payload; all callers are internal and change together.

---

## 3. Current state (grounded)

Two distinct approval paths, both receiving a `dizel_core.tools.FileChange` and returning `bool`:

- **GUI / sidecar:** `ChatService._request_tool_approval(change, emit, tool_id)` (`api/services/chat_service.py:208`) emits an SSE `approval` event and blocks on a `threading.Event` (300 s). `resolve_tool_approval(approval_id, approved)` (`:236`) sets the answer. Wired as `approve_tool=lambda change: cls._request_tool_approval(change, _put, tool_context.key)` (`:444`) inside `stream_chat`, which has `session_id` in scope.
- **TUI / in-process:** `ChatBridge._approve_tool(change)` (`inference/tui/bridge/chat_bridge.py:80`) pushes `ToolApprovalModal(change)` (`ModalScreen[bool]`) and blocks on an Event (180 s). The bridge knows the session via `self._session_id = self.app.session_id` (`:116`). Wired at `:213`.

The engine runs **in a different process** for each surface (TUI in-process; GUI via the FastAPI sidecar). They share `settings.json` on disk but not memory.

**The sandbox already bounds writes.** `_resolve_write(path)` (`dizel_core/tools.py:252`) resolves against `_project_root()` (= `Path.cwd().resolve()`, `:248`) and rejects: out-of-root paths ("File changes are limited to the current project directory."), `BLOCKED_PARTS` (`node_modules`, `.git`, `__pycache__`, `.venv`, `dist`, `.next`), `BLOCKED_SUFFIXES` (binaries/archives), and non-regular files. A `FileChange` only reaches the approve callback *after* passing this. `write_file` action is `create` or `overwrite`; `edit_file` is `edit`. `delete_file` (action `delete`) moves an approved file to the OS recycle bin via `send2trash`; it is root-bounded by `_resolve_delete` (regular files only, no directories, refuses VCS/generated internals) and is `requires_approval=True`. It is deliberately **outside** `WRITE_TOOLS` so trust can never auto-approve it (§4.7).

**Config** is `ConfigManager._DEFAULTS` (`dizel_core/config_manager.py:73`), deep-merged with disk on `load()` and written atomically on `save()`. There is no `tools` section today.

---

## 4. Design

### 4.1 Shared policy module — `dizel_core/tool_trust.py` (new)

Pure, independently testable; no UI or process assumptions.

```
WRITE_TOOLS = {"write_file", "edit_file"}
ALLOW_ONCE = "allow_once"
ALWAYS_SESSION = "always_session"
ALWAYS_PERSISTENT = "always_persistent"
DENY = "deny"

should_auto_approve(change, session_id, *, session_grants, persistent, root) -> bool
    True iff:
      change.tool in WRITE_TOOLS
      AND (persistent OR session_id in session_grants)
      AND _in_root(change.path, root)        # defense-in-depth re-check
    False otherwise (prompt as today).

approved_for(decision) -> bool
    decision != DENY

record_session_grant(session_grants, session_id) -> None     # add id
revoke_session_grant(session_grants, session_id) -> None     # discard id
```

`_in_root` re-resolves `change.path` against `root` and confirms it stays inside (mirrors `_resolve_write`'s bound). The sandbox is the primary guard; this is a cheap second check so a malformed change can never be auto-approved out-of-root.

Persistent flag helpers read/write config through `ConfigManager`:

```
persistent_enabled() -> bool        # ConfigManager.load()["tools"]["always_allow_writes"]
set_persistent(enabled) -> None     # load → set → save (atomic)
```

The flag is read **fresh at decision time** (writes are infrequent) so a grant revoked in one process is honored by the other on its next write without a restart.

### 4.2 Trust stores

- **Session grants (in-memory, per-process):**
  - Sidecar: `ChatService._trusted_sessions: set[str]`, guarded by `_approval_lock`.
  - TUI: `DizelCMDApp._trusted_sessions: set[str]`.
  - Keyed by `session_id`. A new session has a new id → untrusted. Restart clears the set. Deleting a session discards its id.
- **Persistent grant (config):** `tools.always_allow_writes: bool`, default `False`. Shared via `settings.json`.

### 4.3 Config change

Add to `_DEFAULTS`:

```
"tools": { "always_allow_writes": False },
```

The existing deep-merge adds the key for older configs. No migration needed.

### 4.4 Decision API — sidecar

- `resolve_tool_approval(approval_id, decision: str)` replaces the `approved: bool` parameter. It maps `decision → bool` via `approved_for`, and on `ALWAYS_SESSION` adds the entry's `session_id` to `_trusted_sessions`; on `ALWAYS_PERSISTENT` calls `set_persistent(True)`. The approval entry stores `session_id` when created.
- Route `POST /chat/tools/approve` (`api/routes/chat.py`) body becomes `{ approval_id, decision }`. Unknown id → 404 (unchanged).
- `_request_tool_approval(change, emit, tool_id, session_id)` gains `session_id`. **Before** emitting/blocking it calls `should_auto_approve(...)`; if True it emits an audit note and returns True immediately (no SSE approval event, no wait). The `:444` lambda captures `session_id` from `stream_chat`.
- New revoke endpoints: `POST /chat/tools/trust/revoke-session` (drop current session id) and `PUT /chat/tools/trust` `{ persistent: bool }`. A `GET /chat/tools/trust` returns `{ persistent, session_trusted }` for the indicator/Settings.

### 4.5 Decision API — TUI

- `ToolApprovalModal` becomes `ModalScreen[str]`, dismissing a decision string. Four buttons for writes/edits: **Reject** (`deny`), **Allow once** (`allow_once`), **Always this session** (`always_session`), **Always allow** (`always_persistent`). Bindings: `escape`/`n` → deny, `y` → allow_once, `s` → always_session, `a` → always_persistent. For a `delete` change only **Reject** / **Allow once** render and `s`/`a` are unbound (§4.7). The persistent button carries a one-line warning ("Stays on until you revoke it in Settings"). The markup-safe `Text(path)` rendering and the busy/focus behavior are preserved.
- `ChatBridge._approve_tool(change)`: **before** pushing the modal, call `should_auto_approve(change, self._session_id, app._trusted_sessions, persistent_enabled(), root)`; if True, post an audit `SystemLog` and return True. Otherwise push the modal; `done(decision)` records the grant (`record_session_grant` / `set_persistent`) and returns `approved_for(decision)`. Timeout dismisses `"deny"`.

### 4.6 Visibility, revocation, audit

- **Indicator:** GUI `StatusBar` shows "Auto-approving writes" when `persistent` OR the current session is trusted. TUI status line shows a compact marker (e.g. `writes:auto`). Both update on grant/revoke.
- **Revocation UI:** GUI `SettingsDialog` gains a "File writes" group: persistent toggle (with warning text), "Revoke this session" button, current status. TUI `settings_modal` gains the same toggle + revoke alongside the existing tool/bell settings. Revoking persistent calls `set_persistent(False)`; revoking session discards the id.
- **Audit:** every auto-approved write still produces its normal tool row (DONE) **plus** a system-log line `Auto-approved <tool> <path> (session|persistent)`. Auto-approval is never silent.

### 4.7 Delete is always manual (cross-surface invariant)

The user's hard requirement: a deletion is **always** a per-call decision; no setting — including the session and persistent grants above — may ever auto-approve it. Guaranteed structurally, not by convention:

1. **Policy exclusion.** `delete_file ∉ WRITE_TOOLS`. `should_auto_approve` returns False for any delete regardless of `persistent` or `session_grants`. Both surfaces call it before prompting, so a delete always reaches the gate.
2. **No standing-allow UI for delete.** The TUI `ToolApprovalModal` renders only **Reject** / **Allow once** when `change.action == "delete"`; the `always_session` / `always_persistent` buttons and their `s` / `a` bindings are suppressed. The GUI `ToolApprovalDialog` shows only **Reject** / **Approve** for a delete (the `is-delete` destructive gate already implemented); the standing-allow options C adds render for `write_file` / `edit_file` only.
3. **No config path.** `tools.always_allow_writes` governs writes/edits only. No key auto-approves deletes; Settings labels the grant "writes and edits," never "all file operations."
4. **Recoverable by default.** Approved deletes go to the recycle bin (`send2trash`); the sandbox still refuses out-of-root paths, directories, and VCS/generated internals before any prompt.

**Required test (blocks implementation):** with the persistent grant ON *and* a session grant present, a `delete_file` change still prompts on both surfaces (`should_auto_approve(delete_change, ...) is False`), and neither gate renders a standing-allow option for delete.

---

## 5. Threat model and safety

- **Asset:** project files under the engine's cwd root.
- **Adversary:** the local model, possibly steered by injected content, attempting unwanted writes.
- **What the persistent grant removes:** per-write human review for in-root `write_file`/`edit_file`. The model can then create or **overwrite existing project files** (source, configs) without a prompt. This is the accepted trade-off (user chose root-bounded all-writes).
- **Mitigations (layered):**
  1. Inherited sandbox — no out-of-root, binary, archive, VCS-internal, or non-regular-file writes; no exec/shell tool exists. `delete_file` exists but is never auto-approvable (§4.7).
  2. Defense-in-depth `_in_root` re-check in the policy.
  3. Explicit opt-in with a visible warning on the persistent choice.
  4. Always-visible indicator while trust is active.
  5. One-click revoke on both surfaces; persistent revoke is honored cross-process on the next write.
  6. Audit line + transcript row per auto-approval.
- **Session tier:** identical exposure but scoped to one session, in-memory, cleared on restart/new session.
- **Cross-project note:** the persistent grant is a global boolean, not pinned to a root path; it applies to whatever cwd root the engine runs in. Opening a different project keeps the grant active. Documented in the Settings warning; revocable. (Root-pinning was considered and deferred to keep the chosen "all writes" scope simple.)
- **Concurrency:** session set guarded by the sidecar lock; config writes atomic via `ConfigManager.save`.
- **Never auto-approve:** out-of-root changes (rejected upstream + re-checked), non-write tools, and `delete_file` — which always prompts, is excluded from `WRITE_TOOLS`, offers only Approve/Reject at its gate, and has no config key that could auto-approve it (§4.7).

---

## 6. Data flow

**First write, user picks "Always this session":**
tool → `_resolve_write` (sandbox) → approve callback → `should_auto_approve` False → modal/SSE → user picks `always_session` → resolve records session grant, returns True → write happens.

**Later write, same session:**
tool → sandbox → approve callback → `should_auto_approve` True (session grant) → audit note → return True → write happens. No modal, no wait.

**Persistent grant:** same, but `persistent_enabled()` True for any session/process until revoked.

**Revoke:** Settings → `set_persistent(False)` / discard session id → next write prompts again.

---

## 7. Error handling and edge cases

- Unknown `approval_id` on resolve → 404; GUI keeps the dialog and surfaces the error (the existing busy-reset fix is preserved).
- Auto-approve bypasses the Event wait; `cancel_tool_approvals` / `stop` still cancel any *pending* manual approvals.
- Config read failure → `persistent_enabled()` returns False (fail closed: prompt).
- Session changed mid-generation → grants are keyed by id, so a reply for an old session never auto-approves under a new one.
- Modal timeout → `deny` (TUI 180 s, sidecar 300 s), unchanged.
- A grant recorded but the write then fails on disk → normal tool failure path; trust is unaffected.

---

## 8. Testing strategy

All tests use temporary config/data roots — never the real `settings.json` or chat history (standing constraint).

- **Unit `tests/test_tool_trust.py`:** `should_auto_approve` matrix (persistent on/off × session grant present/absent × in/out-of-root × write/non-write tool); `approved_for`; session grant record/revoke; persistent flag round-trip with a temp config; fail-closed on config error. **Delete invariant:** `should_auto_approve` is False for `delete_file` even with persistent ON and a session grant present.
- **Sidecar:** resolve maps each decision correctly; `always_session` records the id and the next same-session write auto-approves with no SSE approval event; `always_persistent` sets the config flag; revoke endpoints clear trust; out-of-root never auto-approves; a `delete_file` change still emits an approval event under full trust.
- **TUI `tests/test_tui_tool_flow.py`:** modal dismisses each of the four decisions; bridge auto-approves a second write in a trusted session without pushing a modal; pilot asserts the indicator and the audit log. Delete gate renders only Reject/Allow once (no `s`/`a`), and a delete still pushes the modal under full trust.
- **GUI smoke:** dialog renders four options for writes; choosing always-session resolves and shows the indicator; a subsequent approval is skipped; revoke restores prompting; persistent toggle round-trips. Delete gate renders only Reject/Approve (no standing-allow) — already covered by the `is-delete` smoke check.
- **Build:** `tsc --noEmit && vite build`; full pytest; preload unit test unaffected.

---

## 9. Files touched (implementation, after review + plan)

- `dizel_core/tool_trust.py` (new) · `dizel_core/config_manager.py` (tools default)
- `api/services/chat_service.py` · `api/routes/chat.py`
- `inference/gui/src/services/api.ts` · `store/chatStore.ts` · `components/ToolApprovalDialog.tsx` · `components/SettingsDialog.tsx` · `components/StatusBar.tsx`
- `inference/tui/bridge/chat_bridge.py` · `panels/modals/tool_approval.py` · `panels/modals/settings_modal.py` · `app.py`
- tests + `inference/gui/scripts/smoke.js`

---

## 10. Resolved decisions

- Persistent scope: **root-bounded, all writes** (user, 2026-11-16). Per-tool and new-files-only were offered and declined.
- Both session and persistent tiers coexist (user asked to *add* "Always allow").
- D (tab badge) and the bell are out of scope here.
- No commit without explicit request; this spec is written but uncommitted.