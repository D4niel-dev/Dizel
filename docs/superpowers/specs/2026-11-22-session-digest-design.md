# Session digest — design spec

**Status:** Draft (2026-11-22). Awaiting spec review before the implementation plan.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** A compact "where you left off" card at the top of a long chat when it is reopened. GUI only, derived entirely from messages already in memory.

---

## 1. Summary

Reopening a 60-message chat drops you at the bottom with no idea what the first 58 were about. The status bar and the context panel now explain the token budget; nothing explains the conversation itself.

The first idea was to reuse the compaction summarizer and cache a digest per session. Investigating showed that is the wrong tool: `MemoryService.summarize()` calls `summarize_old_context()`, whose own docstring says its output is a *prompt fragment* meant for the model and "reading it in the UI is noise". `MemoryService.recent()` deliberately avoids it and uses the last assistant reply instead.

So the digest is **derived, not generated**. No model call, no cache, no schema change, nothing to go stale: it reads the same messages the transcript already has.

---

## 2. Goals and non-goals

### Goals

1. On opening a chat with at least 12 messages, show a card above the transcript with: where it started, where it ended, and the last reply.
2. The card is derived synchronously from the loaded messages — no request, no waiting state.
3. It can be dismissed for the rest of the session, and reopening a different chat resets that dismissal.
4. It reads as a reading aid, not as model output: distinct styling, no "assistant" voice, clearly labelled as a digest.

### Non-goals

- No abstractive/model summary. It cannot hallucinate, but it also will not find themes buried in the middle.
- No per-session persistence. There is no digest to invalidate, and no new field in `history.json`.
- No TUI version yet. The TUI has its own session list and no equivalent transcript header; adding one later is independent.
- No changes to the trimmer, compaction, or the welcome screen's recent-chat lines.

---

## 3. Current state

| Piece | Behaviour today |
|---|---|
| `MemoryService.summarize()` | Extractive prompt fragment: `Previous context: user: … \| assistant: …`. Explicitly not for UI. |
| `MemoryService.recent()` | Uses the last assistant reply, flattened and truncated to 90 chars. |
| `SessionSummary` | `id, title, created, preview, pinned, archived, tags` — no digest field. |
| `ChatView` | Renders the message list; no header card. |
| Session open | Loads messages into the store; no derived state. |

---

## 4. Design

### 4.1 Derivation

A pure helper in the GUI, `buildDigest(messages): Digest | null`. It returns `null`
when there are fewer than 12 messages, so ChatView does not repeat the threshold:

```ts
{
  count: number,      // messages considered
  started: string,    // first non-empty user message, flattened, <= 120 chars
  asking: string,     // last non-empty user message, flattened, <= 120 chars
  answered: string,   // last non-empty assistant message, flattened, <= 160 chars
}
```

- System messages and tool-only entries are skipped.
- "Flattened" means whitespace collapsed to single spaces, same as `MemoryService.recent()`.
- Truncation is at the end: `slice(0, n - 1).rstrip() + "…"`, matching the existing convention.
- If there is no assistant reply yet, the card shows started/asking only.

### 4.2 When it appears

`ChatView` computes the digest from `messages` with `useMemo`. It renders the card when:

```
messages.length >= 12 && !dismissed
```

`dismissed` is local component state keyed by session id — a `useEffect` on `sessionId` resets it. Dismissing is a small × button; it does not modify the chat or its history.

The card is **not** re-shown while streaming: once a turn is in flight the transcript is the thing to read, so the card hides when `streaming` is true.

### 4.3 Layout

A single wide card above the first bubble, visually quieter than a message:

```
┌────────────────────────────────────────────────────┐
│ WHERE YOU LEFT OFF                            ×   │
│ Started with   Fix the Windows launcher line…     │
│ Last asked     Can you also check the TUI?        │
│ Dizel said     The launcher now uses CRLF and…    │
│ 48 messages · this is a reading aid, not a reply  │
└────────────────────────────────────────────────────┘
```

Labels are uppercase, small, muted. Values are `--text-secondary`. No avatar, no role colour, no timestamp. It must never be mistaken for something the model wrote.

### 4.4 Errors

There is no failure path: the helper takes an array and returns strings. An empty or malformed array yields a card with no values, which is treated as "do not render".

---

## 5. Files

| File | Change |
|---|---|
| `inference/gui/src/services/digest.ts` (new) | Pure `buildDigest()` and truncation helper. |
| `inference/gui/src/components/ChatView.tsx` | Renders and dismisses the card. |
| `inference/gui/src/styles/app.css` | Card styles. |
| `tests/test_gui_digest.mjs` (new) or a smoke assertion | Derivation cases. |
| `inference/gui/scripts/smoke.js` | Opens a seeded long chat, asserts the card and its dismissal. |

---

## 6. Data flow

```
session messages (already in the store)
        │
        ▼
buildDigest(messages)   ← pure, synchronous
        │
        ▼
ChatView: messages.length >= 12 && !dismissed && !streaming
        │
        ▼
<DigestCard />  ──×──►  dismiss for this session
```

No API calls. No store writes. No persistence.

---

## 7. Testing

**Unit** — pure helper, run with Node in the GUI test setup:

1. Skips system and tool-only messages.
2. Flattens newlines and repeated whitespace.
3. Truncates at 120/160 chars with a single ellipsis.
4. Fewer than 12 messages → `null` (the helper returns null below the threshold, so ChatView need not repeat it).
5. A chat with no assistant reply still yields started/asking.

**Smoke:** seed a session with 12+ messages, open it, assert the card lists three labels, click ×, assert it disappears, open a different session and assert it can appear again.

---

## 8. Decisions

- **Derived over generated.** No model call and nothing to invalidate; the trade-off is that it cannot summarise the middle of a long conversation.
- **12 messages.** Below that, scrolling back is cheap and a card is clutter. The threshold lives in one constant.
- **Dismiss per session, not persisted.** A user who dislikes the card should not have to find a setting.
- **Hidden while streaming.** The live answer is more important than a recap.
- **GUI only.** The TUI's compact layout has no equivalent slot; adding it there is a separate change.
