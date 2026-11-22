# Vision capability detection — design spec

**Status:** Draft (2026-11-22). Awaiting spec review before the implementation plan.
**Date:** 2026-11-22
**Owner:** D4niel-dev
**Scope:** Decide `model_info.vision` from the provider's own answer instead of a model-name regex, and correct the UI copy that claims images are never sent. `dizel_core` plus two GUI strings.

---

## 1. Summary

The image path is already built. `chat_manager` records `data:image/` attachments with the user turn, `_api_messages()` copies them into the provider payload, `Composer` disables the paperclip unless `status.model_info.vision === true`, and the store refuses with "the loaded model has no vision input".

What is missing is the **source of truth for that flag**. It comes from `model_supports_vision()`, a regex over the model name. Ollama will answer directly: `POST /api/show` returns `capabilities: ["completion", "vision"]`. So a vision model whose name the regex does not recognise is silently blocked, and a name that looks multimodal but is not still passes.

This spec replaces the guess with the provider's answer where one exists, keeps the regex as the fallback, and fixes two strings that describe the old behaviour.

---

## 2. Goals and non-goals

### Goals

1. An Ollama model is marked vision-capable when `/api/show` lists `vision` in `capabilities`, regardless of its name.
2. An Ollama model that does *not* list `vision` is text-only, even if its name matches the regex.
3. Any probe failure (daemon down, old Ollama without `capabilities`, timeout) falls back to today's regex. Never worse than current behaviour.
4. Local GGUF backends stay `vision: false`.
5. The attachment preview says the model cannot read images, not that images are not sent.

### Non-goals

- No image input for local checkpoints. Dizel 1.5 and 1.6 are text-only.
- No manual override toggle. Add one only if a real model trips detection.
- No probing of OpenAI-compatible endpoints. There is no standard capability endpoint; the regex stays.
- No change to attachment count limits, storage, the send path, or the transcript format.
- No new provider.

---

## 3. Current state

| Piece | Behaviour today |
|---|---|
| `providers.model_supports_vision(provider, model)` | Regex. Ollama gets its own narrower pattern. |
| `ChatManager.model_info` | Provider path calls the regex; local path sets `vision: false`. |
| Send path | `_record_user_turn(..., images=...)` then `_api_messages()` copies `images`. Already forwards. |
| `Composer.tsx` | `visionReady = status?.model_info?.vision === true`; paperclip disabled when false. |
| `chatStore` | Rejects attachments with "the loaded model has no vision input". |
| `AttachmentPreview.tsx` | "Not sent to the model — no vision." — wrong for provider models. |

---

## 4. Design

### 4.1 Detection order

1. **Probe** (Ollama only): `POST {base}/api/show` with `{"model": <id>}`. Capable when `"vision"` is in `capabilities`.
2. **Regex fallback**: `model_supports_vision()` — unchanged, used whenever the probe yields no answer.
3. **Local**: `vision: false`, unchanged.

The probe returns `Optional[bool]`: `True` / `False` when the provider answered, `None` when it could not. `None` is not "no" — it means "ask the regex".

### 4.2 Where the probe runs

Not in `model_info`. That property is read on every `/chat/status` poll, and a network call there would put a timeout in the status path.

The probe runs in `ChatManager._load_provider_state()` — the single place `_provider_slug` and `_api_model` are assigned, reached from `__init__` and `reload_provider()`. The result is cached on the manager:

```python
self._vision_flags: dict[tuple[str, str], bool] = {}
```

`model_info` reads the cache for the active `(provider, model)`; on a miss it uses the regex without probing. Because every model or provider change goes through `_load_provider_state`, the cache is warm for the pair that is actually in use.

### 4.3 Timeout

`OllamaProvider` gains a `SHOW_TIMEOUT = 3` seconds constant, used only by `/api/show`. The existing `DEFAULT_TIMEOUT` (20s) is for model listing and would stall startup on a dead daemon.

### 4.4 Failure behaviour

| Situation | Result |
|---|---|
| Ollama not running | `None` → regex |
| Old Ollama, no `capabilities` key | `None` → regex |
| Model not pulled | `None` → regex |
| `/api/show` 5xx | `None` → regex |
| Answered with `capabilities` | Authoritative `True`/`False` |

No exception escapes `probe_vision`.

### 4.5 Copy

- `AttachmentPreview`: **"This model can't read images."**
- `Composer` hint keeps "no vision" (it is a chip, not a sentence).

---

## 5. Files

| File | Change |
|---|---|
| `dizel_core/providers.py` | `OllamaProvider.show_model(model)`; `probe_vision(provider_slug, model, ollama_url)`; `resolve_vision(provider_slug, model, cached)`. |
| `dizel_core/chat_manager.py` | `_vision_flags` cache; probe inside `_load_provider_state()`; `model_info` reads the cache. |
| `inference/gui/src/components/AttachmentPreview.tsx` | One string. |
| `tests/test_vision_capability.py` (new) | Probe parsing, fallback, cache behaviour. |

---

## 6. Data flow

```
settings.json api_router
        │
        ▼
ChatManager._load_provider_state()
        │  provider=ollama → POST /api/show → capabilities
        ▼
_vision_flags[(provider, model)] = True | False
        │
        ▼
model_info.vision  ──►  GET /chat/status  ──►  store.status.model_info.vision
        │
        ▼
Composer.visionReady → paperclip enabled + "vision" hint
                     → store accepts or rejects the attachment
```

---

## 7. Error handling

- The probe is best-effort and silent. It must never raise into `_load_provider_state`, which is on the startup path.
- A probe timeout must not delay startup beyond 3 seconds, and only when Ollama is the configured provider.
- The cache is per-process. A model pulled mid-session is picked up on the next `reload_provider()`, which is what Settings → save already triggers.

---

## 8. Testing

**Unit** (`tests/test_vision_capability.py`):

1. `/api/show` with `["completion","vision"]` → `True`, even when the model name matches no regex.
2. `/api/show` with `["completion"]` on a name the regex likes (`llava`) → `False` (the provider wins).
3. `ConnectionError` → `None`, and `resolve_vision` falls back to the regex result.
4. Response missing `capabilities` → `None`.
5. Cache hit performs no second HTTP call (assert on the mocked session).

**GUI:** extend `smoke.js` with one check that the attachment hint does not claim images are never sent, and that the paperclip is disabled when `model_info.vision` is false.

**Manual:** with Ollama running, select `llava` (paperclip enabled) and `qwen2.5:7b` (disabled), confirming the flag flips without a restart.

---

## 9. Decisions

- **Regex stays.** It is the only signal for OpenAI-compatible endpoints, and the fallback keeps probe failures harmless.
- **No override.** A toggle would be a second source of truth for something the provider can usually answer. Revisit if a real endpoint is misdetected.
- **3-second probe timeout.** Long enough for a loaded daemon, short enough not to be felt at startup.
