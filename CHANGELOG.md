# Changelog

All notable changes to the Dizel app are recorded here. Release tags use the
app version from `dizel_core/version.py`.

## [2.0.4] - 2026-11-22

### Added

- A context panel on the status bar. It shows the real tokenizer count for the
  base prompt, mode instructions, response-length clause, project rules and
  conversation, with a caret tying the panel to the meter. The renderer no
  longer guesses tokens by dividing characters by four.
- A "Where you left off" card at the top of long chats. It is derived from the
  messages already loaded — where the chat started, what was asked last and
  the last reply — with no model call and nothing to cache. It hides while a
  reply is streaming and can be dismissed per session.
- Fast reply speeds: Fast, Turbo and Ultrafast, nested under the Fast mode
  alongside Thinking effort. Each tier carries its own clause, colour and reply
  budget, and the slider can be dragged just like the effort slider.
- A System colour mode in the desktop and web settings. It follows the OS
  light/dark preference and changes live when the OS does.
- The core 15-case benchmark is now documented in the README with its run
  command and the RFC-0008 warning that it is not a release gate.

### Changed

- Response length now changes the system prompt as well as the reply budget.
  Short and Detailed add explicit instructions; Normal stays silent. The
  context panel counts the clause, and the status bar refreshes when any prompt
  choice changes.
- The response-length control lives in its own Reply section of the composer's
  `+` menu. Its flyout lists the three options with their budget multipliers
  and a short description; Settings → Chat keeps the same control, and both
  stay in sync.
- The evaluation framework was modernized: benchmark cases are versioned,
  custom scorers can be registered, manual cases fail loudly instead of
  counting as half a pass, case weights affect the score, and timestamps are
  timezone-aware. The scorecard shape consumed by the apps is unchanged.

### Fixed

- Response length was effectively a no-op: the app sent `short`/`detailed`
  while the budget table only had `low`/`high`, so all three options produced
  the same budget.
- A saved response length had no effect at startup because the clause state was
  only refreshed by a Settings save.
- The context panel stayed stale after changing mode, effort, speed or length.
- The Fast speed slider accepted clicks and arrow keys but could not be
  dragged; synthetic pointer events also threw an uncaught capture error.
- Evaluation benchmark JSON was gitignored and `evaluation/suite.py` untracked,
  so a fresh checkout had no cases to run. The redundant second result store
  was removed.

## [2.0.3] - 2026-11-17

### Added

- Provider-specific usage limits. Every API provider resolves to its own context
  window (OpenAI, Groq, Mistral, xAI, Cohere, Azure and Meta default to 128k;
  Anthropic 200k; Google 1M; AI21 256k; Ollama and custom endpoints 32k), and the
  window can be overridden per provider in Settings → Chat → Usage limit (GUI)
  or Settings → API → Usage limit (TUI). Local checkpoints keep the window they
  report.
- Per-provider usage windows. Each provider keeps its own 12-hour usage counter,
  reset time and compaction count; switching providers switches the meter, and
  switching back restores it. Existing saved usage migrates into the local
  bucket.
- The terminal client now enforces the usage window like the desktop: a spent
  window refuses the turn with a reset notice, 84% triggers automatic
  compaction, and the five-compaction cap is respected.
- The composer's model menu remembers loaded checkpoints. "Recently loaded"
  lists the last eight (one click to reload, missing files greyed) above a
  collapsible "All checkpoints" list. The history is shared, so models loaded in
  the terminal appear too.
- Running tool rows animate their names. The desktop sweeps a highlight across
  the letters of every running tool; the terminal advances the same highlight
  and adds elapsed seconds to the row.
- Standing write decisions ask twice. "Always this session" and "Always allow"
  open an inline confirmation, with the persistent one warning in amber that
  writes in every project will apply until revoked; Escape backs out. The
  terminal requires a second press of `s` or `a`.

### Changed

- The desktop usage tooltip names the provider whose window it shows, and the
  terminal context rail follows the active provider's context window instead of
  the last loaded checkpoint's.
- The single "Provider context length" setting became a per-provider usage
  limit. The docs site's releases page, app screenshot and hover motion were
  refreshed for 2.0.3.

### Fixed

- Desktop generations could end with "Generation ended without a reply." even
  when the model replied: the sidecar signalled the end of the turn when the
  worker started instead of when generation finished. This affected every local
  checkpoint, not just 1.6.
- The Ctrl+K command palette drifted right inside its grid track; it is centered
  again, and a smoke check pins the geometry.
- The Windows taskbar icon could start blank on a restored-maximized window: the
  AppUserModelID is set before any window exists and the icon is re-applied
  after the window is shown.
- The reported usage capacity stayed frozen at 1,024 because the usage snapshot
  endpoint swallowed a NameError; it now reports the active provider's real
  window.
- The site's edition badge no longer wraps at tablet widths, hover arrows can
  actually animate (inline spans ignore transforms), and lifted cards stay above
  their neighbours.

## [2.0.2] - 2026-11-17

### Added

- Desktop command palette on Ctrl+K: a searchable, keyboard-driven launcher for
  chats, settings, and actions.
- In-app release notes in both clients. The desktop About card gains a "What's
  new" entry with a NEW badge and a markdown modal; the terminal adds
  `/release-notes`. Both read the bundled `CHANGELOG.md` through a shared
  `dizel_core/release_notes.py` reader.
- Write-approval trust tiers. A shared `dizel_core/tool_trust.py` policy backs
  four-option gates (Reject, Allow once, Always this session, Always allow) in
  both clients, with sidecar and TUI decision APIs. Persistent "Always allow"
  and per-session trust are revocable, and every standing decision is audited.
  The desktop shows an "Auto-approving writes" status chip and a Settings → Chat
  → File writes section; the terminal shows a `writes:auto` marker and matching
  settings rows.
- `delete_file` tool in both clients. Approved deletions move to the OS recycle
  bin via `send2trash` (recoverable), are confined to the project root, and are
  always approved per call — no setting or trust tier can auto-approve a delete.

### Changed

- All built-in writes and edits are confined to the project root and share the
  trust policy; deletes are structurally excluded from auto-approval.
- `send2trash>=1.8` is now a runtime and test dependency for recoverable deletes.
- The README clarifies that `v2.x.x` release tags version the desktop and
  terminal inference app, not the model. Model versions (Dizel 1.5, 1.6 Flash
  Preview) live in `model/dizel_info.py` and are tracked separately.

### Fixed

- Desktop composer accepts input, including slash commands, once the app has
  booted; sending stays gated on model readiness.
- Desktop write-approval dialog ignores Escape while a decision is in flight,
  resets its busy state in `finally`, and ignores stale approval ids.
- Terminal approval modal renders file paths as literal text instead of letting
  Rich interpret path characters as markup.
- Desktop new-chat IPC listener returns an unsubscribe function and is cleaned
  up on effect teardown, so listeners no longer accumulate across re-renders.

## [2.0.1] - 2026-11-15

### Added

- Terminal auto-update lifecycle: background checks, download and staging under
  `.dizel/updates/`, apply on quit or next launch, `/update`, `/update install`,
  `/check-update`, and launcher `dizel update` / `dizel check-update`. A status
  badge marks a staged release.
- Desktop and terminal update badges for available, downloading, and ready
  states.
- Inline tool-result drawers in both clients. Edit calls show a numbered diff,
  writes show the file content, reads show line ranges, and searches show the
  query, URL, and results. Code is syntax-highlighted.
- Desktop write/edit approval card with the target path, action, diff, and line
  counts.
- TUI session tabs with per-tab close and new-tab controls, a scrollable strip,
  per-tab welcome state, and a session-only context rail.
- TUI Thought view, debug panel, and `/debug` tool fixtures. The desktop
  `/tooltest` fixtures cover grep, read, write, edit, and web search.
- v1.6 reward-filter scoring and DPO pair-selection scripts, plus their tests.

### Changed

- Tool results share one `dizel_core/tools.py` UI payload across the API, TUI,
  and desktop app, including truncation limits.
- Versioning now has one source: `dizel_core/version.py`. `scripts/set_version.py`
  provides `show`, `set`, `bump`, and `check`, stamps the GUI package, lockfile,
  and npm launcher, and both CI and release jobs reject drift. Startup banners
  read the canonical version.
- The npm launcher publishes as `@d4niel-dev/dizel`; install docs use the scoped
  package.
- The docs site spacing and responsive rhythm were corrected, including the
  release page and theme footer.
- v1.6 post-training tooling requires `transformers>=4.51.0` for Qwen3 and
  Skywork reward models.

### Fixed

- Windows `.cmd` launchers are checked out with CRLF on every platform, so the
  terminal launcher and its CI check agree on Linux and Windows.
- Scoped docs CSS resets no longer collapse component margins in the hero,
  footers, and architecture cards.
- Staged terminal updates skip automatic apply on a source checkout with
  uncommitted changes; clean checkouts apply on exit or next launch.
- Tool rows keep their result payload available for inline expansion after a
  turn completes.

### Removed

- Obsolete v1.1/v1.2 dataset preparation and migration scripts, the session
  retitle utility, and the PySide6 smoke test.
