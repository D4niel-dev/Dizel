# Changelog

All notable changes to the Dizel app are recorded here. Release tags use the
app version from `dizel_core/version.py`.

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
