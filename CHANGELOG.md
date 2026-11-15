# Changelog

All notable changes to the Dizel app are recorded here. Release tags use the
app version from `dizel_core/version.py`.

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

- Scoped docs CSS resets no longer collapse component margins in the hero,
  footers, and architecture cards.
- Staged terminal updates skip automatic apply on a source checkout with
  uncommitted changes; clean checkouts apply on exit or next launch.
- Tool rows keep their result payload available for inline expansion after a
  turn completes.

### Removed

- Obsolete v1.1/v1.2 dataset preparation and migration scripts, the session
  retitle utility, and the PySide6 smoke test.
