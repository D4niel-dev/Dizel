# Dizel official site — design spec

**Status:** Direction approved; implementation pending
**Date:** 2026-11-13
**Owner:** D4niel-dev
**Scope:** Rework the static `docs/` GitHub Pages site
**Visual reference:** the interactive research-lab mockup shown in the 2026-11-13 session (Home, Model card, Docs, 390 px preview, design notes). The mockup is the light-theme reference; the night theme re-maps the same tokens and hierarchy. The mockup's inline ring was a stand-in for the brand mark; production uses the official `assets/app/Dizel.png` directly.

---

## 1. Summary

Replace the current single-page 1.6 announcement with a layered official site:

- **Home** — what Dizel is, what runs today, and three clear paths.
- **Model** — a real model card with architecture, training, evaluation, limitations, and citation.
- **Docs** — a documentation shell with a working install/launch path.
- **Releases** — version state, known limits, and links to the repository.
- **Demo** — the existing scripted desktop replica, kept as is.

The design direction is **research lab / technical record**, not a launch page. The signature is an **evidence ledger** where every public claim carries a status, a value, and a source. The site stays static on GitHub Pages, with no runtime framework. It ships in **paper light** and **graphite night** themes, selected by a persistent nav toggle; the demo keeps its own dark interface.

The current landing page is replaced, not extended. The existing `demo.html`, `styles/demo.css`, and `js/demo.js` remain the demo app and are not restyled by this work.

---

## 2. Goals and non-goals

### Goals

1. Make the official site feel like a record of a real project: evidence first, navigation second, atmosphere last.
2. Let each audience find its path in one click: install v1.5, read the model card, or open the docs.
3. State the v1.5/v1.6 difference without blurring it: v1.5 is published and runnable; v1.6 Flash is a preview with unpublished weights and no downstream benchmarks.
4. Give documentation a real surface with a sidebar, on-page TOC, code copy, and a working install path.
5. Remove the AI-launch-page grammar: animated hero, gradient headline, glowing dashboard, fake live metrics, counter animations, decorative card grid, provider marquee.
6. Keep the site fast, accessible, dependency-free at runtime, and deployable to GitHub Pages from `/docs`.
7. Offer a readable night theme that keeps the same evidence-first hierarchy — graphite and paper, not near-black neon.

### Non-goals

- No framework, SSG, or client-side router. The output is static HTML/CSS/JS.
- No Qoder Sites migration. `docs/` is not a Qoder Site and this work does not move hosting.
- No invented benchmarks, release dates, or download links.
- No theme system on the demo. The public toggle does not affect the demo's existing dark interface.
- No fake search. Search is deferred until a real static index exists.
- No rework of the demo app, and no restyle of its dark interface.
- No user accounts, analytics, comments, or backend.

---

## 3. Audiences and jobs

| Audience | What they need first | Primary path |
|---|---|---|
| Local-AI builders | What runs today, hardware needs, install path, privacy | Home → Install v1.5 → Launch |
| Model / ML peers | Architecture, training method, evaluation, limitations | Home → Model card |
| General AI users | What it is, screenshots, a demo, privacy reassurance | Home → Demo / App screenshot |
| Contributors | Repository, training docs, release policy | Docs / Releases → GitHub |

The home page addresses all three by layering: a plain-language thesis, a technical status ledger, and a three-path documentation index. The model card and docs carry the depth.

---

## 4. Information architecture

### Routes

| Route | Page | Job |
|---|---|---|
| `/` | Home | Thesis, evidence ledger, availability, start paths |
| `/model.html` | Model card | Architecture, training, evaluation, limitations, citation |
| `/guide/` | Docs overview | Choose a path through the docs |
| `/guide/install.html` | Install | Requirements and install commands |
| `/guide/launch.html` | Launch | Desktop app and TUI launch |
| `/releases.html` | Releases | Version state, known limits, repository releases |
| `/demo.html` | Demo | Existing scripted desktop replica |
| `/404.html` | Not found | Return to home or docs |

Paths are relative and portable under `https://d4niel-dev.github.io/Dizel/`. The guide lives at `docs/guide/` rather than `docs/docs/` to avoid a doubled path segment.

### Global navigation

Desktop: `Model · Docs · Releases · Demo`, with the brand returning home, a small availability label (`v1.5 available`), the theme toggle, and a GitHub icon.

Mobile: brand, theme toggle, GitHub, menu button. The menu contains the four primary links plus a back-to-home link. The theme toggle stays outside the menu so it is always one tap away.

The theme toggle is an icon button with a stable accessible name, `Dark theme`, and `aria-pressed` reporting whether the night theme is active. The icon previews the destination: a moon in light mode, a sun in dark mode. Its title mirrors the action (`Switch to dark theme` / `Switch to light theme`).

The active primary link uses `aria-current="page"` and a hairline underline. On the home page no section link is current; the brand is home.

### Docs navigation

The eventual sidebar IA is:

- Getting started: Overview, Requirements, Install, Launch
- Desktop app: Launch, Settings, Providers, Voice
- Terminal: Launch, Commands, Sessions, MCP plugins
- Model: Checkpoints, Loading, Modes and tiers
- Reference: CLI flags, Configuration, Troubleshooting

Groups appear in that order, but a row is linked only after its content is migrated. Phase 1 shows only Overview, Install, and Launch; new groups and rows are added as pages land. No dead links, no "coming soon" rows in production.

### Footer

Three link groups and a bottom line:

- Product: Desktop app, Terminal UI, Interface demo
- Research: Model card, Architecture, Evaluation
- Project: Releases, Training log, Repository
- Bottom: version/record, MIT license, citation link, built-in-public line, back to top

---

## 5. Content rules and sources

### Truth rules

1. **Only publish numbers that exist in the repository or a published run.** The current verified numbers are:
   - v1.5 dense: 454.78M total, 28 layers, d_model 1,024, 16/8 Q/KV heads, SwiGLU 3,072, 100,000-token vocabulary.
   - v1.6 Flash Preview: 435.3M total, 293.7M active per token, 20 layers, d_model 768, 12/6 Q/KV heads, 4 experts × 1,536, top-2, 151,936-row vocabulary, 2,048-token context.
   - Pretrain status: step 10,500 of 45,777; 688.1M of 3B tokens; held-out BPB 1.1236; validation loss 3.4479.
2. **Do not invent a benchmark table.** There is no published downstream benchmark result. The evaluation section shows held-out BPB, validation loss, and an explicit "Downstream benchmarks — not published" row.
3. **Label state with words, not only color.** `Published`, `Training`, `Verified`, `Not published`, `Pending` always appear as text next to any color.
4. **Separate the preview from the release.** v1.6 Flash is marked `Preview`; v1.5 is marked `Published`. The primary home action is `Run v1.5 locally`.
5. **Do not imply a release date.** Use `Record 2026-11` and `Last verified at step 10,500`, not a stable-release date.
6. **Attribute evidence.** Ledger and model card values link to the training log, the evaluation script, or the repository README.

### Content migration map

| README section | Destination |
|---|---|
| Requirements, Install, Launch, Headless generation | `guide/install.html`, `guide/launch.html` |
| Checkpoints in this repository | `guide/checkpoints.html` |
| Local inference, Model switching, Tiers and modes | `guide/loading.html`, `guide/modes.html` |
| Providers | `guide/providers.html` |
| Sessions, context, usage, Voice, Keyboard | `guide/sessions.html`, `guide/voice.html` |
| Architecture, Parameter accounting, Why these choices | `model.html#architecture` |
| Training, Resume rules, Release policy, Evaluation | `model.html#training`, `model.html#evaluation` |
| What's coming, six gates, Pro, no vision | `releases.html`, `model.html#limitations` |
| FAQ | `guide/faq.html` in phase 2, or a short FAQ block on `releases.html` if it stays small |
| Updates / Publishing a release | `releases.html` |
| VRAM guide, Overfitting, Extending Dizel | Later guide pages |

The migration copies and edits; it does not bulk-dump README text into HTML. Long command blocks stay in code fences; prose is shortened and linked.

### Voice

- Plain verbs, sentence case, active voice.
- Specific over clever: "Run v1.5 locally", not "Begin your journey".
- A control names what happens: `Install v1.5`, `Read the model card`, `Copy`.
- Empty and unavailable states say why and what to do next.
- No exclamation marks, no "revolutionary", no "unleash".

---

## 6. Visual system

### Concept

A technical report cover, not a marketing hero. The page should feel assembled from the project's own artifacts: checkpoints, training logs, a model card, a terminal, and an app window. One bold move: the most prominent element is a **status ledger**, not an illustration or a dashboard.

### Signature: the evidence ledger

On the home page, the right column of the cover is a bordered table:

| Claim | Status | Value |
|---|---|---|
| v1.5 · Dense control | Published | 454.78M |
| v1.6 Flash · Preview | Training | 22.9% |
| Held-out BPB · Step 10,500 | Verified | 1.1236 |
| Validation loss · Step 10,500 | Verified | 3.4479 |
| Holdout set · 500 documents | Not published | — |
| v1.6 weights · Instruct release | Pending | — |

Header: `Evidence ledger` / `As of 13 Nov 2026`. Footer: "Each value is copied from the last validation run; nothing here polls or animates. Sources: training log, held-out evaluation, and model card."

This is the one risk: a table as the hero. It works because the subject is a model record, not a lifestyle product.

### Color tokens

Primitive values:

| Token | Hex | Use |
|---|---|---|
| `--chalk-050` | `#FBFBF8` | Raised paper surfaces |
| `--chalk-100` | `#F1F2EC` | Page background (cool chalk paper) |
| `--chalk-200` | `#E7E9E1` | Secondary surface |
| `--chalk-300` | `#D8DBD1` | Hairline borders |
| `--ink-900` | `#15171B` | Primary text |
| `--ink-700` | `#3C424A` | Body text |
| `--ink-500` | `#666D77` | Captions, metadata |
| `--peri-500` | `#4457D5` | Links, primary action, focus |
| `--peri-700` | `#3242A5` | Primary hover |
| `--peri-100` | `#E5E8F9` | Selected/soft accent |
| `--amber-700` | `#85560C` | Training state |
| `--green-700` | `#28664A` | Published/verified state |

Semantic roles: `--bg`, `--surface`, `--surface-2`, `--ink`, `--ink-2`, `--ink-3`, `--line`, `--line-2`, `--accent`, `--accent-hover`, `--accent-soft`, `--on-accent`, `--focus`, `--ok`, `--warn`, `--code-bg`, `--code-ink`, `--code-comment`, `--code-key`, `--code-string`, `--shot-mat`, `--selection`.

Brand mark: use the official `assets/app/Dizel.png` directly (736×729, transparent, ~153 KB). Do not create a resized derivative; the browser scales it. It renders at 40×40 with `object-fit: contain`, transparent in both themes, with no plate or tile behind it. Do not redraw it as an inline ring or wrap it in a generic app-icon square; the icon's own purple/red colors are the mark. `Dizel.ico` remains the favicon.

Rules:
- Periwinkle is the only action/link accent.
- Amber and green encode state only, always with a word.
- The peach–pink–lavender gradient is retired. The pink survives only as the logo arc.
- No gradients on text, surfaces, or buttons.
- No glow, blur, glass, or colored shadows in either theme.

### Dark theme: graphite night

`<html data-theme="dark">` re-maps semantic roles; primitives stay shared.

| Semantic role | Light | Dark |
|---|---|---|
| `--bg` | `#F1F2EC` | `#141519` |
| `--surface` | `#FBFBF8` | `#1B1D22` |
| `--surface-2` | `#E7E9E1` | `#23262C` |
| `--ink` | `#15171B` | `#F2F0E9` |
| `--ink-2` | `#3C424A` | `#C7C5BD` |
| `--ink-3` | `#666D77` | `#93958F` |
| `--line` | `#D8DBD1` | `#34373E` |
| `--line-2` | `#BFC3B8` | `#4A4E56` |
| `--accent` | `#4457D5` | `#9EAAFF` |
| `--accent-hover` | `#3242A5` | `#C0C8FF` |
| `--accent-soft` | `#E5E8F9` | `rgba(158,170,255,.14)` |
| `--on-accent` | `#FFFFFF` | `#101115` |
| `--focus` | `#4457D5` | `#C0C8FF` |
| `--ok` | `#28664A` | `#8FD6AC` |
| `--warn` | `#85560C` | `#E5B76D` |
| `--code-bg` | `#17191D` | `#0F1013` |
| `--code-ink` | `#E6E6E0` | `#ECEAE3` |
| `--code-comment` | `#8F959E` | `#8E939D` |
| `--code-key` | `#B7C2FF` | `#B7C2FF` |
| `--code-string` | `#E6C58F` | `#E6C58F` |
| `--shot-mat` | `#FBFBF8` | `#22252B` |
| `--selection` | `rgba(68,87,213,.22)` | `rgba(158,170,255,.28)` |

Dark-theme rules:
- The night theme is **not a cosmic inversion**. No pure black, no neon, no glow. The page is graphite; text is paper; hairlines stay visible; the ledger and tables keep the same weights.
- Set `color-scheme: light` on the light root and `color-scheme: dark` on the dark root so native scrollbars and form controls follow.
- Primary buttons use `--on-accent` text and `--accent-hover` on hover. Do not hardcode white text; the dark primary button is light periwinkle with near-ink text.
- Code blocks use `--code-bg`, which stays darker than the page in both themes, plus a `--line` border. The copy button uses `--surface-2` and `--ink-2`.
- The app screenshot sits on `--shot-mat` with a `--line` border; the mat keeps the dark screenshot from merging into the night page.
- The logo is the official app icon asset; do not recolor, redraw, or plate it. Its purple/red colors stay as authored, at 40×40 in both themes.
- Status colors are tuned per theme and always paired with a text label.
- Contrast is checked in both themes; do not infer dark contrast from light. Body text, `--ink-3`, `--accent`, and `--on-accent` on `--accent` must pass 4.5:1.

### Theme selection

- The theme toggle is available on every public page in the primary nav.
- Preference order on first visit: stored `dizel.site.theme` → `prefers-color-scheme` → light.
- Once the user toggles, the choice is stored and wins over system changes. If there is no stored choice, the site follows system changes live.
- The demo does not read or write this preference; it keeps its own dark theme.
- The toggle is hidden when JavaScript is unavailable; the site remains fully usable in light mode. No dead control.

### Typography

| Role | Family | Notes |
|---|---|---|
| Display | **Newsreader** (self-hosted variable, OFL) | H1/H2, report voice; 500–600 weight; tight tracking |
| Body / UI | **Geist** (existing self-hosted) | Prose, navigation, labels |
| Data / code | **Geist Mono** (existing self-hosted) | Numbers, status labels, code |

Scale:
- Display: 84/76/58/46/34 px via `clamp`.
- Section H2: 46/40/34/28 px.
- Body: 16 px desktop, 15.5–16 px mobile; line-height 1.65–1.72.
- Small: 13–14 px; metadata 10–11 px mono with `.06–.12em` tracking.
- Code: 12–13 px mono.

Measure: 60–70 characters for prose; 53ch for the home deck. Headlines `text-wrap: balance`, paragraphs `text-wrap: pretty`.

Newsreader is added to `docs/assets/fonts/` with its OFL license and preloaded. Google Fonts is not used at runtime.

### Layout

- 1,180 px content shell; 12-column editorial grid.
- Page gutters: `clamp(20px, 4.2vw, 46px)`.
- Section rhythm: `clamp(56px, 7vw, 86px)`.
- Hairline rules separate sections; no card shadows.
- Radii: 4–6 px on buttons, inputs, code; no pill-shaped everything.
- The home cover is a two-column grid: copy left, ledger right. The ledger moves below the copy before the layout stacks.
- The model card is a three-column paper: section rail, main measure, at-a-glance rail.
- The docs shell is a three-column layout: sidebar, main, on-page TOC.

### Motion

- Hover/focus transitions only, 150–200 ms.
- No scroll reveals, no counters, no word-by-word text, no marquee, no parallax.
- The router lab changes state on click with no entrance animation.
- The theme toggle applies immediately. No background/color cross-fade, so text does not appear to melt between themes.
- `prefers-reduced-motion: reduce` disables transitions and smooth scrolling.

### Anti-slop checklist

- [ ] No gradient headline or gradient body text.
- [ ] No glowing or glassy cards.
- [ ] No fake "live" pulse or animated dashboard.
- [ ] No count-up numbers.
- [ ] No word-by-word hero reveal.
- [ ] No provider logo marquee.
- [ ] No generic three-card feature grid.
- [ ] No emoji as icons.
- [ ] No dead controls (search, helpful buttons, version options, theme toggle without JS).
- [ ] No near-black + neon dark theme; the night theme is graphite and paper.
- [ ] No flash of the wrong theme on load or navigation.
- [ ] Every section has a content job, not just visual rhythm.

---

## 7. Page specifications

### 7.1 Home

**Cover**
- Eyebrow: `Model record / Preview / 2026-11`.
- H1: `A small model that stays yours.` Serif, 84 px desktop, `yours.` italic periwinkle.
- Deck: "Dizel 1.6 Flash is a 435.3M-parameter sparse model with 293.7M active per token. It is training in public on one T4. The published v1.5 app runs locally today."
- Actions: primary `Run v1.5 locally` → install guide; secondary `Read the model card`; text link `Training log`.
- Meta: MIT licensed · Windows / macOS / Linux · Python 3.10+.
- Ledger at right as specified above.

**Availability**
- H2: `Two models. One is ready.`
- Copy: "The app runs v1.5 today. The 1.6 preview has no published weights and no downstream benchmarks; its training log is the public evidence. We do not blur those two states."
- Matrix: Model / Status / Parameters / Runs on / Start.
  - Dizel 1.5 · Published · 454.78M all active · Desktop, TUI, GGUF · `Install v1.5`.
  - Dizel 1.6 Flash · Training · 435.3M total / 293.7M active · Weights pending · `Training log`.
- Mobile: the matrix becomes stacked label/value rows, not a horizontal scroll.

**Start here**
- H2 `Start here.` Copy: "Dizel is a model and two clients. Pick the surface that matches how you work."
- Phase 1 rows, all with real destinations: Install Dizel → `guide/install.html`; Use the terminal → `guide/launch.html#tui`; Read the model card → `model.html`; Browse the docs → `guide/index.html`.
- Provider and MCP rows are added when their guide pages land. No dead rows.

**Evidence**
- H2 `The model’s home.` Copy: Electron shell over a local Python engine; runs v1.5; prompts stay on the machine.
- App screenshot in a hairline frame with a caption and repository link.
- Right column `The sparse bet`: 435.3M total, 293.7M active, 4 experts top-2, 2,048 context, 151,936 vocabulary, MIT. Link to architecture.

**Router lab**
- H2 `Route a token.` Copy: dropless router sends a fragment to two of four experts.
- Six token buttons; selecting one updates the token label, two active expert boxes, and the readout `E0 + E2 · 2 of 4 awake`.
- Routes are illustrative; footer says `Illustrative routes at the 25% even-load target.`
- Keyboard: buttons with `aria-pressed`; live readout with `aria-live="polite"`.

**Limitations**
- H2 `What this is not.`
- Four items: not a vision model; not a 1.6 release; not publicly benchmarked; not a long-context model.
- Copy: "A preview page should state its boundaries before asking for trust."

**Footer** as specified.

### 7.2 Model card

**Header**
- Breadcrumb: `Dizel / Model / 1.6 Flash Preview`.
- H1: `Dizel 1.6 Flash Preview`.
- Meta: `Model card · 2026-11 · Preview · MIT`.

**Layout**
- Left rail: Overview, Architecture, Training, Evaluation, Limitations, Citation. Sticky on desktop; active section follows scroll (scrollspy). Hidden on narrow screens.
- Main: 60–70ch measure.
- Right rail: At a glance, preview warning, sources. Sticky on desktop; moves above the main on narrow screens.

**Sections**
1. **Overview** — abstract; status note "Evaluation checkpoint. This is a pretrain snapshot, not an instruct release. Use v1.5 for the published model."; what is verified; what is not.
2. **Architecture** — decoder-only Transformer: pre-RMSNorm, RoPE, GQA, SwiGLU, tied embeddings, dropless top-2 MoE. Block-order figure. Parameter table:
   - 1.5 Dense: 454.78M total/active; 28 layers; d_model 1,024; 16/8 heads; SwiGLU 3,072.
   - 1.6 Flash: 435.3M total; 293.7M active; 20 layers; d_model 768; 12/6 heads; 4 × 1,536 top-2.
   - Why top-2, why GQA, router stability.
3. **Training** — one T4, 65,536-token effective batch, validate every 500 steps on a fixed 500-document holdout, last verified step 10,500. Stage table: Pretrain (training, 22.9%), SFT (not started), token-level KD (not started), DPO (optional). BPB caveat. Progress line with 688.1M / 3B, 10,500 / 45,777, BPB 1.1236.
4. **Evaluation** — held-out BPB 1.1236; validation loss 3.4479; downstream benchmarks not published. Link to evaluation script. Do not add a benchmark table until results exist.
5. **Limitations** — no vision; 2,048-token context; no instruct release; no public downstream benchmarks; holdout not published.
6. **Citation** — working BibTeX entry, labeled as a draft; copy button; stable release gets a permanent record.

**Behavior**
- Left rail scrollspy; right rail sticky; code copy; external links open in new tabs with `rel="noopener"`.
- Tables use `caption`/`scope`; wide tables scroll inside a wrapper on mobile.

### 7.3 Guide / docs

**Layout**
- Docs bar: `Dizel Docs` label and current version. No second brand mark and no duplicate GitHub icon; GitHub lives in the primary nav.
- Sidebar: grouped links; active page has a periwinkle left rule.
- Main: breadcrumb, H1, lede, steps, code blocks, callouts, next/previous.
- Right rail: on-page TOC, edit-this-page link, version/record note.
- On mobile: sidebar becomes a `<details>` documentation menu; right rail hides.

**Phase 1 pages**
- `guide/index.html`: Overview — what Dizel is, model vs app vs TUI, how to choose a path, prerequisites, links.
- `guide/install.html`: Requirements, clone, virtualenv, PyTorch, requirements; training-only `HF_TOKEN` warning; next link.
- `guide/launch.html`: Desktop app and TUI tabs with real commands; `DIZEL_PYTHON` note; next links.
- Sidebar only shows these pages; the model card and repository live in the primary nav and footer. No empty rows.

**Content rules**
- Code blocks are dark ink surfaces; copy button with clipboard fallback; no line-level highlights.
- Commands are copied from README and kept current; no invented flags.
- "Was this page helpful?" is replaced with `Report a docs issue` linking to GitHub. No dead feedback buttons.
- The version selector appears only when a second real docs version exists; phase 1 shows `v1.5` as a label, not an interactive select with a fake preview option.
- Search is omitted until a real static index exists. No non-functional search input.

### 7.4 Releases

- H1 `Releases`.
- Current table: v1.5 (published), v1.6 Flash Preview (training), with parameters and links.
- `Known limits` list: weights pending, downstream benchmarks not published, 2,048 context, no vision.
- `Release policy`: six measured gates, not a date; link to README release policy.
- Link to GitHub releases; no fabricated changelog dates.
- Phase 2 adds per-version changelog entries from README "What's new".

### 7.5 Demo

- Keep `demo.html`, `styles/demo.css`, `js/demo.js`, `assets/demo/` as they are.
- Link from the primary nav and home evidence section.
- The demo's dark theme is intentionally separate from the public paper site.
- Do not add the public site header inside the demo; it is an app replica.
- The demo does not render the theme toggle, does not read `dizel.site.theme`, and is not restyled by the night theme.

### 7.6 404

- Paper page with `Page not found.` and links to Home, Docs, Model card, Repository.
- Reuse the global nav/footer; no illustration.

---

## 8. Components and states

| Component | Default | Hover | Focus | Active/current | Disabled |
|---|---|---|---|---|---|
| Primary button | `--accent` fill, `--on-accent` text | `--accent-hover` | 2px `--focus` ring | pressed accent | muted, no pointer |
| Secondary button | transparent, hairline border | `--surface-2` fill | ring | border `--ink` | muted |
| Text link | `--ink-2`, underline | `--accent` | ring | `--accent` | — |
| Nav link | `--ink-2` | `--ink` + underline | ring | `--ink` + hairline underline + `aria-current` | — |
| Theme toggle | 40×40 icon button (44×44 on mobile), `--ink-2` | `--surface-2`, `--ink` | ring | `aria-pressed="true"` + sun icon | hidden without JS |
| Status tag | mono label + square dot | — | — | color + text | — |
| Ledger row | hairline bottom | — | — | — | — |
| Code copy | `--surface-2`, mono | `--ink` | ring | `Copied` / `Copy failed` | — |
| Docs sidebar link | `--ink-2` | `--accent` | ring | `--accent` + left rule | — |
| Mobile menu | hidden | — | ring | expanded state on button | — |

Implementation rules:
- Buttons and inputs use 5 px radius; status tags are text + dot, not pills.
- Focus is always visible; never `outline: none` without a replacement.
- Hover is never the only affordance.
- Color is never the only status signal.
- The theme toggle is keyboard reachable, has a stable accessible name, and reports state with `aria-pressed`; the icon is decorative and changes to preview the destination theme.
- Tables have captions and header scopes; wide tables scroll inside a wrapper.

---

## 9. Responsive behavior

Breakpoints: **1440 / 1024 / 820 / 600 / 390**.

- 1024: model card right rail collapses below the main; docs TOC hides.
- 820: home cover stacks; ledger moves below copy; availability matrix remains a table; docs sidebar hides and the mobile docs menu appears; app evidence stacks.
- 600: nav links hide behind the menu; the theme toggle stays in the top bar; buttons stack full-width in the cover; ledger keeps all three columns; matrix becomes stacked rows; router route becomes vertical; footer becomes two columns.
- 390: single-column content; 16 px body; 44 px minimum target on mobile controls (including the theme toggle); no horizontal page overflow; code and wide tables scroll inside their own wrappers.
- Sticky rails are disabled below 1024.
- Images use fixed aspect ratios/width-height attributes to avoid layout shift.

---

## 10. Accessibility

Target: **WCAG 2.2 AA**.

- Skip link to `#main` on every page; `main` is focusable.
- Headings are sequential; one H1 per page.
- Contrast: light theme — body ink `#15171B` on `#F1F2EC` (high), ink-3 `#666D77` for metadata only, periwinkle `#4457D5` on paper passes AA, amber/green state colors pass AA when paired with text. Dark theme — paper `#F2F0E9` on graphite `#141519`, ink-3 `#93958F`, accent `#9EAAFF`, and near-ink `#101115` on the light periwinkle primary button all pass 4.5:1. Check both themes with a contrast tool.
- Focus visible on all interactive elements (2 px `--focus`, 3 px offset). The dark focus ring is light periwinkle, not the light-theme blue.
- Touch targets ≥44 px on mobile for nav, menu, theme toggle, buttons, tabs, tokens, and copy.
- The router lab is keyboard-operable; `aria-pressed` on tokens; `aria-live="polite"` on the route readout.
- Tables have captions and `scope`; status is not color-only.
- `prefers-reduced-motion` disables transitions and smooth scrolling.
- Mobile menu sets `aria-expanded` and `aria-controls`.
- Theme toggle: stable accessible name `Dark theme`, `aria-pressed` reports the current mode, the icon is `aria-hidden`, and the title mirrors the action. Keyboard activation works; the state is announced.
- The chosen theme is applied before first paint by an inline head script, so there is no flash of the wrong theme. The preference persists across pages and reloads.
- Without JavaScript the site stays fully usable in light mode; the theme toggle is hidden, not a dead button.
- Links that open a new tab say so visually or in the accessible name.
- Text remains usable at 200% zoom; no fixed-height text containers.
- No autoplaying media; no motion that cannot be paused (there is no such motion).

---

## 11. Performance and SEO

### Fonts and assets

- Self-host **Newsreader** variable WOFF2 + `OFL-Newsreader.txt` in `docs/assets/fonts/`.
- Keep existing Geist and Geist Mono; Outfit remains only for the demo.
- Preload Newsreader and Geist; `font-display: swap`; no Google Fonts at runtime.
- App screenshot: use `assets/site/app-desktop.webp` (49 KB) with width/height; lazy load below the fold.
- OG image: use the existing `assets/site/app-desktop.png` (435 KB) or generate a smaller 1200×630 paper card in phase 2. Do not block launch on a new OG image.
- Icons are inline SVG, one stroke family, no icon font.

### JavaScript

- One `js/site.js` for: mobile menu, docs tabs, copy buttons, router demonstration, model-card scrollspy, optional TOC highlighting, and the theme toggle.
- A small inline theme-init script in the head, before the stylesheet, reads `dizel.site.theme`, falls back to `prefers-color-scheme`, sets `data-theme`, `color-scheme`, and the `js` class, and updates the `theme-color` meta. It is inline, blocking, and adds no network request; it prevents a flash of the wrong theme.
- The toggle writes `dizel.site.theme` and updates the meta. If no preference is stored, the page follows system changes live.
- No framework, no polyfills, no external scripts.
- Both themes live in `site.css`; no theme-specific stylesheet request.
- All content exists without JS; JS only enhances. Without JS the site stays light and the toggle is hidden.

### SEO

- Per-page `<title>`, meta description, canonical, Open Graph, Twitter card.
- `SoftwareApplication` or `WebSite` JSON-LD on home; `FAQPage` only if the FAQ is on the page.
- `theme-color` meta starts at the light paper value, is corrected by the head script, and updates when the toggle changes. The demo keeps its own `#0e0e0f`.
- `sitemap.xml` and `robots.txt` generated/updated.
- 404 page.
- Relative internal links; external links use `rel="noopener"`.
- No `noindex` on real pages.

---

## 12. Technical architecture

### Stack

- Static HTML/CSS/JS on GitHub Pages.
- A tiny **stdlib-only Python layout builder** because the guide will grow beyond a handful of pages and the header/footer/sidebar must stay identical.
- Generated HTML is committed; GitHub Pages serves `/docs` with no build step.

### File tree

```text
docs/
  index.html                  generated home
  model.html                  generated model card
  releases.html               generated releases page
  404.html                    generated not-found page
  demo.html                   existing demo (unchanged)
  guide/
    index.html                generated docs overview
    install.html              generated install page
    launch.html               generated launch page
  styles/
    site.css                  new public-site design system
    demo.css                  existing demo styles (unchanged)
  js/
    site.js                   new shared behavior
    demo.js                   existing demo behavior (unchanged)
  assets/
    fonts/                    Newsreader + OFL, existing Geist/Geist Mono/Outfit
    site/                     app screenshot, OG assets
    app/                      icons
    demo/                     existing demo assets
  _site/
    layouts/
      public.html             nav/footer shell for home, model, releases, 404
      docs.html               docs shell with sidebar/TOC
    partials/
      nav.html
      footer.html
      docs-sidebar.html
      docs-toc.html
      theme-init.html           inline no-flash head script
    pages/
      index.html
      model.html
      releases.html
      404.html
      guide-index.html
      guide-install.html
      guide-launch.html
  tools/
    build_site.py             stdlib-only builder
```

### Builder contract

- `python docs/tools/build_site.py` reads `_site/pages/*.html`, a small JSON front-matter block (title, description, canonical, active nav, layout), and injects the shared partials.
- Placeholders: `{{title}}`, `{{description}}`, `{{canonical}}`, `{{nav}}`, `{{footer}}`, `{{content}}`, `{{active_nav}}`, `{{sidebar}}`, `{{toc}}`, `{{theme_init}}`.
- The public and docs layouts inject `{{theme_init}}` in `<head>` before the stylesheet; the demo is not generated by the builder and does not receive it.
- The `theme-color` meta is emitted before `{{theme_init}}` so the inline script can rewrite it to match the stored or system theme.
- `site.css` defines the light semantic tokens on `:root` and the dark overrides on `[data-theme="dark"]`; components reference semantic roles only.
- `site.js` reads the current theme from `document.documentElement.dataset.theme`, updates `aria-pressed`, swaps the icon, writes `localStorage`, and updates the `theme-color` meta.
- The builder is dependency-free and deterministic; running it twice produces identical output.
- The old `docs/index.html`, `styles/landing.css`, and `js/landing.js` are replaced. Delete them only after the new pages render and the old assets are confirmed unused.
- Do not delete `demo.html`, `styles/demo.css`, `js/demo.js`, `assets/demo/`, or `tools/` demo utilities.
- No CI workflow change; Pages continues to serve `/docs`.

### Why not an SSG or Qoder Sites

- An SSG adds a runtime/toolchain and a deploy pipeline for a site that is fundamentally static.
- Qoder Sites would change hosting and the public URL; not part of this request.
- The builder is small, transparent, and reversible.

---

## 13. Implementation phases

### Phase 1 — Foundation and the official surface

1. Add semantic design tokens for paper light and graphite night, plus the public-site CSS (`site.css`), including mobile/responsive rules.
2. Add Newsreader + OFL license; update font preloads.
3. Build the shared builder, public layout, nav, footer, docs layout, `theme-init` head script, and theme toggle.
4. Replace `index.html` with the home page; add `model.html`, `releases.html`, `404.html`.
5. Add `guide/` overview, install, and launch pages with real README commands.
6. Add `site.js`: mobile menu, docs tabs, copy, router lab, model scrollspy, theme toggle.
7. Add SEO files (sitemap, robots), per-page metadata, `theme-color`, and OG tags.
8. Verify both themes at desktop/mobile, keyboard, contrast, no horizontal overflow, no theme flash, and all links.
9. Only then remove the old landing assets if unused.

### Phase 2 — Documentation depth

1. Migrate remaining README sections into guide pages: checkpoints, loading, modes/tiers, providers, sessions/context, voice, keyboard, CLI, configuration, troubleshooting.
2. Grow the sidebar only as pages land.
3. Add a real static search: build a JSON index of titles/headings and filter client-side; only then show the search input.
4. Add per-version changelog entries from README "What's new".
5. Add a proper 1200×630 OG card.

### Phase 3 — Evidence expansion

1. Add a downstream benchmark table only when results are published, with prompts, scoring, model version, and date.
2. Expand the model card with the fixed holdout or a public evaluation report when available.
3. Add a training figure only with real series data; do not draw an illustrative curve.
4. Add a citation/DOI/archive link when the stable release exists.

---

## 14. Verification

Phase 1 is complete only when:

- `python docs/tools/build_site.py` runs cleanly and is deterministic.
- A local HTTP server serves the site; all internal links and anchors resolve.
- Headless/browser screenshots pass at 1440, 1024, 820, and 390 px **in both themes**:
  - no horizontal page overflow;
  - the cover/ledger stack correctly;
  - the availability matrix stacks on mobile;
  - the docs sidebar becomes the mobile menu;
  - code blocks scroll inside their frames;
  - the screenshot mat, ledger, and status tags remain distinct in dark mode.
- Theme pass:
  - the toggle works on home, model, guide, releases, and 404;
  - the choice persists across navigation and reload;
  - the first visit follows the system preference when no choice is stored;
  - no flash of the wrong theme on load;
  - `aria-pressed`, label, icon, and title match the active theme;
  - the toggle is keyboard-operable and hidden with JavaScript disabled;
  - the `theme-color` meta updates.
- Keyboard pass: skip link, nav, mobile menu, theme toggle, docs tabs, router tokens, copy buttons, model-card rail.
- Contrast check in both themes: paper, muted text, periwinkle links, primary button text, status tags, and code comments.
- `prefers-reduced-motion` shows no transitions; the theme switch itself is instant.
- The demo still opens and behaves as before, with its own dark theme and no public toggle.
- No old landing dependency remains on the new pages.
- The model card contains no invented numbers; every row maps to README/evaluation.
- The docs contain no dead search, version, or feedback controls.

---

## 15. Decisions locked by this spec

- **Static GitHub Pages, no SSG/Qoder Sites.**
- **Newsreader self-hosted** (OFL) with Geist/Geist Mono; Outfit is demo-only.
- **Dark mode included** as a graphite-night re-map of the semantic tokens, with a persistent nav toggle. Preference order: stored choice → system → light. The demo is unaffected.
- **Search deferred** until a real index exists; phase 1 ships no search input.
- **Helpful buttons replaced** with a GitHub "Report a docs issue" link.
- **Version selector deferred** until a second real docs version exists.
- **No benchmark table** until published results exist.
- **Old landing replaced**, demo untouched.
- **No commits** while Dizel 1.6 Flash is still in progress; the builder output and spec stay uncommitted until the maintainer says otherwise.
