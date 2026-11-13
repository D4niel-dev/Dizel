# MCP for GUI + TUI, and a Workspaces motion pass — Design

Date: 2026-11-08
Status: Approved design, ready for implementation planning
Owner: D4niel-dev

## Summary

Dizel gains an MCP client: it connects to user-configured MCP servers over stdio
or Streamable HTTP, exposes their tools to the model through the existing textual
tool loop, and offers full configuration parity in the GUI (Settings → MCP) and
the TUI (`/mcp`). Separately, the GUI Workspaces surfaces get a full motion pass.
The TUI has no Workspaces feature and is untouched by Part B.

## Decisions (from brainstorming)

- Direction: **client only** — Dizel consumes external MCP tools. No MCP server mode.
- Transports: **stdio + Streamable HTTP** (official Python `mcp` SDK).
- Configuration: **full parity** — GUI Settings tab and TUI `/mcp` modal, both
  reading/writing `settings.json` under `mcp.servers`.
- Gating: **per-server enable toggle** only, matching today's tools UX. No per-call
  approval, no per-tool allowlist.
- Workspaces: **full motion pass** over the popover and every secondary panel.

## Current state (evidence)

- Tools today are a textual protocol: `<tool name="NAME">{json}</tool>` defined in
  `dizel_core/tools.py:329-383`; `REGISTRY` at `dizel_core/tools.py:305`; parsing/
  execution at `tools.py:345-376`; prompt block at `tools.py:379`.
- Agent loop: `dizel_core/chat_manager.py:984-1089`, max 3 rounds, gated by
  `tools.is_enabled()` (`tools.py:55`).
- Tool events reach the GUI as SSE `notice` lines (`api/services/chat_service.py:325-336`)
  and the TUI via `inference/tui/bridge/chat_bridge.py:128-138`; both render tool
  blocks already.
- Sidecar: FastAPI in `api/main.py`, SSE chat at `api/routes/chat.py:145`; `/chat/tools`
  at `api/routes/chat.py:31`.
- Settings: `dizel_core/config_manager.py` persists JSON at `.dizel/settings.json`
  (`DIZEL_CONFIG_DIR` overrides); GUI mutates it through the sidecar config routes,
  TUI writes directly.
- GUI settings dialog: tabbed (`TABS` nav, `SettingsDialog.tsx:738`), section rows
  via `Section`/`Row` helpers.
- TUI settings: `inference/tui/panels/settings_modal.py`; commands registry at
  `inference/tui/commands/registry.py`.
- Workspaces: GUI-only popover (`Sidebar.tsx:135-168`) opening a left-docked panel
  (`SecondarySidebar.tsx:393-431`). Motion tokens `--ease`/`--dur` in
  `inference/gui/src/styles/tokens.css:46-47`; kill switches `:root[data-motion="off"]`
  (`app.css:3931-3937`) and `prefers-reduced-motion` (`app.css:3939-3947`).
- No MCP code exists anywhere. The `mcp` SDK (1.27.0) is present in the dev
  environment and will be added to `requirements.txt`.

## Part A — MCP client

### A1. Package layout

New package `dizel_core/mcp/`:

- `config.py` — parse, validate, and normalize `mcp.servers`.
- `manager.py` — connection lifecycle, tool discovery, tool execution.
- `bridge.py` — registers MCP tools into the textual tool registry.

`requirements.txt` gains `mcp>=1.27.0` under a new "MCP client" comment block.

### A2. Configuration schema

Stored in `settings.json`:

```json
{
  "mcp": {
    "servers": [
      {
        "id": "filesystem",
        "name": "Filesystem",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\work"],
        "env": {"GITHUB_TOKEN": "…"},
        "enabled": true
      },
      {
        "id": "linear",
        "name": "Linear",
        "transport": "http",
        "url": "https://mcp.linear.app/mcp",
        "headers": {"Authorization": "Bearer …"},
        "enabled": false
      }
    ]
  }
}
```

Normalization rules (`config.py`):

- Missing/invalid `mcp` → `{"servers": []}`; the app never crashes on bad config.
- `id`: slug `[a-z0-9_-]{1,32}`, lowercased from `name` when absent, deduplicated
  with `-2`, `-3` suffixes.
- `name`: 1–60 chars, required.
- `transport`: `"stdio"` or `"http"`; anything else rejects the server.
- stdio requires `command` (non-empty string); optional `args` (list of strings),
  `env` (string map).
- http requires `url` (http/https); optional `headers` (string map).
- Unknown keys are dropped from the normalized output.
- On read, invalid entries are skipped with a warning; on write (API/TUI),
  validation errors reject the whole save with a readable message.

### A3. Manager

`manager.py` exposes one process-wide `MCPManager`:

- Owns a dedicated daemon thread running its own asyncio event loop; all public
  methods are synchronous, marshalling into the loop with
  `asyncio.run_coroutine_threadsafe(...).result(timeout)`.
- Connection via the official SDK: `stdio_client(StdioServerParameters(...))` for
  stdio, `streamablehttp_client(url, headers=...)` for HTTP, each wrapped in
  `ClientSession` + `initialize()` + `list_tools()`.
- stdio env = `os.environ` merged with configured `env` (configured wins).
- Command resolution: `shutil.which` so `npx` maps to `npx.cmd` on Windows before
  spawning (avoids shell resolution problems).
- Per-server state: `disabled | starting | ready | error`, plus `error` text and
  cached tool list. State changes only through `apply()`, `test()`, or a lazy
  reconnect attempt on call.
- `apply(servers)`: diff against live sessions — stop removed/edited servers,
  start newly enabled ones, mark disabled ones without spawning.
- `test(server_id)`: force reconnect and return fresh status.
- `call(server_id, tool_name, arguments)`: ensure connected (one reconnect
  attempt on error), then `session.call_tool`; text content is joined; non-text
  content becomes `[image: <mime>]`, `[audio omitted]`, `[resource: <uri>]`
  placeholders; `isError` results are prefixed `Error: `.
- Timeouts: connect 15 s, list 15 s, call 60 s. A timeout or transport error
  becomes a tool result string, never an exception into the chat loop.
- Shutdown: close sessions, join the thread (5 s), then kill remaining child
  processes on Windows with `taskkill /T` (same pattern as the sidecar killer in
  `inference/gui/electron/main.cjs:244`).
- No background reconnect loop in v1; a call to an errored server retries once.

### A4. Tool bridge

- `dizel_core/tools.py` gains a generic seam:
  - `_EXTERNAL: Dict[str, Tool]` plus `set_external(tools)`.
  - `execute()` and `prompt_block()` consult `{**REGISTRY, **_EXTERNAL}`
    (builtins first, MCP tools sorted by name).
- `bridge.py` builds `Tool` objects from the manager's ready servers:
  - Names are namespaced `"{server_id}__{tool_name}"`, lowercased with
    non-alphanumerics mapped to `_`; collisions with builtins or between servers
    get `_2`, `_3` suffixes. The existing call regex (`[\w_]+`,
    `tools.py:342`) already matches these names.
  - The rendered schema mirrors `Tool.schema()` (`tools.py:66-68`): MCP
    `inputSchema.properties` become `"name": "type"` (required) or
    `"name": "type?"` (optional, using JSON-schema `required`); the description
    is the tool description's first line, capped at 150 chars.
  - `run` closes over the manager and calls `call(server_id, tool_name, **args)`
    synchronously.
- Refresh points: manager `apply()` and `test()` recompute ready servers and call
  `tools.set_external(...)`. Disabled or errored servers contribute no tools.
- `/chat/tools` (`api/routes/chat.py:31`) returns the merged catalog so the GUI
  ToolsSheet lists MCP tool names as chips automatically.

### A5. Agent loop and events

Unchanged. `chat_manager.send_with_tools` still gates on `tools.is_enabled()`,
parses calls, executes, and feeds results back. MCP calls surface through the
existing `on_tool` callback, so the GUI shows `tool server__tool(args) → …` as a
notice and the TUI renders its existing tool block. No new event types in v1.

### A6. Sidecar API

New `api/routes/mcp.py`, registered in `api/main.py`:

- `GET /mcp/servers` → `{servers: [...]}` where each entry carries the normalized
  config, live `state`, `error`, and `tools: [{name, description}]`.
  **Secret values are never returned**: `env` and `headers` are replaced by
  `env_keys` and `header_keys` arrays.
- `PUT /mcp/servers` → body `{servers: [...]}`, full replace. Validates, persists
  through `ConfigManager`, calls `manager.apply()`, returns the GET payload.
  Validation failures return 422 with a readable `detail`.
- `POST /mcp/servers/{id}/test` → reconnect and return that server's status.
- The manager starts in the app lifespan with the currently enabled servers and
  shuts down on lifespan exit.
- Existing token middleware covers all routes.

### A7. GUI surface

- `SettingsDialog.tsx`: add an `MCP` tab to `TABS`/`TAB_ICONS`/`TAB_HINTS` with a
  plug-style icon from the existing icon set, and a content branch rendering the
  new `McpSettings` component.
- New `inference/gui/src/components/McpSettings.tsx`:
  - Server rows: status dot (mint `ready`, pulsing amber `starting`, red `error`
    with message, dim `disabled`), name, transport badge (`stdio`/`http`), tool
    count, enable switch, and Test / Edit / Delete actions.
  - Add/edit form: name; transport select; stdio fields (command, args one per
    line, env `KEY=VALUE` per line) or http fields (url, headers `KEY=VALUE`);
    inline validation; Save/Cancel. When editing, env/header values start blank
    with a "leave blank to keep" hint — the API never sends values.
  - Empty state: short explanation + Add server button.
  - Delete asks for confirmation (matching existing destructive patterns).
  - Status refreshes on tab open, after every action, and on a 2 s interval while
    the tab is visible.
- `inference/gui/src/services/api.ts`: typed `mcpServers()`, `setMcpServers()`,
  `testMcpServer()` helpers following the existing fetch/token pattern.
- Motion follows the existing conventions (150–180 ms, `--ease`).

### A8. TUI surface

- New `inference/tui/commands/builtins/mcp.py` registers `/mcp` in the command
  registry; it opens a new `McpModal`.
- `inference/tui/panels/mcp_modal.py` (Textual ModalScreen):
  - List rows: state glyph (`●` ready, `◐` starting, `✕` error, `○` disabled),
    name, transport, tool count, enabled hint.
  - Keys: `space` toggle enabled, `t` test, `a` add, `e` edit, `d` delete, `esc`
    close; footer shows the bindings.
  - Add/edit opens a compact form screen: name, transport select, command/args,
    env lines, url/headers lines (args/env/headers are one-per-line text areas).
    Validation errors render inline; save writes via `ConfigManager`, then
    `manager.apply()` and `tools.set_external(...)`.
  - The list refreshes every 1.5 s while the modal is open.
- App lifecycle: `DizelCMDApp` starts the manager on mount with enabled servers
  and shuts it down on unmount. If no servers are configured, the manager starts
  idle (no thread work beyond a no-op loop).
- Help/command listings pick `/mcp` up automatically from the registry.

### A9. Trust and secret handling

- Enabled stdio servers are spawned at app start: that is the explicit trust
  boundary, stated in the GUI/TUI copy ("Enabled servers start with the app").
- `env`/`headers` values live in the user-owned `settings.json` as plain text
  (same file as the rest of the config) and never travel over HTTP — the API
  returns key names only.
- MCP tool descriptions are untrusted text injected into the system prompt;
  prompt injection by a malicious server is possible by design of MCP. Tool
  calls are visible in the transcript, which is the mitigation surface in v1.
- Every exposed MCP tool costs prompt tokens; the UI shows the tool count so the
  cost is visible.

## Part B — Workspaces motion (GUI only)

Files: `inference/gui/src/styles/app.css`, `SecondarySidebar.tsx`, `Sidebar.tsx`.

1. **Panel swap** — `.sub-body` content is keyed by the active view and gets a
   `ws-swap` animation on mount: opacity 0→1 + translateY(6px)→0, 180 ms
   `--ease`, `both`. Switching workspaces re-runs it via the key change.
2. **Staggered entrance** — lists/chips/cards inside the panel receive an
   inline `--i` index capped at 8; rows animate `ws-rise` (opacity 0→1,
   translateY(6px)→0, 220 ms) with `animation-delay: calc(var(--i) * 28ms)`.
3. **Hover lift** — `.sub-row:hover` gets translateY(-1px), stronger border, and
   surface-2 background, 150 ms; `:active` keeps the existing `scale(0.985)`
   press feedback.
4. **Monitor meters** — on panel mount each `.meter` fill sweeps from scaleX(0)
   to its value over 600 ms `cubic-bezier(0.22, 0.7, 0.3, 1)`, transform-origin
   left. Live value updates after mount are not re-animated.
5. **Library add/remove** — new prompt gems enter with scale(0.96)+fade over
   180 ms; removal adds a leaving class (opacity 0, scale(0.96), collapsed via
   the existing grid-rows pattern at `app.css:5075`) and unmounts after 200 ms.
6. **Empty states** — `.sub-empty` fades and rises 80 ms after the panel swap.
7. **Popover polish** — Workspaces popover item icons nudge 1px and scale 1.05
   on hover, 150 ms; the existing press state stays.
8. **Kill switches** — every new animation is covered by the existing
   `:root[data-motion="off"]` and `prefers-reduced-motion` blocks; verify none
   of them shift layout (transform/opacity only, except the Library removal
   collapse which uses the established grid-rows technique).

## Part C — Testing and verification

- `tests/test_mcp_config.py` — normalization, slug/id generation, validation
  failures, invalid-entry skipping.
- `tests/test_mcp_bridge.py` — namespacing, collision suffixes, schema rendering
  from `inputSchema`, dispatch through a fake manager, disabled/errored servers
  contributing nothing.
- `tests/test_mcp_manager.py` — connects to a fixture stdio server
  (`tests/fixtures/fake_mcp_server.py`, a FastMCP app with an `echo` and a
  failing tool), asserts ready state, tool list, call results, error results,
  and clean shutdown with no orphaned process.
- `tests/test_mcp_routes.py` — FastAPI TestClient with a temp `DIZEL_CONFIG_DIR`:
  PUT/GET roundtrip, 422 on invalid config, values withheld (`env_keys` only),
  test route responds.
- GUI: `npm run build` clean; headless screenshot (existing `DIZEL_SHOT`
  harness) of Settings → MCP with a fixture entry; workspace panels captured
  with motion on and with `data-motion="off"`.
- TUI: snapshot of the `/mcp` modal via the existing snapshot loop; a pilot test
  toggling a server against a temp config dir.
- Isolation rules: temp config dirs only, never write the user's real settings,
  no secrets in captures, delete only test artifacts created by the session.

## Risks and limits

- Prompt token cost grows with tool count; mitigated by per-server toggles and
  visible counts.
- A slow-starting stdio server delays the first tool round; surfaced as a tool
  result message rather than a hang (timeouts above).
- Remote MCP auth headers are stored in plain text in the user's own config
  file; acceptable for the local-first trust model, documented in the UI copy.
- Windows process trees must be killed explicitly on shutdown to avoid orphaned
  npx children.

## Non-goals

- MCP server mode, resources, prompts, sampling, elicitation.
- Per-call approval dialogs or per-tool allowlists.
- Workspaces in the TUI; implementing the Image/Presentation stubs.
- Any change to the landing page, browser demo, or model training code.

## Build order

1. Backend core: config, manager, bridge + unit tests.
2. Sidecar routes + GUI MCP tab + ToolsSheet catalog.
3. TUI `/mcp` modal + lifecycle.
4. Workspaces motion pass.
5. Full verification (backend tests, GUI build + captures, TUI snapshots).
