# PROJECT: Dizel — Complete CMD UI Rebuild (OpenCode / KiloCode-Inspired, Textual + Rich)

## CONTEXT
The old CMD UI has been removed from the codebase and must be rebuilt from scratch.

This is **NOT** a restoration of the old interface.
This is a complete redesign into a modern terminal-based AI workspace inspired by OpenCode and KiloCode.

The new CMD UI must feel like a real terminal IDE:
- multi-panel
- keyboard-first
- streaming
- transparent
- structured
- agent-aware
- model-aware
- tool-aware
- with a strong command workflow
- and full backend control

The CMD UI should be a separate product surface from the main PySide6 GUI, but it should share the same backend brain and infrastructure where appropriate.

## IMPORTANT
- Do **NOT** rebuild the deleted CMD UI as-is.
- Do **NOT** make a plain CLI chatbot.
- Do **NOT** make a one-column terminal with simple prompt/response only.
- Do **NOT** hide internal actions or tool usage.
- Do **NOT** fake transparency.
- Do **NOT** make the CMD UI dependent on the GUI.
- Do **NOT** break the GUI.
- Build a fresh, modular, maintainable command workspace from scratch.

## PRIMARY GOAL
Rebuild Dizel CMD UI into a modern terminal-based AI IDE using:
- Textual for layout, widgets, panels, focus handling, and app shell
- Rich for rich text rendering, colored logs, panels, tables, markdown-like formatting, status blocks, and streaming output styling

The result should feel like:
- a terminal workspace
- a command center
- an AI development cockpit
- a structured OpenCode/KiloCode-style interactive environment

## DESIGN PHILOSOPHY
The CMD UI must feel like a terminal IDE, not a shell.

Core experience goals:
- visible workspace layout
- persistent panels
- command + chat hybrid interaction
- full transparency of actions
- streaming responses token-by-token
- visible tool execution
- visible agent/model state
- clear session context
- clear status metrics
- fast keyboard-first workflow
- command palette / quick actions
- multi-pane context visibility
- polished but minimal terminal styling

## TECH STACK REQUIREMENT
1) Use both:
- **Textual** : for the actual terminal app structure, widgets, docking/panel layout, focus, input components, sidebar, main workspace, and event handling
- **Rich** : for high-quality formatted output, markdown-ish renderables, status blocks, logs, tables, progress, highlight panels, and stream rendering

2) Use each tool where it is strongest:
- Textual = UI shell, panes, layouts, interaction
- Rich = text formatting, structured output, logs, status lines, tables, code blocks, streaming visuals

## CORE UI LAYOUT
The CMD UI should be multi-panel and persistent, inspired by OpenCode/KiloCode:

> Suggested layout:

```
■▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎■
 |                       Top Header / Session Bar                        |
 |▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎◇▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎◇▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎|
 |                   |                               |                   |
 |   Left Sidebar    |        Main Workspace         |    Right Panel    |
 |                   |     Conversation / Logs /     |      Context      |
 |                   |          Tool Output          |                   |
 |                   |                               |                   |
 |▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎◇▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎◇▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎|
 |              Bottom Input Bar / Command Line / Hint Bar               |
■▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎▪︎■
```

> The interface should support:
- left panel for sessions, history, commands, agents
- center panel for conversation, tool logs, code, task execution, system messages
- right panel for context, model state, token usage, active tools, files, backend/provider info
- bottom input for prompts and commands
- top bar for active task/session/model/mode/status

The layout must remain usable even when resized.

## VISUAL STYLE REQUIREMENTS
The visual language should be:
- dark
- clean
- structured
- high-contrast enough to read easily
- minimal clutter
- subtle separators
- polished status indicators
- no unnecessary decoration

> Use:
- compact panels
- clear labels
- colored tags for status
- structured sections
- separators
- boxed output where needed
- consistent typography
- keyboard hints in a bottom bar or help overlay

## FULL TRANSPARENCY REQUIREMENT
This is **critical.**

The CMD UI must expose what it is doing internally.

> That means:
- show tool calls
- show agent routing decisions
- show model/mode selection
- show backend provider selection
- show token budget decisions if available
- show context trimming if performed
- show file reads / extraction / summarization / search actions
- show background tasks
- show errors clearly
- show generation state clearly
- show cancellations clearly

Do not hide these steps behind a generic *“thinking”* state only.
The user should be able to see what is happening.

## FUNCTIONAL GOAL
The CMD UI must fully control the backend.

It should not be a passive frontend.

It must control:
- model switching
- mode switching
- provider switching
- prompt building
- tool dispatch
- agent selection
- token budgeting
- context trimming
- streaming generation
- cancellation
- session saving
- history retrieval
- command execution
- config read/write

The backend should be treated as the engine and the CMD UI as the control surface.

## BEHAVIORAL REQUIREMENTS

1. Command + prompt hybrid
The CMD UI must accept both:
- slash commands
- natural language prompts

> Examples:
- `/help`
- `/model Dizel Pro`
- `/mode Planning`
- `/tools`
- `/history`
- `/clear`
- `/reset`
- `/config`
- `/export`
- `/agent`
- “Explain this code”
- “Summarize the file”
- “Search the web for...”

Commands should be parsed and executed explicitly.
Natural language should go through the normal assistant pipeline.

2. Streaming token-by-token
When the backend generates text, the CMD UI must stream it token-by-token if supported.
If full token streaming is not available, display incremental chunk streaming.
Never block the UI while waiting for a full response.

3. Multi-panel context
The CMD UI should always show useful context in the side panels, such as:
- current model
- current mode
- current provider
- current session name
- active tools
- token usage
- context length
- generation status
- active task
- maybe active files or sources
- maybe active agent routing

4. Rich output rendering
All output should be rendered in structured form:
- user input blocks
- assistant output blocks
- system status blocks
- tool execution blocks
- warnings
- errors
- file extraction results
- search results
- summaries
- background task notifications

5. Keyboard-first workflow
The CMD UI should be usable entirely with keyboard.

> Required shortcuts:
- Enter: submit prompt or command
- Up/Down: command history
- Ctrl+L: clear workspace or terminal view
- Ctrl+C or Esc: interrupt generation or cancel current action if safe
- Tab: command completion or focus traversal
- Ctrl+P or similar: open command palette
- /: command entry shortcut or command mode indicator
- Alt/Shift shortcuts are allowed if helpful

6. Command palette
Add a command palette or quick launcher style interface inspired by IDEs.
It should allow:
- quick command search
- model selection
- mode selection
- tool toggles
- session actions
- config shortcuts
- history search
- provider selection

7. Agent / backend transparency
If the system routes a prompt to an agent or special path:
- show which agent handled it
- show why it was routed there if possible
- show whether the prompt was sent to general chat, tool pipeline, dictionary agent, file agent, etc.
- show the final backend target

8. Session management
Support:
- create new session
- rename session
- switch session
- search sessions
- persistent history
- restore session
- export session
- clear session
- pin/favorite session if practical

9. Tool visibility
If a tool is used:
- show tool start
- show tool progress
- show tool completion
- show returned content clearly
- show errors if the tool fails
- show tool source/category if relevant

10. Model/mode switching
The CMD UI must support:
- Dizel Lite
- Dizel Pro
- Mila Lite
- Mila Pro
- mode variants such as Fast / Planning

When switched, the status bar must update immediately.
The prompt builder and backend routing must also update accordingly.

11. Token budget visibility
If the app already uses token budgeting logic, show it in the CMD UI:
- estimated input tokens
- budget selected
- current context window usage
- context trimming status
- output budget if relevant

12. Context trimming
If the conversation exceeds limits:
- show that context was trimmed
- show what kind of trimming was done if possible
- preserve important system context and tool outputs

13. Error handling
The UI must handle:
- invalid commands
- unknown flags
- backend failures
- provider errors
- tool failures
- missing model files
- interrupted generation
- context overflow
- session load failures

Errors must be visible and readable, not silent.

14. Background tasks
The interface should support non-blocking jobs:
- file extraction
- summarization
- search
- indexing
- model loading
- provider handshake
- export generation

These should appear as transparent background operations.

15. Extensibility
The architecture must be modular and future-proof.
It should be easy to add:
- new commands
- new tools
- new models
- new providers
- new side panels
- new output formats
- new shortcuts
- new agents

## RECOMMENDED ARCHITECTURE

> A) Presentation layer
Handles:
- Textual app shell
- multi-panel layout
- focus management
- top bar
- bottom input bar
- sidebars
- command palette
- session view
- contextual right panel

> B) Output formatting layer
Use Rich for:
- colored logs
- panels
- tables
- status blocks
- markdown-like text
- progress indicators
- code blocks
- warnings/errors
- tool logs

> C) Command parsing layer
Handles:
- slash command parsing
- flags
- arguments
- quoted strings
- aliases
- completion
- validation

> D) Session/controller layer
Handles:
- prompt routing
- model switching
- mode switching
- agent selection
- backend dispatch
- tool routing
- stream handling
- cancellation
- persistence

> E) Backend bridge layer
Handles:
- shared logic with the GUI backend
- provider calls
- local model calls
- token budgeting
- prompt building
- context management
- tool integration

> F) Persistence layer
Handles:
- history storage
- config storage
- session restore
- command history
- model/mode preferences

## RECOMMENDED PROJECT STRUCTURE
Create a dedicated CMD UI package with a clear, layered organization, for example:

```
cmd_ui/
├── main.py                     # Entry point
├── app.py                      # Textual app shell and lifecycle
├── state.py                    # Global CMD UI state definitions
├── layout.py                   # Main panel/layout composition
├── styles.py                   # Terminal/theme styling
├── keybindings.py              # Keyboard shortcuts and bindings
├── banners.py                  # Startup banner / welcome screen
├── shortcuts.py                # On-screen shortcut hints
├── session_manager.py          # Active session handling
├── history_store.py            # Persistent command/chat history
├── command_parser.py           # Slash command parsing and validation
├── command_registry.py         # Command registration / lookup
├── prompt_router.py            # Routes natural prompts vs commands
├── model_manager.py            # Active model / mode switching
├── provider_bridge.py          # Backend provider abstraction
├── tool_dispatcher.py          # Tool execution and tool state handling
├── token_status.py             # Token budget / context state display
├── stream_renderer.py          # Token-by-token streaming output
├── output_renderer.py          # Rich-formatted terminal output
├── config_manager.py           # CMD UI runtime config integration
├── commands/
│   ├── __init__.py
│   ├── help.py
│   ├── clear.py
│   ├── exit.py
│   ├── model.py
│   ├── mode.py
│   ├── history.py
│   ├── tools.py
│   ├── config.py
│   ├── export.py
│   ├── reset.py
│   ├── reload.py
│   └── context.py
├── panels/
│   ├── __init__.py
│   ├── workspace_panel.py      # Main conversation / logs area
│   ├── context_panel.py        # Right-side context / state panel
│   ├── session_panel.py        # Session/history list panel
│   ├── status_panel.py         # Top status bar / model state
│   ├── input_panel.py          # Bottom command/prompt input
│   └── command_palette.py      # Quick command/model picker
└── adapters/
    ├── __init__.py
    ├── dizel_adapter.py        # Dizel backend bridge
    ├── mila_adapter.py         # Mila backend bridge
    ├── provider_adapter.py     # Provider routing adapter
    └── tool_adapter.py         # Shared tool execution adapter
```

## COMMAND REQUIREMENTS
Minimum commands to support:
```bash
/help
# Show commands, shortcuts, and usage examples

/clear
# Clear the current visible workspace or terminal log, while optionally preserving session state

/exit
# Close the CMD UI cleanly

/model
# Show the active model or switch models

/mode
# Show the active mode or switch modes

/history
# Show recent prompts, commands, and outputs

/tools
# List available tools and current status

/config
# Show or update runtime configuration

/export
# Export session or conversation logs

/reset
# Reset the current session

/reload
# Reload config or refresh backend state
```

> Additional helpful commands:
```bash
/agent
/provider
/session
/search
/pin
/context
/budget
/interrupt
```

## OPENCODE / KILOCODE STYLE TARGET
The CMD UI should feel like:
- a structured command workspace
- a productive AI coding terminal
- a developer tool
- a visibility-first assistant interface
- a tool you can live inside while coding

It should **NOT** feel like:
- a minimal REPL
- a toy terminal
- a hidden backend console
- a basic prompt loop

## OUTPUT STYLE TARGET
Use Rich to make output readable and attractive:
- colored labels like ```SYSTEM```, ```USER```, ```ASSISTANT```, ```TOOL```, ```ERROR```, ```MODELS```, ```MODES```
- panels for important sections
- tables for model/session status
- separators between turns
- streamed token chunks in-place
- small status badges
- visible timing / token / context metrics where possible

## STATE DISPLAY TARGET
Always show current status somewhere visible:
- ```IDLE```
- ```THINKING```
- ```STREAMING```
- ```TOOL RUNNING```
- ```LOADING MODEL```
- ```TRIMMING CONTEXT```
- ```ERROR```
- ```INTERRUPTED```

## BACKEND CONTROL TARGET
The CMD UI should be able to fully control:
- provider routing
- mode routing
- model routing
- tool invocation
- prompt construction
- token budget selection
- context trimming
- history context injection

The backend should be used consistently across the GUI and CMD where possible to avoid duplication.

## INTEGRATION RULES
- Reuse shared backend logic from core/ and inference/dizel_ui/logic/ where practical.
- Do not duplicate prompt builder or token budget logic unless absolutely necessary.
- Keep the CMD UI decoupled from PySide6.
- The CMD UI should not depend on the GUI runtime.
- The GUI should not depend on the CMD UI runtime.
- Both should speak to the same backend brain.

## IMPLEMENTATION PRIORITIES
Since functionality matters more than polish right now:
1. Make command parsing reliable
2. Make backend routing correct
3. Make streaming stable
4. Make tool visibility transparent
5. Make model/mode switching work
6. Make session/history persistence work
7. Then improve UX polish

## ACCEPTANCE CRITERIA
*The task is complete only if :*
- [ ] The CMD UI is rebuilt from scratch
- [ ] It is multi-panel and keyboard-first
- [ ] It uses Textual + Rich effectively
- [ ] It streams output token-by-token
- [ ] It exposes backend actions transparently
- [ ] It supports commands and natural prompts
- [ ] It supports model/mode switching
- [ ] It supports tool visibility and background tasks
- [ ] It has a persistent session/history system
- [ ] It is modular and extensible
- [ ] It fully controls the backend workflow
- [ ] It feels like a real terminal AI workspace inspired by OpenCode / KiloCode

## FINAL INTENT
Build a new Dizel CMD UI from scratch as a modern, transparent, multi-panel, keyboard-first terminal AI workspace using Textual and Rich, with OpenCode/KiloCode-inspired interaction patterns, full backend control, streaming token-by-token output, visible tool and agent activity, command support, session management, and a modular architecture that can grow with the project.
