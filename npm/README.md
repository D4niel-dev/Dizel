# Dizel npm launcher

```bash
npm install -g @d4niel-dev/dizel@latest
dizel install
dizel doctor
dizel tui
dizel gui
```

`dizel` is a small launcher, not the 9 GB app payload. It downloads the matching
GitHub Release artifacts, verifies their SHA-256, and keeps them under
`%LOCALAPPDATA%\Dizel` (Windows), `~/Library/Application Support/Dizel` (macOS),
or `~/.local/share/dizel` (Linux).

## What `npm install -g` gives you

- a global `dizel` command; npm creates `dizel.cmd` on Windows;
- `dizel install` to download the app version that matches this launcher;
- `dizel update` to refresh the terminal archive from the latest release;
- `dizel tui` to run the terminal client;
- `dizel gui` to open the installed desktop app;
- `dizel deps` to create a virtualenv and install the Python requirements;
- `dizel doctor` to check Node, Python, installed files, and model paths;
- `dizel update` to refresh the terminal archive from the latest release.

## Requirements

- **Node.js 18+** for this launcher.
- **Windows** for the desktop app. The current release publishes an NSIS installer.
- **Python 3.10+** for the TUI and the desktop sidecar. The desktop installer
  includes the Python source but not a Python runtime.
- A model checkpoint. Weights are not part of npm or the launcher; use the
  published v1.5 GGUF or a local `.pt` checkpoint:
  `dizel tui --checkpoint <path>`.

The Python and model requirements are the reason this is a launcher rather than
a single self-contained npm tarball. A frozen Python sidecar is the next release
engineering step toward a true one-command install.

## Commands

```bash
dizel install                 # desktop (Windows) + terminal
dizel install --tui           # terminal only
dizel install --gui           # desktop installer only
dizel install --dry-run       # print asset URLs without downloading
dizel install --yes           # launch the GUI installer without asking
dizel tui [args...]           # run the terminal client
dizel gui                     # launch the installed desktop app
dizel deps                    # create the managed virtualenv and pip install requirements
dizel doctor [--json]         # diagnostics
dizel update                  # refresh the terminal archive from latest
dizel model                   # model setup pointers
dizel version
```

The launcher never stores model weights and never publishes secrets. Release
downloads are restricted to GitHub release hosts and verified against the
manifest SHA-256.
