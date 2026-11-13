/**
 * Electron main process.
 *
 * Owns two things: the window, and the Python sidecar that does the actual
 * inference. The renderer never talks to Python directly — it asks main for
 * the sidecar's port and launch token, then streams over HTTP to 127.0.0.1.
 */

const {
  app,
  BrowserWindow,
  clipboard,
  ipcMain,
  shell,
  session,
  dialog,
  Tray,
  Menu,
  nativeImage,
  globalShortcut,
  screen,
} = require("electron");
const path = require("node:path");
const fs = require("node:fs");
const net = require("node:net");
const crypto = require("node:crypto");
const { spawn, spawnSync } = require("node:child_process");

const IS_DEV = !app.isPackaged && process.env.DIZEL_PROD !== "1";
// inference/gui/electron -> up three to the repo root (was two when this
// lived at desktop/). Packaged builds carry the Python app in resources/app.
const REPO_ROOT = app.isPackaged
  ? path.join(process.resourcesPath, "app")
  : path.resolve(__dirname, "..", "..", "..");
const DEV_URL = "http://localhost:5173";

// Captures are throwaway renders; they must not overwrite real geometry.
const HEADLESS_CAPTURE = Boolean(process.env.DIZEL_SHOT);
const windowState = require("./window-state.cjs");
const windowStatePath = () =>
  process.env.DIZEL_WINDOW_STATE ||
  path.join(app.getPath("userData"), "window-state.json");

const UPDATE_REPO = "D4niel-dev/Dizel";
const UPDATE_RELEASES = `https://github.com/${UPDATE_REPO}/releases`;
const AUTO_CHECK_MS = 6 * 60 * 60 * 1000;
const updatePrefsPath = () => path.join(app.getPath("userData"), "updates.json");

/** @type {BrowserWindow | null} */
let win = null;
let tray = null;
/** @type {{ proc: import('node:child_process').ChildProcess, port: number, token: string } | null} */
let sidecar = null;
/** @type {{ port: number, token: string } | null} */
let sidecarInfo = null;
let quitting = false;
let lastPythonError = "";
/** Last bounds seen while the window was neither maximized nor fullscreen. */
let lastNormalBounds = null;
/** "normal" | "maximized" | "fullscreen" — tracked so transitions don't
 *  overwrite the size the user chose with a transient frame. */
let windowMode = "normal";

// ── Updates ──────────────────────────────────────────────────────────────────

/** @type {import("electron-updater").AppUpdater | null} */
let autoUpdater = null;
let updateTimer = null;
let updateChecking = false;
let updatePrefs = { autoCheck: true, lastCheck: 0 };
let updateState = {
  status: "idle",
  currentVersion: app.getVersion(),
  latestVersion: "",
  notes: "",
  releaseUrl: UPDATE_RELEASES,
  progress: 0,
  error: "",
  autoCheck: true,
  canInstall: false,
  dev: !app.isPackaged,
};

// ── Python discovery ─────────────────────────────────────────────────────────
// The machine has several interpreters and only some have the app's
// dependencies, so candidates are probed rather than assumed.

function probe(exe, args) {
  try {
    const res = spawnSync(exe, [...args, "-c", "import fastapi, uvicorn; print('dizel-ok')"], {
      timeout: 25000,
      encoding: "utf8",
      windowsHide: true,
    });
    if (res.error) lastPythonError = `${exe}: ${res.error.message}`;
    return res.status === 0 && String(res.stdout || "").includes("dizel-ok");
  } catch (err) {
    lastPythonError = `${exe}: ${err.message}`;
    return false;
  }
}

function candidates() {
  const list = [];
  if (process.env.DIZEL_PYTHON) list.push({ exe: process.env.DIZEL_PYTHON, args: [] });

  const venvWin = path.join(REPO_ROOT, ".venv", "Scripts", "python.exe");
  const venvNix = path.join(REPO_ROOT, ".venv", "bin", "python");
  if (fs.existsSync(venvWin)) list.push({ exe: venvWin, args: [] });
  if (fs.existsSync(venvNix)) list.push({ exe: venvNix, args: [] });

  list.push({ exe: "python", args: [] });

  if (process.platform === "win32") {
    list.push({ exe: "py", args: ["-3"] });
    const localAppData = process.env.LOCALAPPDATA;
    if (localAppData) {
      for (const v of ["Python314", "Python313", "Python312", "Python311"]) {
        const exe = path.join(localAppData, "Programs", "Python", v, "python.exe");
        if (fs.existsSync(exe)) list.push({ exe, args: [] });
      }
    }
  }
  return list;
}

function resolvePython() {
  const cachePath = path.join(app.getPath("userData"), "python.json");
  try {
    const cached = JSON.parse(fs.readFileSync(cachePath, "utf8"));
    if (cached.exe && probe(cached.exe, cached.args || [])) return cached;
  } catch {
    /* no cache yet */
  }

  for (const candidate of candidates()) {
    if (probe(candidate.exe, candidate.args)) {
      try {
        fs.mkdirSync(path.dirname(cachePath), { recursive: true });
        fs.writeFileSync(cachePath, JSON.stringify(candidate));
      } catch {
        /* caching is a nicety, not a requirement */
      }
      return candidate;
    }
  }
  return null;
}

// ── Sidecar lifecycle ────────────────────────────────────────────────────────

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const address = srv.address();
      const port = typeof address === "object" && address ? address.port : 0;
      srv.close(() => (port ? resolve(port) : reject(new Error("No free port"))));
    });
  });
}

async function waitForHealth(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/health/`);
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}

async function startSidecar() {
  const python = resolvePython();
  if (!python) {
    throw new Error(
      "No Python interpreter with FastAPI + Uvicorn was found.\n" +
        "Install the app dependencies, or set DIZEL_PYTHON to the right interpreter.\n" +
        (lastPythonError ? `Last probe: ${lastPythonError}` : ""),
    );
  }

  const port = await findFreePort();
  const token = crypto.randomBytes(32).toString("base64url");
  const userData = app.getPath("userData");
  // Packaged installs keep mutable data outside resources/app, so an update
  // can replace the application without touching chats or settings.
  const dataEnv = app.isPackaged
    ? {
        ...(process.env.DIZEL_CONFIG_DIR
          ? {}
          : { DIZEL_CONFIG_DIR: path.join(userData, "config") }),
        ...(process.env.DIZEL_DATA_DIR ? {} : { DIZEL_DATA_DIR: path.join(userData, "data") }),
      }
    : {};

  const proc = spawn(
    python.exe,
    [
      ...(python.args || []),
      "-m",
      "uvicorn",
      "api.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
      "--log-level",
      "warning",
    ],
    {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        ...dataEnv,
        DIZEL_API_TOKEN: token,
        DIZEL_PORT: String(port),
        PYTHONUNBUFFERED: "1",
        PYTHONIOENCODING: "utf-8",
      },
      windowsHide: true,
    },
  );

  proc.stdout?.on("data", (d) => process.stdout.write(`[sidecar] ${d}`));
  proc.stderr?.on("data", (d) => process.stderr.write(`[sidecar] ${d}`));
  proc.on("exit", (code, signal) => {
    sidecar = null;
    sidecarInfo = null;
    if (quitting) return;
    console.error(`[sidecar] exited with code ${code} (${signal || "no signal"})`);
    // Without this the renderer just sees a dead socket and the user watches a
    // spinner until the 900s stream deadline expires.
    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send("sidecar:down", { code, signal: signal || null });
    }
  });

  const healthy = await waitForHealth(port);
  if (!healthy) {
    killSidecar(proc);
    throw new Error("The Python sidecar started but never became healthy.");
  }

  sidecar = { proc, port, token };
  sidecarInfo = { port, token };
  console.log(`[dizel] sidecar ready on 127.0.0.1:${port}`);
  return sidecarInfo;
}

function killSidecar(proc = sidecar?.proc) {
  if (!proc || proc.killed) return;
  try {
    if (process.platform === "win32" && proc.pid) {
      // uvicorn spawns no children, but /T keeps it safe if that ever changes.
      spawnSync("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { windowsHide: true });
    } else {
      proc.kill("SIGTERM");
    }
  } catch {
    /* already gone */
  }
}

// ── Updates ──────────────────────────────────────────────────────────────────

function loadUpdatePrefs() {
  try {
    const stored = JSON.parse(fs.readFileSync(updatePrefsPath(), "utf8"));
    if (typeof stored.autoCheck === "boolean") updatePrefs.autoCheck = stored.autoCheck;
    if (Number.isFinite(stored.lastCheck)) updatePrefs.lastCheck = stored.lastCheck;
  } catch {
    /* first run */
  }
  updateState.autoCheck = updatePrefs.autoCheck;
}

function saveUpdatePrefs() {
  try {
    fs.mkdirSync(path.dirname(updatePrefsPath()), { recursive: true });
    fs.writeFileSync(updatePrefsPath(), JSON.stringify(updatePrefs, null, 2));
  } catch {
    /* preference is a nicety */
  }
}

function broadcastUpdate(patch = {}) {
  updateState = { ...updateState, ...patch };
  for (const window of BrowserWindow.getAllWindows()) {
    window.webContents.send("update:status", updateState);
  }
  return updateState;
}

function parseVersion(value) {
  const match = String(value || "").trim().match(/^v?(\d+(?:\.\d+)*)(?:-([0-9A-Za-z.-]+))?/);
  if (!match) return { core: [0], pre: "" };
  return { core: match[1].split(".").map(Number), pre: match[2] || "" };
}

function compareVersions(left, right) {
  const a = parseVersion(left);
  const b = parseVersion(right);
  const length = Math.max(a.core.length, b.core.length);
  for (let index = 0; index < length; index += 1) {
    const diff = (a.core[index] || 0) - (b.core[index] || 0);
    if (diff !== 0) return diff;
  }
  if (a.pre === b.pre) return 0;
  if (!a.pre) return 1;
  if (!b.pre) return -1;
  return a.pre > b.pre ? 1 : -1;
}

function releaseNotes(value) {
  if (Array.isArray(value)) {
    return value.map((item) => item?.note || "").filter(Boolean).join("\n");
  }
  return String(value || "");
}

async function fetchLatestRelease() {
  const endpoint =
    process.env.DIZEL_UPDATE_FEED ||
    `https://api.github.com/repos/${UPDATE_REPO}/releases/latest`;
  const response = await fetch(endpoint, {
    headers: {
      Accept: "application/vnd.github+json",
      "User-Agent": `Dizel/${app.getVersion()}`,
    },
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`GitHub returned HTTP ${response.status}`);
  const payload = await response.json();
  const tag = String(payload.tag_name || "");
  if (!tag) return null;
  return {
    version: tag.replace(/^v/i, ""),
    tag,
    notes: releaseNotes(payload.body),
    url: String(payload.html_url || UPDATE_RELEASES),
  };
}

async function checkForUpdates(manual = false) {
  if (
    updateChecking ||
    updateState.status === "downloaded" ||
    updateState.status === "downloading"
  ) {
    return updateState;
  }
  updateChecking = true;
  broadcastUpdate({ status: "checking", error: "", progress: 0 });

  try {
    if (app.isPackaged && autoUpdater) {
      await autoUpdater.checkForUpdates();
    } else {
      const release = await fetchLatestRelease();
      if (!release) {
        broadcastUpdate({ status: "up-to-date", latestVersion: "", notes: "" });
      } else if (compareVersions(release.version, app.getVersion()) > 0) {
        broadcastUpdate({
          status: "available",
          latestVersion: release.version,
          notes: release.notes,
          releaseUrl: release.url,
          canInstall: false,
        });
      } else {
        broadcastUpdate({
          status: "up-to-date",
          latestVersion: release.version,
          releaseUrl: release.url,
        });
      }
    }
  } catch (error) {
    const message = String(error?.message || error);
    if (/\b404\b|no published release|latest\.ya?ml/i.test(message)) {
      broadcastUpdate({ status: "up-to-date", error: "" });
    } else {
      broadcastUpdate({ status: "error", error: message });
    }
  } finally {
    updateChecking = false;
    updatePrefs.lastCheck = Date.now();
    saveUpdatePrefs();
  }
  return updateState;
}

function scheduleAutoCheck() {
  if (updateTimer) {
    clearTimeout(updateTimer);
    updateTimer = null;
  }
  if (!updatePrefs.autoCheck) return;
  const elapsed = Date.now() - Number(updatePrefs.lastCheck || 0);
  const wait = Math.max(9000, AUTO_CHECK_MS - elapsed);
  updateTimer = setTimeout(async () => {
    await checkForUpdates(false);
    scheduleAutoCheck();
  }, wait);
}

function setupUpdater() {
  loadUpdatePrefs();
  updateState.currentVersion = app.getVersion();
  updateState.dev = !app.isPackaged;

  if (app.isPackaged) {
    try {
      autoUpdater = require("electron-updater").autoUpdater;
      autoUpdater.logger = {
        info: (message) => console.log(`[updater] ${message}`),
        warn: (message) => console.warn(`[updater] ${message}`),
        // A repo with no latest.yml is a normal pre-release state, not a bug
        // worth dumping a stack trace for.
        error: (message) => {
          const text = String(message);
          if (/latest\.ya?ml|no published release|\b404\b/i.test(text)) return;
          console.error(`[updater] ${text}`);
        },
        debug: () => {},
      };
      autoUpdater.autoDownload = true;
      autoUpdater.autoInstallOnAppQuit = true;
      autoUpdater.allowPrerelease = false;

      autoUpdater.on("checking-for-update", () => {
        broadcastUpdate({ status: "checking", error: "", progress: 0 });
      });
      autoUpdater.on("update-available", (info) => {
        broadcastUpdate({
          status: "available",
          latestVersion: String(info?.version || ""),
          notes: releaseNotes(info?.releaseNotes),
          canInstall: false,
        });
      });
      autoUpdater.on("update-not-available", (info) => {
        broadcastUpdate({
          status: "up-to-date",
          latestVersion: String(info?.version || ""),
          notes: "",
          error: "",
        });
      });
      autoUpdater.on("download-progress", (progress) => {
        broadcastUpdate({
          status: "downloading",
          progress: Math.max(0, Math.min(100, Math.round(progress?.percent || 0))),
        });
      });
      autoUpdater.on("update-downloaded", (info) => {
        broadcastUpdate({
          status: "downloaded",
          latestVersion: String(info?.version || ""),
          notes: releaseNotes(info?.releaseNotes),
          progress: 100,
          canInstall: true,
          error: "",
        });
      });
      autoUpdater.on("error", (error) => {
        const message = String(error?.message || error);
        if (/\b404\b|no published release|latest\.ya?ml/i.test(message)) {
          broadcastUpdate({ status: "up-to-date", error: "" });
        } else {
          broadcastUpdate({ status: "error", error: message });
        }
      });
    } catch (error) {
      console.error("[dizel] updater unavailable:", error);
      updateState.error = String(error?.message || error);
    }
  }

  scheduleAutoCheck();
}


// ── Window ───────────────────────────────────────────────────────────────────

function captureNormalBounds() {
  if (!win || win.isDestroyed() || windowMode !== "normal") return;
  if (win.isMaximized() || win.isFullScreen()) return;
  lastNormalBounds = win.getBounds();
}

function persistWindowState() {
  if (HEADLESS_CAPTURE || !win || win.isDestroyed()) return;
  windowState.save(windowStatePath(), {
    bounds: lastNormalBounds,
    maximized: windowMode === "maximized" || win.isMaximized(),
    fullscreen: windowMode === "fullscreen" || win.isFullScreen(),
  });
}

function createWindow() {
  const restored = HEADLESS_CAPTURE
    ? windowState.sanitize(null, [])
    : windowState.load(windowStatePath(), screen.getAllDisplays());
  const bounds = restored.bounds || {};

  win = new BrowserWindow({
    width: windowState.DEFAULTS.width,
    height: windowState.DEFAULTS.height,
    minWidth: windowState.DEFAULTS.minWidth,
    minHeight: windowState.DEFAULTS.minHeight,
    show: false,
    frame: false,
    backgroundColor: "#212121",
    title: "Dizel",
    // Without this the taskbar shows Electron's default icon, not ours.
    icon: path.join(__dirname, "..", "public", "Dizel_GUI.ico"),
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
    },
  });

  // Apply saved bounds through setBounds/setSize: the constructor and
  // getBounds() disagree by the invisible Windows resize border, which made
  // the window creep on every launch. Both sides now use outer bounds.
  if (bounds.width && bounds.height) {
    if (Number.isFinite(bounds.x) && Number.isFinite(bounds.y)) {
      win.setBounds({ x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height });
    } else {
      win.setSize(bounds.width, bounds.height);
    }
  }
  lastNormalBounds = win.getBounds();
  windowMode = restored.fullscreen
    ? "fullscreen"
    : restored.maximized
      ? "maximized"
      : "normal";

  let saveTimer = null;
  const scheduleSave = () => {
    if (HEADLESS_CAPTURE) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      saveTimer = null;
      persistWindowState();
    }, 400);
  };

  win.once("ready-to-show", () => win?.show());

  const notifyMaximize = () => win?.webContents.send("window:maximized", win.isMaximized());
  win.on("maximize", () => {
    windowMode = "maximized";
    notifyMaximize();
    scheduleSave();
  });
  win.on("unmaximize", () => {
    windowMode = win.isFullScreen() ? "fullscreen" : "normal";
    notifyMaximize();
    scheduleSave();
  });
  win.on("enter-full-screen", () => {
    windowMode = "fullscreen";
    scheduleSave();
  });
  win.on("leave-full-screen", () => {
    windowMode = win.isMaximized() ? "maximized" : "normal";
    scheduleSave();
  });
  win.on("resize", () => {
    captureNormalBounds();
    scheduleSave();
  });
  win.on("move", () => {
    captureNormalBounds();
    scheduleSave();
  });
  win.on("close", persistWindowState);

  win.on("closed", () => {
    win = null;
  });

  // F11 toggles true fullscreen; the chosen mode is restored next launch.
  win.webContents.on("before-input-event", (event, input) => {
    if (input.type !== "keyDown" || input.key !== "F11" || !win) return;
    event.preventDefault();
    if (win.isFullScreen()) {
      win.setFullScreen(false);
    } else {
      windowMode = "fullscreen";
      win.setFullScreen(true);
    }
  });

  // An event fired before the renderer mounts is lost, so the title bar asks
  // again once the page is live.
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("window:maximized", win.isMaximized());
  });

  if (restored.fullscreen) win.setFullScreen(true);
  else if (restored.maximized) win.maximize();
  windowMode = restored.fullscreen
    ? "fullscreen"
    : restored.maximized
      ? "maximized"
      : "normal";

  // Nothing in this app should navigate away or open child windows.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (IS_DEV && url.startsWith(DEV_URL)) return;
    event.preventDefault();
  });

  const debugQuery = process.env.DIZEL_DEBUG === "1" ? { debug: "1" } : undefined;

  if (IS_DEV) {
    win.loadURL(debugQuery ? `${DEV_URL}/?debug=1` : DEV_URL);
  } else {
    win.loadFile(path.join(__dirname, "..", "dist", "index.html"), { query: debugQuery });
  }

  // Headless self-check: DIZEL_SHOT=<path> renders, captures and exits.
  // Used to verify the UI without a human watching the screen.
  if (process.env.DIZEL_SHOT) {
    win.webContents.once("did-finish-load", () => {
      const size = (process.env.DIZEL_SHOT_SIZE || "").split("x").map(Number);
      if (size.length === 2 && size[0] > 0 && size[1] > 0) {
        if (win.isMaximized()) win.unmaximize();
        win.setSize(size[0], size[1]);
      }
      const delay = Number(process.env.DIZEL_SHOT_DELAY || 4000);
      setTimeout(async () => {
        try {
          const scriptFile = process.env.DIZEL_SHOT_JS_FILE;
          const script =
            (scriptFile && fs.existsSync(scriptFile) ? fs.readFileSync(scriptFile, "utf8") : null) ??
            process.env.DIZEL_SHOT_JS;
          if (script) {
            const result = await win.webContents.executeJavaScript(script);
            console.log(`[dizel] shot js → ${JSON.stringify(result)}`);
            await new Promise((r) => setTimeout(r, 1500));
          }
          // An occluded window can hand back a frame from seconds ago, so CDP
          // is used to force a fresh paint; capturePage is the fallback.
          let png = null;
          try {
            const dbg = win.webContents.debugger;
            if (!dbg.isAttached()) dbg.attach("1.3");
            const shot = await dbg.sendCommand("Page.captureScreenshot", {
              format: "png",
              captureBeyondViewport: false,
            });
            png = Buffer.from(shot.data, "base64");
          } catch (error) {
            console.error("[dizel] cdp capture failed, falling back:", error);
          }
          if (!png) {
            const image = await win.webContents.capturePage();
            png = image.toPNG();
          }
          fs.writeFileSync(process.env.DIZEL_SHOT, png);
          console.log(`[dizel] screenshot → ${process.env.DIZEL_SHOT}`);
        } catch (err) {
          console.error("[dizel] screenshot failed:", err);
        }
        app.quit();
      }, delay);
    });
  }
}

// ── App wiring ───────────────────────────────────────────────────────────────

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!win) return;
    if (win.isMinimized()) win.restore();
    win.focus();
  });

  function createTray() {
  // Prefer the .ico on Windows — it carries the small sizes the tray needs, so
  // it stays sharp instead of being resampled down from a 1080px PNG.
  const candidates = [
    path.join(__dirname, "..", "public", "Dizel_GUI.ico"),
    path.join(__dirname, "..", "public", "Dizel_GUI.png"),
  ];

  let image = nativeImage.createEmpty();
  let used = "";
  for (const candidate of candidates) {
    const loaded = nativeImage.createFromPath(candidate);
    if (!loaded.isEmpty()) {
      image = loaded;
      used = candidate;
      break;
    }
  }

  if (image.isEmpty()) {
    // Fail loudly: a silently missing tray is impossible to diagnose later.
    console.warn("[dizel] tray icon could not be loaded from:", candidates.join(", "));
    return;
  }

  // Windows renders the tray at 16px (32px at 200% scaling). Handing it the raw
  // multi-size .ico lets it pick a 256px frame and fall back to the default
  // icon, so pick the frame explicitly.
  const trayImage = image.resize({ width: 32, height: 32, quality: "best" });
  tray = new Tray(trayImage.isEmpty() ? image : trayImage);
  console.log("[dizel] tray icon:", used, trayImage.getSize());
  tray.setToolTip("Dizel GUI");
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Show Dizel", click: () => showWindow() },
      {
        label: "New chat",
        click: () => {
          showWindow();
          win?.webContents.send("app:new-chat");
        },
      },
      { type: "separator" },
      { label: "Quit", click: () => app.quit() },
    ]),
  );
  tray.on("click", () => showWindow());
}

function showWindow() {
  if (!win) {
    createWindow();
    return;
  }
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

app.whenReady().then(async () => {
    if (process.platform === "win32") app.setAppUserModelId("dev.dizel.desktop");

    // Strict CSP for the packaged renderer only — the Vite dev server needs a
    // looser policy for its inline HMR preamble.
    if (!IS_DEV) {
      session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
        callback({
          responseHeaders: {
            ...details.responseHeaders,
            "Content-Security-Policy": [
              // blob: is required for the profile avatar — it is fetched with the
              // launch token and handed to <img> as an object URL.
              "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; " +
                "script-src 'self'; connect-src http://127.0.0.1:*",
            ],
          },
        });
      });
    }

    createWindow();
    createTray();
    setupUpdater();

    // Summon or hide from anywhere. Ctrl+Shift+Space rather than Ctrl+Space
    // because several IMEs claim the latter.
    globalShortcut.register("CommandOrControl+Shift+Space", () => {
      if (win?.isVisible() && win.isFocused()) win.hide();
      else showWindow();
    });

    try {
      const info = await startSidecar();
      win?.webContents.send("sidecar:ready", info);
    } catch (err) {
      console.error("[dizel] sidecar failed:", err);
      win?.webContents.send("sidecar:error", String(err?.message || err));
      dialog.showErrorBox("Dizel could not start", String(err?.message || err));
    }

    win.on("focus", () => win.flashFrame(false));

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });

  app.on("before-quit", () => {
    quitting = true;
    if (updateTimer) clearTimeout(updateTimer);
    killSidecar();
  });

  app.on("will-quit", () => {
    globalShortcut.unregisterAll();
    killSidecar();
  });
}

ipcMain.handle("sidecar:info", () => sidecarInfo);

ipcMain.handle("sidecar:restart", async () => {
  try {
    killSidecar();
    sidecar = null;
    sidecarInfo = null;
    // Give the port a moment to be released before asking for a new one.
    await new Promise((resolve) => setTimeout(resolve, 400));
    const info = await startSidecar();
    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send("sidecar:ready", info);
    }
    return { ok: true, info };
  } catch (error) {
    const message = String(error?.message || error);
    for (const window of BrowserWindow.getAllWindows()) {
      window.webContents.send("sidecar:error", message);
    }
    return { ok: false, error: message };
  }
});

ipcMain.handle("window:state", () => ({
  maximized: Boolean(win?.isMaximized()),
  fullscreen: Boolean(win?.isFullScreen()),
}));

ipcMain.on("window:minimize", () => win?.minimize());
ipcMain.on("window:toggle-maximize", () => {
  if (!win) return;
  if (win.isFullScreen()) {
    win.setFullScreen(false);
    return;
  }
  if (win.isMaximized()) {
    win.unmaximize();
  } else {
    windowMode = "maximized";
    win.maximize();
  }
});
ipcMain.on("window:close", () => win?.close());

ipcMain.on("app:attention", () => {
  // Windows and macOS both flash the taskbar/dock icon; it stops on focus.
  if (win && !win.isFocused()) win.flashFrame(true);
});

ipcMain.handle("clipboard:read-text", () => clipboard.readText());

ipcMain.handle("app:open-external", async (_event, url) => {
  if (typeof url === "string" && /^https?:\/\//i.test(url)) {
    await shell.openExternal(url);
    return true;
  }
  return false;
});

// ── Update bridge ────────────────────────────────────────────────────────────

ipcMain.handle("update:status", () => updateState);

ipcMain.handle("update:check", () => checkForUpdates(true));

ipcMain.handle("update:download", async () => {
  if (app.isPackaged && autoUpdater && updateState.status === "available") {
    try {
      await autoUpdater.downloadUpdate();
    } catch (error) {
      const message = String(error?.message || error);
      if (!/already|in progress/i.test(message)) {
        broadcastUpdate({ status: "error", error: message });
      }
    }
    return updateState;
  }
  await shell.openExternal(updateState.releaseUrl || UPDATE_RELEASES);
  return updateState;
});

ipcMain.handle("update:install", async () => {
  if (app.isPackaged && autoUpdater && updateState.status === "downloaded") {
    // Let the IPC reply reach the renderer before the app tears down.
    // Silent install, then relaunch the updated app.
    setImmediate(() => autoUpdater.quitAndInstall(true, true));
    return { ok: true };
  }
  await shell.openExternal(updateState.releaseUrl || UPDATE_RELEASES);
  return { ok: false, opened: true };
});

ipcMain.handle("update:open-release", async () => {
  await shell.openExternal(updateState.releaseUrl || UPDATE_RELEASES);
  return true;
});

ipcMain.handle("update:set-auto-check", (_event, enabled) => {
  updatePrefs.autoCheck = Boolean(enabled);
  saveUpdatePrefs();
  broadcastUpdate({ autoCheck: updatePrefs.autoCheck });
  scheduleAutoCheck();
  return updateState;
});

// ── System monitor (the Monitor workspace view) ──────────────────────────────

const os = require("node:os");
let cpuSample = null;

function cpuPercent() {
  const cores = os.cpus();
  let idle = 0;
  let total = 0;
  for (const core of cores) {
    for (const value of Object.values(core.times)) total += value;
    idle += core.times.idle;
  }

  if (cpuSample) {
    const idleDelta = idle - cpuSample.idle;
    const totalDelta = total - cpuSample.total;
    cpuSample = { idle, total };
    if (totalDelta > 0) {
      return Math.max(0, Math.min(100, Math.round((1 - idleDelta / totalDelta) * 100)));
    }
    return 0;
  }

  cpuSample = { idle, total };
  return Math.round((1 - idle / total) * 100);
}

ipcMain.handle("system:stats", () => {
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  return {
    cpu: cpuPercent(),
    ram: Math.round((1 - freeMem / totalMem) * 100),
    ramUsedMb: Math.round((totalMem - freeMem) / 1048576),
    ramTotalMb: Math.round(totalMem / 1048576),
    platform: os.platform(),
    uptimeSec: Math.round(os.uptime()),
  };
});

const ATTACHMENT_EXTENSIONS = [
  "txt", "md", "json", "jsonl", "csv", "log", "yml", "yaml", "toml", "ini",
  "py", "ts", "tsx", "js", "jsx", "css", "html", "sql", "sh",
  "rs", "go", "java", "c", "h", "cpp", "cs", "rb", "php", "swift", "kt",
];

ipcMain.handle("dialog:pick-files", async (_event, kind) => {
  const isCheckpoint = kind === "checkpoint";
  const isImage = kind === "image";
  const result = await dialog.showOpenDialog(win, {
    title: isCheckpoint ? "Select a model checkpoint" : isImage ? "Attach images" : "Attach files",
    properties: isCheckpoint ? ["openFile"] : ["openFile", "multiSelections"],
    filters: isCheckpoint
      ? [{ name: "Model checkpoints", extensions: ["gguf", "pt", "safetensors"] }]
      : isImage
        ? [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp", "gif", "bmp"] }]
        : [
            { name: "Text & code", extensions: ATTACHMENT_EXTENSIONS },
            { name: "All files", extensions: ["*"] },
          ],
  });
  return result.canceled ? [] : result.filePaths;
});

// Picks a directory and returns its top-level text files, so "attach a folder"
// doesn't walk a whole tree into the prompt.
ipcMain.handle("dialog:save-json", async (_event, { defaultName, json }) => {
  const result = await dialog.showSaveDialog(win, {
    title: "Export chat",
    defaultPath: defaultName,
    filters: [{ name: "JSON", extensions: ["json"] }],
  });
  if (result.canceled || !result.filePath) return null;

  try {
    await fs.promises.writeFile(result.filePath, json, "utf8");
    return result.filePath;
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
});

ipcMain.handle("dialog:pick-folder", async () => {
  const result = await dialog.showOpenDialog(win, {
    title: "Attach a folder",
    properties: ["openDirectory"],
  });
  if (result.canceled || !result.filePaths[0]) return [];

  try {
    const entries = await fs.promises.readdir(result.filePaths[0], { withFileTypes: true });
    return entries
      .filter((entry) => {
        if (!entry.isFile()) return false;
        const ext = entry.name.split(".").pop()?.toLowerCase() ?? "";
        return ATTACHMENT_EXTENSIONS.includes(ext);
      })
      .slice(0, 12)
      .map((entry) => path.join(result.filePaths[0], entry.name));
  } catch {
    return [];
  }
});

// Reads a text file for attachment context. Capped so a stray 2 GB log can't
// stall the renderer; the caller shows the truncation.
ipcMain.handle("fs:reveal", async (_event, filePath) => {
  if (typeof filePath !== "string" || !filePath) return { ok: false };
  try {
    // Selects the file in Explorer/Finder rather than opening it.
    shell.showItemInFolder(filePath);
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("fs:save-text", async (_event, options) => {
  const { defaultName = "export.md", contents = "" } = options || {};
  try {
    const result = await dialog.showSaveDialog({
      defaultPath: defaultName,
      filters: [
        { name: "Markdown", extensions: ["md"] },
        { name: "JSON", extensions: ["json"] },
        { name: "All files", extensions: ["*"] },
      ],
    });
    if (result.canceled || !result.filePath) return { ok: false, canceled: true };
    await fs.promises.writeFile(result.filePath, contents, "utf8");
    return { ok: true, path: result.filePath };
  } catch (error) {
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("fs:open-path", async (_event, filePath) => {
  if (typeof filePath !== "string" || !filePath) return { ok: false };
  try {
    const problem = await shell.openPath(filePath);
    return problem ? { ok: false, error: problem } : { ok: true };
  } catch (error) {
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("fs:data-url", async (_event, filePath) => {
  // Images are attached for the reader, not the model — there is no multimodal
  // path yet. The renderer needs the bytes to draw a thumbnail, so hand it a
  // data URL rather than text.
  const TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
  };
  const MAX_BYTES = 12 * 1024 * 1024;

  if (typeof filePath !== "string" || !filePath) {
    return { ok: false, error: "No path given." };
  }
  const mime = TYPES[path.extname(filePath).toLowerCase()];
  if (!mime) return { ok: false, error: "Not an image type." };

  try {
    const stat = await fs.promises.stat(filePath);
    if (stat.isDirectory()) return { ok: false, error: "That is a folder." };
    if (stat.size > MAX_BYTES) return { ok: false, error: "Image is larger than 12 MB." };

    const bytes = await fs.promises.readFile(filePath);
    return {
      ok: true,
      size: stat.size,
      dataUrl: `data:${mime};base64,${bytes.toString("base64")}`,
    };
  } catch (error) {
    return { ok: false, error: error.message };
  }
});

ipcMain.handle("fs:read-text", async (_event, filePath) => {
  const MAX_BYTES = 200_000;
  if (typeof filePath !== "string" || !filePath) {
    return { ok: false, error: "No path given." };
  }
  try {
    const stat = await fs.promises.stat(filePath);
    if (stat.isDirectory()) return { ok: false, error: "That is a folder." };
    if (stat.size > 20_000_000) return { ok: false, error: "File is larger than 20 MB." };

    const length = Math.min(stat.size, MAX_BYTES);
    const handle = await fs.promises.open(filePath, "r");
    const buffer = Buffer.alloc(length);
    try {
      await handle.read(buffer, 0, length, 0);
    } finally {
      await handle.close();
    }

    // A NUL byte in the head means this isn't text. Decoding it anyway would
    // inject mojibake into the prompt, which is worse than refusing.
    const probe = buffer.subarray(0, Math.min(length, 8000));
    if (probe.includes(0)) {
      return {
        ok: false,
        error: "This file is binary — the model reads text only.",
      };
    }

    return {
      ok: true,
      content: buffer.toString("utf8").replace(/^\uFEFF/, ""),
      size: stat.size,
      truncated: stat.size > MAX_BYTES,
    };
  } catch (err) {
    return { ok: false, error: String(err?.message || err) };
  }
});
