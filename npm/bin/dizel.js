#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const https = require("node:https");
const crypto = require("node:crypto");
const readline = require("node:readline");
const { spawn, spawnSync } = require("node:child_process");
const pkg = require("../package.json");

const VERSION = pkg.version;
const REPO = "D4niel-dev/Dizel";
const MANIFEST_VERSION_URL = `https://github.com/${REPO}/releases/download/v${VERSION}/dizel-update.json`;
const MANIFEST_LATEST_URL = `https://github.com/${REPO}/releases/latest/download/dizel-update.json`;
const DOCS_URL = "https://d4niel-dev.github.io/Dizel/guide/install.html";
const RELEASES_URL = `https://github.com/${REPO}/releases`;
const IS_WIN = process.platform === "win32";
const IS_MAC = process.platform === "darwin";
const ALLOWED_HOSTS = new Set([
  "github.com",
  "api.github.com",
  "objects.githubusercontent.com",
  "github-releases.githubusercontent.com",
  "release-assets.githubusercontent.com",
]);

function defaultHome() {
  if (IS_WIN) {
    return path.join(process.env.LOCALAPPDATA || os.homedir(), "Dizel");
  }
  if (IS_MAC) {
    return path.join(os.homedir(), "Library", "Application Support", "Dizel");
  }
  const data = process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share");
  return path.join(data, "dizel");
}

const HOME = process.env.DIZEL_HOME
  ? path.resolve(process.env.DIZEL_HOME)
  : defaultHome();
const STATE = path.join(HOME, "state.json");
const DOWNLOADS = path.join(HOME, "downloads");
const INSTALLERS = path.join(HOME, "installers");

function print(message = "") {
  console.log(message);
}

function fail(message) {
  throw new Error(message);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function readState() {
  try {
    return JSON.parse(fs.readFileSync(STATE, "utf8"));
  } catch {
    return {};
  }
}

function saveState(next) {
  ensureDir(HOME);
  fs.writeFileSync(STATE, `${JSON.stringify(next, null, 2)}\n`, "utf8");
}

function venvPythonPath() {
  return path.join(HOME, "venv", IS_WIN ? "Scripts/python.exe" : "bin/python");
}

function installedTui() {
  const state = readState();
  return state.tui && fs.existsSync(path.join(state.tui, "inference", "tui", "main.py"))
    ? state.tui
    : null;
}

function formatBytes(bytes) {
  if (!bytes || bytes < 0) return "unknown size";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function getAllowedResponse(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    let parsed;
    try {
      parsed = new URL(url);
    } catch {
      reject(new Error(`bad URL: ${url}`));
      return;
    }
    if (parsed.protocol !== "https:") {
      reject(new Error(`refusing non-HTTPS download: ${url}`));
      return;
    }
    if (!ALLOWED_HOSTS.has(parsed.hostname)) {
      reject(new Error(`refusing download host: ${parsed.hostname}`));
      return;
    }
    const request = https.get(
      parsed,
      { headers: { "User-Agent": `dizel-npm/${VERSION}` } },
      (response) => {
        const status = response.statusCode || 0;
        if ([301, 302, 303, 307, 308].includes(status) && response.headers.location) {
          response.resume();
          if (redirects >= 5) {
            reject(new Error("too many redirects"));
            return;
          }
          resolve(
            getAllowedResponse(
              new URL(response.headers.location, url).href,
              redirects + 1,
            ),
          );
          return;
        }
        if (status !== 200) {
          response.resume();
          reject(new Error(`HTTP ${status} for ${url}`));
          return;
        }
        resolve(response);
      },
    );
    request.on("error", reject);
    request.setTimeout(30000, () => request.destroy(new Error("request timed out")));
  });
}

function readResponse(response, limit = 4 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    response.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        response.destroy();
        reject(new Error("release manifest is too large"));
        return;
      }
      chunks.push(chunk);
    });
    response.on("end", () => resolve(Buffer.concat(chunks)));
    response.on("error", reject);
  });
}

async function loadManifest({ latest = false } = {}) {
  if (process.env.DIZEL_RELEASE_MANIFEST) {
    return JSON.parse(
      fs.readFileSync(path.resolve(process.env.DIZEL_RELEASE_MANIFEST), "utf8"),
    );
  }
  const response = await getAllowedResponse(
    latest ? MANIFEST_LATEST_URL : MANIFEST_VERSION_URL,
  );
  const body = await readResponse(response);
  return JSON.parse(body.toString("utf8"));
}

function assetUrl(manifest, asset) {
  if (!asset || !asset.asset) fail("release manifest has no downloadable asset");
  return `https://github.com/${REPO}/releases/download/${encodeURIComponent(
    manifest.tag,
  )}/${encodeURIComponent(asset.asset)}`;
}

function sha256File(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(file);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
    stream.on("error", reject);
  });
}

async function downloadAsset(asset, url, destination) {
  ensureDir(path.dirname(destination));
  if (fs.existsSync(destination)) {
    const digest = await sha256File(destination);
    const size = fs.statSync(destination).size;
    if (
      (!asset.sha256 || digest === asset.sha256) &&
      (!asset.size || size === asset.size)
    ) {
      print(`  cached   ${asset.asset}`);
      return;
    }
    fs.rmSync(destination, { force: true });
  }

  print(`  download ${asset.asset} (${formatBytes(asset.size)})`);
  const response = await getAllowedResponse(url);
  const temporary = `${destination}.part`;
  fs.rmSync(temporary, { force: true });
  const hash = crypto.createHash("sha256");
  let bytes = 0;

  await new Promise((resolve, reject) => {
    const output = fs.createWriteStream(temporary);
    response.on("data", (chunk) => {
      hash.update(chunk);
      bytes += chunk.length;
      if (!output.write(chunk)) {
        response.pause();
        output.once("drain", () => response.resume());
      }
    });
    response.on("error", reject);
    response.on("end", () => output.end());
    output.on("error", reject);
    output.on("finish", () => {
      try {
        if (asset.size && bytes !== asset.size) {
          throw new Error(
            `size mismatch for ${asset.asset}: expected ${asset.size}, got ${bytes}`,
          );
        }
        const digest = hash.digest("hex");
        if (asset.sha256 && digest !== asset.sha256) {
          throw new Error(`SHA-256 mismatch for ${asset.asset}`);
        }
        fs.renameSync(temporary, destination);
        resolve();
      } catch (error) {
        reject(error);
      }
    });
  }).catch((error) => {
    fs.rmSync(temporary, { force: true });
    throw error;
  });
  print(`  verified ${asset.asset}`);
}

function extractZip(archive, destination) {
  fs.rmSync(destination, { recursive: true, force: true });
  ensureDir(destination);
  if (IS_WIN) {
    const quote = (value) => `'${value.replace(/'/g, "''")}'`;
    const script = `Expand-Archive -LiteralPath ${quote(
      archive,
    )} -DestinationPath ${quote(destination)} -Force`;
    const result = spawnSync("powershell.exe", ["-NoProfile", "-Command", script], {
      stdio: "inherit",
    });
    if (result.status === 0) return;
    throw new Error("PowerShell could not extract the TUI archive.");
  }
  const python = findPython();
  if (python) {
    const result = spawnSync(
      python.exe,
      [...python.args, "-m", "zipfile", "-e", archive, destination],
      { stdio: "inherit" },
    );
    if (result.status === 0) return;
  }
  const unzip = spawnSync("unzip", ["-q", archive, "-d", destination], {
    stdio: "inherit",
  });
  if (unzip.status !== 0) {
    throw new Error("Could not extract the TUI archive (install unzip or Python).");
  }
}

function findPython() {
  const candidates = [];
  if (process.env.DIZEL_PYTHON) {
    candidates.push({ exe: process.env.DIZEL_PYTHON, args: [] });
  }
  const venv = venvPythonPath();
  if (fs.existsSync(venv)) candidates.push({ exe: venv, args: [] });
  if (IS_WIN) candidates.push({ exe: "py", args: ["-3"] });
  candidates.push({ exe: "python3", args: [] }, { exe: "python", args: [] });

  for (const candidate of candidates) {
    const result = spawnSync(
      candidate.exe,
      [
        ...candidate.args,
        "-c",
        "import sys; print('%d.%d' % sys.version_info[:2]); raise SystemExit(0 if sys.version_info >= (3, 10) else 1)",
      ],
      { encoding: "utf8", windowsHide: true },
    );
    if (!result.error && result.status === 0) {
      return { ...candidate, version: (result.stdout || "").trim() };
    }
  }
  return null;
}

function findGui() {
  if (!IS_WIN) return null;
  const candidates = [];
  if (process.env.LOCALAPPDATA) {
    candidates.push(
      path.join(process.env.LOCALAPPDATA, "Programs", "Dizel", "Dizel.exe"),
      path.join(process.env.LOCALAPPDATA, "Dizel", "Dizel.exe"),
    );
  }
  if (process.env.ProgramFiles) {
    candidates.push(path.join(process.env.ProgramFiles, "Dizel", "Dizel.exe"));
  }
  if (process.env["ProgramFiles(x86)"]) {
    candidates.push(
      path.join(process.env["ProgramFiles(x86)"], "Dizel", "Dizel.exe"),
    );
  }
  return candidates.find((candidate) => fs.existsSync(candidate)) || null;
}

function run(exe, args, options = {}) {
  const result = spawnSync(exe, args, { stdio: "inherit", ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${exe} exited with code ${result.status}`);
  }
}

function confirm(question) {
  if (!process.stdin.isTTY) return Promise.resolve(false);
  return new Promise((resolve) => {
    const interface_ = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    interface_.question(`${question} [Y/n] `, (answer) => {
      interface_.close();
      const value = answer.trim().toLowerCase();
      resolve(value === "" || value === "y" || value === "yes");
    });
  });
}

async function installTui(manifest, options) {
  const asset = manifest.tui;
  const url = assetUrl(manifest, asset);
  const archive = path.join(DOWNLOADS, asset.asset);
  if (options.dryRun) {
    print(`  TUI      ${url}`);
    print(`           ${formatBytes(asset.size)} · sha256 ${asset.sha256 || "—"}`);
    return;
  }
  await downloadAsset(asset, url, archive);
  const destination = path.join(HOME, "tui", manifest.version);
  print(`  extract  ${destination}`);
  extractZip(archive, destination);
  const state = readState();
  state.version = manifest.version;
  state.tui = destination;
  state.installedAt = new Date().toISOString();
  saveState(state);
  print(`  ready    terminal ${manifest.version}`);
}

async function installGui(manifest, options) {
  if (!IS_WIN) {
    print("  desktop  Windows-only in the current release. Use the TUI for now.");
    return false;
  }
  const asset = manifest.gui;
  const url = assetUrl(manifest, asset);
  const installer = path.join(INSTALLERS, asset.asset);
  if (options.dryRun) {
    print(`  desktop  ${url}`);
    print(`           ${formatBytes(asset.size)} · sha256 ${asset.sha256 || "—"}`);
    return true;
  }
  await downloadAsset(asset, url, installer);
  if (options.yes || (await confirm("Launch the Dizel desktop installer now?"))) {
    spawn(installer, [], { detached: true, stdio: "ignore" }).unref();
    print(`  opened   ${installer}`);
  } else {
    print(`  installer saved to ${installer}`);
    print("  run it when ready, then use: dizel gui");
  }
  return true;
}

async function installCommand(options, flags, { latest = false } = {}) {
  const manifest = await loadManifest({ latest });
  if (!manifest.version || !manifest.tag || !manifest.tui || !manifest.gui) {
    fail("release manifest is missing version, tag, GUI, or TUI assets");
  }

  let wantTui = true;
  let wantGui = IS_WIN;
  const onlyTui = flags.has("--tui") && !flags.has("--gui") && !flags.has("--all");
  const onlyGui = flags.has("--gui") && !flags.has("--tui") && !flags.has("--all");
  if (onlyTui) wantGui = false;
  if (onlyGui) wantTui = false;
  if (flags.has("--all")) {
    wantTui = true;
    wantGui = IS_WIN;
  }

  if (options.dryRun && options.json) {
    const payload = {
      version: manifest.version,
      tag: manifest.tag,
      tui: {
        asset: manifest.tui.asset,
        url: assetUrl(manifest, manifest.tui),
        sha256: manifest.tui.sha256 || "",
        size: manifest.tui.size || 0,
      },
      gui: {
        asset: manifest.gui.asset,
        url: assetUrl(manifest, manifest.gui),
        sha256: manifest.gui.sha256 || "",
        size: manifest.gui.size || 0,
      },
    };
    print(JSON.stringify(payload, null, 2));
    return;
  }

  print(`Dizel ${manifest.version} (${manifest.tag})`);
  if (wantTui) await installTui(manifest, options);
  if (wantGui) await installGui(manifest, options);
  if (!wantTui && !wantGui) {
    fail("nothing selected; use --tui, --gui, or --all");
  }

  print("");
  if (wantGui) print("Finish the desktop installer, then open Dizel or run: dizel gui");
  if (wantTui) {
    print("First run: dizel deps");
    print("Then:      dizel tui --checkpoint <path>");
  }
  print(`Model weights are separate. See: ${DOCS_URL}`);
}

async function depsCommand() {
  const tui = installedTui();
  if (!tui) fail("terminal archive is not installed; run: dizel install --tui");
  const requirements = path.join(tui, "requirements.txt");
  if (!fs.existsSync(requirements)) {
    fail(`no requirements.txt in ${tui}`);
  }
  const system = findPython();
  if (!system) {
    fail("Python 3.10+ is required: https://www.python.org/downloads/");
  }
  const venv = path.join(HOME, "venv");
  const venvPython = venvPythonPath();
  if (!fs.existsSync(venvPython)) {
    print(`Creating virtualenv: ${venv}`);
    run(system.exe, [...system.args, "-m", "venv", venv]);
  }
  print("Installing runtime dependencies. This includes PyTorch and can take a while.");
  run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
  run(venvPython, ["-m", "pip", "install", "-r", requirements]);
  print("");
  print("Dependencies ready. Run: dizel tui --checkpoint <path>");
  if (IS_WIN) {
    print("The desktop installer still resolves its own Python; set DIZEL_PYTHON if needed.");
  }
}

function runTui(args) {
  const tui = installedTui();
  if (!tui) fail("terminal archive is not installed; run: dizel install --tui");
  const venvPython = venvPythonPath();
  const python = fs.existsSync(venvPython)
    ? { exe: venvPython, args: [] }
    : findPython();
  if (!python) {
    fail(
      "Python 3.10+ with the Dizel dependencies is required. " +
        "Install Python, then run: dizel deps",
    );
  }
  const result = spawnSync(
    python.exe,
    [...python.args, "-m", "inference.tui.main", ...args],
    {
      cwd: tui,
      stdio: "inherit",
      env: { ...process.env, PYTHONPATH: tui },
    },
  );
  if (result.error) throw result.error;
  process.exitCode = result.status === null ? 1 : result.status;
}

function guiCommand() {
  const exe = findGui();
  if (!exe) {
    if (!IS_WIN) {
      print("The desktop app is Windows-only in the current release.");
      print(`Installers: ${RELEASES_URL}`);
      return;
    }
    fail("desktop app is not installed; run: dizel install --gui");
  }
  spawn(exe, [], { detached: true, stdio: "ignore" }).unref();
  print(`Opened ${exe}`);
}

function modelRoots() {
  return [
    path.join(process.cwd(), "checkpoints"),
    path.join(process.cwd(), "models"),
    path.join(HOME, "checkpoints"),
    path.join(HOME, "models"),
  ];
}

function localModels() {
  const models = [];
  for (const root of modelRoots()) {
    if (!fs.existsSync(root)) continue;
    try {
      for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
        if (!entry.isFile()) continue;
        if (!/\.(pt|gguf|safetensors|bin)$/i.test(entry.name)) continue;
        models.push(path.join(root, entry.name));
      }
    } catch {
      /* unreadable directory */
    }
  }
  return models;
}

function doctorCommand(options) {
  const python = findPython();
  const venv = fs.existsSync(venvPythonPath());
  const tui = installedTui();
  const gui = findGui();
  const models = localModels();
  const report = {
    launcher: VERSION,
    node: process.version,
    platform: `${process.platform}-${process.arch}`,
    home: HOME,
    python: python ? python.version : null,
    pythonExe: python ? python.exe : null,
    venv,
    tui,
    gui,
    models,
    docs: DOCS_URL,
  };

  if (options.json) {
    print(JSON.stringify(report, null, 2));
    return;
  }

  print(`Dizel launcher ${VERSION}`);
  print(`  node      ${process.version}`);
  print(`  platform  ${process.platform}-${process.arch}`);
  print(`  home      ${HOME}`);
  print(`  python    ${python ? `${python.version} (${python.exe})` : "not found — install Python 3.10+"}`);
  print(`  venv      ${venv ? "ready" : "not created — run dizel deps after install"}`);
  print(`  terminal  ${tui || "not installed — dizel install --tui"}`);
  print(`  desktop   ${gui || (IS_WIN ? "not installed — dizel install --gui" : "Windows-only")}`);
  print(`  models    ${models.length ? `${models.length} local checkpoint(s)` : "none found"}`);
  if (models.length) {
    for (const model of models.slice(0, 5)) print(`            ${model}`);
  }
  print("");
  print("Next:");
  if (!tui) print("  dizel install --tui");
  else if (!venv) print("  dizel deps");
  else print("  dizel tui --checkpoint <path>");
  if (IS_WIN && !gui) print("  dizel install --gui");
  if (!models.length) print(`  model guides: ${DOCS_URL}`);
}

function modelCommand() {
  print("Model weights are not part of npm or the launcher.");
  print(`Docs: ${DOCS_URL}`);
  print(`Releases: ${RELEASES_URL}`);
  const models = localModels();
  if (models.length) {
    print("");
    print("Local checkpoints:");
    for (const model of models) print(`  ${model}`);
  }
  print("");
  print("Run: dizel tui --checkpoint <path>");
}

function help() {
  print(`Dizel ${VERSION} — local-first AI model, desktop app, and terminal client`);
  print("");
  print("Usage:");
  print("  dizel install [--tui|--gui|--all] [--yes] [--dry-run]");
  print("  dizel install tui|gui|all");
  print("  dizel tui [args...]");
  print("  dizel gui");
  print("  dizel deps");
  print("  dizel doctor [--json]");
  print("  dizel update");
  print("  dizel model");
  print("  dizel version");
  print("");
  print("The npm package is a launcher. It downloads the matching GitHub Release");
  print("artifacts; weights, Electron, and Python remain separate downloads.");
  print(`Docs: ${DOCS_URL}`);
}

async function main() {
  const argv = process.argv.slice(2);
  const command = argv[0] || "help";

  if (["help", "--help", "-h"].includes(command)) {
    help();
    return;
  }
  if (["version", "--version", "-v"].includes(command)) {
    print(`dizel ${VERSION}`);
    return;
  }
  if (command === "tui") {
    runTui(argv.slice(1));
    return;
  }

  const flags = new Set(
    argv.slice(1).filter((value) => value.startsWith("--")),
  );
  const options = {
    dryRun: flags.has("--dry-run"),
    yes: flags.has("--yes"),
    json: flags.has("--json"),
  };
  const selector = argv[1];
  if (["tui", "gui", "all"].includes(selector)) flags.add(`--${selector}`);

  switch (command) {
    case "install":
      await installCommand(options, flags);
      return;
    case "update": {
      flags.add("--tui");
      await installCommand(options, flags, { latest: true });
      print("The desktop app updates itself; run dizel install --gui to refresh the installer.");
      return;
    }
    case "deps":
      await depsCommand();
      return;
    case "doctor":
      doctorCommand(options);
      return;
    case "gui":
      guiCommand();
      return;
    case "model":
      modelCommand();
      return;
    default:
      help();
      process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(`dizel: ${error.message || error}`);
  process.exitCode = 1;
});
