#!/usr/bin/env node
/**
 * Runs scripts/smoke.js inside a real Electron window and turns the result into
 * an exit code.
 *
 * The GUI has no test framework — `tsc` proves it compiles and `vite build`
 * proves it bundles, but neither proves the app runs. This boots the production
 * bundle, drives the store and the DOM, and fails the command if any check
 * reports `ok: false`.
 *
 *   npm run smoke
 *   npm run smoke -- --delay 20000     # slower machine
 *
 * The per-check logic lives in smoke.js and runs in the renderer, because that is
 * the only place with access to `window.__dizel` and a laid-out DOM.
 */
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const electron = require("electron");
const root = path.resolve(__dirname, "..");

const argOf = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

const shot = path.join(os.tmpdir(), `dizel-smoke-${Date.now()}.png`);

const env = { ...process.env };
// Inherited from an Electron-based parent this makes the binary act as plain
// node and never open a window.
delete env.ELECTRON_RUN_AS_NODE;
env.DIZEL_PROD = "1";
// `?debug=1` is what mounts `window.__dizel` (src/main.tsx). Without it the
// renderer boots fine and every check reports "no store", which looks like a
// broken app rather than a broken harness.
env.DIZEL_DEBUG = "1";
env.DIZEL_SHOT = shot;
env.DIZEL_SHOT_DELAY = argOf("delay", "12000");
env.DIZEL_SHOT_SIZE = argOf("size", "1280x900");
env.DIZEL_SHOT_JS_FILE = path.join(__dirname, "smoke.js");

const TIMEOUT_MS = Number(argOf("timeout", "360000"));

console.log("smoke: booting the app…");

const child = spawn(electron, ["."], { cwd: root, env, stdio: ["ignore", "pipe", "pipe"] });

let stdout = "";
let stderr = "";
child.stdout.on("data", (chunk) => (stdout += chunk));
child.stderr.on("data", (chunk) => (stderr += chunk));

const timer = setTimeout(() => {
  console.error(`smoke: timed out after ${TIMEOUT_MS / 1000}s`);
  child.kill();
}, TIMEOUT_MS);

child.on("close", (code) => {
  clearTimeout(timer);

  const line = stdout.split("\n").find((l) => l.includes("[dizel] shot js → "));
  if (!line) {
    console.error("smoke: the renderer never returned a result.");
    if (code !== 0) console.error(`smoke: electron exited with ${code}`);
    const tail = (stderr || stdout).trim().split("\n").slice(-12).join("\n");
    if (tail) console.error(`--- output ---\n${tail}`);
    process.exit(1);
  }

  const raw = line.slice(line.indexOf("→") + 1).trim();
  let payload;
  try {
    // The harness logs the renderer's return value, which is itself a JSON string.
    payload = JSON.parse(JSON.parse(raw));
  } catch (error) {
    console.error(`smoke: could not parse the result (${error.message})`);
    console.error(raw.slice(0, 400));
    process.exit(1);
  }

  if (payload.fatal) {
    console.error(`smoke: fatal — ${payload.fatal}`);
    process.exit(1);
  }

  const { passed, total, failed, results } = payload;
  for (const r of results) {
    const mark = r.ok ? "  ok  " : " FAIL ";
    const detail = r.ok ? "" : `  ${JSON.stringify(r.detail).slice(0, 150)}`;
    console.log(`${mark} ${r.name}${detail}`);
  }

  console.log("");
  if (failed.length === 0) {
    console.log(`smoke: ${passed}/${total} passed`);
    try {
      fs.unlinkSync(shot);
    } catch {
      /* the capture is a side effect, not an artefact worth keeping */
    }
    process.exit(0);
  }

  console.log(`smoke: ${passed}/${total} passed, ${failed.length} failed`);
  for (const name of failed) console.log(`  - ${name}`);
  console.log(`  screenshot: ${shot}`);
  process.exit(1);
});
