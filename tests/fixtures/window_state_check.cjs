"use strict";

const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const root = path.resolve(__dirname, "..", "..");
const ws = require(path.join(root, "inference", "gui", "electron", "window-state.cjs"));

const displays = [{ workArea: { x: 0, y: 0, width: 1920, height: 1080 } }];

let state = ws.sanitize(
  { bounds: { x: 120, y: 80, width: 1400, height: 900 }, maximized: true },
  displays,
);
assert.deepStrictEqual(state.bounds, { x: 120, y: 80, width: 1400, height: 900 });
assert.strictEqual(state.maximized, true);
assert.strictEqual(state.fullscreen, false);

state = ws.sanitize(
  { bounds: { x: 5000, y: 5000, width: 1200, height: 800 }, maximized: true },
  displays,
);
assert.strictEqual(state.bounds, null, "off-screen bounds must be discarded");
assert.strictEqual(state.maximized, true, "mode survives a bad position");

state = ws.sanitize({ bounds: { x: 0, y: 0, width: 100, height: 100 } }, displays);
assert.deepStrictEqual(state.bounds, { x: 0, y: 0, width: ws.DEFAULTS.minWidth, height: ws.DEFAULTS.minHeight });

state = ws.sanitize({ bounds: { width: 1280, height: 720 } }, displays);
assert.deepStrictEqual(state.bounds, { width: 1280, height: 720 });

state = ws.sanitize({ bounds: { width: "wide", height: 720 } }, displays);
assert.strictEqual(state.bounds, null);

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dizel-window-state-"));
const file = path.join(dir, "window-state.json");
const saved = {
  bounds: { x: 10, y: 20, width: 1300, height: 850 },
  maximized: false,
  fullscreen: true,
};
ws.save(file, saved);
assert.deepStrictEqual(ws.load(file, displays), saved);
fs.rmSync(dir, { recursive: true, force: true });

assert.deepStrictEqual(ws.load(path.join(os.tmpdir(), "missing-window-state.json"), displays), {
  bounds: null,
  maximized: false,
  fullscreen: false,
});

console.log("ok");
