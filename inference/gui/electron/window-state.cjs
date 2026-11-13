/**
 * Window geometry persistence.
 *
 * Kept separate from main.cjs so bounds sanitising can be tested without an
 * Electron runtime. A saved position is accepted only when it still overlaps a
 * connected display; otherwise the window opens centred at its saved size.
 */

"use strict";

const fs = require("node:fs");
const path = require("node:path");

const DEFAULTS = Object.freeze({
  width: 1180,
  height: 780,
  minWidth: 900,
  minHeight: 620,
});

function toInt(value) {
  return Number.isFinite(value) ? Math.round(value) : null;
}

function sanitize(raw, displays) {
  const state = { bounds: null, maximized: false, fullscreen: false };
  if (!raw || typeof raw !== "object") return state;

  state.maximized = raw.maximized === true;
  state.fullscreen = raw.fullscreen === true;

  const saved = raw.bounds;
  if (!saved || typeof saved !== "object") return state;

  const width = toInt(saved.width);
  const height = toInt(saved.height);
  if (width === null || height === null) return state;

  const x = toInt(saved.x);
  const y = toInt(saved.y);
  const bounds = {
    width: Math.max(DEFAULTS.minWidth, width),
    height: Math.max(DEFAULTS.minHeight, height),
  };

  if (x === null || y === null) {
    state.bounds = bounds;
    return state;
  }
  bounds.x = x;
  bounds.y = y;

  const areas = (Array.isArray(displays) ? displays : [])
    .map((display) => (display && (display.workArea || display.bounds)) || null)
    .filter(Boolean);

  // Unplugging a monitor must not strand the window outside every display.
  if (areas.length) {
    const visible = areas.some((area) => {
      const overlapX = Math.min(x + bounds.width, area.x + area.width) - Math.max(x, area.x);
      const overlapY = Math.min(y + bounds.height, area.y + area.height) - Math.max(y, area.y);
      return overlapX >= 120 && overlapY >= 60;
    });
    if (!visible) return state;
  }

  state.bounds = bounds;
  return state;
}

function load(file, displays) {
  try {
    return sanitize(JSON.parse(fs.readFileSync(file, "utf8")), displays);
  } catch {
    return sanitize(null, displays);
  }
}

function save(file, state) {
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const temp = `${file}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(state, null, 2));
    fs.renameSync(temp, file);
  } catch {
    /* window geometry is a nicety, never a launch blocker */
  }
}

module.exports = { DEFAULTS, sanitize, load, save };
