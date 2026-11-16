/* Unit test for the preload bridge's listener contract.
 *
 * Runs under plain Node (node:test) with a mocked `electron`, so it needs no
 * window or sidecar. Every on* bridge must return an unsubscribe that detaches
 * exactly the listener it registered.
 *
 *   node scripts/test-preload.cjs
 */
const test = require("node:test");
const assert = require("node:assert");
const Module = require("node:module");
const path = require("node:path");

const listeners = new Map();
const exposed = {};

const fakeElectron = {
  contextBridge: {
    exposeInMainWorld: (name, api) => {
      exposed[name] = api;
    },
  },
  ipcRenderer: {
    on: (channel, handler) => {
      if (!listeners.has(channel)) listeners.set(channel, new Set());
      listeners.get(channel).add(handler);
    },
    removeListener: (channel, handler) => {
      const set = listeners.get(channel);
      if (set) set.delete(handler);
    },
    invoke: async () => null,
    send: () => {},
  },
  webFrame: { setZoomFactor: () => {} },
  webUtils: { getPathForFile: () => "" },
};

const originalLoad = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === "electron") return fakeElectron;
  return originalLoad.apply(this, arguments);
};
require(path.resolve(__dirname, "../electron/preload.cjs"));
Module._load = originalLoad;

const emit = (channel, ...args) => {
  for (const handler of listeners.get(channel) || []) handler({}, ...args);
};

test("onNewChat returns an unsubscribe that detaches its listener", () => {
  const api = exposed.dizel;
  let calls = 0;
  const off = api.onNewChat(() => {
    calls += 1;
  });

  assert.equal(typeof off, "function", "onNewChat must return an unsubscribe");

  emit("app:new-chat");
  assert.equal(calls, 1, "callback fires while subscribed");

  off();
  emit("app:new-chat");
  assert.equal(calls, 1, "callback must not fire after unsubscribe");
});