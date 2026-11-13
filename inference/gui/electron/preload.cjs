/**
 * Preload — the only bridge between the renderer and the main process.
 * Everything here is an explicit, narrow capability; nothing else is exposed.
 */

const { contextBridge, ipcRenderer, webFrame, webUtils } = require("electron");

contextBridge.exposeInMainWorld("dizel", {
  platform: process.platform,

  /** @returns {Promise<{port: number, token: string} | null>} */
  getSidecar: () => ipcRenderer.invoke("sidecar:info"),

  /** @param {(info: {port: number, token: string}) => void} cb */
  onSidecarReady: (cb) => {
    const handler = (_event, info) => cb(info);
    ipcRenderer.on("sidecar:ready", handler);
    return () => ipcRenderer.removeListener("sidecar:ready", handler);
  },

  /** @param {(info: {code: number|null, signal: string|null}) => void} cb */
  onSidecarDown: (cb) => {
    const handler = (_event, info) => cb(info);
    ipcRenderer.on("sidecar:down", handler);
    return () => ipcRenderer.removeListener("sidecar:down", handler);
  },

  /** @returns {Promise<{ok: boolean, info?: {port: number, token: string}, error?: string}>} */
  restartSidecar: () => ipcRenderer.invoke("sidecar:restart"),

  /** @param {(message: string) => void} cb */
  onSidecarError: (cb) => {
    const handler = (_event, message) => cb(message);
    ipcRenderer.on("sidecar:error", handler);
    return () => ipcRenderer.removeListener("sidecar:error", handler);
  },

  /** Ask the shell to flash the taskbar — used when a reply lands unfocused. */
  attention: () => ipcRenderer.send("app:attention"),

  window: {
    minimize: () => ipcRenderer.send("window:minimize"),
    toggleMaximize: () => ipcRenderer.send("window:toggle-maximize"),
    close: () => ipcRenderer.send("window:close"),
    state: () => ipcRenderer.invoke("window:state"),
    /** @param {(maximized: boolean) => void} cb */
    onMaximizeChange: (cb) => {
      const handler = (_event, maximized) => cb(maximized);
      ipcRenderer.on("window:maximized", handler);
      return () => ipcRenderer.removeListener("window:maximized", handler);
    },
  },

  openExternal: (url) => ipcRenderer.invoke("app:open-external", url),

  updates: {
    getState: () => ipcRenderer.invoke("update:status"),
    check: () => ipcRenderer.invoke("update:check"),
    download: () => ipcRenderer.invoke("update:download"),
    install: () => ipcRenderer.invoke("update:install"),
    openRelease: () => ipcRenderer.invoke("update:open-release"),
    setAutoCheck: (enabled) => ipcRenderer.invoke("update:set-auto-check", enabled),
    onStatus: (cb) => {
      const handler = (_event, state) => cb(state);
      ipcRenderer.on("update:status", handler);
      return () => ipcRenderer.removeListener("update:status", handler);
    },
  },

  /** Text currently on the system clipboard, used by the composer's + menu. */
  readClipboardText: () => ipcRenderer.invoke("clipboard:read-text"),

  /** @param {"attachment" | "checkpoint"} kind */
  pickFiles: (kind) => ipcRenderer.invoke("dialog:pick-files", kind),

  pickFolder: () => ipcRenderer.invoke("dialog:pick-folder"),

  saveJson: (defaultName, json) =>
    ipcRenderer.invoke("dialog:save-json", { defaultName, json }),

  onNewChat: (callback) => {
    ipcRenderer.on("app:new-chat", () => callback());
  },

  /** Electron 32+ removed File.path; this is the supported replacement. */
  pathForFile: (file) => {
    try {
      return webUtils.getPathForFile(file);
    } catch {
      return "";
    }
  },

  /** @returns {Promise<{ok: boolean, content?: string, size?: number, truncated?: boolean, error?: string}>} */
  readTextFile: (filePath) => ipcRenderer.invoke("fs:read-text", filePath),
  /** Image bytes as a data URL, for thumbnails. */
  readDataUrl: (filePath) => ipcRenderer.invoke("fs:data-url", filePath),
  revealInFolder: (filePath) => ipcRenderer.invoke("fs:reveal", filePath),
  openPath: (filePath) => ipcRenderer.invoke("fs:open-path", filePath),
  saveText: (options) => ipcRenderer.invoke("fs:save-text", options),

  systemStats: () => ipcRenderer.invoke("system:stats"),

  /** Font scale is real browser zoom, not a CSS transform. */
  setZoom: (factor) => webFrame.setZoomFactor(factor),
});
