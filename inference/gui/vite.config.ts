import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base:'./' is required — the packaged renderer is loaded from disk (file://),
// so every asset reference must be relative.
export default defineConfig({
  base: "./",
  plugins: [react()],
  // The bundled changelog lives at the repo root (../../../../CHANGELOG.md?raw);
  // the dev server must be allowed to read above the gui package.
  server: { port: 5173, strictPort: true, fs: { allow: ["../.."] } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "chrome120",
  },
});
