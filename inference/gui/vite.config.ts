import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base:'./' is required — the packaged renderer is loaded from disk (file://),
// so every asset reference must be relative.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "chrome120",
  },
});
