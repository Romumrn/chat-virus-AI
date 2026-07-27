import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// The backend runs on :8080. In dev we proxy /api there so the SPA and API
// share an origin (no CORS). SSE needs buffering disabled — proxy passes it
// through untouched.
const API_TARGET = process.env.API_PROXY_TARGET || "http://localhost:8080";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
});
