import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Every API call goes to the real FastAPI backend (docker compose, 127.0.0.1:8000) through this proxy.
// There is no mock layer: if the backend is down, the pages say so.
const API = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    // /api/<path> -> <backend>/<path>. The prefix keeps API paths apart from the app's own routes (/student ...).
    proxy: { "/api": { target: API, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
