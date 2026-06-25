import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_PROXY_TARGET = process.env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8050";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: API_PROXY_TARGET, changeOrigin: true } },
  },
  preview: {
    proxy: { "/api": { target: API_PROXY_TARGET, changeOrigin: true } },
  },
});
