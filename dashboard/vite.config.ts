import { defineConfig } from "vite";

export default defineConfig({
  base: "/static/",
  build: {
    outDir: "../src/reachy_mini_conversation_app/static",
    emptyOutDir: true,
    assetsDir: "assets",
  },
});
