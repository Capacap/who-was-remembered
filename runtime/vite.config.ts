import { defineConfig } from "vite";

// Relative base so the built bundle drops onto any static host or subpath
// without rewriting asset URLs. The piece ships as a folder of static files.
export default defineConfig({
  base: "./",
});
