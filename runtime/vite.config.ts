import { defineConfig } from "vite";

// Relative base so the built bundle drops onto any static host or subpath
// without rewriting asset URLs. The piece ships as a folder of static files.
export default defineConfig({
  base: "./",
  // Bind on the LAN (0.0.0.0 + ::) so a phone on the same wifi can hit the dev
  // server for on-device touch testing. Dev-only; never affects the built bundle.
  server: { host: true },
});
