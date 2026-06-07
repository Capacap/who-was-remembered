// Tiny localhost receiver for harness screenshots. The in-app harness can read
// canvas.toDataURL() but a browser can't write files; this accepts those data
// URLs over HTTP and writes them as PNGs into a target dir (default shots/current).
//
//   node scripts/recv.mjs [outDir]      listens on 127.0.0.1:7654
//
// Then from the ?harness page: for each pose, __HARNESS.pose(name); wait for the
// LOD to settle; POST __HARNESS.shot() to http://localhost:7654/shot?name=<name>.
// CORS is wide open because it only ever binds loopback.

import { createServer } from "node:http";
import { writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = process.argv[2]
  ? join(process.cwd(), process.argv[2])
  : join(ROOT, "shots", "current");
mkdirSync(outDir, { recursive: true });

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "*",
};

createServer((req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS);
    return res.end();
  }
  const url = new URL(req.url, "http://localhost");
  if (req.method === "POST" && url.pathname === "/shot") {
    const name = (url.searchParams.get("name") || "shot").replace(/[^\w.-]/g, "_");
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const b64 = body.replace(/^data:image\/png;base64,/, "");
      const file = join(outDir, `${name}.png`);
      writeFileSync(file, Buffer.from(b64, "base64"));
      console.log(`wrote ${file} (${(b64.length / 1024).toFixed(0)} KB b64)`);
      res.writeHead(200, CORS);
      res.end("ok");
    });
    return;
  }
  res.writeHead(404, CORS);
  res.end("not found");
}).listen(7654, "127.0.0.1", () => console.log(`recv -> ${outDir} on :7654`));
