// Golden-image diff for the screenshot harness (see src/harness.ts).
//
// Compares two directories of PNGs by matching filename and reports per-pose
// mismatched-pixel counts. Writes a highlighted diff PNG for any pose that
// differs. Exits non-zero if anything moved, so it can gate a refactor step.
//
//   node scripts/diff.mjs                 compare shots/current against shots/golden
//   node scripts/diff.mjs --update        promote shots/current to shots/golden
//   node scripts/diff.mjs --tol 40        per-pose pixel floor below which a pose
//                                         counts as clean (default 25)
//   node scripts/diff.mjs a/ b/           compare two explicit directories
//
// The tolerance floor absorbs sub-frame nondeterminism -- a far-field point sprite
// flickering across the alpha-fade threshold, an AA edge -- which is single-digit
// pixels. A real render regression (reordered shader chain, flipped depthWrite,
// changed draw order) moves thousands, so the floor never masks one.
//
// The capture half is done by the in-app harness: load the app with ?harness,
// drive window.__HARNESS to pose + read canvas.toDataURL() per pose, and save the
// PNGs into shots/current. Goldens are a local regression baseline (GPU/machine
// specific) -- re-capture them on the clean tree before a refactor, not committed.

import { readdirSync, readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import pixelmatch from "pixelmatch";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SHOTS = join(ROOT, "shots");

const args = process.argv.slice(2);
const update = args.includes("--update");
const tolIdx = args.indexOf("--tol");
const TOL = tolIdx >= 0 ? Number(args[tolIdx + 1]) : 25;
const dirs = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--tol");
const goldenDir = dirs[0] ? join(process.cwd(), dirs[0]) : join(SHOTS, "golden");
const currentDir = dirs[1] ? join(process.cwd(), dirs[1]) : join(SHOTS, "current");
const diffDir = join(SHOTS, "diff");

const pngsIn = (d) =>
  existsSync(d) ? readdirSync(d).filter((f) => f.endsWith(".png")) : [];

if (update) {
  mkdirSync(goldenDir, { recursive: true });
  const cur = pngsIn(currentDir);
  if (!cur.length) {
    console.error(`no PNGs in ${currentDir} to promote`);
    process.exit(1);
  }
  for (const f of cur) copyFileSync(join(currentDir, f), join(goldenDir, f));
  console.log(`promoted ${cur.length} shot(s) to golden`);
  process.exit(0);
}

const golden = pngsIn(goldenDir);
const current = pngsIn(currentDir);
if (!golden.length) {
  console.error(`no goldens in ${goldenDir} -- capture a baseline first, then --update`);
  process.exit(1);
}
mkdirSync(diffDir, { recursive: true });

let totalDiff = 0;
let failed = 0;
const missing = [];

for (const name of golden) {
  if (!current.includes(name)) {
    missing.push(name);
    continue;
  }
  const a = PNG.sync.read(readFileSync(join(goldenDir, name)));
  const b = PNG.sync.read(readFileSync(join(currentDir, name)));
  if (a.width !== b.width || a.height !== b.height) {
    console.log(
      `${name.padEnd(22)} SIZE MISMATCH golden ${a.width}x${a.height} vs current ${b.width}x${b.height}`,
    );
    failed++;
    continue;
  }
  const diff = new PNG({ width: a.width, height: a.height });
  const n = pixelmatch(a.data, b.data, diff.data, a.width, a.height, {
    threshold: 0.1, // per-pixel colour tolerance (0..1); absorbs sub-LSB AA jitter
  });
  totalDiff += n;
  const pct = ((100 * n) / (a.width * a.height)).toFixed(3);
  if (n > TOL) {
    failed++;
    writeFileSync(join(diffDir, name), PNG.sync.write(diff));
    console.log(`${name.padEnd(22)} ${String(n).padStart(8)} px (${pct}%)  -> shots/diff/${name}`);
  } else if (n > 0) {
    console.log(`${name.padEnd(22)} ${String(n).padStart(8)} px (within tol ${TOL})`);
  } else {
    console.log(`${name.padEnd(22)} ${"clean".padStart(8)}`);
  }
}

for (const name of missing) console.log(`${name.padEnd(22)} MISSING in current`);

console.log("---");
console.log(
  `${failed} of ${golden.length} pose(s) differ; ${totalDiff} pixels total` +
    (missing.length ? `; ${missing.length} missing` : ""),
);
process.exit(failed || missing.length ? 1 : 0);
