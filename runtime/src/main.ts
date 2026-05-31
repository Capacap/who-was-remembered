import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import {
  initTerrain,
  initHeightmap,
  sampleHeight,
  sampleNormal,
  createGround,
  applyDistanceFade,
} from "./terrain";
import { buildSky } from "./sky";

// --- walkable field --------------------------------------------------------
// One instanced box per figure, placed straight from the pipeline's (x, y), with
// a first-person controller so the disc can be walked. The ground is a radial
// hill (see terrain.ts): books are seated on it and tilted to its normal, and
// the player's walk height samples the same function so nothing floats.
// Books are still placeholder primitives. Everything visual here is scaffolding.

const info = document.getElementById("info") as HTMLDivElement;

// world scale: the pipeline derives R_MAX from ~1.4 world units to the metre
// (see DESIGN.md / stage6). Eye height and speeds are in metres, converted once.
const UNITS_PER_METRE = 1.4;
const EYE_HEIGHT = 1.7 * UNITS_PER_METRE; // ~2.4u: stand a head above the sand
const WALK_SPEED = 2.5; // units/sec, DESIGN's deliberately slow pace
const RUN_MULT = 5; // hold-to-run
const FLY_SPEED = 60; // crossing the void on foot is an 80-min walk by design
const FLY_RUN_MULT = 6;

// tier -> (overall size, colour). Books lie flat on the sand, so prominence is
// no longer height: a major is a larger, hotter volume, an ordinary a small
// sandy one. (Cross-disc legibility of majors is now a beacon-VFX problem, not a
// tall-pillar one; the field reads as scattered books, not a skyline.)
const TIER_SCALE = [1.0, 1.4, 2.2]; // ordinary, minor, major
const TIER_COLOR = [0xb89b6e, 0xdcab4c, 0xff5a2c].map((c) => new THREE.Color(c));

// Per-instance variety to break up the uniform-grid read. Rotation/tilt/footprint
// are decorative (seeded, don't move the book). SCATTER does move it: a render-only
// experiment — if it earns its keep it belongs in stage6's jitter, not here.
const TILT_MAX = 0.05; // random lean off the ground normal, radians; small so a flat book keeps full contact
const FOOT_VAR = 0.25; // +/- fraction on cover dimensions
const SCATTER = 0; // spacing is now stage6's job (relaxation pass); renderer draws placement as-is

// mulberry32: cheap deterministic PRNG so the variety is stable across reloads.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function loadPositions(url: string) {
  const buf = await (await fetch(url)).arrayBuffer();
  const view = new DataView(buf);
  const n = view.getUint32(0, true);
  let off = 4;
  const x = new Float32Array(buf, off, n);
  off += n * 4;
  const y = new Float32Array(buf, off, n);
  off += n * 4;
  const tier = new Uint8Array(buf, off, n);
  off += n;
  // geo source / placement confidence: 0 birth, 1 death, 2 citizenship,
  // 3 gazetteer, 4 residue (no recorded location -> placed adrift).
  const geo = new Uint8Array(buf, off, n);
  return { n, x, y, tier, geo };
}

function buildField(field: Awaited<ReturnType<typeof loadPositions>>) {
  const { n, x, y, tier, geo } = field;
  // a closed book lying flat on the sand: broad cover (x, z), slim spine (y, the
  // up axis). ~0.42 x 0.58u at tier 1, smaller than the modern spacing (~0.9u) so
  // neighbours read as distinct dropped objects rather than an overlapping mass.
  // The spine is the up extent, so seating offsets by half of it along the normal.
  const SPINE = 0.12; // book thickness at tier 1, world units
  const box = new THREE.BoxGeometry(0.42, SPINE, 0.58); // width, thickness, length
  const mat = new THREE.MeshLambertMaterial();
  // books dissolve with the ground: the same camera-distance fade to transparent,
  // so the field thins into the dome at the horizon rather than leaving sharp specks
  // floating over ground that has already faded out.
  applyDistanceFade(mat);
  const mesh = new THREE.InstancedMesh(box, mat, n);
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);

  // keep the rendered (jittered) ground positions so the look-at picker aims at
  // where a book actually stands, not its pre-scatter pipeline coordinate.
  const px = new Float32Array(n);
  const pz = new Float32Array(n);

  // residue (geo === 4) has no recorded location; its angle is a hash, not
  // geography. Wash it toward a pale grey so it reads as adrift rather than
  // confidently placed, the spatial echo of the date-uncertainty haze.
  const ADRIFT = new THREE.Color(0xb7b0a2);
  const dummy = new THREE.Object3D();
  const col = new THREE.Color();
  const UP = new THREE.Vector3(0, 1, 0);
  const normal = new THREE.Vector3();
  const qAlign = new THREE.Quaternion();
  const qLocal = new THREE.Quaternion();
  const eul = new THREE.Euler();
  const rnd = mulberry32(0x1234abcd);
  for (let i = 0; i < n; i++) {
    const s = TIER_SCALE[tier[i]];
    const fw = 1 + (rnd() * 2 - 1) * FOOT_VAR;
    const fl = 1 + (rnd() * 2 - 1) * FOOT_VAR;
    // pipeline (x, y) is the ground plane; map to world (x, z), y is up.
    px[i] = x[i] + (rnd() * 2 - 1) * SCATTER;
    pz[i] = y[i] + (rnd() * 2 - 1) * SCATTER;
    const gy = sampleHeight(px[i], pz[i]);
    // sample the normal across the book's own footprint (half-length ~0.3·s) so a
    // large book conforms to the slope it spans instead of one 0.5u patch.
    sampleNormal(px[i], pz[i], normal, 0.3 * s);
    // lay the book flat on the slope: its spine (+y) aligns to the ground normal,
    // a random spin about that axis gives it a dropped heading, and a small lean
    // off the normal keeps it from looking neatly placed.
    qAlign.setFromUnitVectors(UP, normal);
    eul.set(
      (rnd() * 2 - 1) * TILT_MAX,
      rnd() * Math.PI * 2,
      (rnd() * 2 - 1) * TILT_MAX,
    );
    qLocal.setFromEuler(eul);
    dummy.quaternion.copy(qAlign).multiply(qLocal);
    // settle the book INTO the sand: lift the centre by less than half the spine,
    // so the underside sits a touch below grade and any leaning corner rests in
    // the surface rather than hovering over it.
    const lift = (SPINE * s) * 0.3;
    dummy.position.set(
      px[i] + normal.x * lift,
      gy + normal.y * lift,
      pz[i] + normal.z * lift,
    );
    dummy.scale.set(s * fw, s, s * fl);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    col.copy(TIER_COLOR[tier[i]]);
    if (geo[i] === 4) col.lerp(ADRIFT, 0.75);
    mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return { mesh, px, pz, tier: tier as Uint8Array };
}

// Teleporter monuments (26): tall emissive-blue pillars, the cool complement to
// the hot-orange landmark books. They beacon through the fog so the player can
// steer toward a known place from across the disc. Far taller than the 16u major
// books because they're rare and meant to be seen from a long way off.
const TP_HEIGHT = 60;
const TP_FOOT = 5;

interface Teleporter {
  label: string;
  x: number;
  y: number;
  era: string;
  seat: string;
  n: number;
}

async function loadTeleporters(url: string): Promise<Teleporter[]> {
  return (await fetch(url)).json();
}

// Baked heightmap (stage9 heightmap.bin): uint32 resolution, float32 world_size,
// then float32 height[res*res] row-major. See stage9_mesh.py:write_heightmap_bin.
async function loadHeightmap(
  url: string,
): Promise<{ res: number; worldSize: number; data: Float32Array }> {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  const res = dv.getUint32(0, true);
  const worldSize = dv.getFloat32(4, true);
  const data = new Float32Array(buf, 8, res * res);
  return { res, worldSize, data };
}

function buildTeleporters(list: Teleporter[]) {
  const geom = new THREE.BoxGeometry(TP_FOOT, TP_HEIGHT, TP_FOOT);
  // emissive so it reads as a lit beacon at distance rather than a shaded box
  // that the fog swallows; a touch of lambert keeps some form on the near ones.
  // Opaque and unfaded (no applyDistanceFade), so the beacons punch through the
  // haze that dissolves the books and ground: the point of one you steer toward.
  const mat = new THREE.MeshLambertMaterial({
    color: 0x1c3a8c,
    emissive: 0x2f6cff,
    emissiveIntensity: 0.9,
  });
  const group = new THREE.Group();
  for (const tp of list) {
    const pillar = new THREE.Mesh(geom, mat);
    // stands plumb on its flattened plaza (terrain levels a disc here).
    pillar.position.set(tp.x, sampleHeight(tp.x, tp.y) + TP_HEIGHT / 2, tp.y);
    group.add(pillar);
  }
  return group;
}

// --- figure metadata (look-at + inspect) -----------------------------------
// meta.bin is row-aligned with positions.bin, so a picked instanceId indexes it
// directly. String blobs are decoded lazily (only the targeted book), never all
// at once. Layout documented in pipeline/export_runtime.py:export_meta.
const YEAR_MISSING = -32768;

async function loadMeta(url: string) {
  const buf = await (await fetch(url)).arrayBuffer();
  const dv = new DataView(buf);
  const n = dv.getUint32(0, true);
  const nameLen = dv.getUint32(4, true);
  const descLen = dv.getUint32(8, true);
  let off = 12;
  const nameOff = new Uint32Array(buf, off, n + 1);
  off += 4 * (n + 1);
  const descOff = new Uint32Array(buf, off, n + 1);
  off += 4 * (n + 1);
  const birth = new Int16Array(buf, off, n);
  off += 2 * n;
  const death = new Int16Array(buf, off, n);
  off += 2 * n;
  const nameBytes = new Uint8Array(buf, off, nameLen);
  off += nameLen;
  const descBytes = new Uint8Array(buf, off, descLen);

  const decoder = new TextDecoder();
  const name = (i: number) =>
    decoder.decode(nameBytes.subarray(nameOff[i], nameOff[i + 1]));
  const desc = (i: number) =>
    decoder.decode(descBytes.subarray(descOff[i], descOff[i + 1]));
  return { n, birth, death, name, desc };
}

function wikiUrl(name: string): string {
  return "https://en.wikipedia.org/wiki/" + encodeURIComponent(name.replace(/ /g, "_"));
}

function fmtYear(v: number): string {
  return v < 0 ? `${-v} BCE` : `${v}`;
}

function fmtYears(b: number, d: number): string {
  const bb = b === YEAR_MISSING ? null : b;
  const dd = d === YEAR_MISSING ? null : d;
  if (bb !== null && dd !== null) return `${fmtYear(bb)} – ${fmtYear(dd)}`;
  if (dd !== null) return `d. ${fmtYear(dd)}`;
  if (bb !== null) return `b. ${fmtYear(bb)}`;
  return "";
}

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!,
  );
}

// Look-at picker: each tick, find the book nearest the camera whose centre falls
// within a narrow cone of the view direction. A distance cull rejects almost all
// 576k instances before the alignment test, so the brute-force sweep is cheap at
// the throttled cadence. Aims at jittered positions (matching what's drawn).
function createPicker(
  camera: THREE.PerspectiveCamera,
  px: Float32Array,
  pz: Float32Array,
) {
  // reading is close-up only: you have to travel and walk up to a book to learn
  // who it is. Long-range legibility (landmarks visible from afar) is a separate
  // problem for a beacon VFX, not for this radius. ~6 u ≈ 4 m at 1.4 u/m.
  const MAX_DIST = 6;
  const MAX_DIST2 = MAX_DIST * MAX_DIST;
  // wider cone than a precise crosshair: once you're standing at a book, facing
  // its general direction should read it without pixel-perfect aim.
  const COS_CONE = Math.cos((11 * Math.PI) / 180);
  // books lie flat on the sand, so the pick point sits just above the ground;
  // you read a book by walking up and looking down at it.
  const PICK_HEIGHT = 0.2;
  const n = px.length;
  const fwd = new THREE.Vector3();

  return function pick(): number {
    camera.getWorldDirection(fwd);
    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;
    // books within reach share the player's local ground, so aim a fixed height
    // above it rather than y=0 (which the hill makes wrong by hundreds of units).
    const dyAll = sampleHeight(cx, cz) + PICK_HEIGHT - cy;
    let best = -1;
    let bestDist2 = Infinity;
    for (let i = 0; i < n; i++) {
      const dx = px[i] - cx;
      const dz = pz[i] - cz;
      const horiz2 = dx * dx + dz * dz;
      if (horiz2 > MAX_DIST2) continue;
      const dy = dyAll;
      const dist2 = horiz2 + dy * dy;
      const dot = (dx * fwd.x + dy * fwd.y + dz * fwd.z) / Math.sqrt(dist2);
      if (dot < COS_CONE) continue;
      if (dist2 < bestDist2) {
        bestDist2 = dist2;
        best = i;
      }
    }
    return best;
  };
}

// --- compass ----------------------------------------------------------------
// A horizontal strip carrying the world's only true axis: time. ⊙ points to the
// origin (inward = the present), ◯ to the radial outward direction (the past, the
// way back in time); the two are always antipodal. This world has no north, only
// longitude around and time in/out. The player drops their own bookmarks (no
// designer landmarks), which slide on the bar by bearing. Readouts are years,
// never raw units — your depth reads as your era, each bookmark shows its fixed era.
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

interface World {
  R_INNER: number;
  R_MAX: number;
  TIME_SPAN: number;
  RADIUS_ALPHA: number;
  REF_YEAR: number;
}

async function loadWorld(url: string): Promise<World> {
  return (await fetch(url)).json();
}

function createCompass(
  camera: THREE.PerspectiveCamera,
  px: Float32Array,
  pz: Float32Array,
  meta: Awaited<ReturnType<typeof loadMeta>>,
  world: World,
) {
  const root = document.getElementById("compass") as HTMLDivElement;
  const eraEl = document.getElementById("compass-era") as HTMLDivElement;
  const labelEl = document.getElementById("compass-label") as HTMLDivElement;

  const HALF_SPAN = Math.PI / 2; // ±90° of bearing maps across the strip
  const NOTCH = (8 * Math.PI) / 180; // within 8° of dead-ahead -> bloom the label
  const ARRIVE = 2.5; // within this many units a marker hides: "you're here"

  // ⊙ present (inward, toward origin) and ◯ past (outward) — the time axis.
  const originEl = document.createElement("div");
  originEl.className = "cmark origin";
  originEl.textContent = "⊙";
  root.appendChild(originEl);
  const pastEl = document.createElement("div");
  pastEl.className = "cmark past";
  pastEl.textContent = "◯";
  root.appendChild(pastEl);

  const marks = new Map<number, HTMLElement>(); // instanceId -> bookmark glyph
  const fwd = new THREE.Vector3();

  function eraStr(year: number): string {
    if (year >= world.REF_YEAR) return "the present";
    return year < 0 ? `c.${-year} BCE` : `c.${year}`;
  }

  function playerEra(r: number): string {
    if (r <= world.R_INNER) return "the present";
    const t = clamp((r - world.R_INNER) / (world.R_MAX - world.R_INNER), 0, 1);
    const tt = Math.pow(t, 1 / world.RADIUS_ALPHA);
    const year = world.REF_YEAR - tt * world.TIME_SPAN;
    return eraStr(Math.round(year / 10) * 10); // decade-rounded for a calm readout
  }

  function figureEra(i: number): string {
    const y = meta.death[i] !== YEAR_MISSING ? meta.death[i] : meta.birth[i];
    if (y === YEAR_MISSING) return "";
    return y < 0 ? `c.${-y} BCE` : `c.${y}`;
  }

  // place a marker by world-space offset (target - player) relative to facing.
  // returns the signed bearing and distance so the caller can pick the notch label.
  function place(el: HTMLElement, dx: number, dz: number, fa: number) {
    const dist = Math.hypot(dx, dz);
    if (dist < ARRIVE) {
      el.style.display = "none";
      return { rel: Infinity, dist };
    }
    let rel = Math.atan2(dx, dz) - fa;
    rel = Math.atan2(Math.sin(rel), Math.cos(rel)); // normalize to [-π, π]
    // screen-right is the camera's +x, which sits at negative `rel` (see derivation
    // in commit), hence the minus. Markers behind the window clamp to the edge.
    const frac = 0.5 - 0.5 * clamp(rel / HALF_SPAN, -1, 1);
    el.style.left = `${frac * 100}%`;
    el.style.opacity = Math.abs(rel) > HALF_SPAN ? "0.4" : "1";
    el.style.display = "block";
    return { rel, dist };
  }

  function update() {
    camera.getWorldDirection(fwd);
    const fa = Math.atan2(fwd.x, fwd.z);
    const cx = camera.position.x;
    const cz = camera.position.z;
    const r = Math.hypot(cx, cz);

    eraEl.textContent = playerEra(r);

    // track the marker nearest dead-ahead, to bloom its label under the notch.
    let bestRel = Infinity;
    let bestFrac = 0.5;
    let bestText = "";
    const consider = (rel: number, text: string) => {
      if (Math.abs(rel) < bestRel) {
        bestRel = Math.abs(rel);
        bestFrac = 0.5 - 0.5 * clamp(rel / HALF_SPAN, -1, 1);
        bestText = text;
      }
    };

    const o = place(originEl, -cx, -cz, fa); // origin (present) is at world (0, 0)
    consider(o.rel, "the present");
    const p = place(pastEl, cx, cz, fa); // outward radial = deeper into the past
    consider(p.rel, "the past");
    for (const [i, el] of marks) {
      const m = place(el, px[i] - cx, pz[i] - cz, fa);
      const era = figureEra(i);
      consider(m.rel, era ? `${meta.name(i)} · ${era}` : meta.name(i));
    }

    if (bestRel < NOTCH && bestText) {
      labelEl.textContent = bestText;
      labelEl.style.left = `${bestFrac * 100}%`;
      labelEl.style.display = "block";
    } else {
      labelEl.style.display = "none";
    }
  }

  return {
    update,
    has: (i: number) => marks.has(i),
    toggle(i: number): boolean {
      const el = marks.get(i);
      if (el) {
        el.remove();
        marks.delete(i);
        return false;
      }
      const m = document.createElement("div");
      m.className = "cmark bookmark";
      m.textContent = "◆";
      root.appendChild(m);
      marks.set(i, m);
      return true;
    },
  };
}

// --- first-person controller ----------------------------------------------
// PointerLockControls owns the look (Euler camera, pitch clamped internally).
// We own translation: a key-state object drives a velocity each frame. Walk
// mode pins Y to the ground; fly mode frees Y and follows the full look vector.
function createController(camera: THREE.PerspectiveCamera, dom: HTMLElement) {
  const controls = new PointerLockControls(camera, dom);
  const keys = new Set<string>();
  let flying = false;

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.code === "KeyF") flying = !flying;
    keys.add(e.code);
  };
  const onKeyUp = (e: KeyboardEvent) => keys.delete(e.code);
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
  // releasing the lock (Esc) should also drop held keys, or the player keeps
  // drifting after the cursor reappears.
  controls.addEventListener("unlock", () => keys.clear());
  dom.addEventListener("click", () => controls.lock());

  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const move = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);

  function update(dt: number) {
    if (!controls.isLocked) return flying;

    camera.getWorldDirection(forward);
    if (!flying) forward.y = 0; // walk: ignore pitch, move along the ground
    forward.normalize();
    right.crossVectors(forward, UP).normalize();

    move.set(0, 0, 0);
    if (keys.has("KeyW") || keys.has("ArrowUp")) move.add(forward);
    if (keys.has("KeyS") || keys.has("ArrowDown")) move.sub(forward);
    if (keys.has("KeyD") || keys.has("ArrowRight")) move.add(right);
    if (keys.has("KeyA") || keys.has("ArrowLeft")) move.sub(right);
    if (flying) {
      if (keys.has("Space")) move.y += 1;
      if (keys.has("KeyC")) move.y -= 1;
    }

    const running = keys.has("ShiftLeft") || keys.has("ShiftRight");
    const base = flying ? FLY_SPEED : WALK_SPEED;
    const speed = base * (running ? (flying ? FLY_RUN_MULT : RUN_MULT) : 1);

    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(speed * dt);
      camera.position.add(move);
    }
    // walk mode keeps the eye a fixed height above the ground every frame;
    // fly mode leaves Y wherever the player flew it. The walk height reads the
    // baked heightmap (the same field the near patch tessellates), so the feet
    // sit on the visible near ground rather than the analytic surface.
    if (!flying) {
      camera.position.y = sampleHeight(camera.position.x, camera.position.z) + EYE_HEIGHT;
    }
    return flying;
  }

  return { controls, update };
}

async function main() {
  const scene = new THREE.Scene();
  // The ground and books fade to TRANSPARENT at distance (see applyDistanceFade)
  // and dissolve into the sky dome, so the distance always reads as exactly the
  // sky behind it, never tinted, and no footprint edge is ever left to see. A
  // single fog colour was the alternative but it can only tint distant surfaces
  // toward one flat colour (brightening the grey void or darkening the present)
  // and, worse, leaves opaque geometry whose square footprint shows from a height.
  // HORIZON is the tone the land dissolves into: it is handed to the dome as its
  // ground-haze colour so land and sky meet without a colour step.
  const HORIZON = new THREE.Color(0x847b6d);
  scene.background = HORIZON;

  // gradient sky dome, horizon band pinned to the fog colour so distant ground
  // dissolves into it. It recentres on the camera each frame (see the loop), so
  // it reads as infinitely far and the world never shows an edge against it.
  const sky = buildSky(HORIZON);
  scene.add(sky);

  const camera = new THREE.PerspectiveCamera(
    70,
    window.innerWidth / window.innerHeight,
    0.1,
    24000,
  );

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const { controls, update } = createController(camera, renderer.domElement);
  scene.add(controls.object);

  // spawn dead centre (0,0), on the summit of the present, facing outward across
  // the empty plaza. The first view is the whole uneven ring at once: a dense
  // wall of books toward the Western longitudes, near-empty ground toward the
  // gaps. The gaze runs down the slope to the distant desert floor, so the land
  // visibly falls away into the past. Survey, then travel.
  camera.position.set(0, sampleHeight(0, 0) + EYE_HEIGHT, 0);
  camera.lookAt(0, 0, 8000);

  // low sun for long shadows-of-mood later; flat lambert for now.
  scene.add(new THREE.HemisphereLight(0xfff1d0, 0x8a7350, 1.1));
  const sun = new THREE.DirectionalLight(0xffe8c0, 1.4);
  sun.position.set(-400, 300, 200);
  scene.add(sun);

  info.innerHTML = "loading positions…";
  const [field, teleporters, meta, world, heightmap] = await Promise.all([
    loadPositions("positions.bin"),
    loadTeleporters("teleporters.json"),
    loadMeta("meta.bin"),
    loadWorld("world.json"),
    loadHeightmap("heightmap.bin"),
  ]);
  // the heightmap is the ground-height source for the clipmap and the player's
  // feet; init it before anything samples it.
  initHeightmap(heightmap.res, heightmap.worldSize, heightmap.data);
  // the plazas flatten around the teleporters, so terrain needs them before the
  // books or pillars are seated. Books, pillars, the picker and the ring all read
  // sampleHeight/sampleNormal now, so they seat on the same baked surface the
  // clipmap draws (no float-off; the analytic field is only the heightmap fallback).
  initTerrain(teleporters);
  // the ground is a camera-following clipmap tessellated from the heightmap (see
  // terrain.createGround); no static mesh ships any more.
  const ground = createGround();
  ground.update(camera.position.x, camera.position.z);
  scene.add(ground.group);
  // settle the eye onto the baked surface now the heightmap is loaded (spawn was
  // placed on the analytic fallback before the fetch resolved).
  camera.position.y = sampleHeight(camera.position.x, camera.position.z) + EYE_HEIGHT;
  const built = buildField(field);
  scene.add(built.mesh);
  scene.add(buildTeleporters(teleporters));

  // a thin ring marking the modern edge (R_INNER), where year 2000 sits and the
  // books begin. Read from world.json so it tracks the placement, never drifts.
  const pad = new THREE.Mesh(
    new THREE.RingGeometry(world.R_INNER - 2, world.R_INNER + 2, 256),
    new THREE.MeshBasicMaterial({ color: 0x7a6038, side: THREE.DoubleSide }),
  );
  pad.rotation.x = -Math.PI / 2;
  // the ring sits on the flat summit plateau (R_INNER is inside it), so a single
  // height for the whole ring is exact; lift it just clear of the ground.
  pad.position.y = sampleHeight(world.R_INNER, 0) + 0.1;
  scene.add(pad);

  // --- look-at glance + inspect overlay -------------------------------------
  const glance = document.getElementById("glance") as HTMLDivElement;
  const overlay = document.getElementById("overlay") as HTMLDivElement;
  const card = document.getElementById("card") as HTMLDivElement;
  const pick = createPicker(camera, built.px, built.pz);
  const compass = createCompass(camera, built.px, built.pz, meta, world);

  let target = -1; // instanceId under the reticle, or -1
  let overlayOpen = false;

  const renderName = (i: number) => escapeHtml(meta.name(i) || "(untitled)");
  const renderDesc = (i: number) => escapeHtml(meta.desc(i));

  function showGlance(i: number) {
    const desc = renderDesc(i);
    const years = fmtYears(meta.birth[i], meta.death[i]);
    glance.innerHTML =
      `<div class="name">${renderName(i)}</div>` +
      (desc ? `<div class="desc">${desc}</div>` : "") +
      (years ? `<div class="years">${years}</div>` : "");
    glance.style.display = "block";
  }

  function openOverlay(i: number) {
    const desc = renderDesc(i);
    const years = fmtYears(meta.birth[i], meta.death[i]);
    const url = wikiUrl(meta.name(i));
    card.innerHTML =
      `<div class="name">${renderName(i)}</div>` +
      (desc ? `<div class="desc">${desc}</div>` : "") +
      (years ? `<div class="years">${years}</div>` : "") +
      `<div class="actions">` +
      `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">Read on Wikipedia →</a>` +
      `<button id="bookmarkBtn"></button>` +
      `</div>` +
      `<div class="hint">Esc or click outside to close</div>`;
    const btn = card.querySelector("#bookmarkBtn") as HTMLButtonElement;
    const renderBtn = () => {
      const on = compass.has(i);
      btn.textContent = on ? "★ Bookmarked" : "☆ Bookmark";
      btn.classList.toggle("on", on);
    };
    renderBtn();
    btn.addEventListener("click", () => {
      compass.toggle(i);
      renderBtn();
    });
    overlay.style.display = "flex";
    overlayOpen = true;
    glance.style.display = "none";
    controls.unlock(); // free the cursor so the link is clickable
  }

  function closeOverlay() {
    overlay.style.display = "none";
    overlayOpen = false;
  }

  // click on the backdrop (not the card) closes; clicking the card/link doesn't.
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeOverlay();
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "KeyE" && !overlayOpen && controls.isLocked && target >= 0) {
      openOverlay(target);
    } else if (e.code === "Escape" && overlayOpen) {
      closeOverlay();
    }
  });

  const hint =
    "click to look · WASD move · Shift run · F fly · Space/C up·down · E inspect · Esc release";
  const setHud = (flying: boolean) => {
    info.innerHTML = `${field.n.toLocaleString()} figures · ${
      flying ? "flying" : "walking"
    } · centre = year 2000<br>${hint}`;
  };
  setHud(false);

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  const clock = new THREE.Clock();
  let wasFlying = false;
  let sincePick = 0;
  const PICK_INTERVAL = 0.12; // ~8 Hz; the look-at label needn't be per-frame
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1); // clamp after tab-out stalls
    const flying = update(dt);
    if (flying !== wasFlying) {
      setHud(flying);
      wasFlying = flying;
    }

    // look-at picking: only while walking the scene (locked) and not inspecting.
    sincePick += dt;
    if (!overlayOpen && controls.isLocked) {
      if (sincePick >= PICK_INTERVAL) {
        sincePick = 0;
        target = pick();
        if (target >= 0) showGlance(target);
        else glance.style.display = "none";
      }
    } else if (glance.style.display !== "none") {
      glance.style.display = "none";
      target = -1;
    }

    // keep the clipmap centred on the camera: each level re-tessellates only when
    // it crosses one of its own cells, and the discard holes follow every frame.
    ground.update(camera.position.x, camera.position.z);

    compass.update();
    sky.position.copy(camera.position); // keep the dome centred on the viewer
    renderer.render(scene, camera);
  });
}

main();
