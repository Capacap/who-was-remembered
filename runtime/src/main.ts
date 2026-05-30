import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";

// --- walkable field --------------------------------------------------------
// One instanced box per figure, placed straight from the pipeline's (x, y), now
// with a first-person controller so the disc can actually be walked. No terrain
// yet: the ground is flat and getGroundHeight() returns 0, but the player Y is
// already routed through it so the heightmap drops in without touching movement.
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

// tier -> (book height, colour). Majors stand tall and hot so they read as
// reference points from across the disc; ordinary books are low and sandy.
const TIER_HEIGHT = [2, 6, 16]; // ordinary, minor, major
const TIER_COLOR = [0xb89b6e, 0xdcab4c, 0xff5a2c].map((c) => new THREE.Color(c));

// Per-instance variety to break up the uniform-grid read. Rotation/tilt/footprint
// are decorative (seeded, don't move the book). SCATTER does move it: a render-only
// experiment — if it earns its keep it belongs in stage6's jitter, not here.
const TILT_MAX = 0.1; // random lean, radians
const FOOT_VAR = 0.3; // +/- fraction on footprint width
const SCATTER = 3.0; // render-only positional jitter, world units

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

// Terrain seam: the only thing movement needs from the (future) heightmap is the
// ground height under a point. Flat for now; the heightmap sampler slots in here.
function getGroundHeight(_x: number, _z: number): number {
  return 0;
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
  const box = new THREE.BoxGeometry(1.4, 1, 1.4);
  const mat = new THREE.MeshLambertMaterial();
  const mesh = new THREE.InstancedMesh(box, mat, n);
  mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);

  // residue (geo === 4) has no recorded location; its angle is a hash, not
  // geography. Wash it toward a pale grey so it reads as adrift rather than
  // confidently placed, the spatial echo of the date-uncertainty haze.
  const ADRIFT = new THREE.Color(0xb7b0a2);
  const dummy = new THREE.Object3D();
  const col = new THREE.Color();
  const rnd = mulberry32(0x1234abcd);
  for (let i = 0; i < n; i++) {
    const h = TIER_HEIGHT[tier[i]];
    const foot = 1 + (rnd() * 2 - 1) * FOOT_VAR;
    // pipeline (x, y) is the ground plane; map to world (x, z), y is up.
    dummy.position.set(
      x[i] + (rnd() * 2 - 1) * SCATTER,
      h / 2,
      y[i] + (rnd() * 2 - 1) * SCATTER,
    );
    dummy.rotation.set(
      (rnd() * 2 - 1) * TILT_MAX,
      rnd() * Math.PI * 2,
      (rnd() * 2 - 1) * TILT_MAX,
    );
    dummy.scale.set(foot, h, foot);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    col.copy(TIER_COLOR[tier[i]]);
    if (geo[i] === 4) col.lerp(ADRIFT, 0.75);
    mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return mesh;
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

function buildTeleporters(list: Teleporter[]) {
  const geom = new THREE.BoxGeometry(TP_FOOT, TP_HEIGHT, TP_FOOT);
  // emissive so it reads as a lit beacon at distance rather than a shaded box
  // that the fog swallows; a touch of lambert keeps some form on the near ones.
  const mat = new THREE.MeshLambertMaterial({
    color: 0x1c3a8c,
    emissive: 0x2f6cff,
    emissiveIntensity: 0.9,
  });
  const group = new THREE.Group();
  for (const tp of list) {
    const pillar = new THREE.Mesh(geom, mat);
    pillar.position.set(tp.x, TP_HEIGHT / 2, tp.y);
    group.add(pillar);
  }
  return group;
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
    // fly mode leaves Y wherever the player flew it.
    if (!flying) {
      camera.position.y = getGroundHeight(camera.position.x, camera.position.z) + EYE_HEIGHT;
    }
    return flying;
  }

  return { controls, update };
}

async function main() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xd9c9a8);
  // world radius ~7100u (linear time); fog tuned so the far frontier hazes
  // out rather than popping at the draw edge.
  scene.fog = new THREE.FogExp2(0xd9c9a8, 0.00016);

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

  // spawn at the pad rim (R_INNER = 200), facing outward into the modern
  // thicket so the first view is books receding toward the deep-time void.
  camera.position.set(0, EYE_HEIGHT, 200);
  camera.lookAt(0, EYE_HEIGHT, 8000);

  // low sun for long shadows-of-mood later; flat lambert for now.
  scene.add(new THREE.HemisphereLight(0xfff1d0, 0x8a7350, 1.1));
  const sun = new THREE.DirectionalLight(0xffe8c0, 1.4);
  sun.position.set(-400, 300, 200);
  scene.add(sun);

  // ground large enough to cover the full disc (radius ~7100 + scatter tail).
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(18000, 18000),
    new THREE.MeshLambertMaterial({ color: 0xcdbd99 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // a thin ring marking the landing pad edge (R_INNER = 200).
  const pad = new THREE.Mesh(
    new THREE.RingGeometry(198, 202, 192),
    new THREE.MeshBasicMaterial({ color: 0x7a6038, side: THREE.DoubleSide }),
  );
  pad.rotation.x = -Math.PI / 2;
  pad.position.y = 0.1;
  scene.add(pad);

  info.innerHTML = "loading positions…";
  const [field, teleporters] = await Promise.all([
    loadPositions("positions.bin"),
    loadTeleporters("teleporters.json"),
  ]);
  scene.add(buildField(field));
  scene.add(buildTeleporters(teleporters));

  const hint =
    "click to look · WASD move · Shift run · F fly · Space/C up·down · Esc release";
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
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1); // clamp after tab-out stalls
    const flying = update(dt);
    if (flying !== wasFlying) {
      setHud(flying);
      wasFlying = flying;
    }
    renderer.render(scene, camera);
  });
}

main();
