import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// --- field tracer ---------------------------------------------------------
// One instanced box per figure, placed straight from the pipeline's (x, y).
// No terrain, no text, no tiling. The point is to see the field: the landing
// pad at the centre, how density falls off into the past, the empty quadrant,
// and whether the landmark tiers read as beacons. Everything here is throwaway
// scaffolding the real runtime will replace.

const info = document.getElementById("info") as HTMLDivElement;

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

async function main() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xd9c9a8);
  // world radius ~8000u now (linear time); fog tuned so the far frontier hazes
  // out rather than popping at the draw edge.
  scene.fog = new THREE.FogExp2(0xd9c9a8, 0.00016);

  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.5,
    24000,
  );
  camera.position.set(0, 900, 1900);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.maxDistance = 18000;
  controls.maxPolarAngle = Math.PI / 2 - 0.02; // stay above the ground

  // low sun for long shadows-of-mood later; flat lambert for now.
  scene.add(new THREE.HemisphereLight(0xfff1d0, 0x8a7350, 1.1));
  const sun = new THREE.DirectionalLight(0xffe8c0, 1.4);
  sun.position.set(-400, 300, 200);
  scene.add(sun);

  // ground large enough to cover the full disc (radius ~8075 + scatter tail).
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

  // human-scale reference standing on the pad: ~1.7 units tall. The whole point
  // is to read book size and walk scale against a body, so the world coordinate
  // unit gets a felt meaning (R_INNER = 30 -> the pad is ~17 people wide).
  const human = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.3, 1.1, 6, 12),
    new THREE.MeshLambertMaterial({ color: 0x33597f }),
  );
  human.position.set(0, 0.85, 0);
  scene.add(human);

  info.textContent = "loading positions…";
  const field = await loadPositions("positions.bin");
  scene.add(buildField(field));
  info.textContent = `${field.n.toLocaleString()} figures · drag to orbit, scroll to zoom · centre = year 2000`;

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

main();
