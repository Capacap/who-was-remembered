import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import Stats from "three/examples/jsm/libs/stats.module.js";
import {
  initTerrain,
  initHeightmap,
  sampleHeight,
  sampleNormal,
  facetHeight,
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

// tier -> beacon colour only. Size is now a separate axis (book_scale, from
// article length; see loadPositions/buildField), so tier no longer touches it:
// a major is a hotter hue, not a bigger book. A long article about an unknown
// figure is a big plain book; a famous stub is a small hot one. (Cross-disc
// legibility of majors is a beacon-VFX problem, not a size one; the field reads
// as scattered books, not a skyline.)
const TIER_COLOR = [0xb89b6e, 0xdcab4c, 0xff5a2c].map((c) => new THREE.Color(c));

// Geo-navigation colour. Ordinary books (the mass you walk through) are hued by
// their canonical longitude: the PRE-jitter angle baked as the lon byte, so a book
// keeps its home region's hue wherever scatter flung it, and the field's overall
// colour tells you which region lies which way ("redder ahead -> heading east").
// Landmark tiers keep their beacon colours; this only repaints tier 0. A per-book
// lightness jitter stops dense clusters merging into a single slab. Eyeball knobs.
const GEO_SAT = 0.35; // hue vividness (low = desert-muted, high = map-key loud)
const GEO_LIGHT = 0.55; // base lightness of an ordinary book
const GEO_LIGHT_VAR = 0.12; // +/- per-book lightness scatter (the anti-merge speckle)
const HUE_OFFSET = 0.0; // rotate the wheel so a chosen region lands on a chosen hue

// Page edges read a fixed cream regardless of the cover's geo hue. The book mesh
// carries a vertex mask (COLOR_0: white cover, black pages) baked to an aPage
// attribute; the material mixes this colour in where aPage == 1 (see buildField's
// shader patch). Authored sRGB; THREE.Color stores it linear, matching the
// linear diffuseColor the patch injects into.
const PAGE_CREAM = new THREE.Color(0xece2cc);

// Per-instance variety to break up the uniform-grid read. Rotation/tilt/footprint
// are decorative (seeded, don't move the book). SCATTER does move it: a render-only
// experiment — if it earns its keep it belongs in stage6's jitter, not here.
const TILT_MAX = 0.05; // random lean off the ground normal, radians; small so a flat book keeps full contact
const FOOT_VAR = 0.2; // +/- fraction per cover axis (0.8..1.2): size/aspect variety, footprint stays small

// Article length reads as THICKNESS, not ground area. The diagnostic settled this:
// the packed centres are spaced for roughly one small-book-width, so any book that
// consumes more ground than BOOK_FOOTPRINT sits on its neighbours (the modern wedge
// is genuinely that dense; radius is era and can't be spread). So every book keeps a
// near-uniform footprint and a long article instead becomes a FATTER book: the
// book_scale byte drives the spine (vertical) extent, which lives on the empty axis
// and never competes for ground. THICK_MIN/MAX map the scale byte to a spine
// multiplier; the field stays legible and a tome still reads bigger than a pamphlet.
const BOOK_FOOTPRINT = 0.7; // uniform ground footprint scale (the clean diagnostic value)
const THICK_MIN = 0.5;      // spine multiplier at the shortest article (thin pamphlet)
const THICK_MAX = 1.6;      // spine multiplier at the longest (thick hardcover, not a loaf)
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

// One shared glTF loader, Draco-equipped. The Blender exporter compresses meshes
// with KHR_draco_mesh_compression, so the loader needs a Draco decoder or
// loadAsync rejects ("No DRACOLoader instance provided"). The decoder is three's
// own copy, self-hosted under public/draco/ rather than a CDN so the piece loads
// offline; the worker pool spins up lazily on the first compressed mesh. With this
// in place an uncompressed export loads fine too, so the export setting is moot.
const gltfLoader = (() => {
  const draco = new DRACOLoader();
  draco.setDecoderPath("draco/");
  const loader = new GLTFLoader();
  loader.setDRACOLoader(draco);
  return loader;
})();

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
  off += n;
  // canonical longitude angle, 0..255 around the disc (stage6 base_angle, pre-
  // jitter). Drives the per-book geo-navigation hue.
  const lon = new Uint8Array(buf, off, n);
  off += n;
  // book_scale quantized over [BOOK_SCALE_MIN, BOOK_SCALE_MAX] (world.json):
  // the per-book size from article length, decoded in buildField.
  const scale = new Uint8Array(buf, off, n);
  return { n, x, y, tier, geo, lon, scale };
}

// Target cover length in world units: the baked book is uniform-scaled so its
// long axis is this at book_scale 1.0, then the per-book size (from article
// length) and the footprint jitter multiply it. Must match stage8's BOOK_LEN,
// which derives each book's packing radius from the same length.
const BOOK_LENGTH = 0.58;

// Bake one authored book node into the orientation, scale, and attribute layout
// the field wants:
//   - lay it flat (Blender models it upright, thin axis = z) so the cover faces
//     +y and the player looks down onto it;
//   - uniform-scale so the cover length is BOOK_LENGTH, then recentre on origin
//     the way BoxGeometry was, so the existing seat/lift math is unchanged;
//   - convert the COLOR_0 mask (white cover, black pages) to a float aPage
//     attribute (1 = page) and drop the raw colour, so vertexColors stays off and
//     the per-instance geo hue is delivered cleanly through instanceColor.
// Every LOD is baked identically (same long-axis length, same recentre), so the
// distance tiers in buildField share an origin and a book doesn't shift on a swap.
function bakeBook(src: THREE.Mesh): THREE.BufferGeometry {
  const g = (src.geometry as THREE.BufferGeometry).clone();
  // mesh z is the thin axis; rotate it up so the broad cover lies in the x/z plane.
  g.rotateX(Math.PI / 2);
  g.computeBoundingBox();
  let bb = g.boundingBox!;
  const k = BOOK_LENGTH / (bb.max.z - bb.min.z);
  g.scale(k, k, k);
  g.computeBoundingBox();
  bb = g.boundingBox!;
  // recentre x/z; leave y centred too so the seat lift behaves like the old box.
  g.translate(
    -(bb.min.x + bb.max.x) / 2,
    -(bb.min.y + bb.max.y) / 2,
    -(bb.min.z + bb.max.z) / 2,
  );
  // COLOR_0 -> aPage. GLTFLoader maps COLOR_0 to attributes.color (vec4, uint16
  // normalized). Pages were painted black, so a low red channel marks a page vertex.
  const color = g.getAttribute("color");
  const aPage = new Float32Array(color.count);
  for (let i = 0; i < color.count; i++) aPage[i] = color.getX(i) < 0.5 ? 1 : 0;
  g.setAttribute("aPage", new THREE.BufferAttribute(aPage, 1));
  g.deleteAttribute("color");
  return g;
}

// Load the named book LODs from book.glb (one fetch, in ladder order) and bake each.
// book.glb carries book_LOD00 (full), book_LOD01 (mid) and book_LOD02 (an authored
// box); the field uses the first two and synthesises its own box proxy, so LOD02
// isn't requested here.
async function loadBookLods(
  url: string,
  nodes: string[],
): Promise<THREE.BufferGeometry[]> {
  const gltf = await gltfLoader.loadAsync(url);
  return nodes.map((node) => {
    const src = gltf.scene.getObjectByName(node) as THREE.Mesh | undefined;
    if (!src?.isMesh) throw new Error(`no '${node}' mesh in ${url}`);
    return bakeBook(src);
  });
}

// Keep pages cream while the cover takes the per-instance geo hue. Chains onto
// applyDistanceFade's onBeforeCompile (call this AFTER it): reads the baked aPage
// mask and overrides diffuseColor on page vertices, after color_fragment has
// already folded instanceColor into the cover.
function applyPageMask(mat: THREE.Material): void {
  const prev = mat.onBeforeCompile;
  const cream = `vec3(${PAGE_CREAM.r}, ${PAGE_CREAM.g}, ${PAGE_CREAM.b})`;
  mat.onBeforeCompile = (shader, renderer) => {
    if (prev) prev.call(mat, shader, renderer);
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nattribute float aPage;\nvarying float vPage;",
      )
      .replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\nvPage = aPage;",
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying float vPage;",
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>\ndiffuseColor.rgb = mix(diffuseColor.rgb, ${cream}, vPage);`,
      );
  };
}

// --- decorative heads --------------------------------------------------------
// Three sculpted head variants from heads.glb, scattered half-buried in the sand
// with faces to the sky (Stage 10 finds the clear spots; decorations.json carries
// per-head position and jitter). Pure decoration, no data; they thicken the
// dream-logic without disturbing a single book. Plain stone material; the basemesh
// ships no material and its COLOR_n/TEXCOORD_n layers are ignored.
const HEAD_HEIGHT = 10; // world units along the sculpted up axis; eyeball knob.
//   Stage 10's HEAD_RADIUS (the open-sand a head needs) tracks ~half of this.
// each variant's three LODs (full, mid, coarse), the same ladder the books use.
const HEAD_VARIANTS = [
  ["head01_LOD00", "head01_LOD01", "head01_LOD02"],
  ["head02_LOD00", "head02_LOD01", "head02_LOD02"],
  ["head03_LOD00", "head03_LOD01", "head03_LOD02"],
];
// camera-distance thresholds for the LOD swap and a hysteresis band (fraction of
// the threshold) so a head straddling a boundary doesn't flicker. Far larger than
// the books' because a head is far bigger on screen: full out to HEAD_LOD[1], mid
// to HEAD_LOD[2], coarse beyond. Off-screen heads frustum-cull, so the hundreds
// scattered across the deep desert only draw when actually in view. Eyeball knobs.
const HEAD_LOD = [0, 150, 400];
const HEAD_LOD_HYST = 0.1;

// Each LOD uniform-scaled to HEAD_HEIGHT on the sculpted up axis (y) and recentred
// on the origin in all three axes, so a head can be freely laid on its back and
// sunk into the sand by a per-instance transform without the LOD ladder shifting.
function normalizeHead(src: THREE.Mesh): THREE.BufferGeometry {
  const g = (src.geometry as THREE.BufferGeometry).clone();
  g.computeBoundingBox();
  let bb = g.boundingBox!;
  const s = HEAD_HEIGHT / (bb.max.y - bb.min.y);
  g.scale(s, s, s);
  g.computeBoundingBox();
  bb = g.boundingBox!;
  g.translate(
    -(bb.min.x + bb.max.x) / 2,
    -(bb.min.y + bb.max.y) / 2,
    -(bb.min.z + bb.max.z) / 2,
  );
  return g;
}

// returns, per variant, its three baked LOD geometries.
async function loadHeadLods(url: string): Promise<THREE.BufferGeometry[][]> {
  const gltf = await gltfLoader.loadAsync(url);
  return HEAD_VARIANTS.map((lods) =>
    lods.map((name) => {
      const src = gltf.scene.getObjectByName(name) as THREE.Mesh | undefined;
      if (!src?.isMesh) throw new Error(`no '${name}' mesh in ${url}`);
      return normalizeHead(src);
    }),
  );
}

// One head per Stage 10 decoration: laid on its back facing the sky, sunk into
// the sand by its sink fraction, spun by its yaw, sized by its scale. Each is a
// THREE.LOD swapping mesh by camera distance (update() driven each frame); opaque
// and unfaded like the teleporter beacons. Three variants jittered into hundreds.
interface Decoration {
  x: number; // pipeline ground x (-> world x)
  y: number; // pipeline ground y (-> world z)
  v: number; // variant index 0..2
  s: number; // scale multiplier
  rot: number; // yaw, radians
  sink: number; // fraction of the laid head buried below grade
}

async function loadDecorations(url: string): Promise<Decoration[]> {
  return (await fetch(url)).json();
}

function buildHeads(
  variants: THREE.BufferGeometry[][],
  decos: Decoration[],
): {
  group: THREE.Group;
  update: (camera: THREE.Camera) => void;
} {
  const mat = new THREE.MeshLambertMaterial({ color: 0xcbbfa8 }); // sandstone
  const group = new THREE.Group();
  const lods: THREE.LOD[] = [];
  // world-vertical extent of each variant once laid face-up: the head's local
  // z (face depth, which the -90° X rotation swings onto world y) at scale 1.
  // Used to bury the head by its sink fraction.
  const depth = variants.map((geos) => {
    geos[0].computeBoundingBox();
    const b = geos[0].boundingBox!;
    return b.max.z - b.min.z;
  });
  for (const d of decos) {
    const geos = variants[d.v] ?? variants[0];
    const lod = new THREE.LOD();
    geos.forEach((g, lvl) =>
      lod.addLevel(new THREE.Mesh(g, mat), HEAD_LOD[lvl], HEAD_LOD_HYST),
    );
    // lay the head on its back, face to the sky, then spin it about up: YXZ order
    // applies Ry(rot) · Rx(+90). (The sculpted face axis ran the opposite way from
    // the first guess, so the pitch is +90, not -90.)
    lod.rotation.set(Math.PI / 2, d.rot, 0, "YXZ");
    lod.scale.setScalar(d.s);
    // bury `sink` of the laid head: its centre sits at ground + H*(0.5 - sink),
    // so exactly that fraction of the world-vertical extent H is below grade.
    const H = depth[d.v] * d.s;
    const gy = sampleHeight(d.x, d.y) + H * (0.5 - d.sink);
    lod.position.set(d.x, gy, d.y);
    group.add(lod);
    lods.push(lod);
  }
  return {
    group,
    update: (camera) => {
      for (const l of lods) l.update(camera);
    },
  };
}

function buildField(
  field: Awaited<ReturnType<typeof loadPositions>>,
  bookNear: THREE.BufferGeometry, // LOD00, full detail, drawn closest
  bookMid: THREE.BufferGeometry, // LOD01, drawn across the mid band
) {
  const { n, x, y, tier, geo, lon, scale } = field;
  // the scale byte is the normalized article length (export_runtime quantized the
  // [BOOK_SCALE_MIN, BOOK_SCALE_MAX] fraction straight to 0..255), so scale[i]/255
  // is already 0..1; the renderer maps it onto spine THICKNESS, not footprint.
  // the authored book mesh, laid flat: broad cover (x, z), slim spine (y, the up
  // axis). Baked in loadBookLods to ~0.58u long at tier 1, smaller than the modern
  // spacing (~0.9u) so neighbours read as distinct dropped objects. Both LODs share
  // the same baked size/origin, so the box dims (and the seat lift) come from the
  // full mesh and the tiers agree. The spine is the up extent, so seating offsets by
  // half of the mesh's own thickness along the normal.
  bookNear.computeBoundingBox();
  const bb = bookNear.boundingBox!;
  const SPINE = bb.max.y - bb.min.y;

  // Three-tier distance LOD. A 230-tri book instanced 574k times and drawn mostly
  // sub-pixel is hopeless; a book is only a legible shape within tens of units and
  // only READABLE within 6u (the picker's reach), so detail past that is wasted.
  // Each book is drawn by exactly ONE detail tier over an always-present box base:
  // full LOD00 within R_FULL, LOD01 out to R_MID, the bare box beyond. The box is
  // shrunk to PROXY so it hides INSIDE whichever detailed mesh covers it (no z-fight);
  // the two detailed tiers never cover the same book, so they can't fight each other.
  // Splitting the old single detailed tier in two keeps each swap small on screen:
  // full->mid lands where a book is already a few px, mid->box smaller still. A
  // per-book dither on the inner boundary scatters that swap so it isn't a clean ring
  // sweeping the field as the camera moves.
  const PROXY = 0.85;
  const boxGeo = new THREE.BoxGeometry(
    (bb.max.x - bb.min.x) * PROXY,
    SPINE * PROXY,
    (bb.max.z - bb.min.z) * PROXY,
  );
  const R_FULL = 30; // LOD00 within this radius (a book is still >~8px here)
  const R_MID = 120; // LOD01 out to here; books are box-indistinguishable past it
  const BOUND_DITHER = 12; // per-book spread (world units) on the full->mid boundary
  const NEAR_CAP = 6000; // LOD00 instances; the R_FULL disc holds far fewer than this
  const MID_CAP = 24000; // LOD01 instances; overflow in the dense band drops the
  //   farthest-in-band to box (a shorter, still sub-pixel mid radius there), the same
  //   graceful degradation the single tier had.
  const REBUILD_DIST = 25; // refill the near/mid sets only after the camera moves this far

  // books dissolve with the ground: the same camera-distance fade to transparent,
  // so the field thins into the dome at the horizon rather than leaving sharp specks
  // floating over ground that has already faded out. The page mask chains on after
  // the fade so cover vertices keep the geo hue and page edges stay cream; both
  // detailed tiers carry it (both have aPage), the box has no pages so it only fades.
  const farMat = new THREE.MeshLambertMaterial();
  applyDistanceFade(farMat);
  const midMat = new THREE.MeshLambertMaterial();
  applyDistanceFade(midMat);
  applyPageMask(midMat);
  const nearMat = new THREE.MeshLambertMaterial();
  applyDistanceFade(nearMat);
  applyPageMask(nearMat);

  const farMesh = new THREE.InstancedMesh(boxGeo, farMat, n);
  farMesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
  farMesh.frustumCulled = false; // spans the whole disc; never wholly off-screen
  const midMesh = new THREE.InstancedMesh(bookMid, midMat, MID_CAP);
  midMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  midMesh.frustumCulled = false; // rebuilt around the camera, bounds don't apply
  midMesh.count = 0;
  const nearMesh = new THREE.InstancedMesh(bookNear, nearMat, NEAR_CAP);
  nearMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  nearMesh.frustumCulled = false;
  nearMesh.count = 0;

  // keep the rendered (jittered) ground positions so the look-at picker aims at
  // where a book actually stands, not its pre-scatter pipeline coordinate.
  const px = new Float32Array(n);
  const pz = new Float32Array(n);
  // every book's full transform + colour, kept so the near mesh can be refilled
  // from the global set as the camera moves (the far box mesh is written once).
  const fullMat = new Float32Array(n * 16);
  const fullCol = new Float32Array(n * 3);

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
  // separate stream for the lightness jitter so adding it doesn't perturb the
  // shape/tilt variety the main rnd stream already drives.
  const rndL = mulberry32(0x9e3779b9);
  for (let i = 0; i < n; i++) {
    // footprint is near-uniform (ground area is the contested axis); article length
    // becomes spine thickness instead, on the empty vertical axis.
    const thick = THICK_MIN + (scale[i] / 255) * (THICK_MAX - THICK_MIN);
    const fw = 1 + (rnd() * 2 - 1) * FOOT_VAR;
    const fl = 1 + (rnd() * 2 - 1) * FOOT_VAR;
    // pipeline (x, y) is the ground plane; map to world (x, z), y is up.
    px[i] = x[i] + (rnd() * 2 - 1) * SCATTER;
    pz[i] = y[i] + (rnd() * 2 - 1) * SCATTER;
    // seat on the facet the clipmap actually draws underfoot, not the smooth
    // sampleHeight field it only chords: on a convex crest the facet sits below the
    // field, so a sampleHeight seat would float. The tilt still follows the smooth
    // normal (the float is a height problem; the facet's own normal would only add
    // per-book tilt jumps for no gain on a book this small).
    const gy = facetHeight(px[i], pz[i]);
    // sample the normal across the book's own footprint (half-length ~0.3·s) so a
    // large book conforms to the slope it spans instead of one 0.5u patch.
    sampleNormal(px[i], pz[i], normal, 0.3 * BOOK_FOOTPRINT);
    // lay the book flat on the slope: its spine (+y) aligns to the ground normal,
    // a random spin about that axis gives it a dropped heading, and a small lean
    // off the normal keeps it from looking neatly placed.
    qAlign.setFromUnitVectors(UP, normal);
    // tilt (x, z) and a random yaw (y) so no two books share a heading.
    const t1 = (rnd() * 2 - 1) * TILT_MAX;
    const yaw = rnd() * Math.PI * 2;
    const t2 = (rnd() * 2 - 1) * TILT_MAX;
    eul.set(t1, yaw, t2);
    qLocal.setFromEuler(eul);
    dummy.quaternion.copy(qAlign).multiply(qLocal);
    // settle the book INTO the sand: lift the centre by less than half the spine,
    // so the underside sits a touch below grade and any leaning corner rests in
    // the surface rather than hovering over it. A thicker (longer-article) book has
    // a taller spine, so it stands proportionally higher out of the sand.
    const lift = (SPINE * thick) * 0.3;
    dummy.position.set(
      px[i] + normal.x * lift,
      gy + normal.y * lift,
      pz[i] + normal.z * lift,
    );
    // x,z = uniform footprint (+ small jitter); y = spine thickness from article length.
    dummy.scale.set(BOOK_FOOTPRINT * fw, thick, BOOK_FOOTPRINT * fl);
    dummy.updateMatrix();
    farMesh.setMatrixAt(i, dummy.matrix);
    dummy.matrix.toArray(fullMat, i * 16);
    // ordinary books carry the geo-navigation hue (+ lightness speckle); minor
    // and major keep their beacon colours. Residue is washed toward adrift on top
    // of either, so a place-less book never reads as confidently regional.
    if (tier[i] === 0) {
      const hue = ((lon[i] / 256) + HUE_OFFSET) % 1;
      const lj = (rndL() * 2 - 1) * GEO_LIGHT_VAR;
      col.setHSL(hue, GEO_SAT, GEO_LIGHT + lj);
    } else {
      col.copy(TIER_COLOR[tier[i]]);
    }
    if (geo[i] === 4) col.lerp(ADRIFT, 0.75);
    farMesh.setColorAt(i, col);
    col.toArray(fullCol, i * 3);
  }
  farMesh.instanceMatrix.needsUpdate = true;
  if (farMesh.instanceColor) farMesh.instanceColor.needsUpdate = true;

  // Spatial grid over the book positions, built once. The near-set refill walks it
  // outward from the camera cell and fills nearest-cell-first up to NEAR_CAP, so a
  // refill costs ~O(NEAR_CAP) instead of scanning all n and sorting. That matters
  // because the refill fires whenever the camera moves REBUILD_DIST, which at flight
  // speed is every frame: the brute scan + sort held that to ~30fps; this doesn't.
  const CELL = 32; // grid cell size, world units; fine enough that ring order ≈ distance
  let minX = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    if (px[i] < minX) minX = px[i];
    if (px[i] > maxX) maxX = px[i];
    if (pz[i] < minZ) minZ = pz[i];
    if (pz[i] > maxZ) maxZ = pz[i];
  }
  const gw = Math.floor((maxX - minX) / CELL) + 1;
  const gh = Math.floor((maxZ - minZ) / CELL) + 1;
  const cellOf = (i: number): number =>
    Math.floor((pz[i] - minZ) / CELL) * gw + Math.floor((px[i] - minX) / CELL);
  // CSR layout: cellStart[c]..cellStart[c+1] indexes a run of book ids in cellItems.
  const cellStart = new Int32Array(gw * gh + 1);
  for (let i = 0; i < n; i++) cellStart[cellOf(i) + 1]++;
  for (let cI = 0; cI < gw * gh; cI++) cellStart[cI + 1] += cellStart[cI];
  const cellItems = new Int32Array(n);
  const cursor = cellStart.slice(0, gw * gh);
  for (let i = 0; i < n; i++) cellItems[cursor[cellOf(i)]++] = i;

  const RMID2 = R_MID * R_MID;
  const RB2 = REBUILD_DIST * REBUILD_DIST;
  const maxRing = Math.ceil(R_MID / CELL) + 1; // cells beyond this are wholly out of range
  const m = new THREE.Matrix4();
  const c = new THREE.Color();
  let lastX = Infinity;
  let lastZ = Infinity;
  function update(camX: number, camZ: number): void {
    const mdx = camX - lastX;
    const mdz = camZ - lastZ;
    if (mdx * mdx + mdz * mdz < RB2) return;
    lastX = camX;
    lastZ = camZ;
    const cgx = Math.floor((camX - minX) / CELL);
    const cgz = Math.floor((camZ - minZ) / CELL);
    let kNear = 0;
    let kMid = 0;
    const addCell = (gx: number, gz: number): void => {
      if (gx < 0 || gz < 0 || gx >= gw || gz >= gh) return;
      const cI = gz * gw + gx;
      const end = cellStart[cI + 1];
      for (let p = cellStart[cI]; p < end; p++) {
        const i = cellItems[p];
        const dx = px[i] - camX;
        const dz = pz[i] - camZ;
        const d2 = dx * dx + dz * dz;
        if (d2 > RMID2) continue;
        // per-book dither on the full->mid radius so the swap is a fuzzy band of
        // individual books rather than a clean ring sweeping the field. Stable hash
        // of the book index, so a given book's boundary doesn't change frame to frame.
        const h = (Math.imul(i, 2654435761) >>> 0) / 4294967296; // [0, 1)
        const rf = R_FULL + (h - 0.5) * BOUND_DITHER;
        if (d2 < rf * rf && kNear < NEAR_CAP) {
          nearMesh.setMatrixAt(kNear, m.fromArray(fullMat, i * 16));
          nearMesh.setColorAt(kNear, c.fromArray(fullCol, i * 3));
          kNear++;
        } else if (kMid < MID_CAP) {
          // mid band, or a near-band book that overflowed NEAR_CAP (still gets detail)
          midMesh.setMatrixAt(kMid, m.fromArray(fullMat, i * 16));
          midMesh.setColorAt(kMid, c.fromArray(fullCol, i * 3));
          kMid++;
        }
        // else: both detail caps full, the always-drawn box base covers this book
      }
    };
    // expand in Chebyshev rings from the camera cell: nearest cells first, so the
    // near tier (innermost) fills before the mid, and when a cap is hit mid-walk it's
    // the farthest books in that tier that drop to the coarser one.
    const bothFull = () => kNear >= NEAR_CAP && kMid >= MID_CAP;
    for (let rr = 0; rr <= maxRing && !bothFull(); rr++) {
      if (rr === 0) {
        addCell(cgx, cgz);
        continue;
      }
      for (let gx = cgx - rr; gx <= cgx + rr; gx++) {
        addCell(gx, cgz - rr);
        addCell(gx, cgz + rr);
      }
      for (let gz = cgz - rr + 1; gz <= cgz + rr - 1; gz++) {
        addCell(cgx - rr, gz);
        addCell(cgx + rr, gz);
      }
    }
    nearMesh.count = kNear;
    nearMesh.instanceMatrix.needsUpdate = true;
    if (nearMesh.instanceColor) nearMesh.instanceColor.needsUpdate = true;
    midMesh.count = kMid;
    midMesh.instanceMatrix.needsUpdate = true;
    if (midMesh.instanceColor) midMesh.instanceColor.needsUpdate = true;
  }

  const group = new THREE.Group();
  group.add(farMesh);
  group.add(midMesh);
  group.add(nearMesh);
  return { group, px, pz, tier: tier as Uint8Array, update, nearMesh, midMesh };
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
  BOOK_SCALE_MIN: number;
  BOOK_SCALE_MAX: number;
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

  // perf monitor: stats.js panel (click to cycle FPS / ms / MB) plus a text
  // readout of the numbers that actually tell us if the book LOD is working,
  // draw calls, triangles, and the live detailed-book count vs the box field.
  const stats = new Stats();
  stats.dom.style.cssText = "position:fixed;top:0;left:0;z-index:100;";
  document.body.appendChild(stats.dom);
  const perf = document.createElement("div");
  perf.style.cssText =
    "position:fixed;top:48px;left:0;z-index:100;padding:4px 6px;" +
    "font:11px/1.4 monospace;color:#9fe;background:rgba(0,0,0,.55);white-space:pre;";
  document.body.appendChild(perf);

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

  // --- load profiling -------------------------------------------------------
  // Phase wall-clock so load cost is measured, not guessed (console: filter
  // "[load]"). "seat books" is the one to watch: it rebuilds a clipmap facet per
  // figure. performance.now() is ms since the page opened, so the final total reads
  // against navigation start and is comparable to the LCP you see in dev tools.
  const t0 = performance.now();
  let tMark = t0;
  const mark = (label: string) => {
    const now = performance.now();
    console.log(`[load] ${label.padEnd(13)}${(now - tMark).toFixed(0)}ms`);
    tMark = now;
  };

  info.innerHTML = "loading positions…";
  const [field, teleporters, meta, world, heightmap, bookLods, headVariants, decorations] =
    await Promise.all([
      loadPositions("positions.bin"),
      loadTeleporters("teleporters.json"),
      loadMeta("meta.bin"),
      loadWorld("world.json"),
      loadHeightmap("heightmap.bin"),
      loadBookLods("book.glb", ["book_LOD00", "book_LOD01"]),
      loadHeadLods("heads.glb"),
      loadDecorations("decorations.json"),
    ]);
  mark("fetch+decode");
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
  mark("terrain");
  const built = buildField(field, bookLods[0], bookLods[1]);
  mark("seat books");
  built.update(camera.position.x, camera.position.z);
  scene.add(built.group);
  scene.add(buildTeleporters(teleporters));
  const heads = buildHeads(headVariants, decorations); // half-buried scatter (Stage 10)
  scene.add(heads.group);

  mark("props");

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

  mark("wiring");
  console.log(`[load] total ${(performance.now() - t0).toFixed(0)}ms to first frame`);

  const clock = new THREE.Clock();
  let wasFlying = false;
  let sincePick = 0;
  let sinceStat = 0;
  // worst-case ms for the two camera-driven rebuilds, reset each readout window,
  // so a bursty re-tessellation spike shows up instead of being averaged away.
  let groundMs = 0;
  let booksMs = 0;
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
    let tA = performance.now();
    ground.update(camera.position.x, camera.position.z);
    const gm = performance.now() - tA;
    if (gm > groundMs) groundMs = gm;
    tA = performance.now();
    built.update(camera.position.x, camera.position.z);
    const bm = performance.now() - tA;
    if (bm > booksMs) booksMs = bm;
    heads.update(camera); // pick each head's LOD by camera distance (a handful)

    compass.update();
    sky.position.copy(camera.position); // keep the dome centred on the viewer
    renderer.render(scene, camera);

    // read renderer.info AFTER render (it resets per frame), throttled to ~4 Hz so
    // the DOM write doesn't itself cost frames. The near count is the LOD's pulse:
    // it should sit in the low thousands and clamp at NEAR_CAP in the dense band.
    stats.update();
    sinceStat += dt;
    if (sinceStat >= 0.25) {
      sinceStat = 0;
      const r = renderer.info.render;
      perf.textContent =
        `calls  ${r.calls}\n` +
        `tris   ${(r.triangles / 1e6).toFixed(2)}M\n` +
        `books  ${built.nearMesh.count.toLocaleString()} full + ${built.midMesh.count.toLocaleString()} mid / ${field.n.toLocaleString()} box\n` +
        `ground ${groundMs.toFixed(1)}ms (peak)\n` +
        `bookfl ${booksMs.toFixed(1)}ms (peak)`;
      groundMs = 0;
      booksMs = 0;
    }
  });
}

// surface a load failure in the HUD instead of freezing on "loading positions…":
// every asset is awaited in one Promise.all, so any rejection (a missing file, a
// glb the loader can't decode) leaves the page stuck with no visible reason.
main().catch((e) => {
  info.innerHTML = `load failed: ${escapeHtml(String(e?.message ?? e))}`;
  console.error(e);
});
