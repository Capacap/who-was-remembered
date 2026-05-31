import * as THREE from "three";

// --- terrain ----------------------------------------------------------------
// The world is a near-flat desert and the dunes are its relief, not a texture
// laid over a hill. The present (origin) stays calm and level where the books
// are densest; only a whisper of a central rise gives spawn a faint vantage
// before it eases to the desert floor and a transverse dune field takes over.
//
// This lives in the renderer, not the pipeline. The vertical axis carries no
// data (time and longitude are the horizontal x/z), so terrain is decoration by
// the three-tier rule, and the shape is a closed-form radial curve plus local
// flattening at the teleporter plazas, not a hand-sculpted raster. getGroundHeight
// is the single source of truth: the ground mesh, book seating, teleporter bases
// and the player's walk height all sample it, so they agree by construction.

const PEAK_HEIGHT = 60; // a whisper of a central rise, not a summit, world units
const PLATEAU_R = 700; // calm and level here (spawn + plaza + the year-2000 ring)
const BASE_R = 5200; // the rise has eased to the desert floor (0) by here

// Teleporter plazas: each monument stands on a level disc carved into the slope
// at its own elevation, so the pillar sits plumb and the future stone-circle
// visual has flat ground to sit on. FLATTEN_R is the level core; it eases back
// to the hill over FLATTEN_FALLOFF. The falloff is wide enough to span several
// mesh facets, so the transition reads as a deliberate faceted pan rather than a
// single vertex yanked up into a spike.
const FLATTEN_R = 14;
const FLATTEN_FALLOFF = 50;

// Dune field. The terrain is essentially flat desert, so the dunes ARE the
// relief, not an undulation on a hill. A prevailing wind packs them into
// transverse ridges: long crests running crosswind, closely spaced along the
// wind. Sampling the noise anisotropically (short along-wind, long crosswind)
// elongates the ridges, a domain warp lets them meander off straight, and a
// ridged fold sharpens the crest into a crease rather than a soft bump. They
// fade in past the calm present plateau and run at full height across the rest.
//
// Slip-face asymmetry (gentle windward, steep lee) is deferred on purpose:
// clamping a lee slope to the sand's angle of repose wants neighbour info, which
// is natural on a baked raster and awkward pointwise, so it is the first thing
// that will earn the heightmap bake.
const DUNE_AMP = 70; // crest height above the trough, world units
const DUNE_SPACE = 260; // along-wind dune spacing (close)
const DUNE_LEN = 900; // crosswind ridge length scale (long)
const DUNE_OCTAVES = 3;
const WARP_AMP = 120; // how far the crest lines meander off straight
const WARP_SCALE = 1100; // wavelength of that meander
const WIND_ANGLE = 0.7; // prevailing wind bearing, radians
const WIND_X = Math.cos(WIND_ANGLE);
const WIND_Z = Math.sin(WIND_ANGLE);
const NOISE_INNER = PLATEAU_R; // dunes start past the calm present plateau
const NOISE_FULL = 1400; // ... at full height by here

let plazas: { x: number; z: number; h: number }[] = [];

// smootherstep: zero first AND second derivative at both ends, so the grade
// eases in and out without a curvature kink that would show up in the normals.
function smootherstep(t: number): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// --- gradient (Perlin) noise -----------------------------------------------
// Self-contained 2D Perlin so the terrain ships no dependency and stays
// deterministic across reloads. Gradient (not value) noise to avoid the blocky
// axis-aligned look value noise gives on broad dunes.
const PERM = (() => {
  const p = new Uint8Array(512);
  const order = Array.from({ length: 256 }, (_, i) => i);
  let s = 0x9e3779b1 >>> 0; // fixed seed -> stable dunes
  for (let i = 255; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const j = s % (i + 1);
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  for (let i = 0; i < 512; i++) p[i] = order[i & 255];
  return p;
})();

const GRAD = [
  [1, 1],
  [-1, 1],
  [1, -1],
  [-1, -1],
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

function perlin(x: number, y: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const X = xi & 255;
  const Y = yi & 255;
  const xf = x - xi;
  const yf = y - yi;
  const dot = (h: number, dx: number, dy: number) => {
    const g = GRAD[PERM[h] & 7];
    return g[0] * dx + g[1] * dy;
  };
  const u = smootherstep(xf);
  const v = smootherstep(yf);
  const aa = PERM[X] + Y;
  const ba = PERM[X + 1] + Y;
  const x1 = (1 - u) * dot(aa, xf, yf) + u * dot(ba, xf - 1, yf);
  const x2 = (1 - u) * dot(aa + 1, xf, yf - 1) + u * dot(ba + 1, xf - 1, yf - 1);
  return (1 - v) * x1 + v * x2; // roughly [-1, 1]
}

// fractal sum (fBm) of a few octaves in UNIT coordinate space (the caller
// pre-scales), normalized back to ~[-1, 1]. Unit space lets the dune sampler
// stretch the coordinates anisotropically before calling in.
function fbmUnit(x: number, y: number): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < DUNE_OCTAVES; o++) {
    sum += amp * perlin(x * freq, y * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

// integer hash -> [0, 1): per-vertex pseudo-random, used to jitter the ground
// grid in-plane so the triangulation is irregular (organic facets, not a
// mechanical diamond lattice once flat-shaded).
function hash2(i: number, j: number): number {
  let h = (Math.imul(i, 374761393) + Math.imul(j, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// Ground colour tells the world's story radially: a bright pale summit at the
// present, fading through desert sand to a dark grey floor in the deep-past
// void. Radius is time, so this is recency literally lighting the map. A
// low-frequency patch noise breaks the gradient so it reads painted, not banded.
const COLOR_PALE = new THREE.Color(0xefe8d2); // bright dry summit (the present)
const COLOR_SAND = new THREE.Color(0xcabb95); // mid desert
const COLOR_GREY = new THREE.Color(0x6e6860); // faded deep-past floor
const PALE_FADE_R = 1800; // pale is concentrated on the summit, gone by here
const GREY_START_R = 3400; // grey begins creeping in beyond the mid field
const GREY_FULL_R = 6400; // fully grey by here, out in the void
const COLOR_WAVELENGTH = 900; // patch-noise scale for the painted wobble

// Distance fade: the ground's opacity falls to zero between these radii from the
// CAMERA (not the origin), so the far landscape literally dissolves into the sky
// dome instead of standing as a lit silhouette. Colour-matching fog can only
// blend a surface toward the haze colour, never past it, so a sunlit dune crest
// always punches through as a bright ridge; going transparent removes the
// surface entirely, so nothing is left to catch the light. Camera-relative means
// the ground underfoot is always solid and only the horizon dissolves, the same
// whether you stand at the centre or out at the rim.
const FADE_START = 3000; // fully opaque within this distance of the camera
const FADE_END = 6500; // fully gone (sky shows through) beyond this

// the dune offset at a point: anisotropic ridged noise, faded in past the
// present plateau. See the dune-field note above for the construction.
function dune(x: number, z: number, r: number): number {
  const env = smootherstep((r - NOISE_INNER) / (NOISE_FULL - NOISE_INNER));
  if (env <= 0) return 0;
  // meander the crest lines so they aren't ruled straight
  const wx = x + WARP_AMP * perlin(x / WARP_SCALE, z / WARP_SCALE);
  const wz = z + WARP_AMP * perlin(x / WARP_SCALE + 41.3, z / WARP_SCALE + 17.9);
  // rotate into wind-aligned axes and sample anisotropically: dunes pack along
  // the wind (short scale), ridges run crosswind (long scale).
  const s = (wx * WIND_X + wz * WIND_Z) / DUNE_SPACE;
  const t = (-wx * WIND_Z + wz * WIND_X) / DUNE_LEN;
  // ridged fold: a crease at the crest. Squaring tightens the crest and pools
  // the sand flat in the troughs.
  const ridge = 1 - Math.abs(fbmUnit(s, t)); // [0, 1], peaked at the crest
  return env * DUNE_AMP * ridge * ridge;
}

// the bare radial hill, before dunes and plaza flattening.
function hill(r: number): number {
  if (r <= PLATEAU_R) return PEAK_HEIGHT;
  if (r >= BASE_R) return 0;
  return PEAK_HEIGHT * (1 - smootherstep((r - PLATEAU_R) / (BASE_R - PLATEAU_R)));
}

// Pin each plaza's level height to the ACTUAL local ground at the monument's
// spot (hill + dune), computed once so getGroundHeight stays a cheap lookup.
// Using the dune-free hill height instead would level the plaza to the baseline
// while the dunes around it sit metres lower, leaving the monument on a mesa.
export function initTerrain(teleporters: { x: number; y: number }[]): void {
  plazas = teleporters.map((t) => {
    const r = Math.hypot(t.x, t.y);
    return { x: t.x, z: t.y, h: hill(r) + dune(t.x, t.y, r) };
  });
}

export function getGroundHeight(x: number, z: number): number {
  const r = Math.hypot(x, z);
  let h = hill(r) + dune(x, z, r);
  // flatten toward each nearby plaza's level height (hill only, dunes suppressed
  // so the plaza reads as deliberately levelled). Plazas are far apart, so at
  // most one is ever in range; a squared-distance reject skips the other 25.
  const reach = FLATTEN_R + FLATTEN_FALLOFF;
  const reach2 = reach * reach;
  for (const p of plazas) {
    const dx = x - p.x;
    const dz = z - p.z;
    const d2 = dx * dx + dz * dz;
    if (d2 > reach2) continue;
    const d = Math.sqrt(d2);
    const blend = 1 - smootherstep((d - FLATTEN_R) / FLATTEN_FALLOFF);
    h += (p.h - h) * blend;
  }
  return h;
}

const NORMAL_EPS = 0.5; // default central-difference step for the normal, world units

// Surface normal via central differences of getGroundHeight, so it stays correct
// through the plaza blend without anyone deriving the gradient by hand. Books
// sample this once at build to lie along the slope; it is never needed per frame.
// eps is the half-span the difference is taken over: pass the book's footprint so
// a large flat book conforms to the slope it actually spans rather than a point.
export function getGroundNormal(
  x: number,
  z: number,
  out = new THREE.Vector3(),
  eps = NORMAL_EPS,
): THREE.Vector3 {
  const hx = getGroundHeight(x + eps, z) - getGroundHeight(x - eps, z);
  const hz = getGroundHeight(x, z + eps) - getGroundHeight(x, z - eps);
  // surface y = h(x, z); gradient is (∂h/∂x, ∂h/∂z); upward normal is
  // (-∂h/∂x, 1, -∂h/∂z), here with the 1/(2·eps) folded into the normalize.
  return out.set(-hx, 2 * eps, -hz).normalize();
}

export function peakHeight(): number {
  return PEAK_HEIGHT;
}

// A displaced disc covering the full world (radius ~7100 + scatter tail), built
// once by sampling getGroundHeight per vertex so it matches everything standing
// on it. While we evaluate the dune shape the segment count is raised (see SEG
// below) so the crests resolve instead of smoothing into bumps; the final
// resolution is the bake's call. Books seated on the true height don't float
// above the mesh because the mesh samples that same height. The ~30u plazas
// span a couple of facets thanks to the wider plaza falloff.
//
// Flat-shaded and vertex-coloured for a low-poly, hand-painted look (Vane): the
// grid is jittered in-plane so facets read as organic triangles, and the
// material derives a per-face normal so each facet catches the sun distinctly.
export function buildGroundMesh(): THREE.Mesh {
  const SIZE = 18000;
  // TEMP: raised from 768 to ~12u quads so the dune crests are actually visible
  // for shape evaluation. This density would smooth away the low-poly facets in
  // the final look; the heightmap bake will set the real resolution.
  const SEG = 1536;
  const quad = SIZE / SEG;
  const JIT = quad * 0.33; // max in-plane displacement, as a fraction of a quad
  const geom = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
  geom.rotateX(-Math.PI / 2); // into the XZ plane, +y up
  const pos = geom.attributes.position;
  const stride = SEG + 1;
  const colors = new Float32Array(pos.count * 3);
  const c = new THREE.Color();
  for (let v = 0; v < pos.count; v++) {
    const col = v % stride;
    const row = (v / stride) | 0;
    let x = pos.getX(v);
    let z = pos.getZ(v);
    // jitter interior vertices only, so the mesh edge stays gap-free. Displace
    // within a DISK (radius JIT), not a square: a square's diagonal reaches
    // ~1.4x further than its sides, so two neighbours could both lunge along it
    // and collapse the edge between them into a sliver, which flat-shading turns
    // into a garbage-normal streak. A disk caps the reach equally in every
    // direction. sqrt() on the radius keeps the points area-uniform, not bunched
    // at the centre.
    if (col > 0 && col < SEG && row > 0 && row < SEG) {
      const rr = JIT * Math.sqrt(hash2(col, row));
      const th = hash2(col + 7919, row + 104729) * Math.PI * 2;
      x += rr * Math.cos(th);
      z += rr * Math.sin(th);
    }
    const y = getGroundHeight(x, z);
    pos.setXYZ(v, x, y, z);
    // radial narrative: pale summit -> sand -> grey void, then a small painted
    // wobble from low-frequency noise so the bands don't read as clean rings.
    const r = Math.hypot(x, z);
    const pale = 1 - smootherstep(r / PALE_FADE_R);
    const grey = smootherstep((r - GREY_START_R) / (GREY_FULL_R - GREY_START_R));
    c.copy(COLOR_SAND).lerp(COLOR_PALE, pale).lerp(COLOR_GREY, grey);
    const tone = perlin(x / COLOR_WAVELENGTH, z / COLOR_WAVELENGTH); // [-1, 1]
    c.offsetHSL(tone * 0.01, tone * 0.03, tone * 0.04);
    colors[v * 3] = c.r;
    colors[v * 3 + 1] = c.g;
    colors[v * 3 + 2] = c.b;
  }
  pos.needsUpdate = true;
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  // flatShading derives the normal per face from position derivatives, so the
  // stale vertex normals from PlaneGeometry are ignored; no computeVertexNormals.
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    flatShading: true,
  });
  applyDistanceFade(mat);
  return new THREE.Mesh(geom, mat);
}

// Inject a camera-distance opacity fade into a material: it goes fully
// transparent between FADE_START and FADE_END from the camera, so the world
// dissolves into the sky dome behind it instead of toward a fog colour. This is
// the "fog that reveals the dome" idea: distance removes the surface rather than
// tinting it, so nothing distant can read brighter or darker than the sky. The
// same fade is shared by the ground and the books so they vanish together.
// mvPosition (view space) is the vertex relative to the camera, so its length is
// the distance we fade over. Works on InstancedMesh too: the instance matrix is
// already folded into mvPosition.
export function applyDistanceFade(mat: THREE.Material): void {
  mat.transparent = true;
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying float vGroundFade;",
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vGroundFade = clamp(
           (length(mvPosition.xyz) - ${FADE_START.toFixed(1)})
             / ${(FADE_END - FADE_START).toFixed(1)}, 0.0, 1.0);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying float vGroundFade;",
      )
      .replace(
        "#include <dithering_fragment>",
        "#include <dithering_fragment>\ngl_FragColor.a *= 1.0 - vGroundFade;",
      );
  };
}
