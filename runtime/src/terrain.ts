import * as THREE from "three";

// --- terrain ----------------------------------------------------------------
// The world is a near-flat desert and the dunes are its relief, not a texture
// laid over a hill. The present (origin) stays calm and level where the books
// are densest; only a whisper of a central rise gives spawn a faint vantage
// before it eases to the desert floor and a transverse dune field takes over.
//
// This lives in the renderer, not the pipeline. The vertical axis carries no data
// (time and longitude are the horizontal x/z), so terrain is decoration by the
// three-tier rule. The shape is a baked heightmap raster (stage9), and sampleHeight
// is the single elevation source: the clipmap tessellates it, the player's feet
// read it, and books seat on facetHeight (the facet the clipmap actually draws), so
// the visible ground and everything on it agree. The analytic getGroundHeight below
// is now only the pre-heightmap-load fallback.

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

// Crest/trough relief tint, layered on the radial base: crests read scoured pale
// and a touch warm, troughs cooler and darker, so the dunes carry colour and not
// just shading. The signal is sampled from each clipmap level's OWN grid (the
// neighbour offset scales with the cell), so the colour tracks the relief that
// level actually draws and degrades with it instead of aliasing into noise at the
// coarse LODs. All four are eyeball knobs.
const RELIEF_CELLS = 3; // neighbour offset in cells: the relief's read wavelength
const RELIEF_SCALE = 0.25; // slope-difference that reaches the full crest/trough tint
const CREST_LIGHT = 0.1; // crest lightens / trough darkens (the dominant read)
const CREST_SAT = 0.05; // crest bleaches / trough deepens
const CREST_HUE = 0.012; // crest warms / trough cools

// Distance fade: every clipmap level's opacity falls to zero between these radii
// from the CAMERA, so the whole landscape dissolves into the sky dome before it
// reaches any footprint edge. The fade is circular (camera distance), so unlike a
// fog tint of opaque geometry it leaves no square plate to see from a height.
// Colour-matching fog can only blend a surface toward the haze colour, never past
// it, so a sunlit dune crest punches through as a bright ridge; going transparent
// removes the surface entirely, so nothing is left to catch the light or to show
// an edge. The catch the clipmap adds is that five camera-centred transparent
// meshes have no stable depth-sort, so their overlap rings double-blend into a
// flickering band; buildGroundLevel fixes that with an explicit renderOrder
// (finest first) so each overlap is won by the finer level and blended once.
export const FADE_START = 3000; // fully opaque within this distance of the camera
export const FADE_END = 6500; // fully gone (dome shows through) beyond this

// --- camera-following ground LOD (geometry clipmap) -------------------------
// The ground is built entirely from the shipped heightmap as concentric square
// levels centred on the camera: a fine block underfoot, each level outward at
// double the cell size, so on-screen facet size stays roughly bounded however the
// camera roams. A single static mesh can't do that (a global tri budget is either
// coarse underfoot or ruinous to ship, and the decimated disc was coarse exactly
// where you stand). Every level shares one material, colour and jitter, so there
// is no near/far seam in look, only a graduated step in facet size. Each level
// snaps to its own cell so its facets never swim, and the coarser levels discard
// a square hole under the finer level inside them, so the levels never fight in
// the depth buffer.
//
// The level boundary itself is handled by GEOMORPHING (Losasso & Hoppe). In a band
// at each level's outer rim the fine vertices morph (height, and jitter relaxing to
// zero) toward what the coarser level draws there, so by the boundary the fine
// surface IS the coarse surface, vertex for vertex: no height step, no crack, no
// pop on a cell-cross. For that to be exact the coarse side must agree, so each
// level also relaxes its jitter to zero in a flat band around its inner hole edge;
// both sides then meet on a plain shared lattice. The morph target reproduces the
// coarser level's exact triangulated chord (same diagonal split), so the match is
// C0, not approximate. The discard hole leaves a one-ring overlap where the fine
// surface is identical to the coarse chord; a per-level depth bias (polygonOffset,
// coarser = pushed back) makes the finer level deterministically win that overlap,
// so no skirt is needed and there is no coplanar flicker at the seam.
interface GroundLevel {
  cell: number; // quad size, world units
  half: number; // half-extent of the square block (every level is 128 cells across)
  hole: number; // half-width of the central square discarded for the finer level
  holeCell: number; // the finer level's cell: the hole snaps to THIS so its edge
  //                   stays locked to the finer level's snapped real-surface edge
}
const GROUND_LEVELS: GroundLevel[] = [
  { cell: 8, half: 512, hole: 0, holeCell: 0 }, // underfoot; finest, no hole
  { cell: 16, half: 1024, hole: 504, holeCell: 8 }, // hole = finer.half-finer.cell,
  { cell: 32, half: 2048, hole: 1008, holeCell: 16 }, //  so the coarser level picks
  { cell: 64, half: 4096, hole: 2016, holeCell: 32 }, //  up exactly where the finer
  { cell: 128, half: 8192, hole: 4032, holeCell: 64 }, // level's real surface ends
];
const GROUND_JIT_FRAC = 0.33; // in-plane vertex jitter as a fraction of the cell:
//   the Vane look, applied at each level's own scale so the character is uniform
const GROUND_MORPH_CELLS = 6; // rim morph-band width in cells: over this band the
//   fine surface lerps to the coarse one, reaching an exact match at the boundary
const GROUND_RELAX_FLAT = 2; // inner rings (in cells) held fully un-jittered around
//   the hole, so the finer level's morph target lands on a plain coarse lattice
const GROUND_RELAX_CELLS = 4; // width (cells) of the ramp from the flat hole band
//   back to full jitter, so the relaxation isn't a hard line
const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);

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

// Surface normal via central differences of getGroundHeight, the analytic
// counterpart of sampleNormal. Now unused (props read sampleNormal off the baked
// field); kept beside getGroundHeight as its fallback companion until the analytic
// path is retired. eps is the half-span the difference is taken over.
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

// Ground colour at a world point: the radial pale-summit -> sand -> grey-void
// narrative, a low-frequency painted wobble, and a crest/trough relief tint. The
// clipmap tints its vertices with this from world position on each rebuild; it
// stays per-vertex on the CPU by design (no texture, no baked raster), so the
// low-poly vertex-colour look survives across the LOD levels. `relief` is the
// signed, already-normalised local relief (+ on crests, - in hollows) the caller
// reads from that level's grid; 0 leaves the base untouched.
export function groundColor(
  x: number,
  z: number,
  out: THREE.Color,
  relief = 0,
): THREE.Color {
  const r = Math.hypot(x, z);
  const pale = 1 - smootherstep(r / PALE_FADE_R);
  const grey = smootherstep((r - GREY_START_R) / (GREY_FULL_R - GREY_START_R));
  out.copy(COLOR_SAND).lerp(COLOR_PALE, pale).lerp(COLOR_GREY, grey);
  const tone = perlin(x / COLOR_WAVELENGTH, z / COLOR_WAVELENGTH); // [-1, 1]
  out.offsetHSL(tone * 0.01, tone * 0.03, tone * 0.04);
  // crests (k > 0) warm, bleach and lighten; troughs (k < 0) cool, deepen and
  // darken. Hue/sat shift against k's sign, lightness with it.
  const k = relief < -1 ? -1 : relief > 1 ? 1 : relief;
  out.offsetHSL(-k * CREST_HUE, -k * CREST_SAT, k * CREST_LIGHT);
  return out;
}

// --- baked heightmap sampler -------------------------------------------------
// The shipped heightmap.bin (stage9_mesh.write_heightmap_bin) is the world's one
// elevation source: the near patch tessellates it and the player's feet read it,
// so the visible near ground and the walk height agree by construction (the
// seating-consistency dividend the bake was always after). Bilinear, edge-clamped,
// matching the pipeline's `_bilinear` and pixel-centre mapping exactly so the
// runtime surface is the same field stage9 baked, not a lookalike.
let HM: {
  data: Float32Array;
  res: number;
  worldSize: number;
  texel: number;
  half: number;
} | null = null;

export function initHeightmap(
  res: number,
  worldSize: number,
  data: Float32Array,
): void {
  HM = { data, res, worldSize, texel: worldSize / res, half: worldSize / 2 };
}

// Height at a world point from the baked raster. Falls back to the analytic
// getGroundHeight before the heightmap has loaded (e.g. the spawn placement,
// which sits on the flat plateau where the two agree anyway).
export function sampleHeight(x: number, z: number): number {
  if (!HM) return getGroundHeight(x, z);
  const { data, res, texel, half } = HM;
  const col = (x + half) / texel - 0.5;
  const row = (z + half) / texel - 0.5;
  const c0 = Math.floor(col);
  const r0 = Math.floor(row);
  const fc = col - c0;
  const fr = row - r0;
  const last = res - 1;
  const clamp01 = (i: number) => (i < 0 ? 0 : i > last ? last : i);
  const c0c = clamp01(c0);
  const c1c = clamp01(c0 + 1);
  const r0c = clamp01(r0) * res;
  const r1c = clamp01(r0 + 1) * res;
  const f00 = data[r0c + c0c];
  const f01 = data[r0c + c1c];
  const f10 = data[r1c + c0c];
  const f11 = data[r1c + c1c];
  return (
    f00 * (1 - fr) * (1 - fc) +
    f01 * (1 - fr) * fc +
    f10 * fr * (1 - fc) +
    f11 * fr * fc
  );
}

// Surface normal from the baked raster, the heightmap counterpart of
// getGroundNormal: central differences on sampleHeight so props seat to the same
// surface the clipmap draws, not the analytic field that diverges from it. eps
// doubles as the footprint half-width, so a large prop conforms to the slope it
// spans rather than one texel.
export function sampleNormal(
  x: number,
  z: number,
  out = new THREE.Vector3(),
  eps = NORMAL_EPS,
): THREE.Vector3 {
  const hx = sampleHeight(x + eps, z) - sampleHeight(x - eps, z);
  const hz = sampleHeight(x, z + eps) - sampleHeight(x, z - eps);
  return out.set(-hx, 2 * eps, -hz).normalize();
}

// Inject the per-level material edits: the camera-distance opacity fade (every
// level, so the world dissolves circularly into the dome before any footprint
// edge) and a camera-centred square discard hole (when holeHalf > 0) that cuts
// this level away where the finer level inside it sits, so a coarse facet can
// never poke through the fine surface. Every level is transparent; their stable
// ordering is the renderOrder set in buildGroundLevel, not this. Returns the
// hole-centre uniform to be tracked to the camera each frame, or null for the
// finest level (no hole).
function applyGroundMaterial(
  mat: THREE.MeshLambertMaterial,
  holeHalf: number,
): { value: THREE.Vector2 } | null {
  mat.transparent = true;
  const holeCenter = holeHalf > 0 ? { value: new THREE.Vector2(0, 0) } : null;
  const fadeSpan = (FADE_END - FADE_START).toFixed(1);
  // Each level bakes its hole size into the GLSL as a literal, invisible to
  // Three's program-cache key (which sees only onBeforeCompile's source text,
  // identical across levels). Without this, all levels would share whichever
  // program compiled first and render wrong. Keying on holeHalf forces a distinct
  // program per level.
  mat.customProgramCacheKey = () => `ground:${holeHalf}`;
  mat.onBeforeCompile = (shader) => {
    if (holeCenter) shader.uniforms.uHoleCenter = holeCenter;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying float vGroundFade;" +
          (holeCenter ? "\nvarying vec2 vGroundXZ;" : ""),
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vGroundFade = clamp(
           (length(mvPosition.xyz) - ${FADE_START.toFixed(1)}) / ${fadeSpan},
           0.0, 1.0);` +
          (holeCenter
            ? `\n         vGroundXZ = (modelMatrix * vec4(transformed, 1.0)).xz;`
            : ""),
      );
    let frag = shader.fragmentShader.replace(
      "#include <common>",
      "#include <common>\nvarying float vGroundFade;" +
        (holeCenter
          ? "\nuniform vec2 uHoleCenter;\nvarying vec2 vGroundXZ;"
          : ""),
    );
    if (holeCenter) {
      frag = frag.replace(
        "#include <clipping_planes_fragment>",
        `#include <clipping_planes_fragment>
         vec2 holeD = abs(vGroundXZ - uHoleCenter);
         if (max(holeD.x, holeD.y) < ${holeHalf.toFixed(1)}) discard;`,
      );
    }
    shader.fragmentShader = frag.replace(
      "#include <dithering_fragment>",
      "#include <dithering_fragment>\ngl_FragColor.a *= 1.0 - vGroundFade;",
    );
  };
  return holeCenter;
}

// The height a clipmap level of cell C draws at (wx, wz): the grid triangulates
// each C-quad along its (x0,z0)->(x0+C,z0+C) diagonal (matching the index pattern
// in buildGroundLevel), so reproduce that exact triangulated chord rather than a
// bilinear patch. Two callers: the geomorph uses it as the next-coarser level's
// surface to morph toward (C0-exact, because the coarser level is held un-jittered
// around its hole so its near-boundary vertices sit on this lattice), and prop
// seating uses it via facetHeight to drop a book onto the surface the player
// actually sees, not the smooth field that surface only chords.
function chordHeight(wx: number, wz: number, C: number): number {
  const x0 = Math.floor(wx / C) * C;
  const z0 = Math.floor(wz / C) * C;
  const fx = (wx - x0) / C;
  const fz = (wz - z0) / C;
  const h00 = sampleHeight(x0, z0);
  const h11 = sampleHeight(x0 + C, z0 + C);
  if (fx <= fz) {
    const h01 = sampleHeight(x0, z0 + C);
    return h00 + fx * (h11 - h01) + fz * (h01 - h00);
  }
  const h10 = sampleHeight(x0 + C, z0);
  return h00 + fx * (h10 - h00) + fz * (h11 - h10);
}

// --- prop seating: the exact drawn facet --------------------------------------
// Books seat on the surface the clipmap actually DRAWS underfoot, not the smooth
// sampleHeight field that surface only chords. The two diverge two ways on a tight
// convex crest: the flat facet chords below the field (faceting), and the in-plane
// jitter shoves the facet's corners up to GROUND_JIT_FRAC*cell sideways, which on a
// steep face turns into a vertical offset of a couple of units. Seating on
// sampleHeight floats over both; seating on the un-jittered chord still floats over
// the second. So reconstruct the finest level's jittered triangle exactly as
// buildGroundLevel draws it near the camera (full jitter, no morph: the canonical
// close-up surface, which is the view that matters) and drop the book onto that
// plane. No bias, so nothing legitimate is ever buried. The far field, where a
// coarser morphing level is drawn under a static seat, still breathes, but that is
// flight-only and faded; on foot the book you stand by is exact.
const FINEST = GROUND_LEVELS[0].cell;
const FINEST_JIT = FINEST * GROUND_JIT_FRAC;
const _fv: THREE.Vector3[] = Array.from({ length: 16 }, () => new THREE.Vector3());

// One finest-level vertex as drawn at jitterScale = 1: the disk jitter of
// buildGroundLevel keyed on the world cell index (gx, gz), height sampled at the
// jittered position. The un-jittered corner sits at (gx*FINEST, gz*FINEST).
function finestVertex(gx: number, gz: number, out: THREE.Vector3): THREE.Vector3 {
  const rr = FINEST_JIT * Math.sqrt(hash2(gx, gz));
  const th = hash2(gx + 7919, gz + 104729) * Math.PI * 2;
  const x = gx * FINEST + rr * Math.cos(th);
  const z = gz * FINEST + rr * Math.sin(th);
  return out.set(x, sampleHeight(x, z), z);
}

// Height of the plane through triangle (a, b, c) at the XZ point (px, pz), or null
// if the point's XZ projection falls outside the triangle. Barycentric in the XZ
// plane; the small negative tolerance lets a point on a shared edge match so the
// triangulation has no gaps.
function baryHeight(
  a: THREE.Vector3,
  b: THREE.Vector3,
  c: THREE.Vector3,
  px: number,
  pz: number,
): number | null {
  const v0x = b.x - a.x;
  const v0z = b.z - a.z;
  const v1x = c.x - a.x;
  const v1z = c.z - a.z;
  const v2x = px - a.x;
  const v2z = pz - a.z;
  const d00 = v0x * v0x + v0z * v0z;
  const d01 = v0x * v1x + v0z * v1z;
  const d11 = v1x * v1x + v1z * v1z;
  const d20 = v2x * v0x + v2z * v0z;
  const d21 = v2x * v1x + v2z * v1z;
  const denom = d00 * d11 - d01 * d01;
  if (denom === 0) return null;
  const v = (d11 * d20 - d01 * d21) / denom;
  const w = (d00 * d21 - d01 * d20) / denom;
  const u = 1 - v - w;
  const e = -1e-4;
  if (u < e || v < e || w < e) return null;
  return a.y + v * (b.y - a.y) + w * (c.y - a.y);
}

// Seat height: the height of the finest-level facet drawn at (x, z). Split on the
// a->d diagonal exactly as the index buffer does. The point almost always lands in
// one of its OWN cell's two triangles, so test those four corners first (the hit
// path: 4 jittered vertices, no lattice). Only when jitter has pulled the point
// across a cell edge does the slow path build the full 4x4 lattice over the 3x3
// cell block (jitter < cell, so one ring of neighbours is guaranteed to hold the
// triangle) and search it. Falls back to the un-jittered chord if the point slips
// through every triangle.
export function facetHeight(x: number, z: number): number {
  const gx0 = Math.floor(x / FINEST);
  const gz0 = Math.floor(z / FINEST);
  // fast path: the book's own cell.
  const a = finestVertex(gx0, gz0, _fv[0]);
  const b = finestVertex(gx0 + 1, gz0, _fv[1]);
  const c = finestVertex(gx0, gz0 + 1, _fv[2]);
  const d = finestVertex(gx0 + 1, gz0 + 1, _fv[3]);
  const h1 = baryHeight(a, c, d, x, z); // upper-left triangle (a, c, d)
  if (h1 !== null) return h1;
  const h2 = baryHeight(a, d, b, x, z); // lower-right triangle (a, d, b)
  if (h2 !== null) return h2;
  // slow path: jitter moved the containing triangle into a neighbouring cell.
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 4; i++) {
      finestVertex(gx0 - 1 + i, gz0 - 1 + j, _fv[j * 4 + i]);
    }
  }
  for (let cj = 0; cj < 3; cj++) {
    for (let ci = 0; ci < 3; ci++) {
      const na = _fv[cj * 4 + ci];
      const nb = _fv[cj * 4 + ci + 1];
      const nc = _fv[(cj + 1) * 4 + ci];
      const nd = _fv[(cj + 1) * 4 + ci + 1];
      const u1 = baryHeight(na, nc, nd, x, z);
      if (u1 !== null) return u1;
      const u2 = baryHeight(na, nd, nb, x, z);
      if (u2 !== null) return u2;
    }
  }
  return chordHeight(x, z, FINEST);
}

// One clipmap level: a 128-cell grid that re-centres on the camera each time it
// crosses one of ITS cells. Topology is built once; on a cell-cross only the
// per-vertex height, in-plane jitter and colour are resampled and the mesh is
// translated to the snapped origin. Snapping is what stops the facets swimming:
// every vertex lands on a fixed world lattice, and the jitter is keyed off the
// world cell a vertex covers (not its local index) so the irregular lattice is
// welded to the world and only re-indexes as the window slides. The outer band
// morphs to the coarser level (geomorphing) and a depth bias picks the winner at
// the seam; see the clipmap note above. hasCoarser
// is false for the outermost level (nothing to morph to; its rim fades into void).
// update() is cheap and idempotent within a cell, safe to call every frame.
function buildGroundLevel(
  level: GroundLevel,
  index: number,
  hasCoarser: boolean,
): {
  mesh: THREE.Mesh;
  update: (camX: number, camZ: number) => void;
} {
  const { cell, half, hole, holeCell } = level;
  const N = Math.round((2 * half) / cell);
  const stride = N + 1;
  const halfCells = half / cell;
  const jit = cell * GROUND_JIT_FRAC;
  const vcount = stride * stride;
  const positions = new Float32Array(vcount * 3);
  const colors = new Float32Array(vcount * 3);
  for (let iz = 0; iz <= N; iz++) {
    for (let ix = 0; ix <= N; ix++) {
      const v = iz * stride + ix;
      positions[v * 3] = -half + ix * cell;
      positions[v * 3 + 2] = -half + iz * cell;
    }
  }
  const indices = new Uint32Array(N * N * 6);
  let t = 0;
  for (let iz = 0; iz < N; iz++) {
    for (let ix = 0; ix < N; ix++) {
      const a = iz * stride + ix;
      const b = a + 1;
      const cc = a + stride;
      const d = cc + 1;
      indices[t++] = a;
      indices[t++] = cc;
      indices[t++] = d;
      indices[t++] = a;
      indices[t++] = d;
      indices[t++] = b;
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  // flatShading derives the per-face normal from position derivatives in-shader,
  // so the welded lattice still reads faceted and needs no vertex normals.
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    flatShading: true,
  });
  const holeCenter = applyGroundMaterial(mat, hole);
  // The morph leaves the fine outer ring coplanar with the coarse hole edge they
  // meet at. renderOrder (below) draws finer first, but GL_LESS lets a coplanar
  // coarse fragment that rounds to the same depth slip through and z-fight as a
  // flickering 1px line. Push each coarser level back so the tie is broken and the
  // finer level reliably wins the shared edge.
  if (index > 0) {
    mat.polygonOffset = true;
    // units only, no factor: the seam surfaces are identical (coplanar), so a
    // constant depth nudge separates them. A slope-scaled factor term would blow
    // up at the horizon where the ground is viewed edge-on, lighting up the far LOD.
    mat.polygonOffsetFactor = 0;
    mat.polygonOffsetUnits = index * 3;
  }
  const mesh = new THREE.Mesh(geom, mat);
  mesh.frustumCulled = false; // it tracks the camera; its bounds are always in view
  // The levels are all transparent and share the camera's centre, so Three's
  // distance sort can't order them: it flips per frame, double-blending the
  // overlap rings into a flickering band. renderOrder = index draws them finest
  // first, so in every overlap the finer level writes depth first and the coarser
  // one fails the depth test and is never blended on top. This is what actually
  // stabilises the seams; the polygon offset above only breaks the coplanar tie.
  mesh.renderOrder = index;

  const posAttr = geom.attributes.position as THREE.BufferAttribute;
  const colAttr = geom.attributes.color as THREE.BufferAttribute;
  const c = new THREE.Color();
  let snapX = NaN;
  let snapZ = NaN;

  function update(camX: number, camZ: number): void {
    // snap the hole to the FINER level's cell (holeCell), the same snap that level
    // uses, so the hole edge and the finer level's real-surface edge never drift
    // apart and open a gap along the boundary.
    if (holeCenter) {
      holeCenter.value.set(
        Math.round(camX / holeCell) * holeCell,
        Math.round(camZ / holeCell) * holeCell,
      );
    }
    const sx = Math.round(camX / cell) * cell;
    const sz = Math.round(camZ / cell) * cell;
    if (sx === snapX && sz === snapZ) return; // same cell, geometry unchanged
    snapX = sx;
    snapZ = sz;
    mesh.position.set(sx, 0, sz);
    const baseIX = sx / cell - halfCells; // world-cell index of the (0,0) corner
    const baseIZ = sz / cell - halfCells;
    const coarseCell = cell * 2; // the next level out doubles the cell
    const rOut = half - cell; // where the coarser level starts drawing (its hole)
    const morphIn = rOut - GROUND_MORPH_CELLS * cell; // inner edge of the morph band
    const morphW = GROUND_MORPH_CELLS * cell;
    const relaxFlat = hole + GROUND_RELAX_FLAT * cell; // un-jittered out to here
    const relaxW = GROUND_RELAX_CELLS * cell;
    for (let iz = 0; iz <= N; iz++) {
      for (let ix = 0; ix <= N; ix++) {
        const v = iz * stride + ix;
        const lx0 = -half + ix * cell; // un-jittered lattice position; the band
        const lz0 = -half + iz * cell; //   coordinate is measured off this
        let lx = lx0;
        let lz = lz0;
        const rim = ix === 0 || iz === 0 || ix === N || iz === N;
        const d = Math.max(Math.abs(lx0), Math.abs(lz0)); // max-norm ring radius
        // morph weight: 0 inside the band, ramping to 1 at the boundary where the
        // coarser level takes over (only if there IS a coarser level out there).
        const alpha = hasCoarser
          ? clamp01((d - morphIn) / morphW)
          : 0;
        // jitter relaxes to 0 at the rim (so the morph target is un-jittered) and
        // in the flat band around the hole (so this level matches the finer one's
        // morph target). The rim itself is always clean, to keep the square edge.
        let jitterScale = 1 - alpha;
        if (hole > 0) jitterScale = Math.min(jitterScale, clamp01((d - relaxFlat) / relaxW));
        if (rim) jitterScale = 0;
        if (jitterScale > 0) {
          // disk jitter (sqrt for area-uniform), keyed by world cell so it is
          // stable as the level re-centres.
          const gx = baseIX + ix;
          const gz = baseIZ + iz;
          const rr = jit * jitterScale * Math.sqrt(hash2(gx, gz));
          const th = hash2(gx + 7919, gz + 104729) * Math.PI * 2;
          lx += rr * Math.cos(th);
          lz += rr * Math.sin(th);
        }
        const wx = lx + sx;
        const wz = lz + sz;
        let y = sampleHeight(wx, wz);
        // crest/trough relief from this level's own grid: the raw height minus
        // its four neighbours at RELIEF_CELLS cells out, normalised to a signed
        // ~[-1, 1]. Offset scales with the cell, so the read wavelength tracks the
        // LOD and the tint matches the relief this level draws. Uses the raw y,
        // before the morph lerp below.
        const ro = RELIEF_CELLS * cell;
        const relief =
          (y -
            0.25 *
              (sampleHeight(wx - ro, wz) +
                sampleHeight(wx + ro, wz) +
                sampleHeight(wx, wz - ro) +
                sampleHeight(wx, wz + ro))) /
          (RELIEF_SCALE * ro);
        if (alpha > 0) {
          // geomorph: lerp toward the coarser level's chord at this point
          const tgt = chordHeight(wx, wz, coarseCell);
          y += (tgt - y) * alpha;
        }
        positions[v * 3] = lx;
        positions[v * 3 + 1] = y;
        positions[v * 3 + 2] = lz;
        groundColor(wx, wz, c, relief);
        colors[v * 3] = c.r;
        colors[v * 3 + 1] = c.g;
        colors[v * 3 + 2] = c.b;
      }
    }
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    geom.computeBoundingSphere();
  }

  return { mesh, update };
}

// The whole ground: every GROUND_LEVELS entry as a camera-following clipmap level,
// grouped. update() drives them all (each snaps to its own cell, so the coarse
// levels rebuild rarely). Built entirely from the heightmap, so ground.bin and the
// analytic dune code are both retired.
export function createGround(): {
  group: THREE.Group;
  update: (camX: number, camZ: number) => void;
} {
  const group = new THREE.Group();
  const levels = GROUND_LEVELS.map((lvl, i) =>
    buildGroundLevel(lvl, i, i < GROUND_LEVELS.length - 1),
  );
  for (const l of levels) group.add(l.mesh);
  return {
    group,
    update: (camX: number, camZ: number) => {
      for (const l of levels) l.update(camX, camZ);
    },
  };
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
    // radial narrative: pale summit -> sand -> grey void, plus a painted wobble.
    groundColor(x, z, c);
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
