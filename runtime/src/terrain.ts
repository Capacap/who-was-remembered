import * as THREE from "three";
import {
  DAYLIGHT_FRAG_COMMON,
  applyDaylightGLSL,
  type DaylightUniforms,
} from "./daylight";

// --- terrain ----------------------------------------------------------------
// The world is a near-flat desert whose relief (a whisper of a central rise easing
// out to a spiral dune field) is a BAKED heightmap raster (stage9), not computed
// here. This lives in the renderer, not the pipeline: the vertical axis carries no
// data (time and longitude are the horizontal x/z), so terrain is decoration by the
// three-tier rule. sampleHeight is the single elevation source: the ground mesh
// tessellates it, the player's feet read it, and books seat on facetHeight (the facet
// the ground mesh actually draws), so the visible ground and everything on it agree.
//
// An earlier analytic hill+dune field (getGroundHeight/getGroundNormal and their
// dune-shaping knobs) was removed 2026-06-05: it only ever fed a pre-load fallback,
// and its transverse-dune shape no longer even matched the baked spiral dunes, so it
// was stale dead weight. Edit terrain SHAPE in the stage9 bake, never here.

let plazas: { x: number; z: number }[] = [];

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
const SAND_GRADIENT_START = new THREE.Color(0xEFE8D2); // bright dry summit (the present)
const SAND_GRADIENT_MID = new THREE.Color(0xD3BC95); // mid desert
const SAND_GRADIENT_END = new THREE.Color(0x9C8465); // faded deep-past floor
// One smooth era ramp across the whole map: pale summit (the present, at the
// centre) -> sand at the midpoint -> grey deep-past floor at the outer edge.
// Spans [0, ERA_GRADIENT_R]; recency literally lighting the map, now a single
// continuous gradient rather than a pale core with a flat sand band and a
// separate grey rim. Mirrors world.json R_MAX (the deep-past edge).
const ERA_GRADIENT_R = 7100;
const COLOR_WAVELENGTH = 900; // patch-noise scale for the painted wobble

// The vortex eye: a pale glacial teal gathered at the centre, the cold light of
// the present where the knowledge piles up. It is painted into the ALBEDO (lerp in
// groundColor toward COLOR_EYE), lit like the rest of the sand. The old additive
// emissive was dropped 2026-06-05: it was a carry-over from a darker shader era; the
// hemisphere light lifts the near-flat centre enough that the albedo reads without a
// glow crutch fighting the lighting. The coverage never makes a clean radial ring: a
// noise field (EYE_NOISE_*) fingers the boundary in and out even across the flat calm
// zone where there are no dunes to follow, a relief push (EYE_RELIEF_PUSH) lets it run
// down the dune troughs where the terrain has relief, and EYE_VALLEY pools the colour
// in the troughs and recedes it off the crests so it interfingers with the rising dunes.
const COLOR_EYE = new THREE.Color(0x4B6B75); // near-white with a whisper of cool: otherworldly bleached sand
const EYE_R0 = 260; // full eye colour inside here (the vantage pocket)
const EYE_R1 = 1900; // ... gone by here, reaching out into the rising dunes
const EYE_ALBEDO_STRENGTH = 0.85; // albedo lerp toward the eye colour (only shows where lit)
const EYE_RELIEF_PUSH = 320; // world units the boundary shifts per unit relief:
// troughs (relief < 0) pull it inward (more eye), crests (relief > 0) push it out
const EYE_NOISE_AMP = 680; // world units the eye boundary wanders by the shared smear field
const EYE_VALLEY = 1.3; // pool the colour in the dune troughs, recede off the crests:
// scales the mask by (1 - EYE_VALLEY * relief), so crests dim and troughs lift

// The shared "smear" field: one radial displacement (two octaves of Perlin) that the
// vortex eye AND the era gradient both add to the radius before reading their colour, so
// the time-bands and the eye finger in and out TOGETHER as one turbulent flow rather than
// sitting in clean concentric rings. This generalises the eye's old boundary noise to the
// whole palette ("all colours smear the way the eye does"). Each user scales the field by
// its own amplitude (EYE_NOISE_AMP for the narrow eye band, ERA_WARP_AMP for the broad
// gradient), so they wander by different absolute amounts off the same field and stay
// coherent. The era gradient is atmosphere — the books carry the real date — so it is free
// to bleed; push ERA_WARP_AMP up for more dissolve, down toward clean concentric rings.
const WARP_SCALE = 760; // coarse wavelength; a finer octave at 0.4x rides on top
const ERA_WARP_AMP = 680; // world units the era bands finger in/out (the gradient smear)
function smearNoise(x: number, z: number): number {
  return (
    perlin(x / WARP_SCALE, z / WARP_SCALE) +
    0.5 * perlin(x / (WARP_SCALE * 0.4), z / (WARP_SCALE * 0.4))
  );
}

// Crest/trough relief tint, painted into the ALBEDO on the radial base: crests read
// scoured pale and a touch warm, troughs deeper and cooler, so the dunes carry COLOUR
// while the raking sun's flat-shaded N·L owns light/dark. The signal is the vertex
// height minus its neighbours RELIEF_CELLS out on the ground grid, so the colour tracks
// the relief the mesh actually draws. It leans on hue + saturation, which ride straight
// through the lighting (a warm albedo times warm light stays warm), and only a whisper
// of lightness, since the sun already owns that axis and would crush a bigger shift. An
// earlier version carried this tint as an additive emissive (a warm crest rim + a deep
// red trough pool); it was removed because out in the dune field the sun DOES light the
// facets, so the albedo reads fine, and the emissive only fought the lighting and the
// Kuindzhi palette. (The vortex eye is now albedo too, for the same reason.) All
// four are eyeball knobs.
const RELIEF_CELLS = 3; // neighbour offset in cells: the relief's read wavelength
const RELIEF_SCALE = 0.25; // slope-difference that reaches the full crest/trough tint
// Crests and troughs tune INDEPENDENTLY (they need not be each other's mirror): the
// crest knobs fire only where relief > 0, the trough knobs only where relief < 0.
// CREST_LIGHT/TROUGH_DARK are PROPORTIONAL (a fraction of the base lightness), not a
// fixed offset, so a trough darkens by the same fraction on the bright centre cream
// and the dark outer grey rather than crushing the already-dark edge toward black.
// Hue/sat ride through the lighting and can read harder.
const CREST_LIGHT = 1.0; // crest lightens, as a fraction of base lightness
const CREST_SAT = 0.10; // crest bleaches (desaturates the scoured top)
const CREST_HUE = 0.1; // crest warms
const TROUGH_DARK = 0.5; // trough darkens, as a fraction of base lightness
const TROUGH_SAT = 0.25; // trough deepens (saturates the shadowed hollow)
const TROUGH_HUE = -0.1; // trough cools

// Distance fade: the ground's opacity falls to zero between these radii from the
// CAMERA, so the whole landscape dissolves into the sky dome before it reaches the
// mesh edge. The fade is circular (camera distance), so unlike a fog tint of opaque
// geometry it leaves no square plate to see from a height. Colour-matching fog can
// only blend a surface toward the haze colour, never past it, so a sunlit dune crest
// punches through as a bright ridge; going transparent removes the surface entirely,
// so nothing is left to catch the light or to show an edge. The mesh half-extent
// (GROUND_HALF) is sized so this fade always lands before the edge while the player
// is in the content region; far out in the empty void the faded edge can show.
export const FADE_START = 3000; // fully opaque within this distance of the camera
export const FADE_END = 6500; // fully gone (dome shows through) beyond this

// Player pool: ONE weak raking light that follows the player and rakes across the
// dune facets near them, just enough to show the terrain underfoot in the mostly-night
// desert. It is additive in WORLD space, lit by each facet's own world-derived normal
// (cross of the position derivatives the flat shading already computes), so it tilts
// with the dune faces rather than washing them flat -- the rake gradient IS the terrain
// reveal. The centre (uPlayer) is shared with the book glow; this radius is its own.
// Each mode is a two-colour RADIAL gradient (centre hue -> rim hue across the pool),
// the same inner->outer trick the teleporter pads use: walking runs pale yellow at the
// core into orange at the skirt; skating runs cyan into blue. uSkate (0..1, ramped in
// the loop) cross-fades the two palettes -- the colour is the only "you're skating"
// signal. The old wide-warm raking pool (str 0.55 / r 42) plus a separate blue skate
// glow used to compound into a strong orange wash that also lit the books; collapsed to
// a single tight, weak pool. Colours baked as linear GLSL literals like the distance
// fade. Eyeball knobs; rebuild to retune.
const PLAYER_WALK_INNER = new THREE.Color(0xffe6a0); // warm pale yellow at the core
const PLAYER_WALK_OUTER = new THREE.Color(0xff8a3c); // orange at the skirt
const PLAYER_SKATE_INNER = new THREE.Color(0x46d8ff); // cyan core while skating
const PLAYER_SKATE_OUTER = new THREE.Color(0x2f6cff); // blue skirt while skating
const PLAYER_LIGHT_RADIUS = 24; // pool kept tight around the feet (was 42)
const PLAYER_LIGHT_INNER = 2.0; // full reach within this
const PLAYER_LIGHT_STRENGTH = 0.22; // weak: reveal the ground, don't wash it (was 0.55)
const PLAYER_LIGHT_HEIGHT = 6.0; // light's height over the player; lower = more grazing
const _glsl = (c: THREE.Color) => {
  const l = c.clone().convertSRGBToLinear();
  return `vec3(${l.r.toFixed(4)}, ${l.g.toFixed(4)}, ${l.b.toFixed(4)})`;
};
const PLAYER_WALK_INNER_RGB = _glsl(PLAYER_WALK_INNER);
const PLAYER_WALK_OUTER_RGB = _glsl(PLAYER_WALK_OUTER);
const PLAYER_SKATE_INNER_RGB = _glsl(PLAYER_SKATE_INNER);
const PLAYER_SKATE_OUTER_RGB = _glsl(PLAYER_SKATE_OUTER);

// Teleporter floor glow: a blue emissive disc pooled on each plaza, the near-field
// "stand here" marker. The beam fades out within TP_BEAM_FADE_NEAR so it never clips
// the camera up close, which is exactly where it stops telling you where to stand;
// this glow takes over there, so the column and the pad hand off as one beacon. Same
// blue as the beam (main.ts TP_BEAM_COLOR). This one stays an additive emissive (unlike
// the eye, now albedo): a "stand here" marker must read in the barely-lit centre instead
// of crushing dark, and because it IS the ground it can never look superimposed (the
// stone ring it replaced did, being a warm Lambert prop dropped onto the cool centre).
// Evaluated per-fragment over the
// teleporter positions, not per-vertex: the disc is smaller than a GROUND_CELL facet,
// so a vertex attribute would light a single triangle. A slow pulse off uDriftTime
// reads as powered. Eyeball knobs; tune against the vortex teal, which the innermost
// teleporters sit inside (a blue pad there has to fight the near-white eye glow).
const TP_GLOW_INNER = new THREE.Color(0x3df0ff); // cyan at the centre of the pad
const TP_GLOW_OUTER = new THREE.Color(0x2f6cff); // blue at the rim, matches main.ts TP_BEAM_COLOR
const TP_GLOW_RADIUS = 9; // disc footprint, world units; tracks main.ts TP_ENTER_RADIUS (the travel zone)
const TP_GLOW_FALLOFF = 1.6; // intensity exponent over the radius; >1 keeps a bright core with a soft skirt
const TP_GLOW_STRENGTH = 0.85; // additive intensity at the centre
const TP_GLOW_PULSE = 0.18; // pulse depth as a fraction of strength (0 = steady)
const TP_GLOW_PULSE_SPEED = 0.6; // pulse rate against the drifting uDriftTime
// Proximity ramp off the player position: far away the pad is a dim, plain-blue
// beacon; on approach it brightens and the cyan core emerges. Keep in sync with
// main.ts TP_BEAM_APPROACH_* so beam and pad ramp together.
const TP_GLOW_APPROACH_NEAR = 70; // player distance (world u) at which the pad reaches full intensity
const TP_GLOW_APPROACH_FAR = 260; // beyond this the pad sits at the dim far level
const TP_GLOW_FAR_LEVEL = 0.3; // intensity multiplier when far (0..1)
// One-sided daylight lift, mirrors the beam (main.ts TP_BEAM_DAY_BOOST). The pad is
// added AFTER the daylight multiply so a shadow never dims it; this only scales it UP
// when the drifting light crosses the plaza, so pad and beam swell together. Keep the
// value matched to TP_BEAM_DAY_BOOST so the whole beacon breathes as one.
const TP_GLOW_DAY_BOOST = 7.5; // additive multiplier in full daylight (1 = no lift). Eyeball.
const _tgi = TP_GLOW_INNER.clone().convertSRGBToLinear();
const _tgo = TP_GLOW_OUTER.clone().convertSRGBToLinear();
const TP_GLOW_INNER_RGB = `vec3(${_tgi.r.toFixed(4)}, ${_tgi.g.toFixed(4)}, ${_tgi.b.toFixed(4)})`;
const TP_GLOW_OUTER_RGB = `vec3(${_tgo.r.toFixed(4)}, ${_tgo.g.toFixed(4)}, ${_tgo.b.toFixed(4)})`;

// A live uniform carrying the player's world-xz position, shared between the
// ground's raking light and the books' proximity glow so both pools share a centre.
// `lift` rides alongside (only the ground rake reads it): the player's height above
// the ground directly below, 0 when grounded, so a jump or the skate hover lifts the
// rake lamp with the eye instead of leaving it painted flat on the sand.
export interface PlayerUniform {
  value: THREE.Vector2;
  lift: { value: number };
}

// --- ground: a single static mesh tessellated from the heightmap --------------
// One uniform grid of GROUND_CELL facets spanning the world, built once at load and
// never rebuilt. It replaced a five-level camera-following geometry clipmap. The
// clipmap bounded on-screen facet size as the camera roamed, but its overlapping,
// separately faded, camera-tracked levels stacked several subtle seam artifacts that
// resisted every fix: a transparent level writes depth and culls the coarser one
// beneath it, so where its fade made it translucent it revealed the sky dome through
// the seam (an "aura" tracing each level, worst at altitude where all levels fade at
// once); the per-frame hole tracking outran the round-robin per-level rebuild; and
// the resolution change left T-junctions. A single surface has none of it: no
// overlap, nothing to depth-cull, no rebuild, so no seams and a steadier framerate
// (no re-tessellation stall when flying). The cost is spending triangles evenly
// rather than concentrating them underfoot, which at GROUND_CELL = 12 is a few
// million tris over the whole world, nothing for the GPU, and it ships nothing extra
// because the mesh is generated from the same heightmap the clipmap used.
//
// The grid is jittered in-plane (the Vane low-poly look) keyed off the world cell, so
// finestVertex/facetHeight reconstruct the drawn facet exactly and books seat on the
// surface the player sees. flatShading derives the per-face normal in-shader.
const GROUND_CELL = 12; // uniform facet size underfoot, world units
const GROUND_HALF = 9000; // half-extent: covers the world (R_MAX ~7100) + the fade tail
const GROUND_JIT_FRAC = 0.33; // in-plane vertex jitter as a fraction of the cell:
//   the Vane look, an organic faceted triangulation instead of a mechanical lattice

// Record each teleporter's xz so the ground shader can pool a "stand here" glow at
// every plaza (applyGroundMaterial reads these). The plazas are levelled in the
// stage9 heightmap bake itself, so nothing here needs to recompute their height.
export function initTerrain(teleporters: { x: number; y: number }[]): void {
  plazas = teleporters.map((t) => ({ x: t.x, z: t.y }));
}

const NORMAL_EPS = 0.5; // default central-difference step for sampleNormal, world units

// Ground colour at a world point: the radial pale-summit -> sand -> grey-void
// narrative, a low-frequency painted wobble, and a crest/trough relief tint.
// buildGround tints each vertex with this from world position; it stays per-vertex on
// the CPU by design (no texture, no baked raster), so the low-poly vertex-colour look
// holds. `relief` is the signed, already-normalised local relief (+ on crests, - in
// hollows) the caller reads from the grid; 0 leaves the base untouched.
// The vortex-eye coverage at a world point, in [0, 1]: 1 at the gathered centre,
// 0 past the band. The boundary radius is perturbed by two octaves of noise (so
// the edge fingers organically rather than reading as a clean ring) and by local
// relief (so it runs down the dune troughs where the terrain has relief to
// follow). buildGround reads this once per vertex to drive the albedo tint toward
// COLOR_EYE in groundColor.
export function eyeMask(x: number, z: number, relief: number): number {
  const r = Math.hypot(x, z);
  const rEff = r + EYE_RELIEF_PUSH * relief + EYE_NOISE_AMP * smearNoise(x, z);
  const radial = 1 - smootherstep((rEff - EYE_R0) / (EYE_R1 - EYE_R0));
  // pool in the troughs, recede off the crests: relief > 0 (crest) dims, < 0 lifts
  const valley = 1 - EYE_VALLEY * relief;
  return Math.min(1, Math.max(0, radial * valley));
}

const _hsl = { h: 0, s: 0, l: 0 }; // scratch for groundColor's single HSL roundtrip
export function groundColor(
  x: number,
  z: number,
  out: THREE.Color,
  relief = 0,
  eye = 0,
): THREE.Color {
  // one continuous ramp: pale -> sand at the midpoint -> grey across [0, edge].
  // The radius is smeared by the shared field (the same one the eye uses) before the ramp
  // is read, so the era bands finger in and out together with the eye instead of reading as
  // clean concentric rings. Clamp: the displacement can push the sampled radius below 0
  // near the centre.
  const r = Math.hypot(x, z);
  const rEra = r + ERA_WARP_AMP * smearNoise(x, z);
  const t = Math.min(1, Math.max(0, rEra / ERA_GRADIENT_R));
  if (t < 0.5) out.copy(SAND_GRADIENT_START).lerp(SAND_GRADIENT_MID, t * 2);
  else out.copy(SAND_GRADIENT_MID).lerp(SAND_GRADIENT_END, (t - 0.5) * 2);
  // Tint the albedo toward the eye colour; the HSL offsets below then ride on top so
  // the eye still picks up crest/trough shading. It is lit like the rest of the sand
  // now (no emissive); the hemisphere light carries the near-flat centre.
  if (eye > 0) out.lerp(COLOR_EYE, eye * EYE_ALBEDO_STRENGTH);
  const tone = perlin(x / COLOR_WAVELENGTH, z / COLOR_WAVELENGTH); // [-1, 1]
  // Split the relief into a crest amount and a trough amount, each 0..1, so the two
  // tune independently: crests warm, bleach and lighten; troughs cool, deepen and
  // darken. (Setting the TROUGH_* knobs equal to the CREST_* ones recovers the old
  // symmetric mirror.)
  const k = relief < -1 ? -1 : relief > 1 ? 1 : relief;
  const crest = k > 0 ? k : 0;
  const trough = k < 0 ? -k : 0;
  // The painted wobble and the relief tint are HSL offsets; fold them into one
  // getHSL/setHSL roundtrip rather than several (this runs per vertex across the whole
  // ground build, so the saved RGB<->HSL conversions matter).
  out.getHSL(_hsl);
  _hsl.h += tone * 0.01 - crest * CREST_HUE + trough * TROUGH_HUE;
  _hsl.s += tone * 0.03 - crest * CREST_SAT + trough * TROUGH_SAT;
  // Lightness relief is PROPORTIONAL: scale the base by (1 +- fraction) so a trough
  // darkens by the same fraction whether the base is bright centre cream or dark outer
  // grey, instead of a fixed offset that crushes the already-dark edge toward black.
  // The painted wobble stays a small additive variation folded in first.
  _hsl.l = (_hsl.l + tone * 0.04) * (1 + crest * CREST_LIGHT - trough * TROUGH_DARK);
  out.setHSL(_hsl.h, _hsl.s, _hsl.l);
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

// Height at a world point from the baked raster. Before the heightmap has loaded it
// returns 0: the only caller then is the spawn placement, which is re-settled onto the
// baked surface the instant the raster is in (main.ts), so no frame ever shows the 0.
export function sampleHeight(x: number, z: number): number {
  if (!HM) return 0;
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

// Surface normal from the baked raster: central differences on sampleHeight so props
// seat to the same surface the ground mesh draws. eps doubles as the footprint
// half-width, so a large prop conforms to the slope it spans rather than one texel.
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

// Inject the ground material edits, shared by the one static ground mesh: the
// camera-distance opacity fade (so the world dissolves circularly into the dome at
// the far edge, no square plate) and the additive raking pool that follows the
// player, then the drifting daylight multiply. All are per-fragment in world
// space, so they ride on the static mesh unchanged from the clipmap days; only the
// per-level discard hole and its depth machinery are gone with the clipmap.
function applyGroundMaterial(
  mat: THREE.MeshLambertMaterial,
  cell: number,
  uPlayer: PlayerUniform,
  uSkate: { value: number },
  daylight: DaylightUniforms,
): void {
  mat.transparent = true;
  const fadeSpan = (FADE_END - FADE_START).toFixed(1);
  // Teleporter floor glow, built from the plaza positions initTerrain stored (set
  // before buildGround calls this). Evaluated per-fragment over the teleporter array;
  // skipped entirely when there are none, since a zero-length GLSL array is invalid.
  const tpPos = plazas.map((p) => new THREE.Vector2(p.x, p.z));
  const tpDecl = tpPos.length ? `\nuniform vec2 uTpPos[${tpPos.length}];` : "";
  const tpGlow = tpPos.length
    ? `
       {
         float tpGlow = 0.0;
         float tpT = 0.0; // radial fraction (0 centre, 1 rim) of the dominant pad
         float tpApproach = 1.0; // 0 when the player is far from that pad, 1 when near
         for (int i = 0; i < ${tpPos.length}; i++) {
           float t = clamp(distance(vWorldPos.xz, uTpPos[i]) / ${TP_GLOW_RADIUS.toFixed(2)}, 0.0, 1.0);
           float g = pow(1.0 - t, ${TP_GLOW_FALLOFF.toFixed(2)}); // soft falloff across the whole disc
           if (g > tpGlow) {
             tpGlow = g; tpT = t;
             tpApproach = smoothstep(${TP_GLOW_APPROACH_FAR.toFixed(1)}, ${TP_GLOW_APPROACH_NEAR.toFixed(1)}, distance(uPlayer, uTpPos[i]));
           }
         }
         // far: dim, plain-blue. near: brighter, with the cyan core emerging.
         float tpProx = mix(${TP_GLOW_FAR_LEVEL.toFixed(2)}, 1.0, tpApproach);
         vec3 tpInner = mix(${TP_GLOW_OUTER_RGB}, ${TP_GLOW_INNER_RGB}, tpApproach);
         vec3 tpCol = mix(tpInner, ${TP_GLOW_OUTER_RGB}, tpT);
         // slow breathing pulse so the pad reads as powered, not painted.
         float tpPulse = 1.0 - ${TP_GLOW_PULSE.toFixed(2)} * (0.5 - 0.5 * cos(uDriftTime * ${TP_GLOW_PULSE_SPEED.toFixed(3)}));
         // one-sided daylight lift, in lockstep with the beam: never dims (added after the
         // daylight multiply), just swells when a daylight pool crosses the plaza. beaconLit
         // is the point-sample reader (distance-independent), not the ground's daylightAt.
         float tpDayGain = mix(1.0, ${TP_GLOW_DAY_BOOST.toFixed(2)}, beaconLit(vWorldPos.xz) * uDaylightMix);
         gl_FragColor.rgb += tpGlow * tpCol * (${TP_GLOW_STRENGTH.toFixed(2)} * tpPulse * tpProx * tpDayGain);
       }`
    : "";
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uPlayer = uPlayer;
    shader.uniforms.uPlayerLift = uPlayer.lift;
    shader.uniforms.uSkate = uSkate;
    shader.uniforms.uDaylight = daylight.uDaylight;
    shader.uniforms.uDriftTime = daylight.uDriftTime;
    shader.uniforms.uDaylightMix = daylight.uDaylightMix;
    if (tpPos.length) shader.uniforms.uTpPos = { value: tpPos };
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying float vGroundFade;\nvarying float vViewDist;\nvarying vec3 vWorldPos;\nvarying vec2 vFieldXZ;",
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vViewDist = length(mvPosition.xyz);
         vGroundFade = clamp(
           (vViewDist - ${FADE_START.toFixed(1)}) / ${fadeSpan},
           0.0, 1.0);
         vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
         // sample the daylight field on the un-jittered lattice point so it reads off
         // the grid rather than the per-vertex jitter; jitter is under half a cell, so
         // rounding the local position to the cell grid recovers the lattice point.
         vec2 _lat = floor(position.xz / ${cell.toFixed(1)} + 0.5) * ${cell.toFixed(1)};
         vFieldXZ = (modelMatrix * vec4(_lat.x, 0.0, _lat.y, 1.0)).xz;`,
      );
    let frag = shader.fragmentShader.replace(
      "#include <common>",
      "#include <common>\nvarying float vGroundFade;\nvarying float vViewDist;\nvarying vec3 vWorldPos;\nvarying vec2 vFieldXZ;\nuniform vec2 uPlayer;\nuniform float uPlayerLift;\nuniform float uSkate;" +
        tpDecl +
        DAYLIGHT_FRAG_COMMON,
    );
    // Raking pool, additive after lighting (linear space, before the colorspace
    // encode). The world-space face normal comes from the position derivatives the
    // flat shading already relies on, oriented upward; the light sits a little above
    // the player, so facets tilted toward the pool brighten and the gradient sweeps
    // the dunes as the player moves. Falls to zero past the radius, so far fragments
    // pay only the derivative + a few ops.
    frag = frag.replace(
      "#include <opaque_fragment>",
      `#include <opaque_fragment>` +
        // day/night: multiply the lit sand (albedo + relief) by the daylight field (night
        // in the dark, day in the lit islands), sinking toward the night floor into the
        // distance dissolve so the far ground darkens to meet the black storm dome.
        applyDaylightGLSL("vFieldXZ", "vGroundFade", "vViewDist") +
        `{
         // these run AFTER the multiply so they survive the night: the player's own
         // light has to read for navigation through the mostly-night desert, and the
         // vortex eye is a glow meant to read in the unlit centre.
         vec3 wn = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
         if (wn.y < 0.0) wn = -wn;
         // the lamp rides PLAYER_LIGHT_HEIGHT above the player's feet AND uPlayerLift
         // higher again when airborne/hovering, so a jump lifts it off the sand (the
         // rake flattens and spreads); pdist stays xz so the pool footprint holds.
         vec3 toP = vec3(uPlayer.x - vWorldPos.x, ${PLAYER_LIGHT_HEIGHT.toFixed(1)} + uPlayerLift, uPlayer.y - vWorldPos.z);
         float pdist = length(toP.xz);
         // squared so the skirt tapers off well before the radius -- a soft pool around
         // the feet rather than a broad wash filling the grazing view.
         float fall = 1.0 - smoothstep(${PLAYER_LIGHT_INNER.toFixed(1)}, ${PLAYER_LIGHT_RADIUS.toFixed(1)}, pdist);
         fall *= fall;
         // mostly raked (so the dune relief reads) with a small floor so the flat ground
         // right underfoot still lights; the rake gradient is the terrain reveal.
         float rake = 0.3 + 0.7 * max(dot(wn, normalize(toP)), 0.0);
         // radial fraction (0 core -> 1 rim) drives an inner->outer colour gradient like
         // the teleporter pads; uSkate cross-fades the warm (yellow->orange) and the cool
         // (cyan->blue) palettes.
         float prad = clamp(pdist / ${PLAYER_LIGHT_RADIUS.toFixed(1)}, 0.0, 1.0);
         vec3 poolInner = mix(${PLAYER_WALK_INNER_RGB}, ${PLAYER_SKATE_INNER_RGB}, uSkate);
         vec3 poolOuter = mix(${PLAYER_WALK_OUTER_RGB}, ${PLAYER_SKATE_OUTER_RGB}, uSkate);
         vec3 poolCol = mix(poolInner, poolOuter, prad);
         gl_FragColor.rgb += poolCol * (${PLAYER_LIGHT_STRENGTH.toFixed(2)} * fall * rake);
       }` +
        // teleporter floor glow last, AFTER the daylight multiply, so the powered pad
        // holds steady instead of dimming as a shadow drifts over the plaza.
        tpGlow,
    );
    shader.fragmentShader = frag.replace(
      "#include <dithering_fragment>",
      "#include <dithering_fragment>\ngl_FragColor.a *= 1.0 - vGroundFade;",
    );
  };
}

// The height the ground grid draws at (wx, wz): each cell-quad triangulates along its
// (x0,z0)->(x0+C,z0+C) diagonal (matching the index pattern in buildGround), so
// reproduce that exact triangulated chord rather than a bilinear patch. Prop seating
// uses it via facetHeight to drop a book onto the surface the player actually sees,
// not the smooth field that surface only chords. (It also remains the slow-path
// fallback inside facetHeight.)
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
// Books seat on the surface the ground mesh actually DRAWS underfoot, not the smooth
// sampleHeight field that surface only chords. The two diverge two ways on a tight
// convex crest: the flat facet chords below the field (faceting), and the in-plane
// jitter shoves the facet's corners up to GROUND_JIT_FRAC*cell sideways, which on a
// steep face turns into a vertical offset of a couple of units. Seating on
// sampleHeight floats over both; seating on the un-jittered chord still floats over
// the second. So reconstruct the ground grid's jittered triangle exactly as
// buildGround draws it and drop the book onto that plane. No bias, so nothing
// legitimate is ever buried. The mesh is one uniform grid now, so this is exact
// everywhere the book stands, not just near the camera.
const FINEST = GROUND_CELL;
const FINEST_JIT = FINEST * GROUND_JIT_FRAC;
const _fv: THREE.Vector3[] = Array.from({ length: 16 }, () => new THREE.Vector3());

// One ground-grid vertex as drawn: the disk jitter of buildGround keyed on the world
// cell index (gx, gz), height sampled at the jittered position. The un-jittered
// corner sits at (gx*FINEST, gz*FINEST).
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

// Build the ground: one static grid of GROUND_CELL facets over the whole world,
// tessellated from the heightmap at load. Vertices use the world-cell disk jitter of
// finestVertex (so facetHeight reconstructs the drawn facet and books seat exactly),
// heights from sampleHeight, and the radial era colour with the crest/trough relief
// tint. flatShading + vertexColors give the low-poly Vane look; applyGroundMaterial
// adds the distance fade, the raking pool and the drifting daylight tint. Built once
// and never rebuilt: it sits still while the shared uPlayer/daylight uniforms move.
export function buildGround(
  uPlayer: PlayerUniform,
  uSkate: { value: number },
  daylight: DaylightUniforms,
): THREE.Mesh {
  const cell = GROUND_CELL;
  const g0 = -Math.round(GROUND_HALF / cell); // world-cell index of the (0,0) corner
  const N = -2 * g0; // cells across
  const stride = N + 1;
  const vcount = stride * stride;
  const positions = new Float32Array(vcount * 3);
  const colors = new Float32Array(vcount * 3);
  const v3 = new THREE.Vector3();
  const c = new THREE.Color();
  const ro = RELIEF_CELLS * cell; // neighbour offset for the crest/trough relief read
  for (let iz = 0; iz <= N; iz++) {
    for (let ix = 0; ix <= N; ix++) {
      const v = iz * stride + ix;
      finestVertex(g0 + ix, g0 + iz, v3); // jittered world xz + sampled height
      positions[v * 3] = v3.x;
      positions[v * 3 + 1] = v3.y;
      positions[v * 3 + 2] = v3.z;
      // crest/trough relief: this vertex's height minus its four neighbours ro out,
      // normalised to a signed ~[-1, 1], so crests read pale/warm and troughs cool.
      const relief =
        (v3.y -
          0.25 *
            (sampleHeight(v3.x - ro, v3.z) +
              sampleHeight(v3.x + ro, v3.z) +
              sampleHeight(v3.x, v3.z - ro) +
              sampleHeight(v3.x, v3.z + ro))) /
        (RELIEF_SCALE * ro);
      const eye = eyeMask(v3.x, v3.z, relief);
      groundColor(v3.x, v3.z, c, relief, eye);
      colors[v * 3] = c.r;
      colors[v * 3 + 1] = c.g;
      colors[v * 3 + 2] = c.b;
    }
  }
  // two triangles per cell, split on the a->d diagonal exactly as facetHeight expects.
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
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    flatShading: true,
  });
  applyGroundMaterial(mat, cell, uPlayer, uSkate, daylight);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.frustumCulled = false; // one big mesh always wrapping the camera
  return mesh;
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
