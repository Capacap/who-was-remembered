import * as THREE from "three";
import { CLOUD_FRAG_COMMON, cloudApplyGLSL, type CloudUniforms } from "./clouds";

// --- terrain ----------------------------------------------------------------
// The world is a near-flat desert and the dunes are its relief, not a texture
// laid over a hill. The present (origin) stays calm and level where the books
// are densest; only a whisper of a central rise gives spawn a faint vantage
// before it eases to the desert floor and a transverse dune field takes over.
//
// This lives in the renderer, not the pipeline. The vertical axis carries no data
// (time and longitude are the horizontal x/z), so terrain is decoration by the
// three-tier rule. The shape is a baked heightmap raster (stage9), and sampleHeight
// is the single elevation source: the ground mesh tessellates it, the player's feet
// read it, and books seat on facetHeight (the facet the ground mesh actually draws),
// so the visible ground and everything on it agree. The analytic getGroundHeight
// below is now only the pre-heightmap-load fallback.

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
// One smooth era ramp across the whole map: pale summit (the present, at the
// centre) -> sand at the midpoint -> grey deep-past floor at the outer edge.
// Spans [0, ERA_GRADIENT_R]; recency literally lighting the map, now a single
// continuous gradient rather than a pale core with a flat sand band and a
// separate grey rim. Mirrors world.json R_MAX (the deep-past edge).
const ERA_GRADIENT_R = 7100;
const COLOR_WAVELENGTH = 900; // patch-noise scale for the painted wobble

// The vortex eye: a pale glacial teal gathered at the centre, the cold light of
// the present where the knowledge piles up. It reads as EMISSIVE (added after
// lighting), not albedo: the flat vantage centre catches almost none of the
// raking sun, so an albedo tint there is crushed to black no matter how light
// the colour, while an additive glow is lighting-independent and reads in the
// dark. The coverage never makes a clean radial ring: a noise field (EYE_NOISE_*)
// fingers the boundary in and out even across the flat calm zone where there are
// no dunes to follow, a relief push (EYE_RELIEF_PUSH) lets it run down the dune
// troughs where the terrain has relief, and EYE_VALLEY pools the glow in the
// troughs and recedes it off the crests so it interfingers with the rising dunes.
const COLOR_EYE = new THREE.Color(0xe8f3f1); // near-white with a whisper of cool: otherworldly bleached sand
const EYE_R0 = 260; // full eye colour inside here (the vantage pocket)
const EYE_R1 = 1900; // ... gone by here, reaching out into the rising dunes
const EYE_ALBEDO_STRENGTH = 0.85; // albedo lerp toward the eye colour (only shows where lit)
const EYE_EMISSIVE_STRENGTH = 0.6; // additive glow strength (carries the eye in the unlit centre)
const EYE_RELIEF_PUSH = 320; // world units the boundary shifts per unit relief:
// troughs (relief < 0) pull it inward (more eye), crests (relief > 0) push it out
const EYE_NOISE_AMP = 680; // world units the boundary wanders by noise (breaks the ring)
const EYE_NOISE_SCALE = 760; // coarse noise wavelength; a finer octave rides on top
const EYE_VALLEY = 1.3; // pool the colour in the dune troughs, recede off the crests:
// scales the mask by (1 - EYE_VALLEY * relief), so crests dim and troughs lift
const _eye = COLOR_EYE.clone().convertSRGBToLinear();
const EYE_EMISSIVE_RGB = `vec3(${_eye.r.toFixed(4)}, ${_eye.g.toFixed(4)}, ${_eye.b.toFixed(4)})`;

// Crest/trough relief tint, layered on the radial base: crests read scoured pale
// and a touch warm, troughs cooler and darker, so the dunes carry colour and not
// just shading. The signal is the vertex height minus its neighbours RELIEF_CELLS out
// on the ground grid, so the colour tracks the relief the mesh actually draws. All
// four are eyeball knobs.
const RELIEF_CELLS = 3; // neighbour offset in cells: the relief's read wavelength
const RELIEF_SCALE = 0.25; // slope-difference that reaches the full crest/trough tint
const CREST_LIGHT = 0.15; // crest lightens / trough darkens (the dominant read)
const CREST_SAT = 0.07; // crest bleaches / trough deepens
const CREST_HUE = 0.012; // crest warms / trough cools
// The albedo crest/trough tint above is crushed by the raking dusk sun (lighting
// already owns light/dark on the flat-shaded facets, so an albedo shift can't
// compete and the dunes read as one sand colour). So the crest/trough COLOUR is
// carried by an additive emissive term, lighting-independent like the vortex eye:
// crests catch a warm rim, troughs pool a deep red glow. Keyed to the same signed
// relief k, passed per-vertex as aRelief. Two independent strengths so the warm rim
// and the red pool tune separately; zero a strength to drop that half of the tint.
const CREST_EMIS = new THREE.Color(0xffc890); // warm rim glow on crests
const TROUGH_EMIS = new THREE.Color(0xcc2800); // deep red glow pooled in troughs
const CREST_EMIS_STRENGTH = 0.18;
const TROUGH_EMIS_STRENGTH = 0.2;
const _crestE = CREST_EMIS.clone().convertSRGBToLinear();
const _troughE = TROUGH_EMIS.clone().convertSRGBToLinear();
const CREST_EMIS_RGB = `vec3(${_crestE.r.toFixed(4)}, ${_crestE.g.toFixed(4)}, ${_crestE.b.toFixed(4)})`;
const TROUGH_EMIS_RGB = `vec3(${_troughE.r.toFixed(4)}, ${_troughE.g.toFixed(4)}, ${_troughE.b.toFixed(4)})`;

// Time rings: radius is time, so a gentle ripple in lightness (and a hair of
// warm/cool) keyed to radius makes the map read as concentric growth rings /
// strata, a tree- or mountain-cross-section of deep time. Keyed straight off
// radius, not the exact year mapping: the bands are decoration for the narrative,
// not a readable century scale, so even spacing is enough (with RADIUS_ALPHA = 1
// it lands on centuries anyway). Eyeball knobs.
const RING_SPACING = 512; // world units between rings (~a century at alpha = 1)
const RING_LIGHT = 0.03; // lightness swing across a ring (the dominant read)
const RING_HUE = 0.004; // warm/cool swing across a ring (subtle)

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

// Player raking light: a warm pool that follows the player and rakes across the
// dune facets near them, so the near ground reads as reactive (the moving
// gradient on the dunes). It is additive in WORLD space, lit by each facet's own
// world-derived normal (cross of the position derivatives the flat shading already
// computes), so it tilts with the dune faces rather than washing them flat. The
// centre (uPlayer) is shared with the book glow; this radius is its own. Colour is
// baked into the GLSL as a linear literal, like the distance fade. Eyeball knobs.
const PLAYER_LIGHT_COLOR = new THREE.Color(0xffc89c); // warm pool
const PLAYER_LIGHT_RADIUS = 42; // raking fades out by this horizontal distance
const PLAYER_LIGHT_INNER = 2.0; // full reach within this
const PLAYER_LIGHT_STRENGTH = 0.55; // additive intensity at the pool centre
const PLAYER_LIGHT_HEIGHT = 6.0; // light's height over the player; lower = more grazing
const _pl = PLAYER_LIGHT_COLOR.clone().convertSRGBToLinear();
const PLAYER_LIGHT_RGB = `vec3(${_pl.r.toFixed(4)}, ${_pl.g.toFixed(4)}, ${_pl.b.toFixed(4)})`;

// Skate glow: a cool pool that blooms under the player while hovering, a second
// signal (beyond the small hover lift) that Shift is doing something. It shares the
// raking pool's centre and normal but is wider and mostly flat (lightly raked), so it
// reads as the ground glowing beneath you rather than a directional light. Ramped by
// uSkate (0..1) in the loop so it eases in/out with the mode. Eyeball knobs, baked as
// GLSL literals like the warm pool; rebuild to retune.
const SKATE_GLOW_COLOR = new THREE.Color(0x4a86ff); // cool blue pool
const SKATE_GLOW_RADIUS = 16; // a tight pool right under the player
const SKATE_GLOW_INNER = 2.0;
const SKATE_GLOW_STRENGTH = 0.8; // additive intensity at centre, times uSkate
const _sg = SKATE_GLOW_COLOR.clone().convertSRGBToLinear();
const SKATE_GLOW_RGB = `vec3(${_sg.r.toFixed(4)}, ${_sg.g.toFixed(4)}, ${_sg.b.toFixed(4)})`;

// Teleporter floor glow: a blue emissive disc pooled on each plaza, the near-field
// "stand here" marker. The beam fades out within TP_BEAM_FADE_NEAR so it never clips
// the camera up close, which is exactly where it stops telling you where to stand;
// this glow takes over there, so the column and the pad hand off as one beacon. Same
// blue as the beam (main.ts TP_BEAM_COLOR). Additive emissive like the vortex eye, so
// it reads in the barely-lit centre instead of crushing dark, and because it IS the
// ground it can never look superimposed (the stone ring it replaced did, being a warm
// Lambert prop dropped onto the cool emissive centre). Evaluated per-fragment over the
// teleporter positions, not per-vertex: the disc is smaller than a GROUND_CELL facet,
// so a vertex attribute would light a single triangle. A slow pulse off uCloudTime
// reads as powered. Eyeball knobs; tune against the vortex teal, which the innermost
// teleporters sit inside (a blue pad there has to fight the near-white eye glow).
const TP_GLOW_INNER = new THREE.Color(0x3df0ff); // cyan at the centre of the pad
const TP_GLOW_OUTER = new THREE.Color(0x2f6cff); // blue at the rim, matches main.ts TP_BEAM_COLOR
const TP_GLOW_RADIUS = 9; // disc footprint, world units; tracks main.ts TP_ENTER_RADIUS (the travel zone)
const TP_GLOW_FALLOFF = 1.6; // intensity exponent over the radius; >1 keeps a bright core with a soft skirt
const TP_GLOW_STRENGTH = 0.85; // additive intensity at the centre
const TP_GLOW_PULSE = 0.18; // pulse depth as a fraction of strength (0 = steady)
const TP_GLOW_PULSE_SPEED = 0.6; // pulse rate against the drifting uCloudTime
// Proximity ramp off the player position: far away the pad is a dim, plain-blue
// beacon; on approach it brightens and the cyan core emerges. Keep in sync with
// main.ts TP_BEAM_APPROACH_* so beam and pad ramp together.
const TP_GLOW_APPROACH_NEAR = 70; // player distance (world u) at which the pad reaches full intensity
const TP_GLOW_APPROACH_FAR = 260; // beyond this the pad sits at the dim far level
const TP_GLOW_FAR_LEVEL = 0.3; // intensity multiplier when far (0..1)
const _tgi = TP_GLOW_INNER.clone().convertSRGBToLinear();
const _tgo = TP_GLOW_OUTER.clone().convertSRGBToLinear();
const TP_GLOW_INNER_RGB = `vec3(${_tgi.r.toFixed(4)}, ${_tgi.g.toFixed(4)}, ${_tgi.b.toFixed(4)})`;
const TP_GLOW_OUTER_RGB = `vec3(${_tgo.r.toFixed(4)}, ${_tgo.g.toFixed(4)}, ${_tgo.b.toFixed(4)})`;

// A live uniform carrying the player's world-xz position, shared between the
// ground's raking light and the books' proximity glow so both pools share a centre.
export interface PlayerUniform {
  value: THREE.Vector2;
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
// narrative, a low-frequency painted wobble, and a crest/trough relief tint.
// buildGround tints each vertex with this from world position; it stays per-vertex on
// the CPU by design (no texture, no baked raster), so the low-poly vertex-colour look
// holds. `relief` is the signed, already-normalised local relief (+ on crests, - in
// hollows) the caller reads from the grid; 0 leaves the base untouched.
// The vortex-eye coverage at a world point, in [0, 1]: 1 at the gathered centre,
// 0 past the band. The boundary radius is perturbed by two octaves of noise (so
// the edge fingers organically rather than reading as a clean ring) and by local
// relief (so it runs down the dune troughs where the terrain has relief to
// follow). buildGround reads this once per vertex for both the albedo tint and
// the emissive glow, so the two always agree.
export function eyeMask(x: number, z: number, relief: number): number {
  const r = Math.hypot(x, z);
  const noise =
    perlin(x / EYE_NOISE_SCALE, z / EYE_NOISE_SCALE) +
    0.5 * perlin(x / (EYE_NOISE_SCALE * 0.4), z / (EYE_NOISE_SCALE * 0.4));
  const rEff = r + EYE_RELIEF_PUSH * relief + EYE_NOISE_AMP * noise;
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
  const r = Math.hypot(x, z);
  const t = Math.min(1, r / ERA_GRADIENT_R);
  if (t < 0.5) out.copy(COLOR_PALE).lerp(COLOR_SAND, t * 2);
  else out.copy(COLOR_SAND).lerp(COLOR_GREY, (t - 0.5) * 2);
  // Tint the albedo toward the eye colour (the glow that actually carries it in
  // the dark centre is the emissive term, added in applyGroundMaterial). Folded
  // under the HSL offsets below so the eye still picks up crest/trough shading.
  if (eye > 0) out.lerp(COLOR_EYE, eye * EYE_ALBEDO_STRENGTH);
  const tone = perlin(x / COLOR_WAVELENGTH, z / COLOR_WAVELENGTH); // [-1, 1]
  // time rings: a smooth ripple, one cycle per RING_SPACING of radius.
  const ring = Math.cos((r / RING_SPACING) * Math.PI * 2);
  // crests (k > 0) warm, bleach and lighten; troughs (k < 0) cool, deepen and
  // darken. Hue/sat shift against k's sign, lightness with it.
  const k = relief < -1 ? -1 : relief > 1 ? 1 : relief;
  // The painted wobble, the time rings and the relief tint are three HSL offsets;
  // fold them into one getHSL/setHSL roundtrip rather than three (this runs per vertex
  // across the whole ground build, so the two saved RGB<->HSL conversions matter).
  out.getHSL(_hsl);
  _hsl.h += tone * 0.01 + ring * RING_HUE - k * CREST_HUE;
  _hsl.s += tone * 0.03 - k * CREST_SAT;
  _hsl.l += tone * 0.04 + ring * RING_LIGHT + k * CREST_LIGHT;
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
// surface the ground mesh draws, not the analytic field that diverges from it. eps
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

// Inject the ground material edits, shared by the one static ground mesh: the
// camera-distance opacity fade (so the world dissolves circularly into the dome at
// the far edge, no square plate) and the additive raking pool that follows the
// player, then the drifting cloud-shadow multiply. All are per-fragment in world
// space, so they ride on the static mesh unchanged from the clipmap days; only the
// per-level discard hole and its depth machinery are gone with the clipmap.
function applyGroundMaterial(
  mat: THREE.MeshLambertMaterial,
  cell: number,
  uPlayer: PlayerUniform,
  uSkate: { value: number },
  cloud: CloudUniforms,
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
         float tpPulse = 1.0 - ${TP_GLOW_PULSE.toFixed(2)} * (0.5 - 0.5 * cos(uCloudTime * ${TP_GLOW_PULSE_SPEED.toFixed(3)}));
         gl_FragColor.rgb += tpGlow * tpCol * (${TP_GLOW_STRENGTH.toFixed(2)} * tpPulse * tpProx);
       }`
    : "";
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uPlayer = uPlayer;
    shader.uniforms.uSkate = uSkate;
    shader.uniforms.uClouds = cloud.uClouds;
    shader.uniforms.uCloudTime = cloud.uCloudTime;
    shader.uniforms.uCloudMix = cloud.uCloudMix;
    if (tpPos.length) shader.uniforms.uTpPos = { value: tpPos };
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nattribute float aEye;\nattribute float aRelief;\nvarying float vEye;\nvarying float vRelief;\nvarying float vGroundFade;\nvarying float vCloudDist;\nvarying vec3 vWorldPos;\nvarying vec2 vCloudXZ;",
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vEye = aEye;
         vRelief = aRelief;
         vCloudDist = length(mvPosition.xyz);
         vGroundFade = clamp(
           (vCloudDist - ${FADE_START.toFixed(1)}) / ${fadeSpan},
           0.0, 1.0);
         vWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
         // sample the cloud on the un-jittered lattice point so the shadow reads off
         // the grid rather than the per-vertex jitter; jitter is under half a cell, so
         // rounding the local position to the cell grid recovers the lattice point.
         vec2 _lat = floor(position.xz / ${cell.toFixed(1)} + 0.5) * ${cell.toFixed(1)};
         vCloudXZ = (modelMatrix * vec4(_lat.x, 0.0, _lat.y, 1.0)).xz;`,
      );
    let frag = shader.fragmentShader.replace(
      "#include <common>",
      "#include <common>\nvarying float vEye;\nvarying float vRelief;\nvarying float vGroundFade;\nvarying float vCloudDist;\nvarying vec3 vWorldPos;\nvarying vec2 vCloudXZ;\nuniform vec2 uPlayer;\nuniform float uSkate;" +
        tpDecl +
        CLOUD_FRAG_COMMON,
    );
    // Raking pool, additive after lighting (linear space, before the colorspace
    // encode). The world-space face normal comes from the position derivatives the
    // flat shading already relies on, oriented upward; the light sits a little above
    // the player, so facets tilted toward the pool brighten and the gradient sweeps
    // the dunes as the player moves. Falls to zero past the radius, so far fragments
    // pay only the derivative + a few ops.
    frag = frag.replace(
      "#include <opaque_fragment>",
      `#include <opaque_fragment>
       {
         vec3 wn = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
         if (wn.y < 0.0) wn = -wn;
         vec3 toP = vec3(uPlayer.x - vWorldPos.x, ${PLAYER_LIGHT_HEIGHT.toFixed(1)}, uPlayer.y - vWorldPos.z);
         float pdist = length(toP.xz);
         float fall = 1.0 - smoothstep(${PLAYER_LIGHT_INNER.toFixed(1)}, ${PLAYER_LIGHT_RADIUS.toFixed(1)}, pdist);
         float rake = max(dot(wn, normalize(toP)), 0.0);
         gl_FragColor.rgb += ${PLAYER_LIGHT_RGB} * (${PLAYER_LIGHT_STRENGTH.toFixed(2)} * fall * rake);
         // cool pool while skating: wider, mostly-radial blue glow under the player,
         // ramped by uSkate so it blooms in as Shift engages. Lightly raked so the
         // ground reads as glowing beneath you rather than lit by a second sun.
         float bfall = 1.0 - smoothstep(${SKATE_GLOW_INNER.toFixed(1)}, ${SKATE_GLOW_RADIUS.toFixed(1)}, pdist);
         gl_FragColor.rgb += uSkate * ${SKATE_GLOW_RGB} * (${SKATE_GLOW_STRENGTH.toFixed(2)} * bfall * (0.45 + 0.55 * rake));
         // vortex eye: additive glow (colour x mask), lighting-independent so it
         // reads in the barely-lit centre where an albedo tint would be crushed.
         gl_FragColor.rgb += vEye * ${EYE_EMISSIVE_RGB} * ${EYE_EMISSIVE_STRENGTH.toFixed(2)};
         // crest/trough colour: same emissive trick, keyed to signed relief. Crests
         // (vRelief > 0) catch a warm rim, troughs (vRelief < 0) pool a deep red glow,
         // so the dunes carry colour under the raking sun that crushes the albedo tint.
         gl_FragColor.rgb += ${CREST_EMIS_RGB} * (${CREST_EMIS_STRENGTH.toFixed(2)} * max(vRelief, 0.0));
         gl_FragColor.rgb += ${TROUGH_EMIS_RGB} * (${TROUGH_EMIS_STRENGTH.toFixed(2)} * max(-vRelief, 0.0));
       }` +
        // drifting cloud shadow over the dunes, sinking toward the night floor into
        // the distance dissolve so the far ground darkens to meet the black storm dome.
        cloudApplyGLSL("vCloudXZ", "vGroundFade", "vCloudDist") +
        // teleporter floor glow last, AFTER the cloud multiply, so the powered pad
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
// adds the distance fade, the raking pool and the drifting cloud shadow. Built once
// and never rebuilt: it sits still while the shared uPlayer/cloud uniforms move.
export function buildGround(
  uPlayer: PlayerUniform,
  uSkate: { value: number },
  cloud: CloudUniforms,
): THREE.Mesh {
  const cell = GROUND_CELL;
  const g0 = -Math.round(GROUND_HALF / cell); // world-cell index of the (0,0) corner
  const N = -2 * g0; // cells across
  const stride = N + 1;
  const vcount = stride * stride;
  const positions = new Float32Array(vcount * 3);
  const colors = new Float32Array(vcount * 3);
  const eyes = new Float32Array(vcount); // per-vertex eye coverage, for the emissive glow
  const reliefs = new Float32Array(vcount); // per-vertex signed relief k, for the crest/trough emissive
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
      eyes[v] = eye;
      reliefs[v] = relief < -1 ? -1 : relief > 1 ? 1 : relief; // clamped k for the emissive
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
  geom.setAttribute("aEye", new THREE.BufferAttribute(eyes, 1));
  geom.setAttribute("aRelief", new THREE.BufferAttribute(reliefs, 1));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    flatShading: true,
  });
  applyGroundMaterial(mat, cell, uPlayer, uSkate, cloud);
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
