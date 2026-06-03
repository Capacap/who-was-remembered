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
  buildGround,
  applyDistanceFade,
  type PlayerUniform,
} from "./terrain";
import { buildSky } from "./sky";
import {
  buildClouds,
  applyCloudShadow,
  CLOUD_FRAG_COMMON,
  cloudApplyGLSL,
  DAY_GLSL,
  type CloudUniforms,
} from "./clouds";

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
// Grounded movement is velocity-based with acceleration: two modes that share one
// continuous feel, so there's no toggle the player has to discover.
//   walk  - default; high accel AND high decel, low top speed. Crisp and precise for
//           reading: you start and stop almost instantly, the deliberate slow pace.
//   skate - hold Shift; high accel but LOW decel and a high top speed. You build
//           momentum and coast, so crossing the empty antiquity voids is a long glide
//           rather than a dead walk. Release Shift and you drop to walk's high decel,
//           braking to a stop quickly. Book picking is suppressed while skating so the
//           prompts don't strobe as you blow past the field.
// Decel is a RATE applied to the live velocity, never a clamp: releasing Shift at full
// skate speed bleeds the speed off smoothly from wherever you were, it doesn't snap to
// the walk cap. Top speed only limits what acceleration may ADD. Numbers are units/sec
// or units/sec^2 and live on one object so they can be tuned live from the devtools
// console (window.MOVE) without a rebuild.
const MOVE = {
  walk: { max: 5, accel: 30, decel: 50 },
  skate: { max: 80, accel: 30, decel: 15 },
  // the eye lifts this much the instant you start skating: a small, deliberate cue
  // that the button did something, well short of an actual fly-height float. The
  // ground-follow filter eases it in and out, so it reads as rising onto the glide.
  hoverLift: 0.8,
  // ground-follow stiffness: a fixed time-constant filter on eye height. At a walk the
  // terrain target barely moves so the feet stay planted; at skate speed the target
  // changes fast and the same filter smooths the dune bumps that would otherwise jolt
  // the camera (and the stomach). Higher = stiffer.
  followK: 12,
  // circular world bound, set from world.R_MAX once it loads. Past SOFT the outward
  // velocity is shed so you ease along the edge; HARD is the hard clamp. Both sit
  // inside the terrain mesh and the distance fog, so you coast to a stop in haze and
  // never see the ground run out.
  boundSoft: Infinity,
  boundHard: Infinity,
};
(window as unknown as { MOVE: typeof MOVE }).MOVE = MOVE;

// Dev-only free flight (F). Kept as a tool: it clips terrain and breaks the grounded
// premise, so it's off the player HUD, but it's too useful for inspection to cut.
const FLY_SPEED = 60;
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

// Proximity glow: the field rests dark and a book lights up to its full colour
// (plus a soft additive bloom) as the player comes within range, so walking the
// disc carries a travelling pool of light with you and the world reads as
// reactive. The radius is the shared "near the player" notion the terrain
// lighting can later ride on too. A book is fully lit within GLOW_INNER, dark
// beyond GLOW_RADIUS, and ramps between. GLOW_REST_DIM is how dark the resting
// field is (0 = near-black, 1 = full colour always; lower kills the cross-disc
// geo-hue read in exchange for a starker reveal). GLOW_BOOST is the extra
// additive glow at the centre of the pool. Eyeball knobs.
const GLOW_RADIUS = 36; // books dark beyond this horizontal distance from the player
const GLOW_INNER = 2; // tight full-brightness core at the player's feet; smooth taper to GLOW_RADIUS
const GLOW_REST_DIM = 0.12; // resting brightness of a near book outside the pool (0 = black)
const GLOW_REST_FAR = 0.4; // resting brightness once distance-faded; higher than REST_DIM
//   so far books are dim dusty specks, not max-contrast black confetti. The dark
//   specks on bright sand were the worst of the sub-pixel flicker, so lifting the
//   far floor trades a little of the stark dark field for a calmer horizon.
const GLOW_BOOST = 0.6; // additive bloom at the pool centre

// Self-emission so a book is a coloured speck even where the night lighting and the
// proximity dim would otherwise lose it in the dark (the whole field had sunk into
// the black storm scene). EMISSIVE is a fraction of the book's own geo hue added as
// true self-light AFTER the cloud tint, so it pierces the storm shadow rather than
// being multiplied to black under it (a speck bursting through the dark, like the
// sky). EMISSIVE_NEAR is the extra emission the proximity pool adds, so a book by the
// player burns brighter than the distant field (see applyProximityGlow).
const GLOW_EMISSIVE = 0.14; // base self-glow as a fraction of the book's hue
const GLOW_EMISSIVE_NEAR = 0.5; // extra emission at the pool centre

// The field is never quite still, so it reads as alive rather than as plotted data.
// The life is MOTION, not a brightness flicker (scaling the emissive made dim-hued
// books barely move while bright ones winked hard, an inconsistent read). Two effects
// layer onto the player's proximity pool, both driven by the shared uCloudTime (seconds)
// so they stay in step with the drifting weather:
//   - BOB: each book hovers gently above its seat on its OWN hashed phase, so the field
//     shimmers with uncoordinated motion rather than a marching swell. Strictly positive
//     (0..AMP) so a book never dips below the sand, where it would clip and read as the
//     dark "waves" a signed swell produced.
//   - REVEAL: the sun-reveal. Where a daylight break drifts over a book it lights up in
//     step with the sand it stands on: a second tap of the SAME cloudShadow field the
//     ground reads for its day/night tint, cast in the SAME warm DAY colour, so the two
//     are locked to one sky. The ground swings its whole albedo from near-black to bright
//     daylight, so to keep the books from looking flat by comparison the reveal is strong
//     (a book in full sun emits close to its own hue, warmed). It is gated by cloudShadow,
//     so at night it falls to zero and only the steady uEmissive floor remains, the floor
//     that keeps books visible in the dark in the first place.
const BOB_AMP = 0.07; // world units a book hovers above its seat (0..AMP, never below)
const BOB_SPEED = 0.5; // rad/s; period ~12s
const GLOW_REVEAL = 2.0; // sun-reveal strength: book self-light at full daylight

interface GlowUniforms {
  uPlayer: PlayerUniform;
  uGlowRadius: { value: number };
  uGlowInner: { value: number };
  uRestDim: { value: number };
  uRestFar: { value: number };
  uGlowBoost: { value: number };
  uEmissive: { value: number };
  uEmissiveNear: { value: number };
}

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

// Light a book up by proximity to the player. Chains onto whatever onBeforeCompile
// is already set (call AFTER applyDistanceFade / applyPageMask). The book's world
// xz is carried to the fragment shader; the patch then dims the final lit colour
// toward uRestDim with distance and adds a soft additive glow inside the pool.
// Works on the box base, the mid and the near tiers alike (all InstancedMesh, so
// instanceMatrix is in scope), so a book reveals identically whichever LOD draws
// it. The dim/glow run on gl_FragColor after lighting (at <opaque_fragment>),
// before the distance-fade alpha at <dithering_fragment>, so a far book both dims
// and fades. The shared uniforms object is assigned by reference to every patched
// material, so updating uPlayer once per frame moves the pool on all of them.
function applyProximityGlow(
  mat: THREE.Material,
  uni: GlowUniforms,
  cloud: CloudUniforms,
): void {
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader, renderer) => {
    if (prev) prev.call(mat, shader, renderer);
    shader.uniforms.uPlayer = uni.uPlayer;
    shader.uniforms.uGlowRadius = uni.uGlowRadius;
    shader.uniforms.uGlowInner = uni.uGlowInner;
    shader.uniforms.uRestDim = uni.uRestDim;
    shader.uniforms.uRestFar = uni.uRestFar;
    shader.uniforms.uGlowBoost = uni.uGlowBoost;
    shader.uniforms.uEmissive = uni.uEmissive;
    shader.uniforms.uEmissiveNear = uni.uEmissiveNear;
    shader.uniforms.uClouds = cloud.uClouds;
    shader.uniforms.uCloudTime = cloud.uCloudTime;
    shader.uniforms.uCloudMix = cloud.uCloudMix;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vGlowXZ;\nvarying float vCloudDist;\n" +
          "uniform float uCloudTime;\n" +
          // Dave Hoskins hash12: scales the coord down before any fract, so it keeps
          // precision out at the disc's ~7000u edge where fract(sin(dot)*43758) aliases
          // adjacent books to the same value. Drives each book's own bob phase. Returns 0..1.
          "float bookHash(vec2 p){ vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }",
      )
      // Bob each book. project_vertex has already set gl_Position from mvPosition; recompute
      // it in world space so the motion is rigid regardless of the per-book spine scale baked
      // into instanceMatrix (an object-space offset would scale with thickness). Phase keys
      // off the instance ORIGIN, not the per-vertex position, so a whole book moves as one.
      //   _bob:  per-book hover, own hashed phase, 0..AMP so it only ever rises.
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vec4 _O = modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
         vec4 _wpos = modelMatrix * instanceMatrix * vec4(transformed, 1.0);
         vGlowXZ = _wpos.xz;
         float _bob = (0.5 + 0.5 * sin(uCloudTime * ${BOB_SPEED.toFixed(3)} + bookHash(_O.xz) * 6.2831853)) * ${BOB_AMP.toFixed(3)};
         _wpos.y += _bob;
         gl_Position = projectionMatrix * viewMatrix * _wpos;
         vCloudDist = length(mvPosition.xyz);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vGlowXZ;\nvarying float vCloudDist;\nuniform vec2 uPlayer;\n" +
          "uniform float uGlowRadius;\nuniform float uGlowInner;\n" +
          "uniform float uRestDim;\nuniform float uRestFar;\nuniform float uGlowBoost;\n" +
          "uniform float uEmissive;\nuniform float uEmissiveNear;\n" +
          CLOUD_FRAG_COMMON,
      )
      .replace(
        "#include <opaque_fragment>",
        `#include <opaque_fragment>
         float glowD = distance(vGlowXZ, uPlayer);
         float glow = 1.0 - smoothstep(uGlowInner, uGlowRadius, glowD);
         // the resting floor lifts toward uRestFar as the book distance-fades (vGroundFade,
         // set by applyDistanceFade upstream), so far specks lose contrast and stop crawling.
         float restFloor = mix(uRestDim, uRestFar, vGroundFade);
         gl_FragColor.rgb *= mix(restFloor, 1.0, glow);
         gl_FragColor.rgb += gl_FragColor.rgb * glow * uGlowBoost;` +
          // the same drifting cloud shadow the ground takes, so a book darkens with
          // the sand it stands in; faded out into the distance dissolve (vGroundFade).
          cloudApplyGLSL("vGlowXZ", "vGroundFade", "vCloudDist") +
          // self-emission, added AFTER the cloud tint so it is true self-light: it
          // survives cloud shadow (a book is a speck bursting through the dark, like the
          // sky) instead of being multiplied to black under the storm. The steady floor
          // (uEmissive) plus the proximity pool (uEmissiveNear) carry the book's own hue.
          // The sun-reveal rides on top: where the SAME cloudShadow field that lights the
          // ground reads daylight, the book emits its hue warmed by the SAME DAY colour
          // the ground tints to, so a passing sun patch lights book and sand together. It
          // falls to zero at night, leaving only the floor that keeps books visible there.
          `gl_FragColor.rgb += diffuseColor.rgb * (uEmissive + glow * uEmissiveNear);
           float _sun = cloudShadow(vGlowXZ, vCloudDist);
           gl_FragColor.rgb += diffuseColor.rgb * ${DAY_GLSL} * (_sun * ${GLOW_REVEAL.toFixed(3)});`,
      );
  };
}

// Target footprint diameter of the stone circle in world units. The authored
// model is ~4.9u across; scaled up to this so the player can walk inside the ring
// rather than step over it, while still sitting well within the 28u flat plaza
// core the terrain levels around each monument (terrain FLATTEN_R = 14).
const STONE_CIRCLE_DIAMETER = 9;

// Bake the authored stone circle (stone_circle_LOD00) into a ground-ready geometry:
// uniform-scale so its widest footprint axis is STONE_CIRCLE_DIAMETER, recentre x/z
// on the origin, and drop its base to y = 0 so it rests on the plaza when placed at
// ground height. The model has 643 tris and only 26 are ever drawn, so the finer
// LODs aren't worth the swap bookkeeping; LOD00 is used at every distance.
async function loadStoneCircle(url: string): Promise<THREE.BufferGeometry> {
  const gltf = await gltfLoader.loadAsync(url);
  const src = gltf.scene.getObjectByName("stone_circle_LOD00") as
    | THREE.Mesh
    | undefined;
  if (!src?.isMesh) throw new Error(`no 'stone_circle_LOD00' mesh in ${url}`);
  const g = (src.geometry as THREE.BufferGeometry).clone();
  g.computeBoundingBox();
  let bb = g.boundingBox!;
  const k = STONE_CIRCLE_DIAMETER / Math.max(bb.max.x - bb.min.x, bb.max.z - bb.min.z);
  g.scale(k, k, k);
  g.computeBoundingBox();
  bb = g.boundingBox!;
  // recentre the footprint on the origin; sink the base to y = 0 so the stones
  // stand on the ground rather than half-buried or floating when seated.
  g.translate(-(bb.min.x + bb.max.x) / 2, -bb.min.y, -(bb.min.z + bb.max.z) / 2);
  return g;
}

// --- decorative heads --------------------------------------------------------
// Three sculpted head variants from heads.glb, scattered half-buried in the sand
// with faces to the sky (Stage 10 finds the clear spots; decorations.json carries
// per-head position and jitter). Pure decoration, no data; they thicken the
// dream-logic without disturbing a single book. Plain stone material; the basemesh
// ships no material and its COLOR_n/TEXCOORD_n layers are ignored.
// World units along the sculpted up axis (crown to chin). Bounded by the camera,
// not taste: there is no collision, so a head tall enough to reach eye height
// (EYE_HEIGHT ~2.4u) means the player walks INTO the face and the view fills with
// the inside of the skull. The heads are nearly as deep as tall (measured z/y
// ~0.93), so laid on their back the crown rises ~0.93*HEAD_HEIGHT*scale*(1-sink).
// At 2.0 the tallest possible head (scale 1.4, sink 0.30) tops out ~1.8u, below the
// eye, so the camera always glides over the crown instead of into it. A head then
// reads as a boulder among the books (~3-7 book-lengths), never a colossus.
const HEAD_HEIGHT = 2.0;
//   Stage 10's HEAD_RADIUS (the open-sand a head needs) tracks ~half of this.
// each variant's three LODs (full, mid, coarse), the same ladder the books use.
const HEAD_VARIANTS = [
  ["head01_LOD00", "head01_LOD01", "head01_LOD02"],
  ["head02_LOD00", "head02_LOD01", "head02_LOD02"],
  ["head03_LOD00", "head03_LOD01", "head03_LOD02"],
];
// Heads are instanced like the books, NOT one THREE.LOD object each: at 10k heads
// that was 10k scene nodes, 10k update() calls and a draw call per visible head,
// which tanked the frame. Instead every head draws from a single static coarse
// InstancedMesh per variant (the bulk, always on, ~104 tris each), and only the
// handful within HEAD_R_FULL of the camera also draw from a small full-detail pool
// that hides the coarse mesh inside it. Heads sit ~60-125u apart, so that pool is
// nearly always near-empty. Coarse is shrunk by HEAD_COARSE_PROXY so the full mesh
// cleanly occludes it where both draw (the book box uses the same trick).
const HEAD_R_FULL = 40; // full detail within this radius; coarse (104 tris) beyond
const HEAD_NEAR_CAP = 48; // full-detail instances per variant (far more than ever in range)
const HEAD_REBUILD = 20; // refill the near pool only after the camera moves this far
const HEAD_COARSE_PROXY = 0.9; // shrink coarse so the full mesh occludes it, no z-fight

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
  cloud: CloudUniforms,
): {
  group: THREE.Group;
  update: (camera: THREE.Camera) => void;
} {
  const mat = new THREE.MeshLambertMaterial({ color: 0xcbbfa8 }); // sandstone
  // Fade the heads into the dome with distance, the SAME treatment the books get.
  // Without it a head stays a full-brightness opaque speck all the way to the
  // horizon, and a few-pixel high-contrast opaque object crawls against the pixel
  // grid as the camera moves (the "aura"); the books were calm only because they
  // already fade out before they shrink to that size (see GLOW_REST_FAR's note).
  applyDistanceFade(mat);
  applyCloudShadow(mat, cloud, true); // boulders darken under the same drifting shadow
  const group = new THREE.Group();
  const nv = variants.length;
  // world-vertical extent of each variant once laid face-up: the head's local z
  // (face depth, which the +90° X rotation swings onto world y) at scale 1. Used to
  // bury the head by its sink fraction.
  const depth = variants.map((geos) => {
    geos[0].computeBoundingBox();
    const b = geos[0].boundingBox!;
    return b.max.z - b.min.z;
  });

  // Bucket heads by variant and bake each one's world transform once. Lay the head
  // on its back facing the sky (Rx +90, the sculpted face axis ran opposite the
  // first guess), spin it by its yaw (YXZ -> Ry·Rx), scale, and sink it so its
  // centre sits at ground + H*(0.5 - sink), burying exactly that fraction of H.
  const dummy = new THREE.Object3D();
  const mats: number[][] = Array.from({ length: nv }, () => []);
  const xs: number[][] = Array.from({ length: nv }, () => []);
  const zs: number[][] = Array.from({ length: nv }, () => []);
  for (const d of decos) {
    const v = d.v < nv ? d.v : 0;
    const H = depth[v] * d.s;
    dummy.position.set(d.x, sampleHeight(d.x, d.y) + H * (0.5 - d.sink), d.y);
    dummy.rotation.set(Math.PI / 2, d.rot, 0, "YXZ");
    dummy.scale.setScalar(d.s);
    dummy.updateMatrix();
    for (let k = 0; k < 16; k++) mats[v].push(dummy.matrix.elements[k]);
    xs[v].push(d.x);
    zs[v].push(d.y);
  }

  // Per variant: a static coarse InstancedMesh holding ALL its heads (always drawn,
  // never frustum-culled since it spans the disc), plus a small dynamic full-detail
  // pool refilled around the camera. The coarse geometry is shrunk so the full mesh
  // occludes it cleanly where both draw.
  const matF32: Float32Array[] = [];
  const posX: Float32Array[] = [];
  const posZ: Float32Array[] = [];
  const nearMeshes: THREE.InstancedMesh[] = [];
  const tmp = new THREE.Matrix4();
  for (let v = 0; v < nv; v++) {
    const cnt = xs[v].length;
    const mf = new Float32Array(mats[v]);
    matF32.push(mf);
    posX.push(new Float32Array(xs[v]));
    posZ.push(new Float32Array(zs[v]));

    const coarseGeo = variants[v][2].clone();
    coarseGeo.scale(HEAD_COARSE_PROXY, HEAD_COARSE_PROXY, HEAD_COARSE_PROXY);
    const coarse = new THREE.InstancedMesh(coarseGeo, mat, Math.max(cnt, 1));
    coarse.frustumCulled = false;
    coarse.renderOrder = 5; // transparent now (distance fade); draw after the ground
    for (let i = 0; i < cnt; i++) coarse.setMatrixAt(i, tmp.fromArray(mf, i * 16));
    coarse.count = cnt;
    coarse.instanceMatrix.needsUpdate = true;
    group.add(coarse);

    const near = new THREE.InstancedMesh(variants[v][0], mat, HEAD_NEAR_CAP);
    near.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    near.frustumCulled = false;
    near.renderOrder = 5;
    near.count = 0;
    nearMeshes.push(near);
    group.add(near);
  }

  // Refill the full-detail pools only when the camera has moved HEAD_REBUILD. With
  // ~10k heads total a brute scan is trivial (it fires rarely, not per frame).
  const RF2 = HEAD_R_FULL * HEAD_R_FULL;
  const RB2 = HEAD_REBUILD * HEAD_REBUILD;
  const m = new THREE.Matrix4();
  let lastX = Infinity;
  let lastZ = Infinity;
  function update(camera: THREE.Camera): void {
    const cx = camera.position.x;
    const cz = camera.position.z;
    const dx = cx - lastX;
    const dz = cz - lastZ;
    if (dx * dx + dz * dz < RB2) return;
    lastX = cx;
    lastZ = cz;
    for (let v = 0; v < nv; v++) {
      const x = posX[v];
      const z = posZ[v];
      const mf = matF32[v];
      const near = nearMeshes[v];
      let k = 0;
      for (let i = 0; i < x.length && k < HEAD_NEAR_CAP; i++) {
        const ex = x[i] - cx;
        const ez = z[i] - cz;
        if (ex * ex + ez * ez < RF2) {
          near.setMatrixAt(k, m.fromArray(mf, i * 16));
          k++;
        }
      }
      near.count = k;
      near.instanceMatrix.needsUpdate = true;
    }
  }
  return { group, update };
}

function buildField(
  field: Awaited<ReturnType<typeof loadPositions>>,
  bookNear: THREE.BufferGeometry, // LOD00, full detail, drawn closest
  bookMid: THREE.BufferGeometry, // LOD01, drawn across the mid band
  uPlayer: PlayerUniform, // shared player-position uniform (also drives the ground rake)
  cloud: CloudUniforms, // shared cloud-shadow uniforms (also drift over the ground)
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
  // Each book is drawn by EXACTLY ONE mesh. The box draws the whole field; when a book is
  // promoted to a near/mid detail tier (see update()), its box instance is hidden (zeroed
  // matrix), so the box and the detail never cover the same book. That removes the overlap
  // outright, which is what every depth trick here was fighting: a shrunk box read dimmer
  // than the detail (emission scales with on-screen area), so the detail region glowed in a
  // cell-quantized bright zone; a depth-biased box sank behind the terrain at grazing angles
  // and the mid-field vanished. With no overlap the box keeps a plain depth relationship
  // with the ground (never disappears) and there is no box-vs-detail seam to hide.
  const PROXY = 1.0;
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
  // shared proximity-glow uniforms: one object referenced by every book material,
  // so moving uPlayer once per frame lights the pool on all three LOD tiers.
  const glow: GlowUniforms = {
    uPlayer,
    uGlowRadius: { value: GLOW_RADIUS },
    uGlowInner: { value: GLOW_INNER },
    uRestDim: { value: GLOW_REST_DIM },
    uRestFar: { value: GLOW_REST_FAR },
    uGlowBoost: { value: GLOW_BOOST },
    uEmissive: { value: GLOW_EMISSIVE },
    uEmissiveNear: { value: GLOW_EMISSIVE_NEAR },
  };
  const farMat = new THREE.MeshLambertMaterial();
  applyDistanceFade(farMat);
  applyProximityGlow(farMat, glow, cloud);
  const midMat = new THREE.MeshLambertMaterial();
  applyDistanceFade(midMat);
  applyPageMask(midMat);
  applyProximityGlow(midMat, glow, cloud);
  const nearMat = new THREE.MeshLambertMaterial();
  applyDistanceFade(nearMat);
  applyPageMask(nearMat);
  applyProximityGlow(nearMat, glow, cloud);
  // All three now share applyProximityGlow as their outermost onBeforeCompile, so
  // their default program-cache keys (= onBeforeCompile.toString(), closure vars
  // excluded) collide. far has no page mask while mid/near do, so without a
  // distinguishing key three would hand all three whichever program compiled
  // first. Key on the actual patch stack: mid/near are identical (share a program,
  // correct), far is its own. Same defence the ground material uses for its holes.
  farMat.customProgramCacheKey = () => "book:fade+glow+cloud";
  midMat.customProgramCacheKey = () => "book:fade+page+glow+cloud";
  nearMat.customProgramCacheKey = () => "book:fade+page+glow+cloud";

  // Explicit renderOrder. The books are transparent (distance fade) and were all left
  // at renderOrder 0, tied with each other and with the ground, so Three's distance sort
  // flipped per frame. Since a book is now drawn by exactly one mesh (promoted instances
  // are hidden in the box), the tiers never overlap, so this is purely a deterministic
  // transparent-sort order: ground first (renderOrder 0), then the books, below the
  // teleporter beam (renderOrder 10). Keep detail before box so blends are stable.
  const RO_DETAIL = 5; // near/mid detail: after the ground
  const RO_BOX = 7; // the box base: the rest of the field
  const farMesh = new THREE.InstancedMesh(boxGeo, farMat, n);
  // dynamic: update() hides/restores instances as books move in and out of the detail
  // tiers, so the box matrix is rewritten on each rebuild, not just once at build.
  farMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  farMesh.frustumCulled = false; // spans the whole disc; never wholly off-screen
  farMesh.renderOrder = RO_BOX;
  const midMesh = new THREE.InstancedMesh(bookMid, midMat, MID_CAP);
  midMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  midMesh.frustumCulled = false; // rebuilt around the camera, bounds don't apply
  midMesh.count = 0;
  midMesh.renderOrder = RO_DETAIL;
  const nearMesh = new THREE.InstancedMesh(bookNear, nearMat, NEAR_CAP);
  nearMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  nearMesh.frustumCulled = false;
  nearMesh.count = 0;
  nearMesh.renderOrder = RO_DETAIL;

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
    // seat on the facet the ground mesh actually draws underfoot, not the smooth
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
  // a book promoted to a detail tier has its box instance collapsed to a point (zero
  // scale -> degenerate, no fragments), so the box never double-draws it. We track the
  // promoted ids so the next rebuild can restore their box matrix before re-promoting.
  const HIDE = new THREE.Matrix4().makeScale(0, 0, 0);
  const hidden = new Int32Array(NEAR_CAP + MID_CAP);
  let nHidden = 0;
  let lastX = Infinity;
  let lastZ = Infinity;
  function update(camX: number, camZ: number): void {
    const mdx = camX - lastX;
    const mdz = camZ - lastZ;
    if (mdx * mdx + mdz * mdz < RB2) return;
    lastX = camX;
    lastZ = camZ;
    // restore the box instances hidden last rebuild; the walk below re-hides whichever
    // are still promoted, so a book that fell out of the detail tiers reappears in the box.
    for (let h = 0; h < nHidden; h++) {
      const i = hidden[h];
      farMesh.setMatrixAt(i, m.fromArray(fullMat, i * 16));
    }
    nHidden = 0;
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
          farMesh.setMatrixAt(i, HIDE); // detail draws it; collapse the box copy
          hidden[nHidden++] = i;
        } else if (kMid < MID_CAP) {
          // mid band, or a near-band book that overflowed NEAR_CAP (still gets detail)
          midMesh.setMatrixAt(kMid, m.fromArray(fullMat, i * 16));
          midMesh.setColorAt(kMid, c.fromArray(fullCol, i * 3));
          kMid++;
          farMesh.setMatrixAt(i, HIDE); // detail draws it; collapse the box copy
          hidden[nHidden++] = i;
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
    // the restores and re-hides above rewrote a subset of the box matrix; push it once.
    farMesh.instanceMatrix.needsUpdate = true;
  }

  const group = new THREE.Group();
  group.add(farMesh);
  group.add(midMesh);
  group.add(nearMesh);
  return { group, px, pz, tier: tier as Uint8Array, update, nearMesh, midMesh };
}

// Teleporter monuments (26): a ring of standing stones on the ground that the
// player walks into, with a tall emissive-blue light shaft rising from its centre,
// the cool complement to the hot-orange landmark books. The shaft is occluded by
// the dunes like everything else, so it isn't a cross-disc beacon (the compass
// does the long-range wayfinding); it's the reward you crest a ridge to find,
// marking a known place once you're near enough to see it.
const TP_BEAM_HEIGHT = 80; // visible shaft height above the ground
const TP_BEAM_RADIUS = 0.6;
const TP_BEAM_COLOR = new THREE.Color(0x2f6cff);
// A round tube crossing the ground plane reveals its tube shape at the waterline
// (curved bottom rim, the far wall seen through the near one), which breaks the
// flat-pillar illusion up close. So the beam fades out by horizontal camera
// distance: a clean shaft from afar, gone before you are near enough to see the
// intersection. Full strength beyond FAR, gone within NEAR (≈ at the stones).
const TP_BEAM_FADE_NEAR = 18;
const TP_BEAM_FADE_FAR = 50;
// The ground is transparent (distance fade) at the default renderOrder 0. A beam at
// the same order could draw before the ground, which then paints its opaque-near sand
// straight over it (the beam writes no depth, so it can't defend those pixels).
// Drawing the beam after the ground fixes that; the depth TEST against the ground
// (which does write depth) still occludes the beam behind nearer dunes, so physical
// occlusion is preserved. Stays above the ground and the books (renderOrder 5).
const TP_BEAM_RENDER_ORDER = 10;
// Horizontal radius around a circle's centre within which the travel prompt
// arms. The stones span STONE_CIRCLE_DIAMETER (9 -> 4.5 radius) on a flattened
// plaza; a touch wider than the ring so you trigger while standing among the
// stones, not only dead centre.
const TP_ENTER_RADIUS = 7;

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

function buildTeleporters(
  list: Teleporter[],
  circleGeom: THREE.BufferGeometry,
  cloud: CloudUniforms,
) {
  // The stones take the same sandstone as the heads and props, so the monument
  // reads as carved from the desert rather than dropped onto it. Opaque and
  // unfaded (no applyDistanceFade) like the other props, though being flat it is
  // lost to the haze at distance: the beam, not the ring, is the far beacon.
  const stoneMat = new THREE.MeshLambertMaterial({ color: 0xcbbfa8 }); // sandstone
  // Fade the ring into the haze with distance, like the books and heads: an
  // unfaded opaque ring stays a high-contrast speck at the horizon and crawls
  // against the pixel grid (the flickering "aura"). The comment below already
  // expected the ring "lost to the haze at distance"; the fade makes that real
  // instead of leaving max-contrast confetti. The beam stays the far beacon.
  applyDistanceFade(stoneMat);
  applyCloudShadow(stoneMat, cloud, false); // stones darken with the sand around them
  // The beam is an open-ended cylinder shaded as a volumetric light shaft. Two
  // gradients shape it: a silhouette-edge term (alpha ~ |view·normal|) that makes
  // a view ray glowing brightest where it passes through the most of the column
  // and dissolving to nothing at the rounded edges, so there is no hard outline;
  // and a vertical fade that thins the shaft to transparent toward the top, as if
  // the light dissipates as it rises. Additive with no depth write so the layers
  // accumulate into a bright core, but depth TEST stays on: dunes occlude it, and
  // the player crests a ridge to find the light waiting (the compass, not the
  // beam, does the long-range wayfinding).
  const beamGeom = new THREE.CylinderGeometry(
    TP_BEAM_RADIUS,
    TP_BEAM_RADIUS,
    TP_BEAM_HEIGHT,
    16,
    1,
    true, // open-ended: no caps to flare as flat discs when seen from above
  );
  const beamMat = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: TP_BEAM_COLOR },
      uOpacity: { value: 0.4 },
      uHeight: { value: TP_BEAM_HEIGHT },
      uFadeNear: { value: TP_BEAM_FADE_NEAR },
      uFadeFar: { value: TP_BEAM_FADE_FAR },
    },
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      uniform float uHeight;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;
      varying float vT;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vT = (position.y + uHeight * 0.5) / uHeight; // 0 at base, 1 at top
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uFadeNear;
      uniform float uFadeFar;
      varying vec3 vWorldPos;
      varying vec3 vWorldNormal;
      varying float vT;
      void main() {
        vec3 viewDir = normalize(cameraPosition - vWorldPos);
        float edge = abs(dot(viewDir, vWorldNormal));   // 1 through the core, 0 at the silhouette
        float vert = pow(1.0 - clamp(vT, 0.0, 1.0), 1.5); // dissipates toward the top
        // horizontal camera distance (xz only, so looking up the tall shaft doesn't
        // trigger it): fade the whole beam out as you approach, hiding the waterline.
        float camDist = distance(cameraPosition.xz, vWorldPos.xz);
        float camFade = smoothstep(uFadeNear, uFadeFar, camDist);
        gl_FragColor = vec4(uColor, uOpacity * edge * vert * camFade);
      }
    `,
  });
  const group = new THREE.Group();
  for (const tp of list) {
    const h = sampleHeight(tp.x, tp.y);
    // ring of stones resting on the flattened plaza (terrain levels a disc here).
    const circle = new THREE.Mesh(circleGeom, stoneMat);
    circle.position.set(tp.x, h, tp.y);
    circle.renderOrder = 5; // transparent now (distance fade); draw after the ground
    group.add(circle);
    // beam rising from the circle's centre, base at the ground. Drawn after the
    // transparent ground levels (see TP_BEAM_RENDER_ORDER) so they can't overpaint
    // it; it fades out by camera distance (shader) before the waterline shows.
    const beam = new THREE.Mesh(beamGeom, beamMat);
    beam.position.set(tp.x, h + TP_BEAM_HEIGHT / 2, tp.y);
    beam.renderOrder = TP_BEAM_RENDER_ORDER;
    group.add(beam);
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
// PointerLockControls owns the look (Euler camera, pitch clamped internally). We own
// translation. Grounded movement (walk + skate) integrates a persistent velocity with
// per-mode accel/decel and pins the eye to the baked ground through a smoothing filter;
// dev flight (F) keeps the old instant free-Y model untouched.
type MoveMode = "walk" | "skate" | "fly";
function createController(camera: THREE.PerspectiveCamera, dom: HTMLElement) {
  const controls = new PointerLockControls(camera, dom);
  const keys = new Set<string>();
  const vel = new THREE.Vector3(); // carried horizontal velocity (xz; y stays 0)
  let flying = false;

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.code === "KeyF") {
      flying = !flying; // dev fly toggle; drop carried momentum so neither mode lurches
      vel.set(0, 0, 0);
    }
    keys.add(e.code);
  };
  const onKeyUp = (e: KeyboardEvent) => keys.delete(e.code);
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
  // releasing the lock (Esc) drops held keys AND the carried velocity, or the player
  // keeps gliding after the cursor reappears.
  controls.addEventListener("unlock", () => {
    keys.clear();
    vel.set(0, 0, 0);
  });
  dom.addEventListener("click", () => controls.lock());

  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const wish = new THREE.Vector3();
  const target = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0);

  // move `vel` toward `target` by at most `maxDelta`, as a vector so a turn swings the
  // heading and accel/decel scale the magnitude under one rule. This is what makes
  // releasing Shift bleed speed off smoothly instead of snapping to a cap.
  const approach = (maxDelta: number) => {
    target.sub(vel); // the delta we'd ideally apply this frame
    const len = target.length();
    if (len <= maxDelta || len === 0) vel.add(target);
    else vel.addScaledVector(target, maxDelta / len);
  };

  function update(dt: number): MoveMode {
    if (!controls.isLocked) return flying ? "fly" : "walk";

    camera.getWorldDirection(forward);
    if (!flying) forward.y = 0; // grounded: ignore pitch, move along the ground
    forward.normalize();
    right.crossVectors(forward, UP).normalize();

    if (flying) {
      // dev free flight: instant velocity, free Y, Shift boosts. Unchanged.
      wish.set(0, 0, 0);
      if (keys.has("KeyW") || keys.has("ArrowUp")) wish.add(forward);
      if (keys.has("KeyS") || keys.has("ArrowDown")) wish.sub(forward);
      if (keys.has("KeyD") || keys.has("ArrowRight")) wish.add(right);
      if (keys.has("KeyA") || keys.has("ArrowLeft")) wish.sub(right);
      if (keys.has("Space")) wish.y += 1;
      if (keys.has("KeyC")) wish.y -= 1;
      const boost = keys.has("ShiftLeft") || keys.has("ShiftRight") ? FLY_RUN_MULT : 1;
      if (wish.lengthSq() > 0) {
        wish.normalize().multiplyScalar(FLY_SPEED * boost * dt);
        camera.position.add(wish);
      }
      return "fly";
    }

    // grounded: walk by default, skate while Shift is held.
    const skating = keys.has("ShiftLeft") || keys.has("ShiftRight");
    const cfg = skating ? MOVE.skate : MOVE.walk;

    wish.set(0, 0, 0);
    if (keys.has("KeyW") || keys.has("ArrowUp")) wish.add(forward);
    if (keys.has("KeyS") || keys.has("ArrowDown")) wish.sub(forward);
    if (keys.has("KeyD") || keys.has("ArrowRight")) wish.add(right);
    if (keys.has("KeyA") || keys.has("ArrowLeft")) wish.sub(right);

    if (wish.lengthSq() > 0) {
      // accelerate toward the wished heading at top speed
      target.copy(wish.normalize()).multiplyScalar(cfg.max);
      approach(cfg.accel * dt);
    } else {
      // no input: ease toward rest at the mode's decel (skate coasts, walk brakes hard)
      target.set(0, 0, 0);
      approach(cfg.decel * dt);
    }

    camera.position.x += vel.x * dt;
    camera.position.z += vel.z * dt;

    // circular world bound: past SOFT shed the outward velocity so you ease along the
    // edge, and hard-clamp at HARD. Both sit in the fog, so the stop reads as the world
    // thinning out rather than a wall.
    const r = Math.hypot(camera.position.x, camera.position.z);
    if (r > MOVE.boundSoft) {
      const nx = camera.position.x / r;
      const nz = camera.position.z / r;
      const outward = vel.x * nx + vel.z * nz;
      if (outward > 0) {
        vel.x -= outward * nx;
        vel.z -= outward * nz;
      }
      if (r > MOVE.boundHard) {
        camera.position.x = nx * MOVE.boundHard;
        camera.position.z = nz * MOVE.boundHard;
      }
    }

    // pin the eye to the baked ground, smoothed. The skate lift raises it a touch as a
    // tactile cue; the time-constant filter keeps the feet planted at a walk and smooths
    // dune bumps at skate speed where rigid tracking would jolt the camera.
    const lift = skating ? MOVE.hoverLift : 0;
    const targetY = sampleHeight(camera.position.x, camera.position.z) + EYE_HEIGHT + lift;
    camera.position.y += (targetY - camera.position.y) * (1 - Math.exp(-MOVE.followK * dt));

    return skating ? "skate" : "walk";
  }

  // zero the carried velocity (teleport/spawn shouldn't arrive mid-glide).
  const stop = () => vel.set(0, 0, 0);

  return { controls, update, stop };
}

async function main() {
  const scene = new THREE.Scene();
  // The ground and books fade to TRANSPARENT at distance (see applyDistanceFade)
  // and dissolve into the sky dome, so the distance always reads as exactly the
  // sky behind it, never tinted, and no footprint edge is ever left to see. A
  // single fog colour was the alternative but it can only tint distant surfaces
  // toward one flat colour (brightening the grey void or darkening the present)
  // and, worse, leaves opaque geometry whose square footprint shows from a height.
  // The land fades to TRANSPARENT (alpha, not a colour step), so the dome supplies
  // whatever the far desert dissolves into and the two no longer need a matched
  // tone; HORIZON is only the clear-colour behind the dome, rarely seen.
  const HORIZON = new THREE.Color(0x847b6d);
  scene.background = HORIZON;

  // The world sunlight and the sky's warm glow share one bearing, so the bright
  // side of the dome is where the sun actually is, and neither moves as the player
  // walks (see sky.ts). ~9deg elevation: grazing, for long tonal gradients on the
  // dunes. Used below for the DirectionalLight too.
  const SUN_POS = new THREE.Vector3(-700, 130, 380);

  // drifting cloud shadows (see clouds.ts): one shared mask + time, sampled at the
  // world xz of the ground, books, heads and stones so the same shadow falls on a
  // book and the sand under it, AND at the sky dome's pierce points so the storm's
  // breaks open over the lit patches. The scene's main source of large-scale motion.
  // Built before the sky because the dome shares its uniforms.
  const clouds = buildClouds();

  // storm sky dome (see sky.ts): a near-black cloud ceiling whose breaks are cut by
  // the same drifting field that lights the ground, so the sky opens where the dunes
  // beneath are lit. A warm glow stays fixed at the sun bearing; the whole dome
  // deepens with the player's radial depth into the past. It recentres on the camera
  // each frame (see the loop) so it reads as infinitely far and never shows an edge.
  const sky = buildSky(SUN_POS, clouds.uniforms);
  scene.add(sky.mesh);

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

  const { controls, update, stop } = createController(camera, renderer.domElement);
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

  // Dusk rig matching the sky: a low, warm, raking sun so every dune face shows
  // light/dark contrast (a grazing sun maximises the cosine difference between
  // slopes, which is what makes the dunes read as 3D), over a dim, cool hemisphere
  // ambient so the directional term dominates instead of flooding the slopes flat.
  // The sun sits on SUN_POS, the same bearing the sky's warm glow uses, so the lit
  // ground and the bright sky agree.
  scene.add(new THREE.HemisphereLight(0xffd9b3, 0x2b2f47, 0.5));
  const sun = new THREE.DirectionalLight(0xffb066, 2.3);
  sun.position.copy(SUN_POS);
  scene.add(sun);

  // --- load profiling -------------------------------------------------------
  // Phase wall-clock so load cost is measured, not guessed (console: filter
  // "[load]"). "seat books" is the one to watch: it reconstructs a ground facet per
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
  const [field, teleporters, meta, world, heightmap, bookLods, headVariants, decorations, stoneCircle] =
    await Promise.all([
      loadPositions("positions.bin"),
      loadTeleporters("teleporters.json"),
      loadMeta("meta.bin"),
      loadWorld("world.json"),
      loadHeightmap("heightmap.bin"),
      loadBookLods("book.glb", ["book_LOD00", "book_LOD01"]),
      loadHeadLods("heads.glb"),
      loadDecorations("decorations.json"),
      loadStoneCircle("stone_circle.glb"),
    ]);
  mark("fetch+decode");
  // the heightmap is the ground-height source for the ground mesh and the player's
  // feet; init it before anything samples it.
  initHeightmap(heightmap.res, heightmap.worldSize, heightmap.data);
  // the plazas flatten around the teleporters, so terrain needs them before the
  // books or monuments are seated. Books, monuments, the picker and the ring all read
  // sampleHeight/sampleNormal now, so they seat on the same baked surface the
  // ground mesh draws (no float-off; the analytic field is only the heightmap fallback).
  initTerrain(teleporters);
  // one player-position uniform shared by the ground's raking light and the books'
  // proximity glow, so both reactive pools track the same centre (the overlapping
  // light radii). Updated once per frame in the loop.
  const uPlayer: PlayerUniform = { value: new THREE.Vector2(0, 0) };
  // the ground is one static mesh tessellated from the heightmap (terrain.buildGround):
  // no camera-following, no rebuild, no LOD seams. It just sits there; the raking light
  // and cloud shadow ride on shared uniforms updated in the loop.
  const ground = buildGround(uPlayer, clouds.uniforms);
  scene.add(ground);
  // settle the eye onto the baked surface now the heightmap is loaded (spawn was
  // placed on the analytic fallback before the fetch resolved).
  camera.position.y = sampleHeight(camera.position.x, camera.position.z) + EYE_HEIGHT;
  // bound the player inside the content disc, well within the terrain mesh and the
  // distance fog, so skating outward eases to a stop in haze rather than reaching the
  // ground's edge.
  MOVE.boundSoft = world.R_MAX + 300;
  MOVE.boundHard = world.R_MAX + 800;
  mark("terrain");
  const built = buildField(field, bookLods[0], bookLods[1], uPlayer, clouds.uniforms);
  mark("seat books");
  built.update(camera.position.x, camera.position.z);
  scene.add(built.group);
  scene.add(buildTeleporters(teleporters, stoneCircle, clouds.uniforms));
  const heads = buildHeads(headVariants, decorations, clouds.uniforms); // half-buried scatter (Stage 10)
  scene.add(heads.group);

  mark("props");

  // --- look-at glance + inspect overlay -------------------------------------
  const glance = document.getElementById("glance") as HTMLDivElement;
  const overlay = document.getElementById("overlay") as HTMLDivElement;
  const card = document.getElementById("card") as HTMLDivElement;
  const tpPrompt = document.getElementById("tp-prompt") as HTMLDivElement;
  const fade = document.getElementById("fade") as HTMLDivElement;
  const pick = createPicker(camera, built.px, built.pz);
  const compass = createCompass(camera, built.px, built.pz, meta, world);

  let target = -1; // instanceId under the reticle, or -1
  let overlayOpen = false;
  let nearTp = -1; // teleporter circle the player is standing in, or -1

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

  // --- teleporter travel ----------------------------------------------------
  // The 26 circles are an any-to-any fast-travel network across a disc too wide
  // to walk. Standing in a circle arms the prompt (loop below); T opens this
  // menu of the other anchors, nearest first, and a pick jumps you there.
  let traveling = false;
  function travelTo(dest: number) {
    if (traveling) return;
    traveling = true;
    closeOverlay();
    fade.style.opacity = "1"; // fade to black (CSS transition: 0.3s)
    // reposition only once the screen is black, so the pop is never seen; the
    // eye lands on the baked surface at the circle's centre (walk mode re-pins
    // it each frame anyway, fly mode keeps it where we put it).
    window.setTimeout(() => {
      const tp = teleporters[dest];
      camera.position.set(tp.x, sampleHeight(tp.x, tp.y) + EYE_HEIGHT, tp.y);
      stop(); // arrive at rest, not mid-glide
      fade.style.opacity = "0"; // fade back in on the destination
      window.setTimeout(() => {
        traveling = false;
      }, 300);
    }, 300);
  }
  function openTravel(from: number) {
    const here = teleporters[from];
    const rows = teleporters
      .map((tp, i) => ({ i, d: Math.hypot(tp.x - here.x, tp.y - here.y) }))
      .filter((o) => o.i !== from)
      .sort((a, b) => a.d - b.d)
      .map((o) => {
        const tp = teleporters[o.i];
        const line = `${escapeHtml(tp.era)} · ${escapeHtml(tp.seat)} · ${Math.round(o.d).toLocaleString()}u`;
        return (
          `<button class="tp-dest" data-i="${o.i}">` +
          `<span class="tp-label">${escapeHtml(tp.label)}</span>` +
          `<span class="tp-meta">${line}</span>` +
          `</button>`
        );
      })
      .join("");
    card.innerHTML =
      `<div class="name">◎ ${escapeHtml(here.label)}</div>` +
      `<div class="desc">Step through to another circle.</div>` +
      `<div class="tp-list">${rows}</div>` +
      `<div class="hint">Esc or click outside to close</div>`;
    card.querySelectorAll<HTMLButtonElement>(".tp-dest").forEach((btn) => {
      btn.addEventListener("click", () => travelTo(Number(btn.dataset.i)));
    });
    overlay.style.display = "flex";
    overlayOpen = true;
    glance.style.display = "none";
    tpPrompt.style.display = "none";
    controls.unlock(); // free the cursor so destinations are clickable
  }

  // click on the backdrop (not the card) closes; clicking the card/link doesn't.
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeOverlay();
  });
  document.addEventListener("keydown", (e) => {
    if (e.code === "KeyE" && !overlayOpen && controls.isLocked && target >= 0) {
      openOverlay(target);
    } else if (e.code === "KeyT" && !overlayOpen && controls.isLocked && nearTp >= 0) {
      openTravel(nearTp);
    } else if (e.code === "Escape" && overlayOpen) {
      closeOverlay();
    }
  });

  const hint =
    "click to look · WASD move · Shift skate · E inspect · T travel · Esc release";
  const setHud = (mode: MoveMode) => {
    const label = mode === "fly" ? "flying (dev)" : mode === "skate" ? "skating" : "walking";
    info.innerHTML =
      `${field.n.toLocaleString()} figures · ${label} · centre = year 2000<br>${hint}`;
  };
  setHud("walk");

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  mark("wiring");
  console.log(`[load] total ${(performance.now() - t0).toFixed(0)}ms to first frame`);

  const clock = new THREE.Clock();
  let mode: MoveMode = "walk";
  let sincePick = 0;
  let sinceStat = 0;
  // worst-case ms for the two camera-driven rebuilds, reset each readout window,
  // so a bursty re-tessellation spike shows up instead of being averaged away.
  let booksMs = 0;
  const PICK_INTERVAL = 0.12; // ~8 Hz; the look-at label needn't be per-frame
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1); // clamp after tab-out stalls
    const m = update(dt);
    if (m !== mode) {
      setHud(m);
      mode = m;
    }

    // look-at picking: only while walking the scene (locked) and not inspecting.
    // Suppressed while skating so the glance prompt doesn't strobe as books blow past.
    sincePick += dt;
    if (!overlayOpen && controls.isLocked && mode !== "skate") {
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

    // teleporter proximity: arm the travel prompt when standing in a circle.
    // xz only (height is irrelevant) and just 26 anchors, so it runs every frame
    // for an instant prompt. Rebuild the prompt text only when the circle changes.
    {
      let found = -1;
      const cx = camera.position.x;
      const cz = camera.position.z;
      for (let i = 0; i < teleporters.length; i++) {
        const dx = teleporters[i].x - cx;
        const dz = teleporters[i].y - cz;
        if (dx * dx + dz * dz <= TP_ENTER_RADIUS * TP_ENTER_RADIUS) {
          found = i;
          break;
        }
      }
      if (found !== nearTp) {
        nearTp = found;
        if (nearTp >= 0) {
          const tp = teleporters[nearTp];
          tpPrompt.innerHTML =
            `<div class="tp-here">◎ ${escapeHtml(tp.label)}</div>` +
            `<div class="tp-act">press <kbd>T</kbd> to travel</div>`;
        }
      }
      const armed = nearTp >= 0 && !overlayOpen && controls.isLocked && mode !== "skate";
      tpPrompt.style.display = armed ? "block" : "none";
    }

    // the ground is static (built once); only the books refill by LOD as the camera
    // roams. Time the refill for the HUD.
    const tA = performance.now();
    built.update(camera.position.x, camera.position.z);
    const bm = performance.now() - tA;
    if (bm > booksMs) booksMs = bm;
    heads.update(camera); // pick each head's LOD by camera distance (a handful)

    compass.update();
    // travelling pools of light: move the shared player-position uniform (book glow +
    // ground rake) to the player every frame. The book LOD refill is gated by move
    // distance, far too coarse for a smooth pool, so the uniform is driven here.
    uPlayer.value.set(camera.position.x, camera.position.z);
    // drift the cloud shadows across the whole landscape (ground, books, heads,
    // stones all sample the one shared mask + time).
    clouds.update(dt);
    // keep the dome centred on the viewer (so it reads as infinitely far), and
    // deepen it with radial depth into the past. The warm glow's bearing is fixed
    // to the sun, so the dome no longer steers off the camera position.
    sky.mesh.position.copy(camera.position);
    const depth = clamp(
      Math.hypot(camera.position.x, camera.position.z) / world.R_MAX,
      0,
      1,
    );
    sky.update(depth, camera.position.x, camera.position.z);
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
        `bookfl ${booksMs.toFixed(1)}ms (peak)`;
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
