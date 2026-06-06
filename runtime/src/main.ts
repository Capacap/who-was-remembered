import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import Stats from "three/examples/jsm/libs/stats.module.js";
import bookUrl from "./assets/meshes/book.glb?url";
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
  buildDaylight,
  DAYLIGHT_FRAG_COMMON,
  applyDaylightGLSL,
  DAY_GLSL,
  type DaylightUniforms,
} from "./daylight";

// --- walkable field --------------------------------------------------------
// One instanced box per figure, placed straight from the pipeline's (x, y), with
// a first-person controller so the disc can be walked. The ground is the baked
// heightmap (see terrain.ts): books are seated on it and tilted to its normal, and
// the player's walk height samples the same surface so nothing floats.
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
  // that the button did something, well short of an actual fly-height float. PURELY
  // cosmetic: eased in/out and added on top of the physics eye height, so it never
  // feeds the vertical velocity (folding it into the follow target would inject a
  // fake 0.8u/jolt of vy every time Shift was tapped, and could trip a launch).
  hoverLift: 0.8,
  // crest hops. The eye is a point under gravity that the ground pushes UP but never
  // pulls DOWN: climb a dune grounded and you track the surface, but crest it fast and
  // the surface drops out from under you, so you keep the climb's upward speed and arc.
  // gravity pulls the hop back down; launchMin is the climb speed (units/sec) you must
  // exceed to leave the ground at all, so a walk stays planted and only a fast skate-
  // crest pops — raise it if rippled upslopes feel twitchy; vyMax caps the pop so a
  // sharp crest can't fling the eye. All live-tunable via window.MOVE against real dunes.
  gravity: 50,
  launchMin: 4,
  vyMax: 12,
  // ground-follow stiffness: a fixed time-constant filter on eye height. At a walk the
  // terrain target barely moves so the feet stay planted; at skate speed the target
  // changes fast and the same filter smooths the dune bumps that would otherwise jolt
  // the camera (and the stomach). Higher = stiffer.
  followK: 12,
  // seconds of velocity to look ahead when sampling the ground: a steep dune face
  // lifts the eye before you reach it, so a fast climb reads as a climb, not a snap or
  // a clip-through. Scales with speed (negligible at a walk), so it only acts when it
  // matters. The hard floor below is the actual no-clip guarantee; this just smooths it.
  lookAhead: 0.12,
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
// Saturation is the lever that carries BOTH the close-up vividness and the distance
// read: up close, low saturation washes pale; at distance the book goes semi-transparent
// (distance fade) and blends with the tan sand behind it, so a saturated hue stays
// coloured against the dunes while a pale one greys out into them. Pushed up from a washed
// 0.35. Lightness sits at/below 0.5 so the hue reads as colour rather than bleaching toward
// white under the daylight (HSL desaturates perceptually as lightness climbs past 0.5).
const GEO_SAT = 0.6; // hue vividness (low = desert-muted, high = map-key loud)
const GEO_LIGHT = 0.5; // base lightness of an ordinary book
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
const GLOW_RADIUS = 32; // books dark beyond this horizontal distance from the player
const GLOW_INNER = 2; // tight full-brightness core at the player's feet; smooth taper to GLOW_RADIUS
const GLOW_REST_DIM = 0.12; // resting brightness of a near book just outside the pool (0 = black)
const GLOW_REST_FAR = 0.85; // resting brightness a book lifts to with DISTANCE (aerial perspective):
//   a far book settles to ~the same daylight-driven tone as the sand it lies on instead of
//   crushing to the dim near-floor. Crushed-dark distant books read as a dark sheet hung in
//   front of the lit field (they darken, the ground doesn't); lifting them lets distance MUTE
//   them into the field rather than darken them below it. The dim near-floor still rings the
//   player (within HAZE_NEAR) so the pool reveal keeps its drama; the lift ramps in past it.
//   Also calms sub-pixel flicker: a far book near the ground's tone is low-contrast, not black
//   confetti on bright sand. The book's own emissive floor rides on top, so at night the books
//   stay gentle self-lit specks a touch above the dark ground (the self-illuminated read).
const HAZE_NEAR = 250; // camera distance where the aerial lift begins (just past the pool)
const HAZE_FAR = 2000; // ... and reaches GLOW_REST_FAR; the atmospheric-perspective band
const GLOW_BOOST = 1.3; // additive bloom at the pool centre: the reactive light is now
//   the whole "life" of the field (the hover/bob was removed), so the pool is the signal

// Self-emission so a book is a coloured speck even where the night lighting and the
// proximity dim would otherwise lose it in the dark (the whole field had sunk into
// the black storm scene). EMISSIVE is a fraction of the book's own geo hue added as
// true self-light AFTER the daylight tint, so it pierces the storm shadow rather than
// being multiplied to black under it (a speck bursting through the dark, like the
// sky). EMISSIVE_NEAR is the extra emission the proximity pool adds, so a book by the
// player burns brighter than the distant field (see applyProximityGlow).
const GLOW_EMISSIVE = 0.14; // base self-glow as a fraction of the book's hue
const GLOW_EMISSIVE_NEAR = 0.85; // extra emission at the pool centre

// What keeps the field from reading as static plotted data, both REACTIVE rather than a
// constant animation (an earlier per-book hover/bob was removed: with no contact shadows
// to sell the lift it read as aimless drift):
//   - The PROXIMITY POOL (applyProximityGlow, GLOW_* above): books rest dim and blaze to
//     their full colour plus an additive bloom as the player comes within GLOW_RADIUS, so
//     walking the disc carries a travelling light that reacts to where you are. This is the
//     field's "life" now.
//   - REVEAL: the sun-reveal. Where a daylight break drifts over a book it lights up in
//     step with the sand it stands on: a second tap of the SAME daylightAt field the
//     ground reads for its day/night tint, cast in the SAME warm DAY colour, so the two
//     are locked to one sky. The ground swings its whole albedo from near-black to bright
//     daylight, so to keep the books from looking flat by comparison the reveal is strong
//     (a book in full sun emits close to its own hue, warmed). It is gated by daylightAt,
//     so at night it falls to zero and only the steady uEmissive floor remains, the floor
//     that keeps books visible in the dark in the first place.
const GLOW_REVEAL = 2.0; // sun-reveal strength: book self-light at full daylight

// Temperature: as the proximity pool reaches a book it WARMS, a painterly hue rotation
// toward a warm anchor rather than a flat orange tint (which would muddy). Each hue heats
// toward its own warm neighbour -- a cool blue rotates toward cyan, a green toward yellow,
// a purple toward red -- so the reactive light reads as HEAT while staying vivid. The
// anchor sits at YELLOW, not orange: with an orange anchor the shortest hue-arc would send
// blue the wrong way (toward magenta); a yellow anchor is what makes blue->cyan. Strength
// is the max fraction of the arc to the anchor traversed at the pool centre, kept moderate
// so books warm toward their neighbour instead of all converging on yellow. The shift falls
// off with the pool (glow), so a book cools back to its resting hue as the player leaves.
const WARM_HUE = 0.19; // warm anchor on the hue wheel (~69deg, yellow)
const WARM_STRENGTH = 0.18; // max hue-arc fraction rotated toward warm at the pool centre.
// Eased down from 0.3: the warm shift was rotating the red-orange landmark books toward
// yellow-orange, diluting a colour the player reads as a notability signal. The landmark
// hue sits near the warm anchor so its arc is already small; lowering strength quiets it
// while cool colours (far from the anchor) still warm visibly -- blue->cyan survives.

// flatShading on the book materials (set in buildField) gives each authored facet -- the
// cover's beveled lip, the spine sides, the page block -- its own light/dark tone instead of
// the smoothed average, the half of the teleporter-sphere recipe that creates form. It also
// computes normals per-face and ignores the baked vertex normals, so it sidesteps any lingering
// normal quirk in the mesh. (A fresnel rim was tried alongside it and dropped: on the flat tops
// it did nothing, and at distance it just whitened the far field.)

// Facet self-shading. The proximity emissive is otherwise a FLAT albedo add (no normal term),
// so near the player it swamps the Lambert gradient and close books read as flat colour. Carve
// the self-light by a geometric facet term so a spine facing the sun glows brighter than one
// facing away and each book's sides read as a 3D box. The covers stay near-uniform on purpose:
// every book lies cover-up (N=+y), and a flat horizontal plane genuinely can't show a gradient.
// The facet normal is the flat WORLD normal reconstructed from screen-space derivatives of world
// position (true per-facet, matching flatShading) -- no extra vertex normals needed; the normal
// is oriented toward the camera so its sign is winding-safe. Keep SUN_KEY_DIR in sync with
// SUN_POS in the scene setup so the book sides rake the same way the sun lights the ground.
const SUN_KEY_DIR = new THREE.Vector3(-700, 130, 380).normalize(); // = normalize(SUN_POS)
const BOOK_FACET = 0.75; // how hard the self-light is carved by facing (0 = flat as before, 1 = full)
const BOOK_FACET_MIN = 0.3; // floor so a back-facing facet still self-glows (never crushes to black)

// Era temperature: the SAME painterly hue-rotation, but driven by a book's RADIUS (= time)
// instead of the player's proximity. Recency bias is the subject of the piece, so the field
// itself runs warm at the modern centre and cools to cold in the deep-past rim -- the warmth
// IS the bias, baked into the ground. Unlike the proximity pool this is static (radius never
// changes), so it's folded into the resting albedo at build time, not the shader; the
// proximity warm-shift then rides on top, so approaching a cold ancient book warms it back
// toward neutral but never as hot as a modern one at the same range (the bias survives touch).
// Applied to ordinary (tier 0) books ONLY: landmark beacons keep their full notability hue,
// so a hot major reads even harder as a warm beacon against the cold antiquity around it --
// the few remembered names staying bright while their era goes cold (on-theme, not a bug).
// Hue rotation only (S/L untouched), so the saturation-driven distance read is preserved.
const ERA_WARM_HUE = 0.13; // present anchor: a warm gold, a touch oranger than the pool yellow
const ERA_COOL_HUE = 0.55; // deep-past anchor: a cold cyan-blue
const ERA_TEMP_STRENGTH = 0.25; // max hue-arc fraction at the radial extremes (centre / rim)
// 0 (no shift) sits at the mid-radius; the warm half is inside it, the cool half outside.

// shortest signed hue arc from h to anchor, JS-mod-safe (GLSL mod is always positive; % is not)
const mod1 = (x: number) => ((x % 1) + 1) % 1;

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
  daylight: DaylightUniforms,
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
    shader.uniforms.uDaylight = daylight.uDaylight;
    shader.uniforms.uDriftTime = daylight.uDriftTime;
    shader.uniforms.uDaylightMix = daylight.uDaylightMix;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vGlowXZ;\nvarying float vViewDist;\nvarying vec3 vWorldPos;",
      )
      // Carry the book's world xz (for the proximity pool), its full world position (for the
      // facet self-shading normal, reconstructed by derivative in the fragment) and its view
      // distance (for the distance dim). No vertex displacement -- the per-book hover/bob was
      // removed -- so project_vertex's gl_Position stands unchanged.
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vec4 _wpos = modelMatrix * instanceMatrix * vec4(transformed, 1.0);
         vGlowXZ = _wpos.xz;
         vWorldPos = _wpos.xyz;
         vViewDist = length(mvPosition.xyz);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vGlowXZ;\nvarying float vViewDist;\nvarying vec3 vWorldPos;\nuniform vec2 uPlayer;\n" +
          "uniform float uGlowRadius;\nuniform float uGlowInner;\n" +
          "uniform float uRestDim;\nuniform float uRestFar;\nuniform float uGlowBoost;\n" +
          "uniform float uEmissive;\nuniform float uEmissiveNear;\n" +
          DAYLIGHT_FRAG_COMMON +
          // rgb<->hsv (Iñigo Quílez), for the temperature warm-shift in the pool below
          "\nvec3 rgb2hsv(vec3 c){vec4 K=vec4(0.,-1./3.,2./3.,-1.);vec4 p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g));vec4 q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));float d=q.x-min(q.w,q.y);return vec3(abs(q.z+(q.w-q.y)/(6.*d+1e-10)),d/(q.x+1e-10),q.x);}" +
          "\nvec3 hsv2rgb(vec3 c){vec3 r=clamp(abs(mod(c.x*6.+vec3(0.,4.,2.),6.)-3.)-1.,0.,1.);return c.z*mix(vec3(1.),r,c.y);}",
      )
      .replace(
        "#include <opaque_fragment>",
        `#include <opaque_fragment>
         float glowD = distance(vGlowXZ, uPlayer);
         float glow = 1.0 - smoothstep(uGlowInner, uGlowRadius, glowD);
         // aerial perspective: the resting floor lifts toward uRestFar with camera DISTANCE
         // (distLift), so a far book settles to ~the ground's daylight tone instead of crushing
         // to the dim near-floor and reading as a dark sheet in front of the lit field. The dim
         // ring (within HAZE_NEAR) survives so the pool reveal stays dramatic. vGroundFade folds
         // in so the very-far alpha band stays lifted too.
         float distLift = smoothstep(${HAZE_NEAR.toFixed(1)}, ${HAZE_FAR.toFixed(1)}, vViewDist);
         float restFloor = mix(uRestDim, uRestFar, max(vGroundFade, distLift));
         gl_FragColor.rgb *= mix(restFloor, 1.0, glow);
         gl_FragColor.rgb += gl_FragColor.rgb * glow * uGlowBoost;` +
          // the same drifting daylight the ground takes, so a book darkens with
          // the sand it stands in; faded out into the distance dissolve (vGroundFade).
          applyDaylightGLSL("vGlowXZ", "vGroundFade", "vViewDist") +
          // self-emission, added AFTER the daylight tint so it is true self-light: it
          // survives the night (a book is a speck bursting through the dark, like the
          // sky) instead of being multiplied to black under the storm. The steady floor
          // (uEmissive) plus the proximity pool (uEmissiveNear) carry the book's own hue.
          // The sun-reveal rides on top: where the SAME daylightAt field that lights the
          // ground reads daylight, the book emits its hue warmed by the SAME DAY colour
          // the ground tints to, so a passing sun patch lights book and sand together. It
          // falls to zero at night, leaving only the floor that keeps books visible there.
          // facet self-shading: carve the otherwise-flat emissive add by the facet's facing to
          // the sun, so a spine catching the sun glows brighter than one in shade and the sides
          // read 3D. The flat world normal comes from screen-space derivatives of world position
          // (the true per-facet normal under flatShading); _facet floors at BOOK_FACET_MIN so a
          // back-facing side still self-glows, and BOOK_FACET dials the whole effect to zero.
          `vec3 _wn = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
           _wn *= sign(dot(_wn, cameraPosition - vWorldPos)); // orient outward (toward camera) -> sign-safe
           float _facing = dot(_wn, vec3(${SUN_KEY_DIR.x.toFixed(4)}, ${SUN_KEY_DIR.y.toFixed(4)}, ${SUN_KEY_DIR.z.toFixed(4)})) * 0.5 + 0.5;
           float _facet = mix(1.0, mix(${BOOK_FACET_MIN.toFixed(3)}, 1.0, _facing), ${BOOK_FACET.toFixed(3)});
           gl_FragColor.rgb += diffuseColor.rgb * (uEmissive + glow * uEmissiveNear) * _facet;
           float _sun = daylightAt(vGlowXZ, vViewDist);
           gl_FragColor.rgb += diffuseColor.rgb * ${DAY_GLSL} * (_sun * ${GLOW_REVEAL.toFixed(3)}) * _facet;` +
          // temperature: warm the pooled book toward the yellow anchor by the shortest hue
          // arc (blue->cyan, green->yellow, purple->red), scaled by the pool so it heats on
          // approach and cools as the player leaves. Low-saturation page cream barely moves.
          `
           {
             float _warmth = glow * ${WARM_STRENGTH.toFixed(3)};
             vec3 _hsv = rgb2hsv(gl_FragColor.rgb);
             float _dh = mod(${WARM_HUE.toFixed(3)} - _hsv.x + 0.5, 1.0) - 0.5;
             _hsv.x = fract(_hsv.x + _dh * _warmth);
             gl_FragColor.rgb = hsv2rgb(_hsv);
           }`,
      );
  };
}

function buildField(
  field: Awaited<ReturnType<typeof loadPositions>>,
  bookNear: THREE.BufferGeometry, // LOD00, full detail, drawn closest
  bookMid: THREE.BufferGeometry, // LOD01, drawn across the mid band
  uPlayer: PlayerUniform, // shared player-position uniform (also drives the ground rake)
  daylight: DaylightUniforms, // shared daylight uniforms (also drift over the ground)
  world: World, // R_INNER/R_MAX define the era-temperature radial ramp
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
  // flatShading: each authored facet carries its own light/dark tone (the sphere's recipe), so
  // the facet self-shading in applyProximityGlow has real per-face normals to rake. Goes on
  // every tier so a book reads the same through an LOD swap; the box base gets it too.
  const farMat = new THREE.MeshLambertMaterial({ flatShading: true });
  applyDistanceFade(farMat);
  applyProximityGlow(farMat, glow, daylight);
  const midMat = new THREE.MeshLambertMaterial({ flatShading: true });
  applyDistanceFade(midMat);
  applyPageMask(midMat);
  applyProximityGlow(midMat, glow, daylight);
  const nearMat = new THREE.MeshLambertMaterial({ flatShading: true });
  applyDistanceFade(nearMat);
  applyPageMask(nearMat);
  applyProximityGlow(nearMat, glow, daylight);
  // All three now share applyProximityGlow as their outermost onBeforeCompile, so
  // their default program-cache keys (= onBeforeCompile.toString(), closure vars
  // excluded) collide. far has no page mask while mid/near do, so without a
  // distinguishing key three would hand all three whichever program compiled
  // first. Key on the actual patch stack: mid/near are identical (share a program,
  // correct), far is its own. Same defence the ground material uses for its holes.
  farMat.customProgramCacheKey = () => "book:fade+glow+daylight";
  midMat.customProgramCacheKey = () => "book:fade+page+glow+daylight";
  nearMat.customProgramCacheKey = () => "book:fade+page+glow+daylight";

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
  // era-temperature ramp: 0 at the modern inner ring -> 1 at the deep-past R_MAX.
  // eraT 0.5 (mid-radius) is the neutral pivot; warm inside, cool outside.
  const eraSpan = Math.max(1, world.R_MAX - world.R_INNER);
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
      let hue = ((lon[i] / 256) + HUE_OFFSET) % 1;
      // era temperature: rotate the resting hue toward the warm (centre) or cool (rim)
      // anchor by the shortest arc, scaled by distance from the neutral mid-radius. The
      // geo read survives because every book on a given ring shifts by the same amount.
      const eraT = clamp((Math.hypot(px[i], pz[i]) - world.R_INNER) / eraSpan, 0, 1);
      const anchor = eraT < 0.5 ? ERA_WARM_HUE : ERA_COOL_HUE;
      const tempStr = Math.abs(eraT - 0.5) * 2 * ERA_TEMP_STRENGTH;
      hue = mod1(hue + (mod1(anchor - hue + 0.5) - 0.5) * tempStr);
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
const TP_BEAM_RADIUS = 0.6; // radius at the TOP of the shaft
// The beam flares wider where it meets the ground, so its own faceted foot IS the ground
// marker -- there is no separate soft glow pad anymore. That smooth additive pool was the
// last thing breaking the lowpoly look (a blurry light blob in a world of hard facets); the
// flared foot is geometry, faceted like everything else. Pad switched off in terrain
// (TP_GLOW_STRENGTH = 0); raise it back to bring the old pad glow back.
const TP_BEAM_RADIUS_BASE = 3.0; // radius at the foot: the flared base, the marker itself
const TP_BEAM_COLOR = new THREE.Color(0x2f6cff); // blue, up the shaft; matches the pad rim
const TP_BEAM_COLOR_BASE = new THREE.Color(0x3df0ff); // cyan at the foot, handing off to the pad core
// Now the flared faceted foot IS the marker, so the beam must stay visible as you walk in
// (the old pad that carried the near-field cue is gone). It only dissolves once you are
// basically centred on the stand point, sparing the camera the view straight down the
// open cone. Full strength beyond FAR, gone within NEAR. Push these back up toward 18/50
// if the foot reads messy up close (the faceted cone may not need hiding the way the old
// round tube did -- that is the thing to eyeball).
const TP_BEAM_FADE_NEAR = 3;
const TP_BEAM_FADE_FAR = 14;
// Proximity ramp: far away the beam is a dim, plain-blue beacon so it draws the eye
// without dominating the horizon; it brightens to full cyan/blue intensity as the
// player closes in (and the FADE_NEAR clip still takes over once you're on the pad).
// Keep in sync with terrain.ts TP_GLOW_APPROACH_* so beam and pad ramp together.
const TP_BEAM_APPROACH_NEAR = 70; // camera xz distance at which it reaches full intensity
const TP_BEAM_APPROACH_FAR = 260; // beyond this it sits at the dim far level
const TP_BEAM_FAR_LEVEL = 0.5; // intensity multiplier when far (0..1)
// Daylight lift: the shaft is additive, so its alpha IS its brightness. The drifting
// daylight field (daylight.ts) scales that alpha UP when the light crosses the plaza and
// back to 1x (today's resting look) in shadow -- a one-sided boost, never a dim. The pad
// glow beneath it is deliberately held steady against shadow too (terrain TP_GLOW, added
// after the daylight multiply), so flooring the beam at its rest level keeps beam and pad
// agreeing: neither ever drops below findable, the beam just gains in the sun.
const TP_BEAM_DAY_BOOST = 7.5; // alpha multiplier in full daylight (1 = no lift). Eyeball.
// The ground is transparent (distance fade) at the default renderOrder 0. A beam at
// the same order could draw before the ground, which then paints its opaque-near sand
// straight over it (the beam writes no depth, so it can't defend those pixels).
// Drawing the beam after the ground fixes that; the depth TEST against the ground
// (which does write depth) still occludes the beam behind nearer dunes, so physical
// occlusion is preserved. Stays above the ground and the books (renderOrder 5).
const TP_BEAM_RENDER_ORDER = 10;

// Floating-crystal beacon: an alternative to the rising shard -- a faceted diamond hovering
// just out of reach over the plaza, slowly turning and bobbing (the "subtle animation" idea
// finds its home here: a turning jewel reads as alive, not as a gamey pulse). An octahedron
// is faceted by nature, so it belongs in the lowpoly world without any coaxing. Flip
// TP_FLOATING_CRYSTAL to A/B it against the shard; the spin/bob run off the shared drift
// clock in the vertex shader, so no per-frame JS.
const TP_FLOATING_CRYSTAL = true; // true: hovering crystal; false: the rising flared shard
const TP_CRYSTAL_SIZE = 2.0; // octahedron radius before the vertical stretch
const TP_CRYSTAL_STRETCH = 2.3; // taller than wide -> an elongated diamond shard
const TP_CRYSTAL_HOVER = 9.0; // centre height above ground; floats well clear, out of reach
const TP_CRYSTAL_OPACITY = 0.5; // additive base alpha
const TP_CRYSTAL_SPIN = 0.3; // turn rate, rad/s against the drift clock
const TP_CRYSTAL_BOB_AMP = 0.45; // gentle vertical float, world units
const TP_CRYSTAL_BOB_SPEED = 0.5; // bob rate

// Embedded-ball beacon (the current direction, 2026-06-06): the teleporter stops being a
// floating prop and becomes an OBJECT in the field like a book -- a beach-ball-sized icosphere
// half-sunk in the sand. The aim is one interaction grammar (look + E, same as a book) and to
// let the anchor join the topology relaxation pass instead of carving a plaza. It keeps the
// crystalline additive/faceted/daylight-reactive vocabulary of the crystal (so it still reads as
// a teal NODE, not a geo-hued book), just planted and grounded: no hover, no spin/bob -- the
// facet flare off the derivative normal already shimmers as the PLAYER walks past it. Sized and
// glowing a notch hotter than a book so it still carries at a distance; if that proves too weak,
// TP_BEAM_OVERHEAD flips the old vertical shaft back on above the ball. TP_BALL takes priority
// over TP_FLOATING_CRYSTAL / the beam path below (both kept behind their flags for A/B).
const TP_BALL = true; // true: embedded icosphere; false: fall through to crystal/beam
const TP_BALL_RADIUS = 1.4; // world units; ~a couple of book-lengths -- a beach ball among books
const TP_BALL_DETAIL = 1; // icosahedron subdivisions: 0 = 20 chunky faces, 1 = 80 (faceted sphere)
const TP_BALL_BURY = 0.4; // fraction of the DIAMETER below the sand: 0.5 = a clean half-dome
const TP_BALL_COLOR = new THREE.Color(0x2fb6e0); // teal albedo: a NODE hue, distinct from geo books
const TP_BALL_GLOW = 0.18; // steady self-glow as a fraction of the hue; flat add, keeps the gradient
const TP_BALL_NIGHT_FLOOR = 0.5; // how dark the ball goes in night shadow (1 = no darkening). Unlike
// the books it never crushes to NIGHT: a sphere needs its sun/sky shading to survive or it reads
// as a flat silhouette. So it darkens into the field's mood but keeps its form, full-lit in a pool.
// Fresnel rim: brighten the grazing silhouette so the ball reads as a glowing NODE lit from within,
// not a matte stone. View-dependent, added steady (glows at night) and AFTER the daylight floor so
// the edge is the node's own light. Falls on the lit facets untouched -> keeps the form we earned.
const TP_BALL_RIM = 0.7; // rim glow strength
const TP_BALL_RIM_POWER = 2.5; // falloff: higher = thinner, sharper rim
const TP_BALL_RIM_COLOR = new THREE.Color(0x6fe6ff); // a hotter cyan than the albedo -> luminous edge
const TP_BEAM_OVERHEAD = false; // also raise the vertical shaft above the ball (distance fallback)

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

function buildTeleporters(list: Teleporter[], daylight: DaylightUniforms) {
  // The floor marker is now a blue glow baked into the terrain shader at each plaza
  // (terrain TP_GLOW_*), not a prop built here -- being the ground itself it can never
  // read superimposed, and it carries the "stand here" cue up close exactly as the beam
  // fades out to spare the camera. So this builds only the far beacon. Note the pad does
  // NOT dim with the drifting daylight (its glow is added AFTER the ground's daylight
  // multiply, terrain.ts): both pad and beam instead take a one-sided sun BOOST
  // (TP_GLOW_DAY_BOOST / TP_BEAM_DAY_BOOST, kept matched) so the beacon swells when the
  // light crosses the plaza but never drops below findable in shadow.
  // The beam is an open-ended low-sided PRISM shaded as a crystalline light shard --
  // angular to match the faceted lowpoly dunes, not a smooth CG tube (the smoothness
  // was the one thing marking it as foreign in this world). Three terms shape it: a
  // facet term (alpha ~ |view·faceNormal|, the face normal taken from screen-space
  // derivatives the way the terrain does it) so each flat side flares as it turns to
  // face the eye and the silhouette sides still fade out (no hard outline); a vertical
  // fade that thins the shaft to transparent toward the top, as if the light dissipates
  // as it rises; and the colour/proximity grading below. Additive with no depth write so
  // the layers accumulate into a bright core, but depth TEST stays on: dunes occlude it,
  // and the player crests a ridge to find the light waiting (the compass, not the beam,
  // does the long-range wayfinding).
  const beamGeom = new THREE.CylinderGeometry(
    TP_BEAM_RADIUS, // top
    TP_BEAM_RADIUS_BASE, // foot: flared
    TP_BEAM_HEIGHT,
    5, // few sides: an angular shard, not a round tube. Odd count breaks the dead-on symmetry
    1,
    true, // open-ended: no caps to flare as flat discs when seen from above
  ); // radiusTop < radiusBottom: a faceted cone, narrow aloft, flaring at the foot
  const beamMat = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: TP_BEAM_COLOR },
      uColorBase: { value: TP_BEAM_COLOR_BASE },
      uOpacity: { value: 0.4 },
      uHeight: { value: TP_BEAM_HEIGHT },
      uFadeNear: { value: TP_BEAM_FADE_NEAR },
      uFadeFar: { value: TP_BEAM_FADE_FAR },
      uApproachNear: { value: TP_BEAM_APPROACH_NEAR },
      uApproachFar: { value: TP_BEAM_APPROACH_FAR },
      uFarLevel: { value: TP_BEAM_FAR_LEVEL },
      // shared with the books and the ground: same field, same drift clock, same dev
      // kill switch, so the beam lifts on the exact light that crosses its plaza.
      uDayBoost: { value: TP_BEAM_DAY_BOOST },
      uDaylight: daylight.uDaylight,
      uDriftTime: daylight.uDriftTime,
      uDaylightMix: daylight.uDaylightMix,
    },
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      uniform float uHeight;
      varying vec3 vWorldPos;
      varying float vT;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vT = (position.y + uHeight * 0.5) / uHeight; // 0 at base, 1 at top
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      ${DAYLIGHT_FRAG_COMMON}
      uniform vec3 uColor;
      uniform vec3 uColorBase;
      uniform float uOpacity;
      uniform float uFadeNear;
      uniform float uFadeFar;
      uniform float uApproachNear;
      uniform float uApproachFar;
      uniform float uFarLevel;
      uniform float uDayBoost;
      varying vec3 vWorldPos;
      varying float vT;
      void main() {
        vec3 viewDir = normalize(cameraPosition - vWorldPos);
        // flat FACE normal from screen-space derivatives (the terrain's trick): the prism's
        // few flat sides each shade as a panel, brightest when turned to face the eye and
        // fading at the grazing silhouette sides -- a faceted shard, not a smooth tube.
        vec3 faceN = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        float edge = abs(dot(viewDir, faceN));   // 1 on a face turned to the eye, 0 at the silhouette
        float vert = pow(1.0 - clamp(vT, 0.0, 1.0), 1.5); // dissipates toward the top
        // horizontal camera distance (xz only, so looking up the tall shaft doesn't
        // trigger it): fade the whole beam out as you approach, hiding the waterline.
        float camDist = distance(cameraPosition.xz, vWorldPos.xz);
        float camFade = smoothstep(uFadeNear, uFadeFar, camDist);
        // the flared foot is the marker now, so keep it bright -- only the very bottom
        // sliver softens, just to ease the contact line where the cone meets the ground.
        float baseFade = smoothstep(0.0, 0.03, vT);
        // proximity: dim, plain-blue when far; bright, cyan-footed when near.
        float approach = smoothstep(uApproachFar, uApproachNear, camDist); // 0 far, 1 near
        float prox = mix(uFarLevel, 1.0, approach);
        vec3 footCol = mix(uColor, uColorBase, approach); // cyan foot only emerges on approach
        vec3 col = mix(footCol, uColor, clamp(vT, 0.0, 1.0)); // foot grading to blue up the shaft
        // one-sided daylight lift: 1x at rest (in shadow, matching the steady pad), up to
        // uDayBoost when a daylight pool crosses the plaza, so the beacon breathes with the
        // world instead of holding a constant blue. beaconLit is the point-sample, distance-
        // independent field reader (not the ground's daylightAt, which coarsens with distance
        // and would starve this FAR beacon). uDaylightMix folds in the dev flat-lit switch.
        float lit = beaconLit(vWorldPos.xz);
        float dayGain = mix(1.0, uDayBoost, lit * uDaylightMix);
        gl_FragColor = vec4(col, uOpacity * edge * vert * camFade * baseFade * prox * dayGain);
      }
    `,
  });
  // Floating crystal: a faceted diamond hovering over the plaza. Same crystalline shading
  // language as the shard (additive, faceted off the derivative face normal, daylight +
  // proximity reactive) but it floats clear of the ground, so no waterline/camFade machinery
  // is needed. The slow spin + bob live in the vertex shader off the shared drift clock.
  const crystalGeom = new THREE.OctahedronGeometry(TP_CRYSTAL_SIZE, 0);
  crystalGeom.scale(1, TP_CRYSTAL_STRETCH, 1); // stretch tall: a diamond, not a ball
  const crystalMat = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: TP_BEAM_COLOR },
      uColorCore: { value: TP_BEAM_COLOR_BASE },
      uOpacity: { value: TP_CRYSTAL_OPACITY },
      uApproachNear: { value: TP_BEAM_APPROACH_NEAR },
      uApproachFar: { value: TP_BEAM_APPROACH_FAR },
      uFarLevel: { value: TP_BEAM_FAR_LEVEL },
      uDayBoost: { value: TP_BEAM_DAY_BOOST },
      uSpin: { value: TP_CRYSTAL_SPIN },
      uBobAmp: { value: TP_CRYSTAL_BOB_AMP },
      uBobSpeed: { value: TP_CRYSTAL_BOB_SPEED },
      uDaylight: daylight.uDaylight,
      uDriftTime: daylight.uDriftTime,
      uDaylightMix: daylight.uDaylightMix,
    },
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: /* glsl */ `
      uniform float uDriftTime;
      uniform float uSpin;
      uniform float uBobAmp;
      uniform float uBobSpeed;
      varying vec3 vWorldPos;
      void main() {
        float a = uDriftTime * uSpin;
        float s = sin(a), c = cos(a);
        vec3 p = position;
        p.xz = mat2(c, -s, s, c) * p.xz;               // slow turn about the vertical
        vec4 wp = modelMatrix * vec4(p, 1.0);
        wp.y += sin(uDriftTime * uBobSpeed) * uBobAmp; // gentle float
        vWorldPos = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */ `
      ${DAYLIGHT_FRAG_COMMON}
      uniform vec3 uColor;
      uniform vec3 uColorCore;
      uniform float uOpacity;
      uniform float uApproachNear;
      uniform float uApproachFar;
      uniform float uFarLevel;
      uniform float uDayBoost;
      varying vec3 vWorldPos;
      void main() {
        vec3 viewDir = normalize(cameraPosition - vWorldPos);
        // flat FACE normal from derivatives: each facet flares as it turns to the eye, so the
        // turning crystal sparkles facet by facet rather than glowing as a smooth ball.
        vec3 faceN = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        float edge = abs(dot(viewDir, faceN));
        float camDist = distance(cameraPosition.xz, vWorldPos.xz);
        float approach = smoothstep(uApproachFar, uApproachNear, camDist); // 0 far, 1 near
        float prox = mix(uFarLevel, 1.0, approach);
        vec3 col = mix(uColor, uColorCore, edge); // cyan on the facing facets, blue at grazing
        // one-sided daylight lift, same as the shard: swells when a pool crosses the plaza.
        float lit = beaconLit(vWorldPos.xz);
        float dayGain = mix(1.0, uDayBoost, lit * uDaylightMix);
        gl_FragColor = vec4(col, uOpacity * edge * prox * dayGain);
      }
    `,
  });
  // Embedded ball: a SOLID, opaque, flat-shaded icosphere lit by the scene hemisphere + warm sun
  // (MeshLambertMaterial, flatShading) so the facets carry a real light/dark gradient -- that is
  // what makes a sphere read as a sphere, and it is the one thing a flat book can do without and
  // a ball cannot. It does NOT take the books' daylight chain, which multiplies toward near-black
  // NIGHT and would crush all that shading flat (fatal on a curved surface). Instead daylight is a
  // FLOORED darkening (beaconLit, the point-sample reader): the ball sinks into the field's night
  // mood but never below TP_BALL_NIGHT_FLOOR, so its form always survives, and lifts to full in a
  // sun pool. A small teal self-glow rides on top as the node's own light. Opaque + depth-writing,
  // so the buried hemisphere is occluded by the ground and books behind it sort right.
  const ballGeom = new THREE.IcosahedronGeometry(TP_BALL_RADIUS, TP_BALL_DETAIL);
  const ballMat = new THREE.MeshLambertMaterial({
    color: TP_BALL_COLOR,
    flatShading: true,
  });
  ballMat.onBeforeCompile = (shader) => {
    shader.uniforms.uDaylight = daylight.uDaylight;
    shader.uniforms.uDriftTime = daylight.uDriftTime;
    shader.uniforms.uDaylightMix = daylight.uDaylightMix;
    shader.uniforms.uRimColor = { value: TP_BALL_RIM_COLOR };
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vBallXZ;",
      )
      // carry world xz for the daylight field reader.
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vBallXZ = (modelMatrix * vec4(transformed, 1.0)).xz;`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vBallXZ;\nuniform vec3 uRimColor;\n" +
          DAYLIGHT_FRAG_COMMON,
      )
      .replace(
        "#include <opaque_fragment>",
        `#include <opaque_fragment>
         // floored daylight: darken into the field at night but keep the sphere's sun/sky form,
         // lift to full in a pool. beaconLit is the distance-independent point reader; uDaylightMix
         // is the dev flat-lit kill switch (-> no darkening).
         float _day = beaconLit(vBallXZ);
         float _mul = mix(${TP_BALL_NIGHT_FLOOR.toFixed(3)}, 1.0, _day);
         gl_FragColor.rgb *= mix(1.0, _mul, uDaylightMix);
         // node self-glow: a flat add, so it brightens without flattening the facet gradient.
         gl_FragColor.rgb += diffuseColor.rgb * ${TP_BALL_GLOW.toFixed(3)};
         // fresnel rim: the grazing silhouette glows, so the ball reads as lit from within. normal
         // is the flat facet normal (flatShading), so the rim is faceted too; added steady, after
         // the daylight floor, as the node's own light. vViewPosition points fragment->camera.
         float _fres = pow(1.0 - clamp(dot(normalize(vViewPosition), normal), 0.0, 1.0), ${TP_BALL_RIM_POWER.toFixed(2)});
         gl_FragColor.rgb += uRimColor * (_fres * ${TP_BALL_RIM.toFixed(3)});`,
      );
  };
  // centre height so a (1 - TP_BALL_BURY) fraction of the diameter clears the sand.
  const ballCentreY = TP_BALL_RADIUS * (1.0 - 2.0 * TP_BALL_BURY);
  const group = new THREE.Group();
  for (const tp of list) {
    const h = sampleHeight(tp.x, tp.y);
    if (TP_BALL) {
      // a beach-ball icosphere half-sunk in the sand at the anchor, a solid teal node.
      const ball = new THREE.Mesh(ballGeom, ballMat);
      ball.position.set(tp.x, h + ballCentreY, tp.y);
      group.add(ball);
      if (TP_BEAM_OVERHEAD) {
        // optional vertical shaft rising above the ball as a distance beacon.
        const beam = new THREE.Mesh(beamGeom, beamMat);
        beam.position.set(tp.x, h + ballCentreY + TP_BEAM_HEIGHT / 2, tp.y);
        beam.renderOrder = TP_BEAM_RENDER_ORDER;
        group.add(beam);
      }
    } else if (TP_FLOATING_CRYSTAL) {
      // a faceted diamond hovering just out of reach over the plaza, slowly turning.
      const crystal = new THREE.Mesh(crystalGeom, crystalMat);
      crystal.position.set(tp.x, h + TP_CRYSTAL_HOVER, tp.y);
      crystal.renderOrder = TP_BEAM_RENDER_ORDER;
      group.add(crystal);
    } else {
      // beam rising from the plaza centre, base at the ground. Drawn after the
      // transparent ground levels (see TP_BEAM_RENDER_ORDER) so they can't overpaint
      // it; it fades out by camera distance (shader) before the waterline shows.
      const beam = new THREE.Mesh(beamGeom, beamMat);
      beam.position.set(tp.x, h + TP_BEAM_HEIGHT / 2, tp.y);
      beam.renderOrder = TP_BEAM_RENDER_ORDER;
      group.add(beam);
    }
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

// What the reticle is over: a book to inspect or a teleporter ball to step
// through. One grammar -- look at it, press E -- so both flow through the picker
// and the single #glance prompt; the kind only decides which panel E opens.
type Aim = { kind: "book" | "tp"; i: number } | null;

// Look-at picker: each tick, find the book nearest the camera whose seat lies
// within a thin cylinder around the view ray. A distance cull rejects almost all
// 576k instances before the alignment test, so the brute-force sweep is cheap at
// the throttled cadence. Aims at jittered positions (matching what's drawn). The
// 26 teleporter balls are swept the same way (as spheres, not points) right after
// and compete for the reticle by eye-distance, so a book in front of a ball wins.
function createPicker(
  camera: THREE.PerspectiveCamera,
  px: Float32Array,
  pz: Float32Array,
  teleporters: Teleporter[],
) {
  // reading is close-up only: you have to travel and walk up to a book to learn
  // who it is. Long-range legibility (landmarks visible from afar) is a separate
  // problem for a beacon VFX, not for this reach. ~6 u ≈ 4 m at 1.4 u/m.
  const MAX_DIST = 6;
  const MAX_DIST2 = MAX_DIST * MAX_DIST;
  // The tolerance is a world-space radius around the aim ray, not an angle: a
  // fixed angle holds a constant SCREEN tolerance while the book shrinks with
  // distance, so a cone that fits up close balloons to several book-widths at
  // MAX_DIST. A fixed world radius subtends a shrinking angle, so its screen
  // tolerance tracks the book's own. Sized a touch over the cover half-length
  // (BOOK_LENGTH·BOOK_FOOTPRINT/2 ≈ 0.2u) for forgiveness without pixel-aim.
  const PICK_RADIUS = 0.45;
  const PICK_RADIUS2 = PICK_RADIUS * PICK_RADIUS;
  const n = px.length;
  const fwd = new THREE.Vector3();

  // The teleporter ball is a far larger target than a book and one you walk up to
  // as a landmark, so it gets a more generous reach and the tolerance is its own
  // silhouette (the ball radius, with a little forgiveness) rather than a hair-thin
  // cylinder. Its centre sits TP_BALL embed-offset above the baked ground; the ball
  // never moves, so precompute the aim sphere once (sampleHeight is fixed by now).
  const TP_AIM_REACH = 16;
  const TP_AIM_RADIUS = TP_BALL_RADIUS + 0.3;
  const TP_AIM_RADIUS2 = TP_AIM_RADIUS * TP_AIM_RADIUS;
  const tpCentreY = TP_BALL_RADIUS * (1.0 - 2.0 * TP_BALL_BURY);
  const tpAim = teleporters.map((tp) => ({
    x: tp.x,
    z: tp.y,
    cy: sampleHeight(tp.x, tp.y) + tpCentreY,
  }));

  return function pick(): Aim {
    camera.getWorldDirection(fwd);
    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;
    let best = -1;
    let bestDist2 = Infinity;
    for (let i = 0; i < n; i++) {
      const dx = px[i] - cx;
      const dz = pz[i] - cz;
      const horiz2 = dx * dx + dz * dz;
      if (horiz2 > MAX_DIST2) continue;
      // each book sits on its own ground, not the player's: on a slope a book a
      // few units away can be a couple of units higher or lower than underfoot,
      // so sampling the player's feet (or y=0) mis-aims it. Aim at the book's own
      // seat. Only the handful within MAX_DIST survive the cull, so this is cheap.
      const dy = sampleHeight(px[i], pz[i]) - cy;
      // split the offset into along-ray (t, how far ahead) and perpendicular
      // (how far off the aim line). Reject anything behind the eye, beyond reach,
      // or outside the cylinder radius, then keep the nearest survivor.
      const t = dx * fwd.x + dy * fwd.y + dz * fwd.z;
      if (t <= 0 || t > MAX_DIST) continue;
      const dist2 = horiz2 + dy * dy;
      const perp2 = dist2 - t * t;
      if (perp2 > PICK_RADIUS2) continue;
      if (dist2 < bestDist2) {
        bestDist2 = dist2;
        best = i;
      }
    }
    // teleporter balls: same along-ray / perpendicular split, but tested as a
    // sphere of the ball's own radius so the whole silhouette is targetable.
    let bestTp = -1;
    let bestTpDist2 = Infinity;
    for (let i = 0; i < tpAim.length; i++) {
      const a = tpAim[i];
      const dx = a.x - cx;
      const dy = a.cy - cy;
      const dz = a.z - cz;
      const t = dx * fwd.x + dy * fwd.y + dz * fwd.z;
      if (t <= 0 || t > TP_AIM_REACH) continue;
      const dist2 = dx * dx + dy * dy + dz * dz;
      const perp2 = dist2 - t * t;
      if (perp2 > TP_AIM_RADIUS2) continue;
      if (dist2 < bestTpDist2) {
        bestTpDist2 = dist2;
        bestTp = i;
      }
    }
    // both can sit under the reticle (a book just in front of a ball); the nearer
    // to the eye wins, so an actual book you're nose-to-nose with takes priority.
    if (bestTp >= 0 && (best < 0 || bestTpDist2 < bestDist2)) {
      return { kind: "tp", i: bestTp };
    }
    if (best >= 0) return { kind: "book", i: best };
    return null;
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
  // Inset the pitch limits one degree off true vertical: at the poles the look
  // direction's horizontal component collapses and the compass bearing degenerates.
  const POLE_GUARD = Math.PI / 180; // 1°
  controls.minPolarAngle = POLE_GUARD;
  controls.maxPolarAngle = Math.PI - POLE_GUARD;
  const keys = new Set<string>();
  const vel = new THREE.Vector3(); // carried horizontal velocity (xz; y stays 0)
  let flying = false;

  // vertical eye state, kept separate from camera.position.y so the cosmetic skate
  // lift can ride on top without feeding the physics. eyeY is the lift-free physics
  // height; vy is non-zero only mid-hop; vSurf is the smoothed rate the ground rises
  // under the body (the speed a crest launch inherits). lastAppliedY is what we last
  // wrote to camera.position.y, so an external write (spawn settle, teleport) shows up
  // as a mismatch next frame and we resync onto it instead of fighting it.
  let eyeY = camera.position.y;
  let vy = 0;
  let grounded = true;
  let prevBodyTop = camera.position.y;
  let vSurf = 0;
  let liftCur = 0;
  let lastAppliedY = camera.position.y;

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

    // pin the eye to the baked ground, smoothed, with a ballistic launch over fast-
    // crested dunes. Sample at the body AND a velocity-scaled look-ahead: the max lifts
    // the eye onto a rising slope before you reach it (a climb, not a snap or a clip),
    // while the body sample alone governs leaving and landing.
    const gHere = sampleHeight(camera.position.x, camera.position.z);
    const gAhead = sampleHeight(
      camera.position.x + vel.x * MOVE.lookAhead,
      camera.position.z + vel.z * MOVE.lookAhead,
    );
    const bodyTop = gHere + EYE_HEIGHT; // the surface under the feet, eye-high

    // resync if something outside the controller moved the eye (spawn settle, teleport):
    // those write the bare surface + EYE_HEIGHT, so adopt it as the base and drop any
    // carried vertical state rather than smoothing/launching across the discontinuity.
    if (Math.abs(camera.position.y - lastAppliedY) > 1e-4) {
      eyeY = camera.position.y;
      vy = 0;
      grounded = true;
      vSurf = 0;
      prevBodyTop = bodyTop;
    }

    // smoothed rate the ground rises under the body: the climb speed a crest inherits.
    const vyBody = (bodyTop - prevBodyTop) / dt;
    vSurf += (vyBody - vSurf) * (1 - Math.exp(-MOVE.followK * dt));
    prevBodyTop = bodyTop;

    if (grounded) {
      if (vSurf > MOVE.launchMin && vyBody < vSurf * 0.5) {
        // crest: the ground was climbing fast and has now turned over (the raw rate
        // collapsed below half the smoothed climb) → leave it carrying that speed.
        grounded = false;
        vy = Math.min(vSurf, MOVE.vyMax);
      } else {
        // tracking the surface: the time-constant filter keeps the feet planted at a
        // walk and smooths dune bumps at skate speed where rigid tracking would jolt.
        const followY = Math.max(gHere, gAhead) + EYE_HEIGHT;
        eyeY += (followY - eyeY) * (1 - Math.exp(-MOVE.followK * dt));
        if (eyeY < bodyTop) eyeY = bodyTop; // hard floor: never clip the body on a climb
      }
    }
    if (!grounded) {
      vy -= MOVE.gravity * dt;
      eyeY += vy * dt;
      if (eyeY <= bodyTop) {
        eyeY = bodyTop; // land on the true surface under the body
        vy = 0;
        grounded = true;
      }
    }

    // cosmetic skate lift, eased in/out, applied on top of the physics eye height.
    const liftTarget = skating ? MOVE.hoverLift : 0;
    liftCur += (liftTarget - liftCur) * (1 - Math.exp(-MOVE.followK * dt));
    camera.position.y = eyeY + liftCur;
    lastAppliedY = camera.position.y;

    return skating ? "skate" : "walk";
  }

  // zero the carried velocity (teleport/spawn shouldn't arrive mid-glide).
  const stop = () => vel.set(0, 0, 0);

  // flight is a dev affordance; the pause menu drives it through the same path as
  // the F key (drop carried momentum so neither mode lurches), and reads it back to
  // keep the menu checkbox in sync with the key.
  const getFlying = () => flying;
  const setFlying = (v: boolean) => {
    if (v === flying) return;
    flying = v;
    vel.set(0, 0, 0);
  };

  return { controls, update, stop, getFlying, setFlying };
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

  // drifting daylight (see daylight.ts): one shared field + time, sampled at the
  // world xz of the ground, books and stones so the same shadow falls on a
  // book and the sand under it, AND at the sky dome's pierce points so the storm's
  // breaks open over the lit patches. The scene's main source of large-scale motion.
  // Built before the sky because the dome shares its uniforms.
  const daylight = buildDaylight();

  // storm sky dome (see sky.ts): a near-black cloud ceiling whose breaks are cut by
  // the same drifting field that lights the ground, so the sky opens where the dunes
  // beneath are lit. A warm glow stays fixed at the sun bearing; the whole dome
  // deepens with the player's radial depth into the past. It recentres on the camera
  // each frame (see the loop) so it reads as infinitely far and never shows an edge.
  const sky = buildSky(daylight.uniforms, SUN_POS);
  scene.add(sky.group);

  const camera = new THREE.PerspectiveCamera(
    70,
    window.innerWidth / window.innerHeight,
    0.1,
    // far plane must clear the day sky layer's diagonal corner (sky.ts CEIL_R*sqrt2 ~ 37k)
    // so the enlarged layer reaches near the horizon without its corners being clipped.
    40000,
  );

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const { controls, update, stop, getFlying, setFlying } = createController(
    camera,
    renderer.domElement,
  );
  scene.add(controls.object);
  // dev hook, same convention as window.MOVE: lets the camera be posed/inspected from the
  // devtools console (or a headless screenshot) without pointer lock or a rebuild.
  (window as unknown as { CAM: THREE.PerspectiveCamera }).CAM = camera;

  // perf monitor: stats.js panel (click to cycle FPS / ms / MB) plus a text
  // readout of the numbers that actually tell us if the book LOD is working,
  // draw calls, triangles, and the live detailed-book count vs the box field.
  // dev-only readout, hidden by default so the shipped view is clean; ` (Backquote)
  // toggles it. Until the options menu exists this is the one dev affordance that
  // earns a key, since the perf numbers are how we confirm the book LOD is working.
  const stats = new Stats();
  stats.dom.style.cssText = "position:fixed;top:0;left:0;z-index:100;display:none;";
  document.body.appendChild(stats.dom);
  const perf = document.createElement("div");
  perf.style.cssText =
    "position:fixed;top:48px;left:0;z-index:100;padding:4px 6px;display:none;" +
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
  // slopes, which is what makes the dunes read as 3D), over a genuinely COOL sky
  // ambient. The cool sky colour is the load-bearing choice: the warm sun rakes the
  // dune faces warm, while flat ground (and the near-flat vantage centre, which
  // catches almost no direct sun) is lit mostly by this cool ambient, so it reads
  // cool. That warm/cool split is what lets the vortex eye read cold WITHOUT an
  // emissive, and stops a pale albedo collapsing to "the colour of the only light"
  // (it was all-warm before: the old sky colour 0xffd9b3 was warm, so nothing cool
  // reached the desert and white read as sand). The sun sits on SUN_POS, the same
  // bearing the sky's warm glow uses, so the lit ground and the bright sky agree.
  scene.add(new THREE.HemisphereLight(0x7e95bd, 0x2b2f47, 0.5));
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
  const [field, teleporters, meta, world, heightmap, bookLods] =
    await Promise.all([
      loadPositions("positions.bin"),
      loadTeleporters("teleporters.json"),
      loadMeta("meta.bin"),
      loadWorld("world.json"),
      loadHeightmap("heightmap.bin"),
      loadBookLods(bookUrl, ["book_LOD00", "book_LOD01"]),
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
  const uPlayer: PlayerUniform = { value: new THREE.Vector2(0, 0), lift: { value: 0 } };
  // skate state for the ground's cool glow pool: ramped 0..1 in the loop so the blue
  // blooms in/out with the mode rather than snapping on with the Shift key.
  const uSkate = { value: 0 };
  // the ground is one static mesh tessellated from the heightmap (terrain.buildGround):
  // no camera-following, no rebuild, no LOD seams. It just sits there; the raking light
  // and daylight ride on shared uniforms updated in the loop.
  const ground = buildGround(uPlayer, uSkate, daylight.uniforms);
  scene.add(ground);
  // settle the eye onto the baked surface now the heightmap is loaded (spawn was
  // placed on the analytic fallback before the fetch resolved).
  camera.position.y = sampleHeight(camera.position.x, camera.position.z) + EYE_HEIGHT;
  mark("terrain");
  const built = buildField(field, bookLods[0], bookLods[1], uPlayer, daylight.uniforms, world);
  mark("seat books");

  // Wall the player just past the outermost book. R_MAX (the nominal time-radius) is
  // the wrong number: the date-uncertainty frontier scatters books well beyond it (the
  // furthest sits near 8080 vs an R_MAX of 7100), so the wall is derived from the
  // actual render positions and self-corrects on a data regen. Clamped inside the mesh
  // half-extent so the stop lands in fog over real ground, never at the mesh edge.
  let maxR2 = 0;
  for (let i = 0; i < built.px.length; i++) {
    const d = built.px[i] * built.px[i] + built.pz[i] * built.pz[i];
    if (d > maxR2) maxR2 = d;
  }
  const outermost = Math.sqrt(maxR2);
  const meshLimit = heightmap.worldSize / 2 - 200; // keep clear of the mesh edge
  MOVE.boundSoft = Math.min(outermost + 250, meshLimit - 450);
  MOVE.boundHard = Math.min(outermost + 700, meshLimit);

  built.update(camera.position.x, camera.position.z);
  scene.add(built.group);
  scene.add(buildTeleporters(teleporters, daylight.uniforms));

  mark("props");

  // --- look-at glance + inspect overlay -------------------------------------
  const glance = document.getElementById("glance") as HTMLDivElement;
  const reticle = document.getElementById("reticle") as HTMLDivElement;
  const controlsHud = document.getElementById("controls") as HTMLDivElement;
  const overlay = document.getElementById("overlay") as HTMLDivElement;
  const card = document.getElementById("card") as HTMLDivElement;
  const fade = document.getElementById("fade") as HTMLDivElement;
  const pick = createPicker(camera, built.px, built.pz, teleporters);
  const compass = createCompass(camera, built.px, built.pz, meta, world);

  let aim: Aim = null; // what the reticle is over (book or teleporter), or null
  let overlayOpen = false;

  const renderName = (i: number) => escapeHtml(meta.name(i) || "(untitled)");
  const renderDesc = (i: number) => escapeHtml(meta.desc(i));

  // The look-at prompt for whatever's under the reticle. A teleporter borrows the
  // book's glance layout exactly -- name / detail / era / action -- so the two read
  // as one grammar; only the words and the verb ("travel" vs "inspect") differ.
  function showGlance(a: { kind: "book" | "tp"; i: number }) {
    if (a.kind === "tp") {
      const tp = teleporters[a.i];
      glance.innerHTML =
        `<div class="name">◎ ${escapeHtml(tp.label)}</div>` +
        (tp.seat ? `<div class="desc">${escapeHtml(tp.seat)}</div>` : "") +
        (tp.era ? `<div class="years">${escapeHtml(tp.era)}</div>` : "") +
        `<div class="act"><span class="key">E</span> travel</div>`;
      glance.style.display = "block";
      return;
    }
    const i = a.i;
    const desc = renderDesc(i);
    const years = fmtYears(meta.birth[i], meta.death[i]);
    glance.innerHTML =
      `<div class="name">${renderName(i)}</div>` +
      (desc ? `<div class="desc">${desc}</div>` : "") +
      (years ? `<div class="years">${years}</div>` : "") +
      `<div class="act"><span class="key">E</span> inspect</div>`;
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
    reticle.classList.remove("armed");
    controls.unlock(); // free the cursor so the link is clickable
  }

  function closeOverlay() {
    overlay.style.display = "none";
    overlayOpen = false;
    // straight back to walking, not the bare-cursor limbo. The overlay unlocked the
    // pointer programmatically (not a user Esc), so no post-Esc throttle applies here;
    // every caller (scrim click, Esc keydown, a travel pick) is a live user gesture, so
    // the re-lock request is honoured. (A book/tp open suppressed the pause menu via
    // overlayOpen, so this is the only thing that owns the unlocked moment.)
    controls.lock();
  }

  // --- teleporter travel ----------------------------------------------------
  // The 26 balls are an any-to-any fast-travel network across a disc too wide to
  // walk. You look at a ball (the picker arms the glance, same as a book) and press
  // E to open this menu of the other anchors, nearest first; a pick jumps you there.
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
      `<div class="desc">Visit another time and place.</div>` +
      `<div class="tp-list">${rows}</div>` +
      `<div class="hint">Esc or click outside to close</div>`;
    card.querySelectorAll<HTMLButtonElement>(".tp-dest").forEach((btn) => {
      btn.addEventListener("click", () => travelTo(Number(btn.dataset.i)));
    });
    overlay.style.display = "flex";
    overlayOpen = true;
    glance.style.display = "none";
    controls.unlock(); // free the cursor so destinations are clickable
  }

  // click on the backdrop (not the card) closes; clicking the card/link doesn't.
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeOverlay();
  });
  // ── dev toggles ─────────────────────────────────────────────────────────
  // Each dev affordance has ONE setter so the keyboard shortcut and the pause-menu
  // checkbox drive identical state; the menu re-reads the is* getters on open to stay
  // in sync with whatever the keys did since. (Flight is the controller's, via F.)
  const setFlatLit = (on: boolean) => {
    // toggle the daylight (storm-shadow) field off, so the ground shows its full
    // Lambert-lit albedo with no day/night darkening — judges the sand colours without
    // the world shrouded in moving shadow. uDaylightMix is shared into the ground and
    // book materials, so one flip neutralises the whole field.
    daylight.uniforms.uDaylightMix.value = on ? 0 : 1;
  };
  const isFlatLit = () => daylight.uniforms.uDaylightMix.value === 0;
  const setPerf = (on: boolean) => {
    // show/hide the perf readout (stats.js + the LOD numbers). Hidden by default so the
    // shipped view is clean; this is how we confirm the book LOD is working.
    stats.dom.style.display = on ? "block" : "none";
    perf.style.display = on ? "block" : "none";
  };
  const isPerf = () => stats.dom.style.display !== "none";

  // ── pause menu ──────────────────────────────────────────────────────────
  // The unlocked state IS the menu: releasing the pointer lock (Esc) over the scene
  // surfaces it; Resume or a scrim click re-locks. The teleporter overlay also unlocks
  // (to free the cursor for its links) but sets overlayOpen first, so we suppress the
  // menu there and let the card own that unlocked moment. We DON'T bind Esc-to-resume:
  // browsers throttle requestPointerLock for ~1s after an Esc-exit, so a keyboard
  // re-lock would silently fail — the click paths land well after that window. Hiding
  // is driven by the 'lock' event (not optimistically), so a throttled lock leaves the
  // menu up to click again rather than stranding a bare cursor.
  const pauseEl = document.getElementById("pause") as HTMLDivElement;
  const devModeEl = document.getElementById("dev-mode") as HTMLInputElement;
  const devOptsEl = document.getElementById("dev-opts") as HTMLDivElement;
  const devFlyEl = document.getElementById("dev-fly") as HTMLInputElement;
  const devFlatEl = document.getElementById("dev-flat") as HTMLInputElement;
  const devPerfEl = document.getElementById("dev-perf") as HTMLInputElement;
  const spinnerEl = document.getElementById("pause-spinner") as HTMLSpanElement;

  // The menu only ever opens on an Esc exit (overlay unlocks are suppressed above), so
  // the post-Esc re-lock cooldown is ALWAYS ticking when it appears — a documented fixed
  // 1250ms (Chromium kEffectiveUserEscapeDuration). Spin a little indicator for that
  // window so a too-early Resume click reads as "preparing", not broken; clear it on a
  // timer. This is purely the visual hint — the actual re-lock stays attempt-driven, so
  // if the constant ever differs the lock still takes correctly, only the spinner's
  // dwell would be off. The spinner is absolutely positioned, so toggling it never
  // reflows the card.
  const COOLDOWN_HINT_MS = 1250;
  let cooldownTimer = 0;
  const syncDevOpts = () => {
    devOptsEl.style.display = devModeEl.checked ? "block" : "none";
  };
  const showPause = () => {
    devFlyEl.checked = getFlying();
    devFlatEl.checked = isFlatLit();
    devPerfEl.checked = isPerf();
    syncDevOpts();
    pauseEl.style.display = "flex";
    spinnerEl.classList.add("show");
    if (cooldownTimer) clearTimeout(cooldownTimer);
    cooldownTimer = window.setTimeout(() => {
      cooldownTimer = 0;
      spinnerEl.classList.remove("show");
    }, COOLDOWN_HINT_MS);
  };
  // Resume re-locks the pointer. The pause menu is reached via Esc — the one exit that
  // trips Chrome's post-Esc throttle (the book/tp overlays unlock programmatically, so
  // they re-lock instantly; a pause resumed inside the throttle can't be quite that
  // quick). Rather than guess the throttle's length and gate on it — an over-long guess
  // eats good clicks (the bug that made this feel laggy), too short re-spams the error —
  // we just ATTEMPT the lock on every click: it takes the instant the browser allows,
  // usually the first click, and a too-early click is a silent no-op (the menu hides
  // only on the real 'lock' event, so it stays up and the next click takes). We call
  // requestPointerLock directly to OWN its promise, so a throttled reject is a caught
  // no-op rather than an uncaught SecurityError; three's own pointerlockerror logger is
  // dropped just below so the throttled attempt prints nothing either.
  const resume = () => {
    const p = renderer.domElement.requestPointerLock() as Promise<void> | undefined;
    if (p && typeof p.then === "function") p.catch(() => {});
  };
  // three logs a console.error from its own pointerlockerror handler on every failed
  // lock; since our resume makes deliberate throttled attempts we handle via the caught
  // promise, that log is pure noise. Remove it (guarded: a future three rename just
  // brings the log back, it can't crash).
  const errLogger = (controls as unknown as { _onPointerlockError?: EventListener })
    ._onPointerlockError;
  if (errLogger)
    renderer.domElement.ownerDocument.removeEventListener("pointerlockerror", errLogger);

  // Return to start: a lost player jumps back to the spawn centre. Reuses the
  // teleporter's fade-to-black so the position pop is never seen, and restores the
  // FULL opening framing — eye on the summit at (0,0) AND the outward gaze across the
  // plaza (lookAt; PointerLockControls reads the camera each pointermove, so deltas
  // resume from the reset heading). resume() runs SYNCHRONOUSLY on the click so the
  // re-lock rides this gesture (a setTimeout re-lock would be rejected as
  // kRequiresUserGesture); the reposition waits for the black frame.
  const returnToStart = () => {
    resume(); // re-lock back to walking, on this click's user gesture
    fade.style.opacity = "1"; // fade to black (CSS transition: 0.3s)
    window.setTimeout(() => {
      camera.position.set(0, sampleHeight(0, 0) + EYE_HEIGHT, 0);
      camera.lookAt(0, 0, 8000);
      stop(); // arrive at rest, not mid-glide
      fade.style.opacity = "0"; // fade back in at the centre
    }, 300);
  };

  devModeEl.addEventListener("change", syncDevOpts);
  devFlyEl.addEventListener("change", () => setFlying(devFlyEl.checked));
  devFlatEl.addEventListener("change", () => setFlatLit(devFlatEl.checked));
  devPerfEl.addEventListener("change", () => setPerf(devPerfEl.checked));
  (document.getElementById("pause-resume") as HTMLButtonElement).addEventListener(
    "click",
    resume,
  );
  (document.getElementById("pause-respawn") as HTMLButtonElement).addEventListener(
    "click",
    returnToStart,
  );
  pauseEl.addEventListener("click", (e) => {
    if (e.target === pauseEl) resume(); // scrim click resumes; clicks on the card don't
  });
  controls.addEventListener("unlock", () => {
    if (!overlayOpen) showPause();
  });
  controls.addEventListener("lock", () => {
    pauseEl.style.display = "none";
    spinnerEl.classList.remove("show");
    if (cooldownTimer) {
      clearTimeout(cooldownTimer);
      cooldownTimer = 0;
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.code === "KeyE" && !overlayOpen && controls.isLocked && aim) {
      if (aim.kind === "tp") openTravel(aim.i);
      else openOverlay(aim.i);
    } else if (e.code === "KeyL") {
      setFlatLit(!isFlatLit());
    } else if (e.code === "Backquote") {
      setPerf(!isPerf());
    } else if (e.code === "Escape" && overlayOpen) {
      closeOverlay();
    }
  });

  // the bottom-left readout was only ever load/debug text; clear it now the world
  // is up. #info stays in the DOM for the loading messages and the load-failure
  // handler (main().catch). Controls are self-explanatory via the contextual
  // prompts; a full reference belongs in the options menu, not a persistent line.
  info.innerHTML = "";

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
  // intro controls: shown within CTL_R_SHOW of the spawn centre, hidden past
  // CTL_R_HIDE (the gap is hysteresis so walking the boundary can't flicker the
  // timed CSS fade). Both inside R_INNER (~600) so they're gone before the books.
  const CTL_R_SHOW = 360;
  const CTL_R_HIDE = 460;
  let ctlShown = false;
  // worst-case ms for the two camera-driven rebuilds, reset each readout window,
  // so a bursty re-tessellation spike shows up instead of being averaged away.
  let booksMs = 0;
  const PICK_INTERVAL = 0.12; // ~8 Hz; the look-at label needn't be per-frame
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.1); // clamp after tab-out stalls
    mode = update(dt);

    // look-at picking: only while walking the scene (locked) and not inspecting.
    // Suppressed while skating so the glance prompt doesn't strobe as books blow past.
    sincePick += dt;
    if (!overlayOpen && controls.isLocked && mode !== "skate") {
      if (sincePick >= PICK_INTERVAL) {
        sincePick = 0;
        aim = pick();
        if (aim) showGlance(aim);
        else glance.style.display = "none";
        reticle.classList.toggle("armed", aim !== null); // affordance: actionable
      }
    } else if (glance.style.display !== "none") {
      glance.style.display = "none";
      reticle.classList.remove("armed");
      aim = null;
    }

    // intro controls: toggle .show at the centre radius (hysteresis below); the
    // CSS transition does the timed fade. Hidden while an overlay is open.
    {
      const rCentre = Math.hypot(camera.position.x, camera.position.z);
      if (ctlShown && (rCentre > CTL_R_HIDE || overlayOpen)) ctlShown = false;
      else if (!ctlShown && rCentre < CTL_R_SHOW && !overlayOpen) ctlShown = true;
      controlsHud.classList.toggle("show", ctlShown);
    }

    // the ground is static (built once); only the books refill by LOD as the camera
    // roams. Time the refill for the HUD.
    const tA = performance.now();
    built.update(camera.position.x, camera.position.z);
    const bm = performance.now() - tA;
    if (bm > booksMs) booksMs = bm;

    compass.update();
    // travelling pools of light: move the shared player-position uniform (book glow +
    // ground rake) to the player every frame. The book LOD refill is gated by move
    // distance, far too coarse for a smooth pool, so the uniform is driven here.
    uPlayer.value.set(camera.position.x, camera.position.z);
    // height of the eye above the ground directly below (hop + cosmetic skate hover):
    // lifts the rake lamp with the player so the pool reacts to a jump instead of
    // staying painted flat. Same sampleHeight the controller grounds on, so it reads 0
    // when planted.
    uPlayer.lift.value = Math.max(
      0,
      camera.position.y - EYE_HEIGHT - sampleHeight(camera.position.x, camera.position.z),
    );
    // ease the skate glow toward on/off so the blue pool blooms in and out instead of
    // popping with the Shift key (time-constant filter, same shape as the eye-lift).
    uSkate.value += ((mode === "skate" ? 1 : 0) - uSkate.value) * (1 - Math.exp(-6 * dt));
    // drift the daylight across the whole landscape (ground, books and
    // stones all sample the one shared mask + time).
    daylight.update(dt);
    // recentre the sky on the viewer (night dome wraps it, day layer rides overhead)
    // and deepen it with radial depth into the past. The day layer reads the field at
    // absolute world xz, so its openings stay world-locked as the player walks.
    const depth = clamp(
      Math.hypot(camera.position.x, camera.position.z) / world.R_MAX,
      0,
      1,
    );
    sky.update(depth, camera.position);
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
