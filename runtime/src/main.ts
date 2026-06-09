import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/examples/jsm/loaders/DRACOLoader.js";
import Stats from "three/examples/jsm/libs/stats.module.js";
import bookUrl from "./assets/meshes/book.glb?url";
import {
  initHeightmap,
  sampleHeight,
  sampleNormal,
  facetHeight,
  buildGround,
  applyDistanceFade,
  FADE_START,
  FADE_END,
  type PlayerUniform,
} from "./terrain";
import { buildSky } from "./sky";
import { createTouchControls } from "./touch";
import { createWind } from "./wind";
import { HARNESS_ON, setupHarness } from "./harness";
import {
  buildDaylight,
  DAYLIGHT_FRAG_COMMON,
  applyDaylightGLSL,
  DAY_GLSL,
  type DaylightUniforms,
} from "./daylight";

// --- walkable field --------------------------------------------------------
// One book per figure, placed straight from the pipeline's (x, y) and walkable
// with a first-person controller. Each book is drawn in one of three distance
// tiers off a shared baked footprint: a near LOD01 book mesh, a mid LOD02 book
// mesh (both InstancedMesh), and a far GL-points carpet (one rectangular dot per
// figure) that covers the whole field cheaply. The ground is the baked heightmap
// (see terrain.ts): books are seated on it and tilted to its normal, and the
// player's walk height samples the same surface so nothing floats.

const info = document.getElementById("info") as HTMLDivElement;

// world scale: the pipeline derives R_MAX from ~1.4 world units to the metre
// (see stage6_place.py). Eye height and speeds are in metres, converted once.
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

// The far base is GL points (see applyFarPointsShading / buildField): the always-drawn book
// speck past R_MID. SIZE is the camera-facing sprite's world size fed to three's size-
// attenuation (gl_PointSize = SIZE * (drawingBufferHeight/2) / dist), so it shrinks with
// distance like a real book; it's tuned to match a book's apparent size AT the R_MID boundary
// (footprint ~0.58u * 1/tan(fov/2)) so the speck is the same on-screen size as the mid mesh it
// takes over from -- the match holds at all distances (both scale as 1/dist). MIN_PX is the
// on-screen floor: kept at the natural ~1px so a far speck never inflates into a bright stipple
// (the books' rest-floor dim + the distance fade are what shade and dissolve it, mirrored in
// applyFarPointsShading). These are the knobs to tune the far field's weight on-device.
const FAR_POINT_SIZE = 0.4; // world size for size-attenuation (half a book's apparent footprint at R_MID — undersized on purpose so the speck reads as a distant book, not a dot, and more of the field falls below MIN_PX into the alpha-fade)
const FAR_POINT_MIN_CSS = 2.5; // floor on the on-screen speck in CSS px (NOT framebuffer px --
// pushed through the live pixel ratio into the uMinPx uniform on every resize). A framebuffer-px
// floor was a sub-CSS-pixel dot on a low-res / low-renderScale mobile buffer, so the whole far
// field (incl. the nearest R_INNER ring ~600u from the centre spawn) was invisible regardless of
// device pixel ratio. A CSS-px floor is physically the same size on any screen -- the headline
// mobile fix. ~2.5 reads as a clear speck without fattening the dense field into a blob.
// FAR_POINT_ALPHA_FLOOR — the lower bound on that sub-pixel alpha fade. The energy-true fade
// (floor 0) crushes a speck's alpha to its fractional framebuffer coverage; on a small mobile
// buffer the WHOLE field (incl. the nearest R_INNER ring the player must orient toward, ~600u
// from the centre spawn) is sub-pixel, so it fades to black and the player has nothing to walk
// toward. This floors the fade so every live speck keeps at least this alpha: the density
// gradient (present = many overlapping specks = a luminous mass; antiquity = sparse faint
// specks) now comes from how MANY books land per pixel, not from per-speck dimming. 1.0 = no
// thinning at all (pure "embrace the band"); lower it if the dense present blows out to flat fog.
const FAR_POINT_ALPHA_FLOOR = 0.85;
// The lit-multiply the books get and a raw point doesn't: a PointsMaterial is unlit, so a point
// starts from the full vertex colour, while a book's MeshLambert base = colour * (hemisphere +
// sun irradiance). Computed for an up-facing book cover (the dominant visible facet): cool sky
// 0x7e95bd*0.5 + warm sun 0xffb066*2.3*NdotL(+Y=0.161), Lambert /PI, in linear space (colour
// management on). The result is DIM and WARM — without it the points read brighter and greener
// than the books they continue. Tunable: nudge toward the books' average lit tone on-device.
const FAR_LIT = new THREE.Vector3(0.151, 0.099, 0.097);
// DIAGNOSTIC: flatten every far POINT to pure red and every detail BOOK mesh to pure blue (alpha,
// fades and coverage untouched) so the point<->mesh handoff is unmistakable on-device -- where red
// shows past the blue edge, the point is the visible layer; a red crescent inside the blue disc is
// a hole; a blue ring with no red beyond it is a mesh that out-ran its point. Set false to ship.
const DIAG_LOD_COLORS = false;
// DEV_DEFAULT: open every dev affordance on load (perf HUD up, pause menu's dev section
// pre-expanded) so a test build needs no menu poking. Scene-altering toggles (fly/flat) stay
// OFF so the default view is still the real one. Flip false to ship the clean player view.
const DEV_DEFAULT = true;
// Near the camera the books are SOLID and sit OVER the dot floor, but a dot sprite peeks past its
// flat book at grazing angle. So fade the dot floor IN over [DOT_NEAR_FADE_IN, FAR_MESH_FULL] of
// live horizontal distance: gone across the solid-book core (no peeking), full by FAR_MESH_FULL
// where the books begin to dissolve and the floor is needed. Safe on raw distance (unlike the
// abandoned cap-blind fade) because the cap fills nearest-first, so within FAR_MESH_FULL a mesh is
// ALWAYS present to cover the faded-out dot -- the fade-out can never expose a gap here.
const DOT_NEAR_FADE_IN = 30;
// The mid<->points handoff: the far point is a binary brightness FLOOR (full wherever it's DRAWN,
// hidden only once a real mesh book has reached full alpha to replace it), under which the mesh
// fades in by HORIZONTAL distance from the player. NOT an alpha crossfade on the point -- three
// earlier failures taught why:
//   - a complementary alpha fade does NOT conserve brightness (thin foreshortened sliver vs a
//     book's real area), so the 50/50 crossover dips into a dark band before the books arrive;
//   - the point's fade ran on a fixed DISTANCE schedule, blind to whether a book was actually
//     there -- so where MID_CAP runs out (the dense recent eras hold far more than 24000 books in
//     the mid disc) the orphaned specks faded into an EMPTY band, then the whole annulus snapped to
//     mesh on one rebuild when the cap finally reached it ("section pops in");
//   - both the mesh fade and the point fade keyed on 3D VIEW distance, so flying straight up made
//     every book within the mesh disc exceed CROSS_OUT and vanish -- a black hole under the player.
// So the rule is membership-driven and altitude-invariant:
//   - MESH fades 0->1 over [FAR_MESH_FULL, FAR_CROSS_OUT] of LIVE HORIZONTAL glowD, full inward of
//     FAR_MESH_FULL -- independent of camera height.
//   - POINT is a binary brightness FLOOR: full wherever DRAWN, no alpha fade. update() -- the one
//     place that KNOWS whether a book actually got a mesh slot (so it respects the MID_CAP, unlike a
//     blind distance schedule) -- hides it (aShow=0) only where a real mesh book covers it.
// The catch that bred TWO crescents: update() runs only on a rebuild (every REBUILD_DIST of travel),
// so its hide decision is a SNAPSHOT taken against the rebuild centre, while the mesh's alpha is
// LIVE against the current player. The hide must therefore be robust to a full REBUILD_DIST of drift
// before the next refill:
//   - hiding at FAR_MESH_FULL (v1) had ZERO margin -- a 30u walk dragged a hidden book straight into
//     the live [FAR_MESH_FULL, FAR_CROSS_OUT] fade band, where the mesh had faded but the point was
//     still hidden -> a dark TRAILING crescent (hole, no dots).
//   - a live distance fade on the POINT (v2) removed the snapshot but was blind to the cap: in dense
//     eras the cap fills nearest-first from the stale centre and falls SHORT of FAR_MESH_FULL on the
//     leading side, so the point faded out where no mesh arrived -> a dim crescent (dots, but faint).
// Fix: keep the binary membership-driven floor, but hide only within FAR_HIDE_R = FAR_MESH_FULL -
// REBUILD_DIST of the centre. Then a hidden book, even after a full REBUILD_DIST of drift, is still
// within FAR_MESH_FULL of the live player -> mesh at FULL alpha, covering it. And a book the cap
// never meshed keeps its full point -> a SPECK, never a hole or a faded gap. The point switches on
// (FAR_HIDE_R..out) over an identically-coloured FULL mesh book, so the switch is invisible; only
// past FAR_MESH_FULL, where the mesh fades, does the always-full point become the visible layer.
// Result: coverage shortfalls degrade to specks (not empty sections, pops, or crescents), from any
// camera height, in any density.
//
// The remaining "pop": a book only JOINS the mesh set on a rebuild (every REBUILD_DIST of travel),
// but its alpha is live. If the join radius (RMID_OUTER) equals the fade-out radius (FAR_CROSS_OUT),
// a book that was just outside last rebuild can be a full REBUILD_DIST inside the fade band by the
// time the next rebuild adds it -- so it appears at alpha ~1.0 (popped in), not 0. The fade band
// can't smooth a book that materialises INSIDE it. Fix: the join radius must sit a full REBUILD_DIST
// BEYOND the fade-out, so a book always joins while still INVISIBLE (glowD >= FAR_CROSS_OUT, alpha 0)
// and then glides in via the live alpha as the player approaches. That's the invisible margin
// [FAR_CROSS_OUT, RMID_OUTER]: meshed-but-transparent books waiting to fade in. It costs no extra
// geometry -- RMID_OUTER/MID_CAP are unchanged; we just pull the VISIBLE fade inward to 70 and leave
// the outer 30u of the already-meshed disc transparent. Invariant: FAR_CROSS_OUT <= RMID_OUTER -
// REBUILD_DIST (and the join dither is gone -- MID_DITHER=0 -- since joins are now invisible, so
// there's no ring of simultaneous pops left to fuzz). Kept loosely in sync with R_MID.
const FAR_CROSS_OUT = 70; // mesh alpha 0 / point is the only VISIBLE layer beyond this HORIZONTAL distance. Sits REBUILD_DIST inside RMID_OUTER(100) so books join (at RMID_OUTER) while invisible and glide in -- never materialise mid-fade
const FAR_MESH_FULL = 45; // mesh full here and inward; the point is also full from here OUTWARD. Fade band [45,70]

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
  // All three LODs are page-painted, but guard a missing channel anyway (an unpainted
  // LOD is an easy authoring slip): default aPage to 0 (all cover) rather than
  // dereferencing a missing attribute.
  const color = g.getAttribute("color");
  const vcount = color ? color.count : g.getAttribute("position").count;
  const aPage = new Float32Array(vcount);
  if (color) for (let i = 0; i < vcount; i++) aPage[i] = color.getX(i) < 0.5 ? 1 : 0;
  g.setAttribute("aPage", new THREE.BufferAttribute(aPage, 1));
  if (color) g.deleteAttribute("color");
  return g;
}

// Load the named book LODs from book.glb (one fetch, in ladder order) and bake each.
// book.glb carries book_LOD00 (full, ~308 tri), book_LOD01 (mid, ~56 tri) and book_LOD02
// (a 12-tri book box, the mid-band tier above the far points carpet); all three bake through the same uniform
// pipeline so their footprint and origin match exactly. NOTE the bakeBook clone takes
// geometry only — node transforms are dropped — so every LOD must have its SCALE APPLIED
// in Blender. LOD02 once shipped with an unapplied object scale and baked to a fat cube;
// the fix was Ctrl+A->Scale on the source, not code. All three carry COLOR_0 page paint.
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
// Works on the mid and near tiers alike (both InstancedMesh, so
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
           }` +
          // mid<->far-points crossfade (mesh side): fade the book IN as the camera approaches so a
          // freshly-promoted book ramps up under its still-drawn far point instead of popping. The
          // rebuild promotes a whole boundary band on a single frame every REBUILD_DIST; keyed to
          // live view distance this dissolves that wave. Books well inside CROSS_IN are alpha 1
          // (full); the near tier (always inside) is unaffected. Compounds with applyDistanceFade's
          // far dissolve, which patches the alpha again downstream. Keyed on glowD (HORIZONTAL
          // player distance, computed above for the pool), NOT vViewDist -- so the fade tracks the
          // same axis the rebuild classifies on and is altitude-invariant (flying up no longer
          // makes the whole mesh disc exceed CROSS_OUT and vanish). Reaches FULL at FAR_MESH_FULL
          // and inward, so a meshed book the point hides (aShow=0) is already fully opaque. NB:
          // edge0<edge1 then invert -- GLSL smoothstep is UNDEFINED for edge0>=edge1 (garbage on
          // Mali: full-alpha pops + empty chunks), so never write smoothstep(OUT, IN, x).
          `gl_FragColor.a *= 1.0 - smoothstep(${FAR_MESH_FULL.toFixed(1)}, ${FAR_CROSS_OUT.toFixed(1)}, glowD);` +
          (DIAG_LOD_COLORS ? "\n           gl_FragColor.rgb = vec3(0.0, 0.0, 1.0); // DIAG: detail meshes = blue" : ""),
      );
  };
}

// The far base material: each book as a camera-facing GL point. It can't run the mesh book's
// facet/page/glow stack (a point has no faces and no instanceMatrix) and doesn't need to --
// past R_MID a book is a sub-pixel speck. A point is also the RIGHT primitive at grazing
// desert angle, where a ground-flat quad would go edge-on and vanish; the box only stayed
// visible by its vertical spine, which a screen-facing sprite keeps for free at ~1/36th the
// vertices. It keeps only what reads at speck scale: the per-book colour (vertexColors), the
// drifting daylight tint faded to night with distance (the same daylightAt + fade the ground
// and detail books take, so the far field darkens in the same swaths and dissolves into the
// dome), and a small self-emissive floor so a speck still burns through the night.
// Promoted books (drawn by the near/mid detail tiers) carry aShow = 0, collapsing their speck
// so the base never double-draws a book the detail already covers. Not applyDistanceFade:
// the points fragment shader has no <dithering_fragment> for it to patch, so the fade is
// folded in here.
function applyFarPointsShading(
  mat: THREE.PointsMaterial,
  uPlayer: PlayerUniform, // shared player position -- the dots fade IN with live horizontal distance
  daylight: DaylightUniforms,
  uMinPx: { value: number }, // CSS->framebuffer size floor, re-derived on every resize/dpr change
): void {
  mat.transparent = true;
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uPlayer = uPlayer;
    shader.uniforms.uMinPx = uMinPx;
    shader.uniforms.uDaylight = daylight.uDaylight;
    shader.uniforms.uDriftTime = daylight.uDriftTime;
    shader.uniforms.uDaylightMix = daylight.uDaylightMix;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nattribute float aShow;\nuniform float uMinPx;\nvarying vec2 vGlowXZ;\nvarying float vViewDist;\nvarying float vGroundFade;\nvarying float vPointFade;\nvarying float vSquash;",
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>
         vec3 _wp = (modelMatrix * vec4(transformed, 1.0)).xyz;
         vGlowXZ = _wp.xz;
         vViewDist = length(mvPosition.xyz);
         vGroundFade = clamp(
           (vViewDist - ${FADE_START.toFixed(1)}) / ${(FADE_END - FADE_START).toFixed(1)}, 0.0, 1.0);
         // Foreshorten the sprite the way the flat ground foreshortens: a camera-facing point
         // is always a SQUARE, but a book lies flat, so from above it shows its full cover (square,
         // matches) and at grazing angle it collapses to a thin sliver (the square does NOT —
         // that's the ground-level tell). vSquash = |view-ray.y| is exactly that foreshortening:
         // ~1 looking straight down, ~0 at the horizon. The fragment keeps only the central
         // vertical band of this height. Floored so a fully-grazing book keeps a hairline (its
         // spine thickness) instead of vanishing.
         vSquash = clamp(abs(normalize(_wp - cameraPosition).y), 0.12, 1.0);`,
      )
      // aShow zeroes a promoted book's speck; three's size-attenuation block then scales the
      // rest by distance. After it (before logdepthbuf_vertex), floor the on-screen size to
      // uMinPx so a far speck never vanishes. uMinPx is a CSS-pixel floor pushed through the
      // framebuffer pixel ratio every resize (NOT a baked framebuffer-px constant): the size-
      // attenuation result is in framebuffer px, so a constant floor was a TINY dot on a low-res /
      // low-renderScale mobile buffer and the field was invisible "across all pixel ratios". A
      // CSS-px floor is physically consistent on any screen. vPointFade = trueSize/uMinPx (<1 when
      // sub-pixel) still rides along, floored to FAR_POINT_ALPHA_FLOOR so a floored speck stays
      // bright -- the dense/sparse gradient comes from overlap COUNT, not per-speck alpha.
      .replace("gl_PointSize = size;", "gl_PointSize = size * aShow;")
      .replace(
        "#include <logdepthbuf_vertex>",
        `float _natural = gl_PointSize; // true attenuated size (0 if promoted/aShow=0)
         // Promoted specks (aShow=0) stay 0; everything else floors to FAR_POINT_ALPHA_FLOOR.
         vPointFade = _natural > 0.0 ? clamp(${FAR_POINT_ALPHA_FLOOR.toFixed(2)} + ${(1 - FAR_POINT_ALPHA_FLOOR).toFixed(2)} * (_natural / uMinPx), 0.0, 1.0) : 0.0;
         if (_natural > 0.0) gl_PointSize = max(_natural, uMinPx);\n\t#include <logdepthbuf_vertex>`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nuniform vec2 uPlayer;\nvarying vec2 vGlowXZ;\nvarying float vViewDist;\nvarying float vGroundFade;\nvarying float vPointFade;\nvarying float vSquash;\n" +
          DAYLIGHT_FRAG_COMMON,
      )
      .replace(
        "#include <opaque_fragment>",
        `#include <opaque_fragment>
           // The unlit point's base IS the raw vertex colour; the books multiply that same colour
           // by the scene's hemisphere+sun irradiance. Apply the SAME multiply (FAR_LIT) here, so
           // the points' base matches the books' lit base before the shared restFloor/emissive
           // stack below — without it the field reads brighter and greener than the books it
           // continues. Only the reflected base is lit; the additive emissive/sun-reveal below are
           // self-light and stay un-multiplied, exactly as on the books.
           gl_FragColor.rgb *= vec3(${FAR_LIT.x.toFixed(4)}, ${FAR_LIT.y.toFixed(4)}, ${FAR_LIT.z.toFixed(4)});` +
          // Match the detail books' distance shading EXACTLY, or the far field reads brighter
          // than the books it continues (the box base did this via applyProximityGlow; the
          // points dropped it and lit up the intentionally-dark dim ring). A point never enters
          // the glow pool -- the nearest is past R_MID, well beyond GLOW_RADIUS -- so glow is
          // always 0 and the pool/boost/warm/facet terms all drop out, leaving just the resting
          // floor: crushed to REST_DIM in the near dim ring, lifted toward REST_FAR with camera
          // distance (aerial perspective) and with the ground fade.
          `float _distLift = smoothstep(${HAZE_NEAR.toFixed(1)}, ${HAZE_FAR.toFixed(1)}, vViewDist);
           float _restFloor = mix(${GLOW_REST_DIM.toFixed(3)}, ${GLOW_REST_FAR.toFixed(3)}, max(vGroundFade, _distLift));
           gl_FragColor.rgb *= _restFloor;` +
          // daylightAt() is the carpet's costliest per-fragment term (two 4-octave fbm warps +
          // per-layer texture taps). The tint below and the sun-reveal further down used to call
          // it SEPARATELY with identical args -- two evals per fragment for one value. Compute it
          // ONCE here and feed both. Bit-identical output; halves the daylight cost. (The book
          // glow at applyProximityGlow still double-calls -- same free dedup available there.)
          `float _day = daylightAt(vGlowXZ, vViewDist);` +
          // the same drifting daylight the ground + books take, faded to night with distance
          applyDaylightGLSL("vGlowXZ", "vGroundFade", "vViewDist", "_day") +
          // self-emission AFTER the tint (true self-light) so a speck survives the night as its
          // own dim hue, plus the sun-reveal -- both as on the books, minus the facet self-shade
          // (a point has no facet). These keep the mid->points handoff continuous.
          `gl_FragColor.rgb += diffuseColor.rgb * ${GLOW_EMISSIVE.toFixed(3)};
           float _sun = _day; // reuse the single daylightAt above (dedup)
           gl_FragColor.rgb += diffuseColor.rgb * ${DAY_GLSL} * (_sun * ${GLOW_REVEAL.toFixed(3)});
           // distance dissolve into the dome (the box took this from applyDistanceFade's
           // dithering_fragment splice, which the points shader lacks), the sub-pixel coverage
           // fade so the dense mid-field thins to stipple instead of a solid carpet, and the
           // vertical foreshortening band: keep the central vSquash-tall slab, soft-edged. At
           // top-down vSquash~1 so the whole sprite survives (square, matches a book's cover);
           // at grazing vSquash~0 so only a thin horizontal sliver remains (a flat book edge-on).
           // A floored 1px speck is a single centre fragment (_vy~0) so it always survives intact.
           float _vy = abs(gl_PointCoord.y - 0.5) * 2.0;
           float _band = 1.0 - smoothstep(vSquash, vSquash + 0.20, _vy);
           // Near fade-in: drop the dot floor across the solid-book core so a dot sprite can't peek
           // past its flat book, ramping to full by FAR_MESH_FULL where the books start dissolving.
           // Keyed on LIVE horizontal distance -- safe here (unlike the abandoned cap-blind fade)
           // because within FAR_MESH_FULL the cap (nearest-first) ALWAYS has a mesh covering the
           // faded dot, so this can't expose a gap. Beyond FAR_MESH_FULL the factor is 1 and the dot
           // is the full floor under the fading / absent mesh.
           float _nearFade = smoothstep(${DOT_NEAR_FADE_IN.toFixed(1)}, ${FAR_MESH_FULL.toFixed(1)}, distance(vGlowXZ, uPlayer));
           // The far speck stays a brightness FLOOR otherwise (no distance fade-OUT): update() still
           // hides a meshed book's point (aShow=0) within FAR_HIDE_R for the giant-near-sprite cull,
           // staleness-safe and cap-aware. A distance fade-OUT on the point caused a dark band /
           // empty-on-overflow / fly-up hole / dense-area crescent -- see the FAR_* notes above.
           gl_FragColor.a *= (1.0 - vGroundFade) * vPointFade * _band * _nearFade;` +
          (DIAG_LOD_COLORS ? "\n           gl_FragColor.rgb = vec3(1.0, 0.0, 0.0); // DIAG: points = red" : ""),
      );
  };
}

function buildField(
  field: Awaited<ReturnType<typeof loadPositions>>,
  bookNear: THREE.BufferGeometry, // LOD01 (near tier; LOD00 dropped), drawn closest
  bookMid: THREE.BufferGeometry, // LOD02 box, drawn across the mid band (the far base is GL points)
  uPlayer: PlayerUniform, // shared player-position uniform (also drives the ground rake)
  daylight: DaylightUniforms, // shared daylight uniforms (also drift over the ground)
  world: World, // R_INNER/R_MAX define the era-temperature radial ramp
  uFarMinPx: { value: number }, // CSS->framebuffer far-speck size floor, updated on resize
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

  // Distance LOD: two detail mesh tiers over an always-present GL points base. A 230-tri book
  // instanced 574k times and drawn mostly sub-pixel is hopeless; a book is only a legible shape
  // within tens of units and only READABLE within 6u (the picker's reach), so detail past that
  // is wasted. Each book is drawn by exactly ONE layer: full LOD00 within R_FULL, LOD01 out to
  // R_MID, a points speck beyond. A promoted book's speck is collapsed (aShow=0), so the base
  // never covers the same book as a detail mesh (no z-fight / double-draw); the two detailed
  // tiers never cover the same book either. Splitting the detail in two keeps each swap small:
  // full->mid lands where a book is a few px, mid->points smaller still. A per-book dither on
  // the inner boundary scatters that swap so it isn't a clean ring sweeping the field.
  // The far base was a box mesh tier (LOD02) tiled for frustum culling; it was the measured
  // vertex-bound cost on mobile (~19 vs ~40fps) and the recency-concentrated field barely
  // culled, so it's now one GL points cloud (1 vertex/book vs ~36) -- see applyFarPointsShading
  // and the cloud build below. A camera-facing point is also the right primitive at grazing
  // desert angle, where the box only stayed visible by its vertical spine.
  const R_FULL = 30; // LOD00 within this radius (a book is still >~8px here)
  // The mid mesh must cover out to the FIXED transition radius (FAR_CROSS_OUT=100) at PEAK density
  // or the transition floats inward where the cap runs out (the "sporadic" load-in). Measured peak
  // is ~54k books in a 100u disc at "the present" (the densest era), so the mid tier is LOD02 (the
  // 24-vert box, NOT LOD01's 118) and MID_CAP is sized to that peak: 54k LOD02 = ~1.3M verts, far
  // cheaper than 54k LOD01 (~6.4M) and even lighter than the old 24k-LOD01 mid -- LOD02 keeps the
  // baked cover/page vertex colours and at >=30u its missing bevels are sub-pixel. R_MID is the JOIN
  // radius and sits REBUILD_DIST beyond the VISIBLE fade-out (FAR_CROSS_OUT=70): the outer 30u
  // [70,100] is the invisible margin where books are meshed-but-transparent (alpha 0), so a book
  // joins while invisible and glides in instead of popping mid-fade. The cap still only has to cover
  // the 100u disc (~49k mid at peak, headroom under 54k) -- same budget as before, the margin is
  // just the outer slice of the disc we already paid for.
  const R_MID = 100; // mid JOIN radius (= RMID_OUTER, MID_DITHER=0); speck beyond. Visible fade-out is FAR_CROSS_OUT=70, REBUILD_DIST inside this
  const BOUND_DITHER = 12; // per-book spread (world units) on the full->mid boundary
  const NEAR_CAP = 6000; // LOD00 instances; the R_FULL disc holds far fewer than this
  const MID_CAP = 54000; // LOD02 instances; sized to the peak density in the 100u mid disc so the
  //   transition is reliable. Any residual overflow drops to the points floor (a full speck, never
  //   a hole -- membership-driven hiding), so slight under-sizing degrades gracefully.
  const REBUILD_DIST = 30; // refill the near/mid sets after the camera moves this far. The invisible
  //   margin (RMID_OUTER - FAR_CROSS_OUT = 30) must be >= this, so a book can't cross from outside
  //   the join radius to inside the VISIBLE fade in one rebuild step -> it always joins at alpha 0.

  // books dissolve with the ground: the same camera-distance fade to transparent,
  // so the field thins into the dome at the horizon rather than leaving sharp specks
  // floating over ground that has already faded out. The page mask chains on after
  // the fade so cover vertices keep the geo hue and page edges stay cream; both detail
  // tiers carry it. The points base can't mask faces (a point has none) and doesn't need
  // to — a speck is one colour — but it takes the same distance fade (folded into its own
  // shader) so it dissolves into the dome with the rest.
  // shared proximity-glow uniforms: one object referenced by every detail book material,
  // so moving uPlayer once per frame lights the pool on both detail LOD tiers.
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
  // both detail tiers so a book reads the same through an LOD swap. The far base is no longer
  // a mesh tier -- it's a GL points cloud (farPointsMat), built once the positions exist.
  const farPointsMat = new THREE.PointsMaterial({
    vertexColors: true,
    sizeAttenuation: true,
    size: FAR_POINT_SIZE,
  });
  applyFarPointsShading(farPointsMat, uPlayer, daylight, uFarMinPx);
  // depthWrite OFF on the whole field. The dots are a CONTINUOUS floor that the books layer over
  // (points render first, RO_CARPET < RO_DETAIL); the books must NOT depth-occlude the dots. The bug
  // this kills: a transparent fragment still writes depth, so the invisible-margin books (alpha 0
  // in [FAR_CROSS_OUT, RMID_OUTER]) were punching holes in the dot field while drawing nothing --
  // an occlusion ring that JUMPED with the rebuild snapshot ("sparse dot ring"). With no depth
  // write the books just alpha-blend over the dot floor: a full book covers its dot (same colour,
  // seamless), a fading book lets the dot show through proportionally (brightness conserved for
  // free), and an invisible book does nothing. Trade-off: books no longer depth-occlude EACH OTHER,
  // so overlaps resolve by draw order -- acceptable here (stage 8 forbids subsumption, books barely
  // overlap, and they're near-opaque within FAR_MESH_FULL).
  farPointsMat.depthWrite = false;
  const midMat = new THREE.MeshLambertMaterial({ flatShading: true });
  applyDistanceFade(midMat);
  applyPageMask(midMat);
  applyProximityGlow(midMat, glow, daylight);
  midMat.depthWrite = false;
  const nearMat = new THREE.MeshLambertMaterial({ flatShading: true });
  applyDistanceFade(nearMat);
  applyPageMask(nearMat);
  applyProximityGlow(nearMat, glow, daylight);
  nearMat.depthWrite = false;
  // The two detail tiers run the identical patch stack (fade+page+glow), so they legitimately
  // share one program. Their default cache keys (= onBeforeCompile.toString(), closure
  // vars excluded) collide, which here is correct — but three's collision is by accident
  // of stringification, so we still pin an explicit shared key to make the sharing
  // intentional and robust to a future patch divergence. Same defence the ground uses.
  const bookProgramKey = () => "book:fade+page+glow+daylight";
  midMat.customProgramCacheKey = bookProgramKey;
  nearMat.customProgramCacheKey = bookProgramKey;

  // Explicit renderOrder, and the POINTS now draw BEFORE the detail meshes (was the reverse). With
  // depthWrite off across the field, draw order alone decides layering: the dot floor must paint
  // first so the books alpha-blend OVER it (a full book hides its dot, a fading book reveals it).
  // Ground first (renderOrder 0, writes depth), then the points floor, then the books over them,
  // all below the teleporter beam (renderOrder 10).
  const RO_CARPET = 4; // the far points: the continuous dot floor, painted first
  const RO_DETAIL = 5; // near/mid detail: layered OVER the dot floor
  // The far base (every book's lowest LOD) was the dominant GPU cost — measured
  // vertex-bound (~19fps with it on mobile, ~40 without; flat-lit and half pixel-ratio
  // both did nothing, so neither lighting nor fill). It was a box mesh tiled for frustum
  // culling, but the field is recency-concentrated at the centre where the player surveys,
  // so culling barely bit (only ~1.6% culls past half-radius from spawn). It's now a single
  // GL points cloud (built after the positions exist): one vertex per book vs the box's ~36,
  // attacking the vertex cost directly and uniformly across the whole field. No tiling —
  // 574k point-vertices is cheap to run in full, so the frustum-cull split (and its draw
  // calls) is gone too.
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
  // from the global set as the camera moves (the far points carpet is written once).
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
    col.toArray(fullCol, i * 3);
  }

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

  // Far base as ONE GL points cloud: one camera-facing sprite per book, drawn at the book's
  // standing position. No tiling — 574k point-vertices is cheap to run in full, so the box's
  // frustum-cull split (and its draw-call overhead) is unnecessary. Positions are the book
  // translation (lifted seat) lifted to the spine centre, so a speck sits where its book mass
  // is; colour reuses the per-book fullCol; aShow gates promoted books (set in update()).
  const farGeo = new THREE.BufferGeometry();
  const farPos = new Float32Array(n * 3);
  const farShow = new Float32Array(n).fill(1); // 1 = drawn by the base; 0 = promoted to detail
  for (let i = 0; i < n; i++) {
    // fullMat translation (cols 12..14) is the book's seated centre; raise by half a spine so
    // the speck floats at the book's middle rather than its underside.
    farPos[i * 3] = fullMat[i * 16 + 12];
    farPos[i * 3 + 1] = fullMat[i * 16 + 13] + SPINE * 0.5;
    farPos[i * 3 + 2] = fullMat[i * 16 + 14];
  }
  farGeo.setAttribute("position", new THREE.BufferAttribute(farPos, 3));
  // reuse the per-book colours (vertexColors); the points shader multiplies diffuse by these.
  farGeo.setAttribute("color", new THREE.BufferAttribute(fullCol, 3));
  const farShowAttr = new THREE.BufferAttribute(farShow, 1);
  farShowAttr.setUsage(THREE.DynamicDrawUsage); // promotion flips a few thousand per rebuild
  farGeo.setAttribute("aShow", farShowAttr);
  // Density-adaptive carpet thinning. The far base draws one point primitive per book, and
  // the measured mobile bottleneck is primitive throughput (vertex+assembly+binning), worst
  // when looking ACROSS the field where the deep dense clusters stack into a few horizon
  // pixels -- thousands of primitives for a handful of resolvable specks. In a crowded cell
  // the books overlap into the same pixels, so dropping a fraction is invisible while cutting
  // primitives exactly where they pile up. Keyed on LOCAL CELL DENSITY (not distance), so it
  // bakes ONCE: a book's crowding doesn't move with the player. Sparse cells (antiquity) are
  // untouched -- the honest void keeps every dot. The near/mid detail tiers redraw full
  // density within R_MID regardless, so walking up to a thinned cluster restores every book;
  // this only thins the far base where the surplus is sub-pixel anyway.
  // It thins by a constant FRACTION (drop every Nth), NOT a cap-to-count: a cap would flatten
  // every dense cell to the same count and erase the present-vs-past density gradient that is
  // the subject. A constant fraction keeps a present cell proportionally denser than a moderate
  // one -- same gradient, half the primitives. KEEP_OF_N = keep 1 of every N in cells over MIN
  // (2 = "every other"). MIN spares the sparse cells. Both tunable.
  const CARPET_DENSE_MIN = 8; // books/32u-cell above which a cell is thinned
  const CARPET_KEEP_OF_N = 6; // keep 1 in N of a dense cell's books when thinning is ON (mobile
  //   default below). 6 is the on-device sweet spot: enough primitive cut to move the worst case
  //   to ~25-30fps, with the R_MID density step acceptable at phone resolution. (2 = every other.)
  const carpetKeep = new Uint8Array(n).fill(1);
  for (let c = 0; c < gw * gh; c++) {
    const s = cellStart[c];
    const e = cellStart[c + 1];
    if (e - s <= CARPET_DENSE_MIN) continue;
    for (let k = s; k < e; k++) {
      if ((k - s) % CARPET_KEEP_OF_N !== 0) carpetKeep[cellItems[k]] = 0;
    }
  }
  let carpetKept = 0;
  for (let i = 0; i < n; i++) if (carpetKeep[i]) carpetKept++;
  const carpetIdxArr = new Uint32Array(carpetKept);
  for (let i = 0, w = 0; i < n; i++) if (carpetKeep[i]) carpetIdxArr[w++] = i;
  const carpetIdx = new THREE.BufferAttribute(carpetIdxArr, 1);
  console.log(
    `[perf] carpet thinning: ${carpetKept}/${n} kept (${((100 * carpetKept) / n).toFixed(1)}%, dropped ${n - carpetKept} in cells > ${CARPET_DENSE_MIN})`,
  );
  const farPoints = new THREE.Points(farGeo, farPointsMat);
  // Thinning is DEVICE-ADAPTIVE: on by default for coarse-pointer (mobile) devices, which
  // need the primitive cut AND whose low resolution hides the R_MID density step; OFF on
  // desktop, which has the GPU budget to draw the full cloud and where the step is plainly
  // visible at full res. The thinned index always exists (built above), so the toggle below
  // can A/B it on either device. carpetIdx applied = thinned; null = full non-indexed cloud.
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  let carpetThin = coarse;
  farGeo.setIndex(carpetThin ? carpetIdx : null);
  const setCarpetThin = (on: boolean): void => {
    carpetThin = on;
    farGeo.setIndex(on ? carpetIdx : null);
  };
  const isCarpetThin = (): boolean => carpetThin;
  // whole-disc bounds always intersect the frustum and the cloud is cheap to run in full, so
  // don't pay computeBoundingSphere or per-frame cull tests — never culled, always drawn.
  farPoints.frustumCulled = false;
  farPoints.renderOrder = RO_CARPET;

  // No join dither on the mid->points boundary anymore. It used to fuzz the JOIN radius so books
  // didn't all swap to mesh on one ring (a visible pop), but with the invisible margin a book joins
  // at R_MID while still alpha 0 (FAR_CROSS_OUT is REBUILD_DIST inside R_MID), so the join is
  // invisible -- there's no ring of pops left to scatter, and the live alpha fade does all the
  // visible smoothing. (The full->mid LOD swap at R_FULL is a different, VISIBLE geometry change and
  // keeps its BOUND_DITHER.) RMID_OUTER = R_MID is the membership cap.
  const MID_DITHER = 0; // join is invisible (alpha 0 at R_MID), so no dither needed
  const RMID_OUTER = R_MID + MID_DITHER * 0.5;
  const RMID2 = RMID_OUTER * RMID_OUTER;
  // Hide a book's point only where a real mesh book COVERS it: meshed AND within FAR_HIDE_R of the
  // rebuild centre. FAR_HIDE_R = FAR_MESH_FULL - REBUILD_DIST is the staleness margin -- a book
  // hidden here is, even after a full REBUILD_DIST of drift before the next refill, still within
  // FAR_MESH_FULL of the LIVE player, where the mesh is at full alpha and covers the removed point
  // (so the snapshot can never punch a hole; the v1 hide at FAR_MESH_FULL had no margin). The point
  // switches on at FAR_HIDE_R over a FULL mesh book, so the switch is invisible; it only becomes the
  // visible layer past FAR_MESH_FULL where the mesh fades. A book the cap never meshed isn't hidden
  // -> keeps its full point, a speck not a gap (membership-driven, not a blind distance fade).
  const FAR_HIDE_R = Math.max(0, FAR_MESH_FULL - REBUILD_DIST);
  const FAR_HIDE_R2 = FAR_HIDE_R * FAR_HIDE_R;
  const RB2 = REBUILD_DIST * REBUILD_DIST;
  const maxRing = Math.ceil(RMID_OUTER / CELL) + 1; // cells beyond this are wholly out of range
  const m = new THREE.Matrix4();
  const c = new THREE.Color();
  // a book promoted to a detail tier has its far speck collapsed (aShow=0 -> gl_PointSize 0,
  // not rasterised), so the base never double-draws it. We track the promoted ids so the next
  // rebuild can restore their speck before re-promoting.
  const hidden = new Int32Array(NEAR_CAP + MID_CAP);
  let nHidden = 0;
  // near-tier draw order: book ids collected during the (near-first) ring walk, then sorted
  // back-to-front before the GPU write so overlaps composite by painter's order. Reused each
  // rebuild (cleared, not reallocated). See the sort below for the why.
  const nearOrder: number[] = [];
  let lastX = Infinity;
  let lastZ = Infinity;
  // far-base master switch (dev toggle, for the perf baseline). The points cloud is cheap
  // enough that the old per-tile view-distance cull is gone; this is just on/off.
  let carpetShown = true;
  // book-mesh master switch (dev toggle). Pairs with carpetShown: toggling the dot carpet and the
  // book meshes independently is how we ATTRIBUTE the GPU cost between the two field layers --
  // read the gpu line with each off in turn (e.g. at a horizon angle where overdraw peaks).
  let booksShown = true;
  function update(camX: number, camZ: number): void {
    const mdx = camX - lastX;
    const mdz = camZ - lastZ;
    if (mdx * mdx + mdz * mdz < RB2) return;
    lastX = camX;
    lastZ = camZ;
    // restore the far specks hidden last rebuild; the walk below re-hides whichever are still
    // promoted, so a book that fell out of the detail tiers reappears in the far base.
    for (let h = 0; h < nHidden; h++) farShow[hidden[h]] = 1;
    nHidden = 0;
    const cgx = Math.floor((camX - minX) / CELL);
    const cgz = Math.floor((camZ - minZ) / CELL);
    let kNear = 0;
    let kMid = 0;
    nearOrder.length = 0;
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
        // per-book dither on BOTH boundaries so each swap is a fuzzy band of individual books
        // rather than a clean ring sweeping the field. Two independent stable hashes of the
        // book index (so a given book's boundaries don't change frame to frame, and the two
        // boundaries don't correlate): h for full->mid, h2 for mid->points.
        const h = (Math.imul(i, 2654435761) >>> 0) / 4294967296; // [0, 1)
        const h2 = (Math.imul(i ^ 0x9e3779b9, 2246822519) >>> 0) / 4294967296; // [0, 1)
        const rf = R_FULL + (h - 0.5) * BOUND_DITHER;
        const rm = R_MID + (h2 - 0.5) * MID_DITHER;
        if (d2 < rf * rf && kNear < NEAR_CAP) {
          // defer the GPU write -- the draw order is decided by a back-to-front sort after the
          // walk (see below). The near-first ring walk still decides MEMBERSHIP + the cap here.
          nearOrder.push(i);
          kNear++;
          // hide the speck where the mesh book covers it: within FAR_HIDE_R (= FAR_MESH_FULL -
          // REBUILD_DIST), the staleness-safe radius inside which the live mesh stays full alpha
          // through a whole rebuild interval. Books between FAR_HIDE_R and FAR_MESH_FULL keep their
          // full point as the floor under the still-full mesh (the switch is invisible there); past
          // FAR_MESH_FULL the point is the visible layer as the mesh fades. Near books are well
          // inside, so they always collapse.
          if (d2 < FAR_HIDE_R2) {
            farShow[i] = 0;
            hidden[nHidden++] = i;
          }
        } else if (d2 < rm * rm && kMid < MID_CAP) {
          // mid band, or a near-band book that overflowed NEAR_CAP (still gets detail)
          midMesh.setMatrixAt(kMid, m.fromArray(fullMat, i * 16));
          midMesh.setColorAt(kMid, c.fromArray(fullCol, i * 3));
          kMid++;
          if (d2 < FAR_HIDE_R2) {
            farShow[i] = 0; // mesh covers it (staleness-safe radius) -> hide the redundant speck
            hidden[nHidden++] = i;
          } // else: keep the full speck as the floor; the mesh fades in/out under it past here
        }
        // NB: a book that falls through here (mid cap full, so unmeshed) keeps farShow[i]=1 -> a
        // full speck, never an empty hole. That's the whole point of membership-driven hiding.
        // else: past the dithered mid radius (stays a speck), or both detail caps full — the
        // always-drawn far points base covers this book
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
    // Book-on-book overlap fix. The field draws with depthWrite OFF (so books never punch holes
    // in the dot floor -- see the farPointsMat note), which means overlapping books composite by
    // DRAW ORDER, not depth. The ring walk above fills near-first = front-to-back, so a farther
    // book paints OVER a nearer one (the "random overlap" bug). Sort the near set back-to-front
    // (farthest first) before writing, so overlaps composite by painter's order -- exact for flat
    // books strewn on the ground (no deep interpenetration to form cycles; stage 8 forbids
    // subsumption). Also makes the [FAR_MESH_FULL, FAR_CROSS_OUT] alpha-fade band blend in the
    // right order for free. Decoupled from the walk: membership / caps / aShow were already decided
    // above (near-first); only the GPU write order changes, and kNear == nearOrder.length. Near-only
    // -- mid books are small/distant box LODs whose overlaps are sub-pixel, not worth a 54k-instance
    // sort every rebuild. Horizontal distance (the axis the tiers classify on): books hug the ground
    // so it matches eye-distance ordering, and it's free of camera height. Runs only on a rebuild.
    nearOrder.sort((a, b) => {
      const ax = px[a] - camX;
      const az = pz[a] - camZ;
      const bx = px[b] - camX;
      const bz = pz[b] - camZ;
      return bx * bx + bz * bz - (ax * ax + az * az);
    });
    for (let k = 0; k < nearOrder.length; k++) {
      const i = nearOrder[k];
      nearMesh.setMatrixAt(k, m.fromArray(fullMat, i * 16));
      nearMesh.setColorAt(k, c.fromArray(fullCol, i * 3));
    }
    nearMesh.count = kNear;
    nearMesh.instanceMatrix.needsUpdate = true;
    if (nearMesh.instanceColor) nearMesh.instanceColor.needsUpdate = true;
    midMesh.count = kMid;
    midMesh.instanceMatrix.needsUpdate = true;
    if (midMesh.instanceColor) midMesh.instanceColor.needsUpdate = true;
    // the restores + re-hides above flipped aShow on a few thousand books; one upload of the
    // (small, 1 float/book) attribute pushes them all. Only fires on a rebuild (camera moved
    // REBUILD_DIST), not per frame.
    farShowAttr.needsUpdate = true;
  }

  const group = new THREE.Group();
  group.add(farPoints);
  group.add(midMesh);
  group.add(nearMesh);
  // far-base dev visibility toggle (for the perf baseline). The points cloud is cheap enough
  // that the old per-tile view-distance cull is gone. n is the total book count for the readout.
  const setCarpetVisible = (on: boolean): void => {
    carpetShown = on;
    farPoints.visible = on;
  };
  const isCarpetVisible = (): boolean => carpetShown;
  // the two detail tiers move together: they're the same conceptual layer (the legible book
  // meshes), split only by LOD distance, so one switch hides both for the perf baseline.
  const setBooksVisible = (on: boolean): void => {
    booksShown = on;
    nearMesh.visible = on;
    midMesh.visible = on;
  };
  const isBooksVisible = (): boolean => booksShown;
  return {
    group,
    px,
    pz,
    tier: tier as Uint8Array,
    update,
    nearMesh,
    midMesh,
    farPoints,
    setCarpetVisible,
    isCarpetVisible,
    setBooksVisible,
    isBooksVisible,
    setCarpetThin,
    isCarpetThin,
  };
}

// Teleporter monuments (26): the player walks into one to jump across the field.
// Each is an embedded ball -- a beach-ball-sized icosphere half-sunk in the sand at
// the anchor -- so the teleporter is an OBJECT in the field like a book, sharing one
// interaction grammar (look + E) and joining the topology relaxation pass instead of
// carving a plaza. It is a SOLID, opaque, flat-shaded icosphere lit by the scene
// hemisphere + warm sun (MeshLambertMaterial, flatShading) so the facets carry a real
// light/dark gradient -- that is what makes a sphere read as a sphere, the one thing a
// flat book can do without and a ball cannot. It does NOT take the books' daylight chain,
// which multiplies toward near-black NIGHT and would crush all that shading flat (fatal on
// a curved surface). Instead daylight is a FLOORED darkening (beaconLit, the point-sample
// reader): the ball sinks into the field's night mood but never below TP_BALL_NIGHT_FLOOR,
// so its form always survives, and lifts to full in a sun pool. A teal self-glow + fresnel
// rim ride on top as the node's own light, so it reads as a glowing NODE, sized and glowing
// a notch hotter than a book so it still carries at a distance.
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

// Curated opening-vista anchors (export_runtime.export_spawn_anchors): recognisable
// recent landmark figures. The player spawns at the inner rim on one anchor's bearing
// facing outward, so the first thing in view is a name they know a few steps ahead --
// then the walk outward runs back in time into less-familiar, sparser ground. x/y are
// the figure's layout position; bearing and spawn radius are derived in applySpawn.
interface SpawnAnchor {
  qid: string;
  title: string;
  x: number;
  y: number;
}

async function loadSpawnAnchors(url: string): Promise<SpawnAnchor[]> {
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
  // Builds the far beacon at each anchor: an embedded ball (see the TP_BALL_* notes
  // above). The ball IS the marker -- there is no separate floor glow pad or light
  // shaft anymore; both were retired for breaking the hard-faceted lowpoly look.
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
    // a beach-ball icosphere half-sunk in the sand at the anchor, a solid teal node.
    const ball = new THREE.Mesh(ballGeom, ballMat);
    ball.position.set(tp.x, h + ballCentreY, tp.y);
    group.add(ball);
  }
  return group;
}

// --- figure metadata (look-at + inspect) -----------------------------------
// meta.bin is row-aligned with positions.bin, so a picked instanceId indexes it
// directly. String blobs are decoded lazily (only the targeted book), never all
// at once. Layout documented in pipeline/export_runtime.py:export_meta.
const YEAR_MISSING = -32768;

// Fetch the gzipped sibling of a .bin and inflate it in-stream. Shipping it
// pre-gzipped (rather than relying on the host to compress octet-stream, which
// itch and others won't) gets the ~34MB string corpus to ~12MB on every host
// and under Cloudflare Pages' 25 MiB per-file cap.
//
// The compressed file is named `foo.gz.bin`, NOT `foo.bin.gz`, on purpose: a
// `.gz` extension makes many static servers (Vite dev, nginx, some CDNs) send
// `Content-Encoding: gzip`, so the browser transparently inflates the body and
// our DecompressionStream would then double-inflate and throw. Ending in `.bin`
// keeps it an opaque octet-stream everywhere, so we always do the single inflate
// ourselves — deterministic across hosts. Falls back to the plain `.bin` if the
// compressed sibling is absent, so an old artifact + new code still loads.
async function fetchMaybeGzip(url: string): Promise<ArrayBuffer> {
  const gzUrl = url.replace(/\.bin$/, ".gz.bin");
  const gz = await fetch(gzUrl);
  if (gz.ok && gz.body) {
    const stream = gz.body.pipeThrough(new DecompressionStream("gzip"));
    return await new Response(stream).arrayBuffer();
  }
  return await (await fetch(url)).arrayBuffer();
}

async function loadMeta(url: string) {
  const buf = await fetchMaybeGzip(url);
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

  // touch input: a coarse-pointer device has no pointer lock and no keyboard, so an
  // analog path feeds the SAME wish/look machinery as the keyboard + PointerLockControls.
  // touchActive stands in for controls.isLocked (the active-scene gate); moveAxis is the
  // stick deflection (-1..1 strafe/forward, magnitude = speed); look deltas accumulate and
  // are drained each frame here (mouse look is PLC's own pointermove). See touch.ts.
  let touchActive = false;
  const moveAxis = new THREE.Vector2();
  let touchSkate = false;
  let lookDX = 0;
  let lookDY = 0;
  const TOUCH_LOOK = 0.004; // rad per drag-px (mouse is 0.002; touch wants a little more)
  const PI_2 = Math.PI / 2;
  const teuler = new THREE.Euler(0, 0, 0, "YXZ");
  const coarse = window.matchMedia("(pointer: coarse)").matches;

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
  // smoothed magnitude of the eye's vertical velocity (dune rises/falls + hops), so the
  // wind's speed signal swells as you skate over terrain, not just along the flat.
  let prevEyeY = camera.position.y;
  let vVert = 0;

  const onKeyDown = (e: KeyboardEvent) => {
    // ignore game hotkeys while a text field is focused (the dev book-search box):
    // otherwise typing a name toggles fly on the F and seeds the movement key set.
    const tgt = e.target as HTMLElement | null;
    if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA")) return;
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
  // desktop enters look-mode by clicking the canvas; on touch the tap surface IS the
  // controls (requestPointerLock is meaningless there), so skip it on coarse pointers.
  if (!coarse) dom.addEventListener("click", () => controls.lock());

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
    if (!controls.isLocked && !touchActive) return flying ? "fly" : "walk";

    // touch look: apply the accumulated drag with the same Euler('YXZ') + pole clamp
    // PLC uses for the mouse, read fresh from the quaternion so it composes with a
    // lookAt reset (spawn / return to start) exactly as mouse look does.
    if (touchActive && (lookDX !== 0 || lookDY !== 0)) {
      teuler.setFromQuaternion(camera.quaternion);
      teuler.y -= lookDX * TOUCH_LOOK;
      teuler.x -= lookDY * TOUCH_LOOK;
      teuler.x = Math.max(
        PI_2 - controls.maxPolarAngle,
        Math.min(PI_2 - controls.minPolarAngle, teuler.x),
      );
      camera.quaternion.setFromEuler(teuler);
      lookDX = 0;
      lookDY = 0;
    }

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

    // grounded: walk by default, skate while Shift is held (or the touch skate toggle).
    const skating = keys.has("ShiftLeft") || keys.has("ShiftRight") || touchSkate;
    const cfg = skating ? MOVE.skate : MOVE.walk;

    wish.set(0, 0, 0);
    if (keys.has("KeyW") || keys.has("ArrowUp")) wish.add(forward);
    if (keys.has("KeyS") || keys.has("ArrowDown")) wish.sub(forward);
    if (keys.has("KeyD") || keys.has("ArrowRight")) wish.add(right);
    if (keys.has("KeyA") || keys.has("ArrowLeft")) wish.sub(right);
    // analog touch stick: deflection sets both heading and (via wish length) speed.
    if (touchActive && moveAxis.lengthSq() > 0) {
      wish.addScaledVector(forward, moveAxis.y);
      wish.addScaledVector(right, moveAxis.x);
    }

    if (wish.lengthSq() > 0) {
      // accelerate toward the wished heading. Keyboard wishes are length >=1 → full
      // speed (unchanged); the analog stick scales the cap by its deflection (<=1).
      const defl = Math.min(1, wish.length());
      target.copy(wish).normalize().multiplyScalar(cfg.max * defl);
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
      prevEyeY = eyeY; // adopt the new height; a teleport isn't a dune to whoosh over
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

    // vertical eye speed (physics height, lift-free), lightly smoothed: this is what
    // makes the wind gust as you crest and drop dunes at speed.
    vVert += (Math.abs((eyeY - prevEyeY) / dt) - vVert) * (1 - Math.exp(-8 * dt));
    prevEyeY = eyeY;

    // cosmetic skate lift, eased in/out, applied on top of the physics eye height.
    const liftTarget = skating ? MOVE.hoverLift : 0;
    liftCur += (liftTarget - liftCur) * (1 - Math.exp(-MOVE.followK * dt));
    camera.position.y = eyeY + liftCur;
    lastAppliedY = camera.position.y;

    return skating ? "skate" : "walk";
  }

  // zero the carried velocity (teleport/spawn shouldn't arrive mid-glide).
  const stop = () => vel.set(0, 0, 0);

  // speed the wind couples to (units/sec): horizontal glide plus the eye's vertical
  // rate, so cresting dunes at a skate adds its own gust on top of the flat-out rush.
  const getSpeed = () => Math.hypot(vel.x, vel.z, vVert);

  // flight is a dev affordance; the pause menu drives it through the same path as
  // the F key (drop carried momentum so neither mode lurches), and reads it back to
  // keep the menu checkbox in sync with the key.
  const getFlying = () => flying;
  const setFlying = (v: boolean) => {
    if (v === flying) return;
    flying = v;
    vel.set(0, 0, 0);
  };

  // touch input surface (driven by touch.ts via main). setTouchActive mirrors the
  // lock state for coarse-pointer play; going inactive drops the analog input + any
  // carried velocity so a paused player never resumes mid-glide.
  const setTouchActive = (v: boolean) => {
    touchActive = v;
    if (!v) {
      moveAxis.set(0, 0);
      lookDX = 0;
      lookDY = 0;
      vel.set(0, 0, 0);
    }
  };
  const setMoveAxis = (x: number, y: number) => moveAxis.set(x, y);
  const addLook = (dx: number, dy: number) => {
    lookDX += dx;
    lookDY += dy;
  };
  const setTouchSkate = (v: boolean) => {
    touchSkate = v;
  };

  return {
    controls,
    update,
    stop,
    getSpeed,
    getFlying,
    setFlying,
    setTouchActive,
    setMoveAxis,
    addLook,
    setTouchSkate,
  };
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

  // preserveDrawingBuffer only under the screenshot harness, so canvas.toDataURL()
  // returns the last rendered frame rather than a cleared buffer. Off normally
  // (it can cost a copy per frame); see harness.ts.
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: HARNESS_ON,
  });
  // Render scale — the biggest mobile lever. The scene is fill-bound (full-screen
  // sky/daylight/ground shaders) and the box field's vertex cost can't be culled (the
  // books are too concentrated for frustum/distance culling to bite), so the win has to
  // come from fragments, not geometry. Lowering the pixel ratio cuts fragment work
  // across the WHOLE scene at once. devicePixelRatio is capped at 2 (denser retina is
  // wasted on this flat-shaded look), then scaled by renderScale; touch-primary devices
  // default lower since they're both higher-DPI and weaker. Re-applied on resize so a
  // browser-zoom dpr change is picked up too. Player-facing quality setting later; for
  // now a sane auto-default + a dev cycle key (P) to measure the win against the gpu line.
  const PR_CAP = 2;
  const isTouch = window.matchMedia("(pointer: coarse)").matches;
  // default to one of the menu's offered steps (1 / 0.75 / 0.5) so the control reflects
  // a highlighted value out of the box; touch starts at 0.75 (reads decent, real saving).
  let renderScale = isTouch ? 0.75 : 1.0;
  // Far-speck size floor, carried in FRAMEBUFFER px (what gl_PointSize wants) but authored in CSS
  // px (FAR_POINT_MIN_CSS) and re-derived from the live pixel ratio on every resize/dpr/renderScale
  // change -- so a far book stays the same PHYSICAL size on any screen. buildField wires it into the
  // points shader; without the resize update a dpr/zoom change would silently shrink the field again.
  const uFarMinPx = { value: FAR_POINT_MIN_CSS };
  const applyRenderScale = (): void => {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, PR_CAP) * renderScale);
    renderer.setSize(window.innerWidth, window.innerHeight);
    uFarMinPx.value = FAR_POINT_MIN_CSS * renderer.getPixelRatio();
  };
  const setRenderScale = (s: number): void => {
    renderScale = s;
    applyRenderScale();
  };
  applyRenderScale();
  document.body.appendChild(renderer.domElement);

  const {
    controls,
    update,
    stop,
    getSpeed,
    getFlying,
    setFlying,
    setTouchActive,
    setMoveAxis,
    addLook,
    setTouchSkate,
  } = createController(camera, renderer.domElement);
  scene.add(controls.object);
  // dev hook, same convention as window.MOVE: lets the camera be posed/inspected from the
  // devtools console (or a headless screenshot) without pointer lock or a rebuild.
  (window as unknown as { CAM: THREE.PerspectiveCamera }).CAM = camera;

  // Procedural desert wind (see wind.ts): zero-payload Web Audio, volume on the pause
  // slider, rush coupled to skate speed in the render loop. The AudioContext can only
  // start inside a user gesture, so unlock it on the first interaction — the same tap/
  // click/key that the player must make to enter the scene anyway. Idempotent and
  // one-shot per event type.
  const wind = createWind();
  const unlockAudio = () => wind.resume();
  for (const ev of ["pointerdown", "keydown", "touchstart"] as const) {
    window.addEventListener(ev, unlockAudio, { once: true });
  }

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

  // GPU timer (dev/baseline instrument). The render loop runs on setAnimationLoop,
  // which is vsync-capped, so FPS and even the stats.js MS panel can't reveal how
  // much GPU the book field actually costs while we sit under the refresh budget,
  // and a CPU performance.now() around render() only times command submission, not
  // the GPU work. EXT_disjoint_timer_query_webgl2 measures the true GPU duration of
  // the render() call independent of vsync — the one number that moves when we cull
  // triangles, so it's how we decide whether frustum tiling is worth building.
  // One query in flight at a time, polled before the next is begun (a result isn't
  // readable in its own frame); falls back to "n/a" where the ext is unavailable
  // (notably Firefox, which gates it for fingerprinting). Result is in nanoseconds.
  const glCtx = renderer.getContext() as WebGL2RenderingContext;
  const timerExt = glCtx.getExtension("EXT_disjoint_timer_query_webgl2") as
    | { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }
    | null;
  let gpuQueryInFlight: WebGLQuery | null = null;
  let gpuMs = -1; // -1 = no sample yet / unsupported

  // Placeholder pose until the anchors load (applySpawn below moves the player to
  // the rim). Centre + outward keeps the controller's first frames sane.
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
  const [field, teleporters, spawnAnchors, meta, world, heightmap, bookLods] =
    await Promise.all([
      loadPositions("positions.bin"),
      loadTeleporters("teleporters.json"),
      loadSpawnAnchors("spawn_anchors.json"),
      loadMeta("meta.bin"),
      loadWorld("world.json"),
      loadHeightmap("heightmap.bin"),
      // near = LOD01 (118v), mid = LOD02 box (24v); LOD00 (628v) is skipped -- it was the heaviest
      // tier and LOD01 reads the same at the close ranges, so the whole field is now ~2.6M verts.
      loadBookLods(bookUrl, ["book_LOD01", "book_LOD02"]),
    ]);
  mark("fetch+decode");
  // the heightmap is the ground-height source for the ground mesh and the player's
  // feet; init it before anything samples it.
  initHeightmap(heightmap.res, heightmap.worldSize, heightmap.data);
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
  // Spawn at a FIXED distance from centre on a curated landmark's bearing, facing
  // outward. The anchor supplies direction ONLY, not distance: tying spawn radius to
  // each anchor's own radius (they span ~606-835) propagated that spread into the
  // framing -- low-radius anchors shoved the player back into the empty plaza, high
  // ones dropped them mid-field. A constant radius just inside the rim puts every
  // opening at the threshold of the present: books begin a step ahead, the empty
  // plaza is behind for the turn-around, and the recognisable anchor (always beyond
  // the rim, r>=606) sits somewhere ahead in view. Walking outward runs back in time
  // into less-familiar, sparser ground -- the thesis as a gradient under the feet.
  // Now the heightmap is loaded so sampleHeight is real (placeholder pose used the
  // analytic fallback).
  const SPAWN_RADIUS = world.R_INNER - 20; // threshold of the present, just inside the rim
  let spawnIdx = 0;
  function applySpawn(a: SpawnAnchor) {
    const r = Math.hypot(a.x, a.y) || 1;
    const dx = a.x / r; // outward unit bearing (data x -> world x, data y -> world z)
    const dz = a.y / r;
    const px = dx * SPAWN_RADIUS;
    const pz = dz * SPAWN_RADIUS;
    camera.position.set(px, sampleHeight(px, pz) + EYE_HEIGHT, pz);
    camera.lookAt(px + dx * 8000, 0, pz + dz * 8000);
    stop(); // arrive at rest (matters for the dev cycle key mid-walk)
  }
  // Random anchor per session, except under the screenshot harness where the poses
  // must stay deterministic (it overrides the camera each frame anyway).
  if (spawnAnchors.length) {
    spawnIdx = HARNESS_ON ? 0 : Math.floor(Math.random() * spawnAnchors.length);
    applySpawn(spawnAnchors[spawnIdx]);
  } else {
    // no anchors: keep the centre placeholder, just settle its eye onto the baked surface.
    camera.position.y = sampleHeight(camera.position.x, camera.position.z) + EYE_HEIGHT;
  }
  mark("terrain");
  const built = buildField(field, bookLods[0], bookLods[1], uPlayer, daylight.uniforms, world, uFarMinPx);
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
  // Controls legend: prominent (.intro) for the first INTRO_MS of play, then a quiet
  // persistent corner line (.show); toggled off is remembered in localStorage. The
  // intro timer is armed once on the first scene enter and never replays.
  const INTRO_MS = 7000;
  let legendOn = localStorage.getItem("showControls") !== "0";
  let introElapsed = false;
  let introTimer = 0;
  // touch has no pointer lock, so a coarse-pointer device tracks "in the scene" with
  // its own flag; inScene() unifies the two so the picker/inspect gate reads the same
  // on both. The flag is driven by sceneEnter/sceneLeave alongside the controller's.
  let touchActive = false;
  const inScene = () => controls.isLocked || touchActive;
  // contextual wording: no Esc / no keycaps on touch, so prompts say "tap".
  const closeHint = isTouch
    ? "Tap outside to close"
    : "Esc or click outside to close";

  // The touch UI is created later (it needs the pause/inspect entry points), but the
  // scene-transition helpers reference it, so declare the handle up front. sceneLeave/
  // sceneEnter are the platform-agnostic "an overlay takes over" / "back to play" pair:
  // desktop drives the UI off PointerLock's lock/unlock events, touch drives it off the
  // touchActive flag + showing/hiding the controls (no lock to hang events on). Every
  // overlay/pause path calls these instead of controls.lock()/unlock() directly.
  let touch: ReturnType<typeof createTouchControls> | null = null;
  const sceneLeave = () => {
    if (isTouch) {
      touchActive = false;
      setTouchActive(false);
      touch?.setEnabled(false);
    } else {
      controls.unlock();
    }
  };
  const sceneEnter = () => {
    // first entry into play arms the one-shot intro timer: the legend rides bright for
    // INTRO_MS then settles to its subdued persistent state (guarded so it never replays).
    if (!introTimer) {
      introTimer = window.setTimeout(() => {
        introElapsed = true;
      }, INTRO_MS);
    }
    if (isTouch) {
      touchActive = true;
      setTouchActive(true);
      touch?.setEnabled(true);
      pauseEl.style.display = "none"; // no 'lock' event on touch to hide it for us
    } else {
      controls.lock();
    }
  };

  const renderName = (i: number) => escapeHtml(meta.name(i) || "(untitled)");
  const renderDesc = (i: number) => escapeHtml(meta.desc(i));

  // The look-at prompt for whatever's under the reticle. A teleporter borrows the
  // book's glance layout exactly -- name / detail / era / action -- so the two read
  // as one grammar; only the words and the verb ("travel" vs "inspect") differ.
  // the verb line: a keycap on desktop ("E inspect"), tap wording on touch
  // ("Tap to inspect") — same affordance, the device's own idiom.
  const actLine = (verb: string) =>
    isTouch
      ? `<div class="act">Tap to ${verb}</div>`
      : `<div class="act"><span class="key">E</span> ${verb}</div>`;
  function showGlance(a: { kind: "book" | "tp"; i: number }) {
    if (a.kind === "tp") {
      const tp = teleporters[a.i];
      // Name only the place and era, never a single figure: the gate sits on its
      // members' centroid, not on any one of them, so calling out "Leo Tolstoy"
      // promised a person who isn't here. The label is a region, which is honest.
      glance.innerHTML =
        `<div class="name">◎ ${escapeHtml(tp.label)}</div>` +
        (tp.era ? `<div class="years">${escapeHtml(tp.era)}</div>` : "") +
        actLine("travel");
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
      actLine("inspect");
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
      `<div class="hint">${closeHint}</div>`;
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
    sceneLeave(); // free the cursor (desktop) / freeze + hide touch UI so the card is tappable
  }

  function closeOverlay() {
    overlay.style.display = "none";
    overlayOpen = false;
    // straight back to walking, not the bare-cursor limbo. The overlay unlocked the
    // pointer programmatically (not a user Esc), so no post-Esc throttle applies here;
    // every caller (scrim click, Esc keydown, a travel pick) is a live user gesture, so
    // the re-lock request is honoured. (A book/tp open suppressed the pause menu via
    // overlayOpen, so this is the only thing that owns the unlocked moment.) On touch
    // this re-arms the controls flag and re-shows the stick instead.
    sceneEnter();
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
        const line = `${escapeHtml(tp.era)} · ${Math.round(o.d).toLocaleString()}u`;
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
      `<div class="hint">${closeHint}</div>`;
    card.querySelectorAll<HTMLButtonElement>(".tp-dest").forEach((btn) => {
      btn.addEventListener("click", () => travelTo(Number(btn.dataset.i)));
    });
    overlay.style.display = "flex";
    overlayOpen = true;
    glance.style.display = "none";
    sceneLeave(); // free the cursor / freeze + hide touch UI so destinations are tappable
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
  const devCarpetEl = document.getElementById("dev-carpet") as HTMLInputElement;
  const devBooksEl = document.getElementById("dev-books") as HTMLInputElement;
  const spinnerEl = document.getElementById("pause-spinner") as HTMLSpanElement;

  // Resolution (render scale) — player-facing, NOT behind the dev gate. A segmented
  // choice over the same steps the P dev key cycles; the active one wears the accent.
  // syncQuality re-reads renderScale so the menu reflects a default or a P-key change.
  const qualityOpts = Array.from(
    document.querySelectorAll<HTMLButtonElement>("#quality-seg button"),
  );
  const syncQuality = (): void => {
    for (const b of qualityOpts) {
      b.classList.toggle("active", Number(b.dataset.scale) === renderScale);
    }
  };
  for (const b of qualityOpts) {
    b.addEventListener("click", () => {
      setRenderScale(Number(b.dataset.scale));
      syncQuality();
    });
  }
  syncQuality();

  // Binary settings render as a switch + a value word (see .switch in index.html). The
  // whole .switch-wrap is the tap target so a phone gets a generous hit area. onChange
  // fires with the requested state; sync() repaints aria-checked + the word. optimistic
  // (default) repaints immediately on tap; pass false when the real state is confirmed
  // asynchronously (fullscreen) so a rejected/raced request can't leave the switch lying.
  const makeSwitch = (
    switchId: string,
    valId: string,
    onWord: string,
    offWord: string,
    onChange: (on: boolean) => void,
    optimistic = true,
  ): { sync: (on: boolean) => void } => {
    const btn = document.getElementById(switchId) as HTMLButtonElement;
    const val = document.getElementById(valId) as HTMLSpanElement;
    const wrap = btn.closest(".switch-wrap") as HTMLElement;
    const sync = (on: boolean): void => {
      btn.setAttribute("aria-checked", on ? "true" : "false");
      val.textContent = on ? onWord : offWord;
    };
    wrap.addEventListener("click", () => {
      const next = btn.getAttribute("aria-checked") !== "true";
      onChange(next);
      if (optimistic) sync(next);
    });
    return { sync };
  };

  // Far-book level of detail — player-facing toggle. ON = Full (draw every far book);
  // OFF = Low, which thins the far point carpet in dense cells (the mobile primitive-
  // throughput lever, see buildField). setCarpetThin already defaults Low on coarse-
  // pointer devices and Full on desktop, so syncLod just reflects that on open.
  const lodSwitch = makeSwitch("lod-switch", "lod-val", "Full", "Low", (full) =>
    built.setCarpetThin(!full),
  );
  const syncLod = (): void => lodSwitch.sync(!built.isCarpetThin());
  syncLod();

  // Controls hint — player-facing show/hide toggle. The render loop reads legendOn; the
  // choice persists to localStorage across reloads.
  const legendSwitch = makeSwitch("legend-switch", "legend-val", "Shown", "Hidden", (on) => {
    legendOn = on;
    localStorage.setItem("showControls", legendOn ? "1" : "0");
  });
  legendSwitch.sync(legendOn);

  // Wind volume — player-facing range, 0 disables. The slider IS the off switch (the
  // wind is part of the piece, so it ships audible with a low default); the choice
  // persists. The render loop drives the actual sound; this only sets the target.
  const windSlider = document.getElementById("wind-vol") as HTMLInputElement;
  const savedWind = localStorage.getItem("windVolume");
  const initWindVol = savedWind !== null ? Number(savedWind) : 0.6;
  windSlider.value = String(Math.round(initWindVol * 100));
  wind.setVolume(initWindVol);
  windSlider.addEventListener("input", () => {
    const v = Number(windSlider.value) / 100;
    wind.setVolume(v);
    localStorage.setItem("windVolume", String(v));
  });

  // Fullscreen toggle — the mobile answer to accidental nav-button/back taps (going
  // fullscreen hides the system bars on Android). Feature-detected: iPhone Safari has
  // no Fullscreen API, so the row stays hidden there (the iOS path is Add-to-Home-
  // Screen via the manifest). The seg mirrors the live document.fullscreenElement, so
  // an external exit (swipe/Esc) re-syncs the buttons.
  const fsRow = document.getElementById("fullscreen-row") as HTMLDivElement;
  const docEl = document.documentElement as HTMLElement & {
    webkitRequestFullscreen?: () => Promise<void>;
  };
  const fsRequest = docEl.requestFullscreen || docEl.webkitRequestFullscreen;
  const fsExit =
    document.exitFullscreen ||
    (document as Document & { webkitExitFullscreen?: () => Promise<void> })
      .webkitExitFullscreen;
  const fsElement = () =>
    document.fullscreenElement ||
    (document as Document & { webkitFullscreenElement?: Element })
      .webkitFullscreenElement;
  if (fsRequest && fsExit) {
    fsRow.hidden = false;
    // optimistic=false: the switch follows the REAL fullscreenchange event, not the tap,
    // so a rejected or raced request (permission state, external Esc/swipe exit) always
    // leaves the switch showing the truth rather than an optimistic lie.
    const fsSwitch = makeSwitch(
      "fullscreen-switch",
      "fullscreen-val",
      "On",
      "Off",
      (want) => {
        // the tap is the user gesture the API requires; swallow rejections so the menu
        // never throws.
        if (want && !fsElement()) fsRequest.call(docEl).catch(() => {});
        else if (!want && fsElement()) fsExit.call(document).catch(() => {});
      },
      false,
    );
    const syncFs = (): void => fsSwitch.sync(!!fsElement());
    document.addEventListener("fullscreenchange", syncFs);
    document.addEventListener("webkitfullscreenchange", syncFs);
    syncFs();
  }

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
    devCarpetEl.checked = built.isCarpetVisible();
    devBooksEl.checked = built.isBooksVisible();
    syncQuality();
    syncLod();
    syncDevOpts();
    pauseEl.style.display = "flex";
    // the readiness spinner is the post-Esc re-lock cooldown hint; touch has no pointer
    // lock and thus no cooldown, so its Resume takes instantly — skip the spinner there.
    if (!isTouch) {
      spinnerEl.classList.add("show");
      if (cooldownTimer) clearTimeout(cooldownTimer);
      cooldownTimer = window.setTimeout(() => {
        cooldownTimer = 0;
        spinnerEl.classList.remove("show");
      }, COOLDOWN_HINT_MS);
    }
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
    if (isTouch) {
      sceneEnter(); // re-arm the controls + hide the menu; no pointer lock to request
      return;
    }
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
      // back to this session's opening vista (the rim anchor), not the empty centre.
      if (spawnAnchors.length) applySpawn(spawnAnchors[spawnIdx]);
      else camera.position.set(0, sampleHeight(0, 0) + EYE_HEIGHT, 0);
      fade.style.opacity = "0"; // fade back in at the spawn
    }, 300);
  };

  devModeEl.addEventListener("change", syncDevOpts);
  devFlyEl.addEventListener("change", () => setFlying(devFlyEl.checked));
  devFlatEl.addEventListener("change", () => setFlatLit(devFlatEl.checked));
  devPerfEl.addEventListener("change", () => setPerf(devPerfEl.checked));
  devCarpetEl.addEventListener("change", () => built.setCarpetVisible(devCarpetEl.checked));
  devBooksEl.addEventListener("change", () => built.setBooksVisible(devBooksEl.checked));

  // --- dev: find a book by name ----------------------------------------------
  // Type a name, get matching books; a row jumps you straight there, the ☆ drops a
  // ◆ on the compass (the same bookmark the inspect card toggles) so you can walk to
  // it. There's no title->index map and 576k titles, so the lowercased search index
  // is built once on first use and the filter is debounced. Behind the dev gate.
  const findInput = document.getElementById("dev-find-input") as HTMLInputElement;
  const findResults = document.getElementById("dev-find-results") as HTMLDivElement;
  const FIND_LIMIT = 40;
  let findIndex: string[] | null = null;
  let findTimer = 0;

  // jump the player onto a book. Mirrors Return to start: re-lock on this click's
  // gesture, then reposition under cover of the fade so the pop is never seen.
  const teleportToBook = (i: number) => {
    resume();
    fade.style.opacity = "1";
    window.setTimeout(() => {
      const x = built.px[i];
      const z = built.pz[i];
      camera.position.set(x, sampleHeight(x, z) + EYE_HEIGHT, z);
      stop();
      fade.style.opacity = "0";
    }, 300);
  };

  const renderFind = () => {
    const q = findInput.value.trim().toLowerCase();
    if (!q) {
      findResults.innerHTML = "";
      return;
    }
    if (!findIndex) {
      findIndex = new Array<string>(meta.n);
      for (let i = 0; i < meta.n; i++) findIndex[i] = meta.name(i).toLowerCase();
    }
    const hits: number[] = [];
    for (let i = 0; i < findIndex.length && hits.length < FIND_LIMIT; i++) {
      if (findIndex[i].includes(q)) hits.push(i);
    }
    if (!hits.length) {
      findResults.innerHTML = `<div class="dev-find-empty">no match</div>`;
      return;
    }
    findResults.innerHTML = hits
      .map((i) => {
        const years = fmtYears(meta.birth[i], meta.death[i]);
        const on = compass.has(i);
        return (
          `<div class="dev-find-row">` +
          `<button class="dff-go" data-i="${i}" title="Jump here">` +
          `<span class="dfn">${escapeHtml(meta.name(i))}</span>` +
          (years ? `<span class="dfy">${escapeHtml(years)}</span>` : "") +
          `</button>` +
          `<button class="dff-mark${on ? " on" : ""}" data-i="${i}" ` +
          `title="Bookmark on compass">${on ? "★" : "☆"}</button>` +
          `</div>`
        );
      })
      .join("");
  };

  findResults.addEventListener("click", (e) => {
    const t = e.target as HTMLElement;
    const go = t.closest(".dff-go") as HTMLElement | null;
    if (go) {
      teleportToBook(Number(go.dataset.i));
      return;
    }
    const mark = t.closest(".dff-mark") as HTMLElement | null;
    if (mark) {
      const on = compass.toggle(Number(mark.dataset.i));
      mark.textContent = on ? "★" : "☆";
      mark.classList.toggle("on", on);
    }
  });

  findInput.addEventListener("input", () => {
    if (findTimer) clearTimeout(findTimer);
    findTimer = window.setTimeout(renderFind, 120);
  });
  // DEV_DEFAULT: surface the perf HUD and pre-expand the dev section so a test build is ready
  // to read the instant it loads -- no Esc-into-menu-and-tick-Developer-mode each run.
  if (DEV_DEFAULT) {
    setPerf(true);
    devModeEl.checked = true;
    syncDevOpts();
    // Pinned dev toolbar. On touch the B/M/L/F keys don't exist, and the pause-menu checkboxes
    // can't drive the perf measurement: pausing covers the scene, so you can't watch the live
    // gpu line while flipping a layer. These one-tap buttons drive the SAME setters with the
    // scene visible -- the on-device way to run the dots-vs-books attribution. Bottom-left,
    // clear of the perf readout (top-left) and the touch look-zone (right half). Ships out with
    // DEV_DEFAULT. Each button repaints from the is* getter on tap; toggles made via the pause
    // menu instead won't refresh a stale label until the next tap (acceptable for dev tooling).
    const bar = document.createElement("div");
    bar.style.cssText =
      "position:fixed;left:4px;bottom:4px;z-index:101;display:flex;gap:4px;font:11px/1 monospace;";
    const mkBtn = (
      label: string,
      isOn: () => boolean,
      set: (on: boolean) => void,
    ): void => {
      const b = document.createElement("button");
      const paint = (): void => {
        const on = isOn();
        b.textContent = `${label} ${on ? "·on" : "off"}`;
        b.style.cssText =
          "padding:7px 9px;border:1px solid #9fe;border-radius:3px;font:inherit;cursor:pointer;" +
          (on
            ? "background:rgba(0,0,0,.55);color:#9fe;"
            : "background:rgba(0,0,0,.82);color:#566;");
      };
      b.addEventListener("click", () => {
        set(!isOn());
        paint();
      });
      paint();
      bar.appendChild(b);
    };
    mkBtn("dots", () => built.isCarpetVisible(), (on) => built.setCarpetVisible(on));
    mkBtn("books", () => built.isBooksVisible(), (on) => built.setBooksVisible(on));
    mkBtn("thin", () => built.isCarpetThin(), (on) => built.setCarpetThin(on));
    mkBtn("flat", () => isFlatLit(), (on) => setFlatLit(on));
    mkBtn("fly", () => getFlying(), (on) => setFlying(on));
    document.body.appendChild(bar);
  }
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

  // ── touch controls (coarse-pointer devices) ─────────────────────────────
  // Build the analog surface and wire it to the same entry points the keyboard/mouse
  // uses: the stick + drag feed the controller, a tap on an armed reticle inspects or
  // travels, the pause chip drops to the menu. The glance/close wording already says
  // "tap" (isTouch); here we also swap the intro hint and start in the scene, since
  // there's no click-to-lock gesture to wait for on touch.
  if (isTouch) {
    // mirror the desktop legend's keycap+action rows, with touch gestures as the
    // "keys" (the #controls grid aligns the action column regardless of key width).
    // Skate + pause have their own on-screen chips; pause is echoed here for the cue.
    controlsHud.innerHTML =
      `<div class="ctl"><span class="key">Left ½</span><span class="act">Move</span></div>` +
      `<div class="ctl"><span class="key">Right ½</span><span class="act">Look</span></div>` +
      `<div class="ctl"><span class="key">Tap</span><span class="act">Inspect</span></div>` +
      // ‖ not ⏸: the pause character renders as the phone's orange colour emoji; the
      // double-bar is plain text and mirrors the drawn corner button. See index.html.
      `<div class="ctl"><span class="key">‖</span><span class="act">Pause</span></div>`;
    touch = createTouchControls({
      onMove: (x, y) => setMoveAxis(x, y),
      onLook: (dx, dy) => addLook(dx, dy),
      onSkateToggle: (on) => setTouchSkate(on),
      onTap: () => {
        if (overlayOpen || !touchActive || !aim) return;
        if (aim.kind === "tp") openTravel(aim.i);
        else openOverlay(aim.i);
      },
      onPause: () => {
        sceneLeave();
        showPause();
      },
    });
    sceneEnter(); // begin active: show the stick + arm the controller's touch path
  }

  document.addEventListener("keydown", (e) => {
    if (e.code === "KeyE" && !overlayOpen && controls.isLocked && aim) {
      if (aim.kind === "tp") openTravel(aim.i);
      else openOverlay(aim.i);
    } else if (e.code === "KeyL") {
      setFlatLit(!isFlatLit());
    } else if (e.code === "Backquote") {
      setPerf(!isPerf());
    } else if (e.code === "KeyB") {
      // baseline instrument (dev-only, throwaway): hide the entire far dot carpet.
      // carpet-off is the ceiling of perfect frustum culling — watch the gpu line with
      // it on vs off, looking outward at spawn, to see what 574k points actually cost
      // before we decide whether tiling can recover enough of it to be worth building.
      built.setCarpetVisible(!built.isCarpetVisible());
      console.log(`[perf] dot carpet ${built.isCarpetVisible() ? "shown" : "HIDDEN"}`);
    } else if (e.code === "KeyM") {
      // perf-attribution partner to B: hide the book MESHES (near+mid) so B(dots)+M(books)
      // isolate each field layer's GPU cost. Watch the gpu line: all-on, then dots-only,
      // then books-only, at a horizon angle where the overdraw stack is deepest.
      built.setBooksVisible(!built.isBooksVisible());
      console.log(`[perf] book meshes ${built.isBooksVisible() ? "shown" : "HIDDEN"}`);
    } else if (e.code === "KeyP") {
      // dev cycle for the render-scale (pixel-ratio) lever: 1.0 -> 0.75 -> 0.5, so the
      // gpu line can be read at each step. The auto-default (0.67 on touch) is off-cycle;
      // pressing P enters the measurement steps. The player-facing control lands in the
      // pause menu's quality section once the win is confirmed.
      const steps = [1.0, 0.75, 0.5];
      const idx = steps.indexOf(renderScale);
      setRenderScale(steps[(idx + 1) % steps.length]);
      console.log(
        `[perf] render scale ${renderScale} -> pixel ratio ${renderer.getPixelRatio().toFixed(2)}`,
      );
    } else if (e.code === "KeyN") {
      // dev cycle for the opening-vista anchors: step through each (applySpawn + log
      // the figure) to eyeball every spawn and cut the duds. Shift steps backward.
      if (spawnAnchors.length) {
        const n = spawnAnchors.length;
        spawnIdx = (spawnIdx + (e.shiftKey ? n - 1 : 1)) % n;
        applySpawn(spawnAnchors[spawnIdx]);
        console.log(`[spawn] ${spawnIdx + 1}/${n}  ${spawnAnchors[spawnIdx].title}`);
      }
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
    applyRenderScale(); // re-reads dpr (zoom) and renderScale, then resizes
  });

  mark("wiring");
  console.log(`[load] total ${(performance.now() - t0).toFixed(0)}ms to first frame`);

  // Screenshot harness (dev-only, ?harness): when active it pins the camera to a
  // fixed pose and freezes the drift clock so frames are reproducible. Inert
  // otherwise. See harness.ts.
  const harness = setupHarness({
    camera,
    daylight: daylight.uniforms,
    teleporters,
    sampleHeight,
    eyeHeight: EYE_HEIGHT,
    world,
    renderer,
  });

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
    // harness pins the camera in place of the controller; otherwise walk normally.
    mode = harness.active ? harness.applyPose() : update(dt);

    // wind: the rush layer tracks ground speed (normalised by the skate top speed, so a
    // walk sits near silent and only a fast skate brings it up); then ease everything.
    wind.setSpeed(getSpeed() / MOVE.skate.max);
    wind.update(dt);

    // look-at picking: only while walking the scene (locked) and not inspecting.
    // Suppressed while skating so the glance prompt doesn't strobe as books blow past.
    sincePick += dt;
    if (!overlayOpen && inScene() && mode !== "skate") {
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

    // controls legend: a dim persistent line (.show) while in play, brightened
    // (.intro) for the first INTRO_MS until the intro timer flips introElapsed.
    // Suppressed while an inspect overlay is up or the hint is toggled off — but
    // the pause menu KEEPS it (bright), so the reference is there when you stop to
    // look, and the pause "Controls hint" toggle demonstrates itself live.
    {
      const paused = pauseEl.style.display !== "none";
      const live = legendOn && !overlayOpen && (inScene() || paused);
      controlsHud.classList.toggle("show", live);
      controlsHud.classList.toggle("intro", live && (paused || !introElapsed));
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
    // harness: override the advance with a fixed clock so frames are reproducible.
    if (harness.active) harness.freeze();
    // recentre the sky on the viewer (night dome wraps it, day layer rides overhead)
    // and deepen it with radial depth into the past. The day layer reads the field at
    // absolute world xz, so its openings stay world-locked as the player walks.
    const depth = clamp(
      Math.hypot(camera.position.x, camera.position.z) / world.R_MAX,
      0,
      1,
    );
    sky.update(depth, camera.position);

    // GPU-time the render: poll the previous frame's query, then wrap this render in
    // a fresh one. A query can't be read in its own frame, so we keep just one in
    // flight and read it a frame or two later — fine for a ~4 Hz readout.
    if (timerExt) {
      if (gpuQueryInFlight) {
        const done = glCtx.getQueryParameter(gpuQueryInFlight, glCtx.QUERY_RESULT_AVAILABLE);
        const disjoint = glCtx.getParameter(timerExt.GPU_DISJOINT_EXT);
        if (done || disjoint) {
          if (done && !disjoint) {
            gpuMs = glCtx.getQueryParameter(gpuQueryInFlight, glCtx.QUERY_RESULT) / 1e6;
          }
          glCtx.deleteQuery(gpuQueryInFlight);
          gpuQueryInFlight = null;
        }
      }
      if (!gpuQueryInFlight) {
        gpuQueryInFlight = glCtx.createQuery();
        glCtx.beginQuery(timerExt.TIME_ELAPSED_EXT, gpuQueryInFlight);
        renderer.render(scene, camera);
        glCtx.endQuery(timerExt.TIME_ELAPSED_EXT);
      } else {
        renderer.render(scene, camera);
      }
    } else {
      renderer.render(scene, camera);
    }

    // read renderer.info AFTER render (it resets per frame), throttled to ~4 Hz so
    // the DOM write doesn't itself cost frames. The near count is the LOD's pulse:
    // it should sit in the low thousands and clamp at NEAR_CAP in the dense band.
    stats.update();
    sinceStat += dt;
    if (sinceStat >= 0.25) {
      sinceStat = 0;
      const r = renderer.info.render;
      const gpuStr = !timerExt ? "n/a (no ext)" : gpuMs < 0 ? "…" : `${gpuMs.toFixed(2)}ms`;
      // the far base is now one points cloud: it runs the vertex shader on all n every frame
      // (the promoted ones collapse to 0px), so "drawn" is just the field total minus the
      // promoted detail books — no per-tile cull count to report anymore.
      const farBooks = field.n - built.nearMesh.count - built.midMesh.count;
      perf.textContent =
        `gpu    ${gpuStr}\n` +
        `scale  ${renderScale.toFixed(2)}x (pr ${renderer.getPixelRatio().toFixed(2)})\n` +
        `calls  ${r.calls}\n` +
        `tris   ${(r.triangles / 1e6).toFixed(2)}M\n` +
        `near   ${built.nearMesh.count.toLocaleString()} full + ${built.midMesh.count.toLocaleString()} mid${built.isBooksVisible() ? "" : " (HIDDEN)"}\n` +
        `far    ${farBooks.toLocaleString()} / ${field.n.toLocaleString()} pts${built.isCarpetVisible() ? "" : " (HIDDEN)"}\n` +
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
