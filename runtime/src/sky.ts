import * as THREE from "three";
import { CloudUniforms, SKY_CLOUD_COMMON } from "./clouds";

// --- sky --------------------------------------------------------------------
// Two layers, built to read as one weather over the desert without pretending to a
// physical alignment a dome shader cannot give. A dome is direction-based (every pixel
// an angle, infinitely far); the ground's day/night is place-based (a value at a world
// XZ). There is no clean map between them, so casting the field onto a dome always
// smears at the horizon or pinwheels at the zenith. The fix is to stop faking it on
// the dome and put the OPENINGS on real geometry at a finite altitude:
//
//   1. NIGHT DOME  - a stable inward sphere recentred on the camera: a dark gradient
//      with a cube-face starfield. It is the backdrop; it does not move as you walk.
//   2. DAY LAYER   - a large flat sheet at altitude CEIL_H, recentred on the camera's
//      XZ but sampling the SAME drifting field the ground uses (clouds.ts), at the same
//      scale, by true world XZ. Where the field opens it paints a day-sky glow; else it
//      is transparent and the night dome shows through. Because it is real geometry the
//      GPU's own perspective makes overhead openings large and distant ones recede to
//      the horizon (no smear, no pole), and because the field is read at absolute world
//      XZ the openings are world-locked: walking gives parallax, not a sliding sky.
//
// It does not matter that an opening is not exactly over its lit dune; both ride the
// same field at the same scale and drift together, so the eye reads one system. The
// dome carries TIME radially: into the past it dims and the day openings close toward
// night (DEEP_NIGHT), so the deep past is a dark starfield.

// --- night dome --------------------------------------------------------------
const NIGHT_LOW = new THREE.Color(0x141a2e); // deep blue at the skyline
const NIGHT_TOP = new THREE.Color(0x05060e); // near-black indigo overhead
const STAR_COLOR = new THREE.Color(0xcfe0ff); // cool white

const HORIZON_Y = -0.12; // horizon line ~7deg below eye level
const SKY_TOP = 0.9; // elevation over which the gradient reaches the zenith

// Stars on a cube-face grid (no pole, no seam). DENSITY is cells per face-uv unit
// (face uv spans [-1,1], so cells across a face = 2*DENSITY); THRESH is how many cells
// hold a star (higher = sparser); SIZE is the point radius in cell units; TW_SPEED is
// the twinkle rate. Stars fade in off the skyline (STAR_RISE).
const STAR_DENSITY = 22.0;
const STAR_THRESH = 0.86;
const STAR_SIZE = 0.03;
const STAR_BRIGHT = 0.9;
const STAR_TW_SPEED = 0.6;
const STAR_RISE = 0.16;

const DEPTH_DARKEN = 0.55; // how much the dome dims at the deepest past

// Large enough that the farthest content stays inside it, and inside the far plane.
const RADIUS = 20000;

// --- day layer ---------------------------------------------------------------
// The opening sky seen where the deck breaks: a bright warm pale at the rim easing to
// a soft day blue overhead. Eyeball knobs.
const DAY_LOW = new THREE.Color(0xf0e9d8); // warm pale near the horizon
const DAY_TOP = new THREE.Color(0x9fc2ec); // soft day blue overhead
// The day layer is blended ADDITIVELY so the openings read as illuminated sky bursting
// through the night rather than opaque painted cloud: the colour adds to the dome and
// its stars instead of covering them. DAY_GAIN pushes the open cores to bloom hot.
const DAY_GAIN = 1.5;

const CEIL_H = 1800; // layer altitude (world units; lower = openings read bigger/nearer)
const CEIL_R = 15000; // layer half-size; reaches ~7deg elevation at this altitude

// Where the field opens into day. The field is two noise layers clustered near 0.5,
// so its real range is only ~0.35..0.65; the band must sit inside that or nothing ever
// opens. Starting at the ground's day onset (clouds COVER_LO = 0.5) couples the
// openings to the lit patches: a break sits where the ground below is already turning
// to day, which reads as the two agreeing.
const OPEN_LO = 0.55;
const OPEN_HI = 0.75;

// Fade the layer out toward its rim (as a fraction of CEIL_R) so the flat sheet
// dissolves into the dome near the horizon instead of ending on a visible edge, and
// the aliasing-prone grazing rim never shows.
const FADE_FROM = 0.5;

const DEEP_NIGHT = 0.25; // opening opacity multiplier at the rim (closes into the past)

// Break the field's visible tile repeat (every SCALE1 = 1700u) with a domain warp:
// the sample point is nudged by a lower-frequency reading of the same field, so the
// regular lattice of openings dissolves into organic shapes. Seen wide and head-on at
// altitude, the repeat reads far more than it does on the ground at a grazing angle.
// WARP_SCALE shrinks the world xz for the warp source (smaller = larger, smoother
// warp); WARP_AMT is the nudge in world units (larger = more break, looser coupling).
const WARP_SCALE = 0.75;
const WARP_AMT = 500;

export interface Sky {
  group: THREE.Group;
  // depth: 0 at the present (origin), 1 at the rim (deepest past). camPos: the camera
  // world position, to recentre both layers (the day layer keeps the camera's XZ and
  // sits at CEIL_H) so the dome wraps the viewer and the day field stays world-locked.
  update: (depth: number, camPos: THREE.Vector3) => void;
}

// cloud: the shared cloud uniforms (texture + drift time) so the day layer reads the
// same field, time and wind as the ground's lit patches.
export function buildSky(cloud: CloudUniforms): Sky {
  const uDepth = { value: 0 };

  // --- night dome ---
  const domeMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
    fog: false,
    uniforms: {
      uNightLow: { value: NIGHT_LOW.clone().convertSRGBToLinear() },
      uNightTop: { value: NIGHT_TOP.clone().convertSRGBToLinear() },
      uStarColor: { value: STAR_COLOR.clone().convertSRGBToLinear() },
      uDepth,
      uHorizonY: { value: HORIZON_Y },
      uSkyTop: { value: SKY_TOP },
      uCloudTime: cloud.uCloudTime, // only for the star twinkle
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        // the dome is never rotated, so the local position is the world direction.
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vDir;
      uniform vec3 uNightLow;
      uniform vec3 uNightTop;
      uniform vec3 uStarColor;
      uniform float uDepth;
      uniform float uHorizonY;
      uniform float uSkyTop;
      uniform float uCloudTime;

      float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 345.45));
        p += dot(p, p + 34.345);
        return fract(p.x * p.y);
      }

      // stars on a cube-face grid: pick the major axis, grid the face uv, take the
      // brightest hashed point in the 3x3 neighbourhood. No pole, no back-seam.
      float starField(vec3 dir) {
        vec3 ad = abs(dir);
        vec2 cuv; float face;
        if (ad.x >= ad.y && ad.x >= ad.z) { cuv = dir.zy / ad.x; face = 0.0; }
        else if (ad.y >= ad.z) { cuv = dir.xz / ad.y; face = 17.0; }
        else { cuv = dir.xy / ad.z; face = 41.0; }
        vec2 g = cuv * ${STAR_DENSITY.toFixed(1)};
        vec2 base = floor(g);
        float best = 0.0;
        for (int j = -1; j <= 1; j++) {
          for (int i = -1; i <= 1; i++) {
            vec2 cell = base + vec2(float(i), float(j));
            vec2 key = cell + face;
            float h = hash21(key);
            float present = step(${STAR_THRESH.toFixed(2)}, h);
            vec2 jit = vec2(hash21(key + 5.0), hash21(key + 9.0));
            float d = length(g - (cell + jit));
            float pt = smoothstep(${STAR_SIZE.toFixed(2)}, 0.0, d);
            float bright = 0.4 + 0.6 * hash21(key + 13.0);
            float tw = 0.7 + 0.3 * sin(uCloudTime * ${STAR_TW_SPEED.toFixed(2)} + h * 31.0);
            best = max(best, present * pt * bright * tw);
          }
        }
        return best;
      }

      void main() {
        float e = vDir.y - uHorizonY;
        float t = smoothstep(0.0, uSkyTop, e);
        vec3 col = mix(uNightLow, uNightTop, t);

        float s = starField(vDir) * ${STAR_BRIGHT.toFixed(2)};
        s *= smoothstep(0.0, ${STAR_RISE.toFixed(2)}, e);
        col += uStarColor * s;

        col = mix(col, col * (1.0 - ${DEPTH_DARKEN.toFixed(2)}), uDepth);

        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
        float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        gl_FragColor.rgb += (dither - 0.5) / 255.0;
      }
    `,
  });
  const dome = new THREE.Mesh(new THREE.SphereGeometry(RADIUS, 48, 24), domeMat);
  dome.frustumCulled = false;
  dome.renderOrder = -10;

  // --- day layer ---
  const dayMat = new THREE.ShaderMaterial({
    side: THREE.DoubleSide, // seen from below
    transparent: true,
    depthWrite: false,
    depthTest: true, // sit behind any terrain it passes behind near the horizon
    blending: THREE.AdditiveBlending, // openings ADD light: illuminated sky, not cloud
    fog: false,
    uniforms: {
      uDayLow: { value: DAY_LOW.clone().convertSRGBToLinear() },
      uDayTop: { value: DAY_TOP.clone().convertSRGBToLinear() },
      uDepth,
      // shared with the ground's cloud shadows: same texture, same drift time, so the
      // openings and the lit dunes are one field. Must be the SAME objects.
      uClouds: cloud.uClouds,
      uCloudTime: cloud.uCloudTime,
    },
    vertexShader: /* glsl */ `
      varying vec2 vWorldXZ;
      varying vec2 vLocalXY;
      void main() {
        // true world xz of this fragment (the sheet is recentred on the camera but the
        // field is read at absolute world xz, so the openings are world-locked).
        vWorldXZ = (modelMatrix * vec4(position, 1.0)).xz;
        vLocalXY = position.xy; // local plane coords; length taken per fragment
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vWorldXZ;
      varying vec2 vLocalXY;
      uniform vec3 uDayLow;
      uniform vec3 uDayTop;
      uniform float uDepth;
      ${SKY_CLOUD_COMMON}
      void main() {
        // domain warp to break the tile lattice: nudge the sample by a lower-frequency
        // reading of the same field. Still the same drifting field, so the openings
        // keep belonging with the ground; they just no longer fall on a grid.
        vec2 w = vec2(
          skyCover(vWorldXZ * ${WARP_SCALE.toFixed(2)} + 11.3),
          skyCover(vWorldXZ * ${WARP_SCALE.toFixed(2)} + 41.7)
        ) - 0.5;
        float open = smoothstep(${OPEN_LO.toFixed(2)}, ${OPEN_HI.toFixed(2)}, skyCover(vWorldXZ + w * ${WARP_AMT.toFixed(1)}));
        float rt = clamp(length(vLocalXY) / ${CEIL_R.toFixed(1)}, 0.0, 1.0); // 0 overhead -> 1 rim
        float fade = 1.0 - smoothstep(${FADE_FROM.toFixed(2)}, 1.0, rt);
        // day-sky gradient: blue overhead easing to warm pale toward the horizon,
        // pushed by DAY_GAIN so the open cores bloom hot under additive blending.
        vec3 day = mix(uDayTop, uDayLow, rt) * ${DAY_GAIN.toFixed(2)} * (1.0 - ${DEPTH_DARKEN.toFixed(2)} * uDepth);
        float a = open * fade * mix(1.0, ${DEEP_NIGHT.toFixed(2)}, uDepth);
        gl_FragColor = vec4(day, a);
        #include <colorspace_fragment>
      }
    `,
  });
  const day = new THREE.Mesh(new THREE.PlaneGeometry(CEIL_R * 2, CEIL_R * 2), dayMat);
  day.rotation.x = -Math.PI / 2; // lay it flat overhead
  day.frustumCulled = false;
  day.renderOrder = -9;

  const group = new THREE.Group();
  group.add(dome, day);

  function update(depth: number, camPos: THREE.Vector3): void {
    uDepth.value = depth;
    dome.position.copy(camPos); // wrap the viewer
    day.position.set(camPos.x, CEIL_H, camPos.z); // ride overhead, field stays world-locked
  }

  return { group, update };
}
