import * as THREE from "three";
import { CloudUniforms, SKY_CRACK_COMMON } from "./clouds";

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
// The day sky seen THROUGH the cracks in the night, not a glow laid over it. The
// earlier version blended ADDITIVELY, which let the stars shine through the bright
// patches: luminous blobs over a dark sky read as clouds, never as holes. This layer
// instead paints the day OPAQUE inside each opening, so it covers the night dome and
// its stars behind a hard-edged rim. That occlusion is the cue that flips the read
// from "cloud deck" to "window into a daytime sky".
const DAY_LOW = new THREE.Color(0xcfdcef); // pale blue toward the horizon
const DAY_TOP = new THREE.Color(0x5b8fd6); // clear day blue overhead
const DAY_CLOUD = new THREE.Color(0xfdfbf4); // warm white of the day clouds in the gap
const SUN_COL = new THREE.Color(0xfff1cf); // the sun disc + halo glimpsed in a crack

// Puffy day clouds glimpsed inside the windows. They are NOT painted on this plane: that
// glued them to the openings (coplanar world-locked fields shift together under walking,
// so the cloud read as paint on the glass). Instead the puffs are sampled in VIEW
// DIRECTION, a far cloudscape locked to where you look, not where you stand. The
// world-locked openings then slide across this near-static sky as you walk, which is the
// parallax of distant cloud seen behind a near window.
const DAY_CLOUD_DIR_SCALE = 2.0; // view-direction -> cloud uv (smaller = larger puffs)
const DAY_CLOUD_LO = 0.46; // coverage below this is clear blue
const DAY_CLOUD_HI = 0.78; // coverage above this is solid white cloud
const DAY_CLOUD_AMT = 0.85; // how far the puffs push the blue toward white

// The glimpsed sun's elevation, DECOUPLED from the terrain's sun. The world sun sits low
// (~9deg) for the raking dune relief and the unlit vortex centre; glimpsed at that angle
// it lands in the fading horizon rim and reads wrong. The sky sun keeps the world sun's
// azimuth but is lifted here so it reads as a sun up in the day sky. Surreal by design.
const SKY_SUN_ELEV = 0.5; // radians (~29deg)

const CEIL_H = 1800; // layer altitude (world units; lower = openings read bigger/nearer)
// Layer half-size. Kept large so the plane's rim sits only ~4deg above the horizon (a
// flat plane asymptotes to 0deg and its geometry ends at the rim, so the rim elevation is
// how low the cracks can reach). The diagonal corner (CEIL_R*sqrt2) must stay inside the
// camera's far plane, which is sized to suit in main.ts.
const CEIL_R = 26000;

// Where the crack opens into day. The crack field is ~0 inside the shards and ~1 on
// the seams, so the band selects the seam WIDTH: a low LO catches the broad approach to
// a crack, HI the hairline core. Widen the band for fat cracks, tighten it for hairline
// fractures.
const OPEN_LO = 0.78;
const OPEN_HI = 0.95;

// Fade the layer out toward its rim (as a fraction of CEIL_R) so the flat sheet
// dissolves into the dome instead of ending on a visible edge, and the aliasing-prone
// grazing rim never shows. Pushed close to the rim so the cracks carry almost all the
// way down to the horizon and only the final sliver dissolves.
const FADE_FROM = 0.92;

const DEEP_NIGHT = 0.25; // opening opacity multiplier at the rim (closes into the past)

export interface Sky {
  group: THREE.Group;
  // depth: 0 at the present (origin), 1 at the rim (deepest past). camPos: the camera
  // world position, to recentre both layers (the day layer keeps the camera's XZ and
  // sits at CEIL_H) so the dome wraps the viewer and the day field stays world-locked.
  update: (depth: number, camPos: THREE.Vector3) => void;
}

// cloud: the shared cloud uniforms (texture + drift time) so the day layer reads the
// same field, time and wind as the ground's lit patches.
export function buildSky(cloud: CloudUniforms, sunPos: THREE.Vector3): Sky {
  const uDepth = { value: 0 };

  // the glimpsed sun: world sun's azimuth, but lifted to SKY_SUN_ELEV (see above).
  const sunH = Math.hypot(sunPos.x, sunPos.z);
  const skySunDir = new THREE.Vector3(
    sunPos.x,
    Math.tan(SKY_SUN_ELEV) * sunH,
    sunPos.z,
  ).normalize();

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
    // NORMAL blending, not additive: the day is painted OPAQUE inside an opening so it
    // covers the night dome and its stars, and the opening edge is a hard rim. That
    // occlusion is what reads as a hole punched through the night, not a glowing cloud.
    blending: THREE.NormalBlending,
    fog: false,
    uniforms: {
      uDayLow: { value: DAY_LOW.clone().convertSRGBToLinear() },
      uDayTop: { value: DAY_TOP.clone().convertSRGBToLinear() },
      uDayCloud: { value: DAY_CLOUD.clone().convertSRGBToLinear() },
      uSunCol: { value: SUN_COL.clone().convertSRGBToLinear() },
      uSunDir: { value: skySunDir },
      uCamPos: { value: new THREE.Vector3() },
      uDepth,
      // the shared Voronoi crack field: the SAME web the ground now lights through, so the
      // openings overhead and the daylight seams below read as one system.
      uCrack: cloud.uCrack,
      uCloudTime: cloud.uCloudTime,
    },
    vertexShader: /* glsl */ `
      varying vec2 vWorldXZ;
      varying vec2 vLocalXY;
      varying vec3 vWorldPos;
      void main() {
        // true world position of this fragment (the sheet is recentred on the camera but
        // the field is read at absolute world xz, so the openings are world-locked). The
        // full 3D position feeds the view ray for the sun glimpse.
        vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
        vWorldXZ = vWorldPos.xz;
        vLocalXY = position.xy; // local plane coords; length taken per fragment
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec2 vWorldXZ;
      varying vec2 vLocalXY;
      varying vec3 vWorldPos;
      uniform vec3 uDayLow;
      uniform vec3 uDayTop;
      uniform vec3 uDayCloud;
      uniform vec3 uSunCol;
      uniform vec3 uSunDir;
      uniform vec3 uCamPos;
      uniform float uDepth;
      ${SKY_CRACK_COMMON}

      // value-noise fbm: used both to warp the crack lattice and to paint the day clouds
      // glimpsed inside a gap (its own field, separate from the crack mask).
      float h21(vec2 p){ p = fract(p * vec2(123.34, 345.45)); p += dot(p, p + 34.345); return fract(p.x * p.y); }
      float vnoise(vec2 p){
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        float a = h21(i), b = h21(i + vec2(1,0)), c = h21(i + vec2(0,1)), d = h21(i + vec2(1,1));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      }
      float fbm2(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ v += a * vnoise(p); p *= 2.03; a *= 0.5; } return v; }

      void main() {
        // analytic antialiasing: thresholding the crack field sparkles where many cells
        // fall in one pixel (grazing angles near the horizon, thin sub-pixel seams). Widen
        // the smoothstep band by the field's own screen-space rate of change, so the edge
        // is always at least ~a pixel wide: crisp and thin up close (tiny fwidth), softly
        // dissolved at distance (large fwidth) instead of breaking into sparkles. The warp
        // that makes the seams wander now lives inside crackField (shared with the ground).
        float cf = crackField(vWorldXZ);
        float aa = fwidth(cf);
        float open = smoothstep(${OPEN_LO.toFixed(2)} - aa, ${OPEN_HI.toFixed(2)} + aa, cf);
        float rt = clamp(length(vLocalXY) / ${CEIL_R.toFixed(1)}, 0.0, 1.0); // 0 overhead -> 1 rim
        float fade = 1.0 - smoothstep(${FADE_FROM.toFixed(2)}, 1.0, rt);

        // the view ray for this fragment: drives both the direction-locked far cloudscape
        // and the sun glimpse, so both stay fixed to where you look while the world-locked
        // openings slide across them as you walk.
        vec3 viewDir = normalize(vWorldPos - uCamPos);

        // day-sky behind the gap: clear blue overhead easing to pale toward the horizon.
        vec3 day = mix(uDayTop, uDayLow, rt);
        // puffy white day clouds, sampled in VIEW DIRECTION (a far cloudscape), not on this
        // plane: a gnomonic projection so they compress toward the horizon and parallax
        // against the near openings instead of being glued to them.
        float vy = max(viewDir.y, 0.12); // clamp the grazing rim so coords don't blow up
        vec2 cuv = (viewDir.xz / vy) * ${DAY_CLOUD_DIR_SCALE.toFixed(2)} + vec2(0.013, -0.007) * uCloudTime;
        float puff = smoothstep(${DAY_CLOUD_LO.toFixed(2)}, ${DAY_CLOUD_HI.toFixed(2)}, fbm2(cuv));
        day = mix(day, uDayCloud, puff * ${DAY_CLOUD_AMT.toFixed(2)});

        // the sun, glimpsed only through a crack toward its bearing. Direction-based, so
        // the disc stays fixed while the world-locked openings slide across it.
        float sd = max(dot(viewDir, uSunDir), 0.0);
        float disc = smoothstep(0.9986, 0.9997, sd);
        float halo = pow(sd, 220.0);
        day += uSunCol * (disc + halo * 0.5);

        day *= (1.0 - ${DEPTH_DARKEN.toFixed(2)} * uDepth);
        float a = open * fade * mix(1.0, ${DEEP_NIGHT.toFixed(2)}, uDepth);
        // let the sun disc punch through the rim fade so a low gap can still show it.
        a = max(a, open * disc * mix(1.0, ${DEEP_NIGHT.toFixed(2)}, uDepth));
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
    (dayMat.uniforms.uCamPos.value as THREE.Vector3).copy(camPos); // view ray for the sun
  }

  return { group, update };
}
