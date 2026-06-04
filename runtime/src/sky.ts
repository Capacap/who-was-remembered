import * as THREE from "three";
import { CloudUniforms, WIND_DIR } from "./clouds";

// --- sky --------------------------------------------------------------------
// A brooding overcast after Kuindzhi: a dark cloud ceiling where the only bright thing in
// the world is the selective light on the ground below. It is ONE dome carrying two
// things, kept conceptually separate:
//
//   1. GRADIENT    - a muted glow toward the zenith easing to a dark horizon. The infinite
//      backdrop; it does not move as you walk.
//   2. CLOUD DECKS - dark drifting masses composited over the gradient. These are NOT a
//      separate finite quad: each deck is a flat-overhead-sheet projection of the view
//      direction (dir.xz / dir.y, the ray-through-a-horizontal-plane), so the dome already
//      behaves like an infinite cloud sheet hung above the player, with none of a finite
//      quad's edge to hide. The decks drift along the SHARED ground wind (WIND_DIR), so the
//      clouds overhead and the lit swaths below move the same world direction.
//
// The dome carries TIME radially: into the past it dims, so the deep past is a darker,
// emptier sky.

// --- dome --------------------------------------------------------------------
// The storm gradient lives on the dome (it is the real backdrop). SKY_OVER is the muted
// glow toward the zenith, SKY_HORIZON the dark band at the skyline; the dome ramps
// between them by elevation.
const SKY_OVER = new THREE.Color(0x35372b); // muted glow toward the zenith
const SKY_HORIZON = new THREE.Color(0x12110b); // dark earthy band at the skyline

const HORIZON_Y = -0.12; // horizon line ~7deg below eye level
const SKY_TOP = 0.9; // elevation over which the gradient reaches the zenith

const DEPTH_DARKEN = 0.55; // how much the dome dims at the deepest past

// --- storm clouds (on the dome) ----------------------------------------------
// Dark drifting masses sampled in view direction by a flat-plane projection (so the
// clouds recede toward the horizon in perspective). They darken the gradient toward
// CLOUD_DARK, the grey-green glow showing through the breaks. Several decks are stacked
// (compositing far to near) at different scales, drift speeds and height falloffs so they
// do not move in lockstep and the cloudscape reads with depth and parallax as you turn.
// Each is densest overhead and faded to nothing toward the horizon: that matches the
// reference's clearer low band AND hides the projection's hard stretch at grazing angles.
const CLOUD_DARK = new THREE.Color(0x0a0a0e); // near-black brooding masses
const CLOUD_LO = 0.35; // fbm below this is a clear break (shared by all decks)
const CLOUD_HI = 0.78; // fbm above this is a full mass

// Each deck: [projection scale (smaller = larger masses), drift SPEED along the shared
// wind, height floor, height ceil, darkening amount]. The drift vector is WIND_DIR * speed,
// computed in the GLSL splice below, so every deck moves along the same world axis as the
// ground shadows, only at its own speed. Listed far (large, slow, low) to near (fine, fast,
// high) and composited in that order so the near deck sits on top.
const CLOUD_LAYERS: [number, number, number, number, number][] = [
  [1.0, 0.004, 0.1, 0.4, 0.85], // far: broadest, slowest, hugs the horizon band
  [1.7, 0.008, 0.16, 0.5, 0.8], // mid
  [2.6, 0.014, 0.24, 0.6, 0.72], // near: finest, fastest, sits highest
];

// Large enough that the farthest content stays inside it, and inside the far plane.
const RADIUS = 20000;

export interface Sky {
  group: THREE.Group;
  // depth: 0 at the present (origin), 1 at the rim (deepest past). camPos: the camera
  // world position, used to recentre the dome on the viewer.
  update: (depth: number, camPos: THREE.Vector3) => void;
}

// cloud: the shared cloud uniforms (only uCloudTime here, driving the storm drift so it
// moves with the ground's lit patches); _sunPos is unused now the glimpsed sun is gone.
export function buildSky(cloud: CloudUniforms, _sunPos: THREE.Vector3): Sky {
  const uDepth = { value: 0 };

  // --- night dome ---
  const domeMat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: false,
    fog: false,
    uniforms: {
      uSkyHorizon: { value: SKY_HORIZON.clone().convertSRGBToLinear() },
      uSkyOver: { value: SKY_OVER.clone().convertSRGBToLinear() },
      uCloudDark: { value: CLOUD_DARK.clone().convertSRGBToLinear() },
      uDepth,
      uHorizonY: { value: HORIZON_Y },
      uSkyTop: { value: SKY_TOP },
      uCloudTime: cloud.uCloudTime, // drives the cloud drift
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
      uniform vec3 uSkyHorizon;
      uniform vec3 uSkyOver;
      uniform vec3 uCloudDark;
      uniform float uDepth;
      uniform float uHorizonY;
      uniform float uSkyTop;
      uniform float uCloudTime;

      float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 345.45));
        p += dot(p, p + 34.345);
        return fract(p.x * p.y);
      }

      // value-noise fbm for the drifting cloud masses.
      float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        float a = hash21(i), b = hash21(i + vec2(1,0)), c = hash21(i + vec2(0,1)), d = hash21(i + vec2(1,1));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
      }
      float fbm2(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { v += a * vnoise(p); p *= 2.03; a *= 0.5; } return v; }

      // one cloud deck: a flat-plane projection of the view direction (so it recedes toward
      // the horizon) sampled by fbm, faded out below floorY..ceilY so the stretched grazing
      // band stays clear. Returns mass density [0,1].
      float cloudDeck(vec3 dir, float scale, vec2 drift, float floorY, float ceilY) {
        float vy = max(dir.y, 0.12); // clamp the grazing rim so the projection stays finite
        vec2 cuv = (dir.xz / vy) * scale + drift * uCloudTime;
        float m = smoothstep(${CLOUD_LO.toFixed(2)}, ${CLOUD_HI.toFixed(2)}, fbm2(cuv));
        return m * smoothstep(floorY, ceilY, dir.y);
      }

      void main() {
        float e = vDir.y - uHorizonY;
        float t = smoothstep(0.0, uSkyTop, e); // 0 at the horizon -> 1 at the zenith
        vec3 col = mix(uSkyHorizon, uSkyOver, t);

        // dark drifting storm decks, composited far to near; each recedes toward the horizon
        // and fades out near it. The gradient glow shows through the breaks.
        ${CLOUD_LAYERS.map(([sc, sp, fl, cl, amt]) => {
          const dx = WIND_DIR[0] * sp;
          const dy = WIND_DIR[1] * sp;
          return `col = mix(col, uCloudDark, cloudDeck(vDir, ${sc.toFixed(2)}, vec2(${dx.toFixed(5)}, ${dy.toFixed(5)}), ${fl.toFixed(2)}, ${cl.toFixed(2)}) * ${amt.toFixed(2)});`;
        }).join("\n        ")}

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

  // The cloud layer (dark drifting masses on a finite sheet) returns here, on top of the
  // dome's gradient, once the gradient reads right.

  const group = new THREE.Group();
  group.add(dome);

  function update(depth: number, camPos: THREE.Vector3): void {
    uDepth.value = depth;
    dome.position.copy(camPos); // wrap the viewer
  }

  return { group, update };
}
