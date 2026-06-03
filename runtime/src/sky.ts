import * as THREE from "three";

// --- sky --------------------------------------------------------------------
// A gradient dome standing in for the open sky. It carries the world's one real
// axis, TIME, but only radially: the whole dome darkens as the player travels
// outward from the present (origin) into the deep past (rim), so the light itself
// reads as your era. Azimuth carries nothing, on purpose. In this world the angle
// around the disc is LONGITUDE (geography), so steering a time cue by bearing would
// paint time onto the geography axis; time stays radial, where the rest of the
// world keeps it, and the compass does the wayfinding.
//
// The warm/cool split across the dome is a fixed dusk, not a time cue: a warm glow
// sits low on the horizon WHERE THE SUN IS, cooling and darkening toward the
// anti-sun side. The glow shares its bearing with the scene's DirectionalLight
// (passed in), so the bright side of the sky and the lit side of the dunes agree,
// and because the bearing is fixed in world space the sky never rotates as you
// walk. (An earlier version steered the glow toward the world origin, recomputed
// from the camera each frame; near the centre a few steps swung the bearing tens of
// degrees and the whole sky whipped around, reading as a sun teleporting.)
//
// The dome is a single inward-facing sphere recentred on the camera each frame (see
// main's loop), so it sits at a fixed apparent distance and never clips. It is NOT
// rotated, so a dome point's normalized local position is its world direction:
// vDir.y is the sine of elevation and vDir.xz is the world bearing, which is why a
// fixed world-space sun bearing lands the glow in the same place from anywhere.
//
// The land fades to fully transparent into this dome (applyDistanceFade), so the
// far desert dissolves into whatever the dome paints at the horizon, no colour seam.

// Base vertical gradient. Surreal-dusk rather than bright desert: a pale warm
// skyline easing up to a dusty indigo zenith. Eyeball knobs.
const SKY_LOW = new THREE.Color(0xb8a6a0); // pale warm grey at the skyline
const ZENITH = new THREE.Color(0x444d72); // dusty indigo overhead

// The dusk tints. SUN_WARM blooms low toward the sun bearing; SKY_COOL cools and
// darkens the anti-sun side.
const SUN_WARM = new THREE.Color(0xffc27a); // warm glow toward the sun
const SKY_COOL = new THREE.Color(0x1c2440); // cold dark away from it

// Horizon line sits below eye level so looking level shows more sky than ground
// and the skyline reads as dropping away. SKY_TOP is how far above the line the
// sky takes to reach the zenith.
const HORIZON_Y = -0.12; // ~7deg below eye level
const SKY_TOP = 0.7;

// How strongly each tint paints, and how tightly the warm glow hugs the horizon
// (larger LOW_BAND = the warmth climbs higher up the dome).
const WARM_STRENGTH = 0.85;
const COOL_STRENGTH = 0.55;
const LOW_BAND = 0.5; // elevation over which the horizon glow fades out
const DEPTH_DARKEN = 0.6; // how much the whole dome dims at the deepest past

// Large enough that the farthest content (rim seen from the opposite rim,
// ~14000u) stays inside it, and inside the camera's far plane (24000u).
const RADIUS = 20000;

export interface Sky {
  mesh: THREE.Mesh;
  // depth: 0 at the present (origin), 1 at the rim (deepest past). Deepens the
  // whole dome with era. The warm glow's bearing is fixed (the sun), so nothing
  // about the dome tracks the camera's xz any more.
  update: (depth: number) => void;
}

// sunPos: the scene's DirectionalLight position; its xz bearing is where the warm
// glow sits, so the bright side of the sky matches the lit side of the ground.
export function buildSky(sunPos: THREE.Vector3): Sky {
  const r = Math.hypot(sunPos.x, sunPos.z) || 1;
  const uToSun = { value: new THREE.Vector2(sunPos.x / r, sunPos.z / r) };
  const uDepth = { value: 0 };
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false, // pure backdrop; never occludes the world
    fog: false,
    uniforms: {
      // colours converted to linear working space so that, after the colorspace
      // encode in the fragment shader, they render exactly as authored. A raw
      // ShaderMaterial gets none of three's automatic colour management.
      uSkyLow: { value: SKY_LOW.clone().convertSRGBToLinear() },
      uZenith: { value: ZENITH.clone().convertSRGBToLinear() },
      uWarm: { value: SUN_WARM.clone().convertSRGBToLinear() },
      uCool: { value: SKY_COOL.clone().convertSRGBToLinear() },
      uToSun, // world-xz bearing toward the sun (normalized, fixed)
      uDepth, // player's normalized radial depth into the past
      uHorizonY: { value: HORIZON_Y },
      uSkyTop: { value: SKY_TOP },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        // local position is the direction from the dome centre (= the camera),
        // so its normalized y is the sine of elevation and its xz is the world
        // bearing (the dome is never rotated).
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vDir;
      uniform vec3 uSkyLow;
      uniform vec3 uZenith;
      uniform vec3 uWarm;
      uniform vec3 uCool;
      uniform vec2 uToSun;
      uniform float uDepth;
      uniform float uHorizonY;
      uniform float uSkyTop;
      void main() {
        // elevation measured from the dome's horizon line (below eye level).
        float e = vDir.y - uHorizonY;
        // base vertical gradient: skyline -> zenith. Below the horizon it holds
        // the skyline tone, which is what the faded far land dissolves into.
        vec3 col = mix(uSkyLow, uZenith, pow(smoothstep(0.0, uSkyTop, e), 1.3));

        // bearing toward the sun: +1 looking at it, -1 looking away.
        float az = dot(normalize(vDir.xz + vec2(1e-5)), uToSun);
        // both tints hug the horizon and fade out with elevation.
        float lowBand = 1.0 - smoothstep(0.0, ${LOW_BAND.toFixed(2)}, e);

        // warm glow toward the sun.
        float toward = smoothstep(0.0, 1.0, az);
        col = mix(col, uWarm, lowBand * toward * ${WARM_STRENGTH.toFixed(2)});
        // cold deepening on the anti-sun side.
        float away = smoothstep(0.0, 1.0, -az);
        col = mix(col, uCool, lowBand * away * ${COOL_STRENGTH.toFixed(2)});

        // the whole dome dims as the player travels into the past, so the light
        // reads as the era you are standing in.
        col = mix(col, col * (1.0 - ${DEPTH_DARKEN.toFixed(2)}), uDepth);

        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(RADIUS, 48, 24), mat);
  mesh.frustumCulled = false; // it wraps the camera; never cull it

  function update(depth: number): void {
    uDepth.value = depth;
  }

  return { mesh, update };
}
