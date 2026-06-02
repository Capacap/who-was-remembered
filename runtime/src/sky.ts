import * as THREE from "three";

// --- sky --------------------------------------------------------------------
// A gradient dome standing in for the open sky, now carrying the world's one
// real axis: TIME. The disc is radial time (present at the origin, deep past at
// the rim), and the sky encodes it. A warm "present" glow sits low on the
// horizon in the direction of the world origin, so wherever you stand the sky
// tells you which way the present lies; the opposite side (the deep past, the
// way out) cools and darkens. As you travel outward the whole dome deepens, so
// the light itself reads as your era. None of this is physical sky; it is the
// time axis painted overhead.
//
// The dome is a single inward-facing sphere recentred on the camera each frame
// (see main's loop), so it sits at a fixed apparent distance and never clips. It
// is NOT rotated, so a dome point's normalized local position is its world
// direction: vDir.y is the sine of elevation, vDir.xz is the world bearing, and
// the present-glow can be steered toward the world origin by bearing alone.
//
// The land fades to fully transparent into this dome (applyDistanceFade), so the
// far desert dissolves into whatever the dome paints at the horizon. The old
// shared ground-haze tone (passed in, matched at the seam) is retired: the fade
// is alpha, not a colour step, so the land can dissolve into a warm or cold
// horizon without a visible join.

// Base vertical gradient. Surreal-dusk rather than bright desert: a pale warm
// skyline easing up to a dusty indigo zenith. Eyeball knobs.
const SKY_LOW = new THREE.Color(0xb8a6a0); // pale warm grey at the skyline
const ZENITH = new THREE.Color(0x444d72); // dusty indigo overhead

// The time tints. PRESENT blooms warm low in the sky toward the origin; PAST
// cools and darkens the far (outward) side.
const PRESENT = new THREE.Color(0xffc27a); // warm dawn toward the present
const PAST = new THREE.Color(0x1c2440); // cold dark toward the deep past

// Horizon line sits below eye level so looking level shows more sky than ground
// and the skyline reads as dropping away. SKY_TOP is how far above the line the
// sky takes to reach the zenith.
const HORIZON_Y = -0.12; // ~7deg below eye level
const SKY_TOP = 0.7;

// How strongly each time tint paints, and how tightly the present-glow hugs the
// horizon (larger LOW_BAND = the warmth climbs higher up the dome).
const PRESENT_STRENGTH = 0.85;
const PAST_STRENGTH = 0.55;
const LOW_BAND = 0.5; // elevation over which the horizon glow fades out
const DEPTH_DARKEN = 0.6; // how much the whole dome dims at the deepest past

// Large enough that the farthest content (rim seen from the opposite rim,
// ~14000u) stays inside it, and inside the camera's far plane (24000u).
const RADIUS = 20000;

export interface Sky {
  mesh: THREE.Mesh;
  // camX/camZ: player world position (the dome's centre). depth: 0 at the
  // present (origin), 1 at the rim (deepest past). Steers the present-glow
  // toward the origin and deepens the dome with era.
  update: (camX: number, camZ: number, depth: number) => void;
}

export function buildSky(): Sky {
  const uToOrigin = { value: new THREE.Vector2(0, 0) };
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
      uPresent: { value: PRESENT.clone().convertSRGBToLinear() },
      uPast: { value: PAST.clone().convertSRGBToLinear() },
      uToOrigin, // world-xz direction from the camera to the origin (normalized)
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
      uniform vec3 uPresent;
      uniform vec3 uPast;
      uniform vec2 uToOrigin;
      uniform float uDepth;
      uniform float uHorizonY;
      uniform float uSkyTop;
      void main() {
        // elevation measured from the dome's horizon line (below eye level).
        float e = vDir.y - uHorizonY;
        // base vertical gradient: skyline -> zenith. Below the horizon it holds
        // the skyline tone, which is what the faded far land dissolves into.
        vec3 col = mix(uSkyLow, uZenith, pow(smoothstep(0.0, uSkyTop, e), 1.3));

        // bearing toward the present: +1 looking at the origin, -1 toward the rim.
        float az = dot(normalize(vDir.xz + vec2(1e-5)), uToOrigin);
        // both time tints hug the horizon and fade out with elevation.
        float lowBand = 1.0 - smoothstep(0.0, ${LOW_BAND.toFixed(2)}, e);

        // warm present-glow toward the origin.
        float toward = smoothstep(0.0, 1.0, az);
        col = mix(col, uPresent, lowBand * toward * ${PRESENT_STRENGTH.toFixed(2)});
        // cold deepening toward the deep past (the outward radial).
        float away = smoothstep(0.0, 1.0, -az);
        col = mix(col, uPast, lowBand * away * ${PAST_STRENGTH.toFixed(2)});

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

  function update(camX: number, camZ: number, depth: number): void {
    // direction from the player to the world origin (the present), on the ground
    // plane. At the origin itself there is no direction; leave it zero so the
    // present-glow simply vanishes (you are already in the present).
    const r = Math.hypot(camX, camZ);
    if (r > 1e-3) uToOrigin.value.set(-camX / r, -camZ / r);
    else uToOrigin.value.set(0, 0);
    uDepth.value = depth;
  }

  return { mesh, update };
}
