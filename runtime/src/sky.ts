import * as THREE from "three";
import { CloudUniforms, SKY_CLOUD_COMMON } from "./clouds";

// --- sky --------------------------------------------------------------------
// A storm ceiling, not an open dusk. The dome is a near-black layer of cloud with
// sparse breaks where light bursts through, and those breaks are cut by the SAME
// drifting field that lights the ground (see clouds.ts): the sky opens where the
// dunes beneath are lit, and the two pass over together as the weather drifts. The
// references are Kuindzhi and the storm-light watercolours, where the sky and the
// land are one system: a single break in a black deck lights a patch of meadow and
// shows as a silver burst overhead. Sky and ground here share one cloud field for
// exactly that.
//
// The dome still carries the world's one real axis, TIME, but only radially: the
// whole thing dims as the player travels outward from the present (origin) into the
// deep past (rim), so the light reads as your era. Azimuth carries no time cue, on
// purpose; the angle around the disc is LONGITUDE (geography), so a bearing-steered
// time cue would paint time onto the geography axis. The dusk warm/cool split and
// the cloud breaks vary by bearing, but those are weather, not time.
//
// The warm glow low on the horizon sits WHERE THE SUN IS, sharing its bearing with
// the scene's DirectionalLight (passed in), so the lit side of the sky and the lit
// side of the dunes agree. The bearing is fixed in world space, so the sky never
// rotates as you walk. A thin warm skyline strip survives under the black mass (the
// sunset band both references hold); it is also what the far land dissolves into.
//
// The dome is a single inward-facing sphere recentred on the camera each frame (see
// main's loop), so it sits at a fixed apparent distance and never clips. It is NOT
// rotated, so a dome point's normalized local position is its world direction:
// vDir.y is the sine of elevation and vDir.xz is the world bearing. To find where a
// dome ray meets the cloud deck we cast it onto a plane at height CLOUD_H above the
// viewer and sample the field at that world xz; uCamXZ (the camera's world xz) puts
// that sample in the same world frame the ground samples, so the breaks register.

// Base vertical gradient of the OPEN sky revealed through a break: a pale warm
// skyline easing up to a dusty indigo zenith. Eyeball knobs.
const SKY_LOW = new THREE.Color(0xb8a6a0); // pale warm grey at the skyline
const ZENITH = new THREE.Color(0x444d72); // dusty indigo overhead

// The dusk tints on the open sky. SUN_WARM blooms low toward the sun bearing;
// SKY_COOL cools the anti-sun side.
const SUN_WARM = new THREE.Color(0xffc27a); // warm glow toward the sun
const SKY_COOL = new THREE.Color(0x1c2440); // cold dark away from it

// The cloud ceiling itself: near-black and close to neutral (a warm base read as a
// brown haze overhead, not pitch cloud). CEIL_LOW is the base, CEIL_TOP the zenith.
// The two are kept close and dark; the breaks, not a steep gradient, carry the
// variation. BREAK_HOT is the silver-warm core of a wide break. A dim warm
// HORIZON_GLOW hugs the skyline so the far dunes read as silhouettes against a sky
// that is barely lighter than they are.
const CEIL_LOW = new THREE.Color(0x0d0c0b); // cloud base, near-black neutral
const CEIL_TOP = new THREE.Color(0x040405); // pitch overhead
const BREAK_HOT = new THREE.Color(0xffe9c4); // hot core where a break is widest
const HORIZON_GLOW = new THREE.Color(0x3d2a1c); // dim warm sliver at the skyline

// Horizon line sits below eye level so looking level shows more sky than ground.
// SKY_TOP is how far above the line the open sky takes to reach the zenith.
const HORIZON_Y = -0.12; // ~7deg below eye level
const SKY_TOP = 0.7;

// Open-sky tint strengths and how tightly the warm glow hugs the horizon.
const WARM_STRENGTH = 0.85;
const COOL_STRENGTH = 0.55;
const LOW_BAND = 0.5; // elevation over which the horizon glow fades out
const DEPTH_DARKEN = 0.6; // how much the whole dome dims at the deepest past

// The cloud deck. CLOUD_H is the layer height a dome ray is cast onto to find its
// coverage sample (world units; lower = the breaks read closer and bigger overhead).
// SKY_OPEN_LO/HI is the break threshold on the field's raw coverage: it is much
// higher and narrower than the ground's day/night band (clouds COVER_LO = 0.5), so
// only the brightest peaks open and the deck stays mostly black with sparse specks
// of light. Any sky break therefore sits over ground that is already well lit.
const CLOUD_H = 1200;
const SKY_OPEN_LO = 0.62;
const SKY_OPEN_HI = 0.92;

// Breaks are faded out below this elevation band. Near the horizon the pierce point
// races outward (t = CLOUD_H / vDir.y blows up), so a break smears radially into a
// vertical streak; fading the breaks in only well above the skyline keeps them in
// the upper sky where the geometry is stable and leaves the horizon a clean wall.
const BREAK_RISE_LO = 0.08;
const BREAK_RISE_HI = 0.32;

// The ceiling gradient is spread over this elevation so it never saturates inside
// the visible dome (max elevation above the horizon line is ~1.12 at the zenith).
// A wider spread means a gentler ramp with no visible knee where it would flatten.
const CEIL_SPREAD = 1.6;

// The dim warm horizon glow: how far up the dome it reaches (HORIZON_BAND, tight to
// the skyline) and how strongly it lifts the dark deck (HORIZON_STRENGTH, weak, so
// the light is barely perceptible). It is biased a touch brighter toward the sun.
const HORIZON_BAND = 0.1;
const HORIZON_STRENGTH = 0.55;

// Large enough that the farthest content (rim seen from the opposite rim,
// ~14000u) stays inside it, and inside the camera's far plane (24000u).
const RADIUS = 20000;

export interface Sky {
  mesh: THREE.Mesh;
  // depth: 0 at the present (origin), 1 at the rim (deepest past); dims the whole
  // dome with era. camX/camZ: the camera's world xz, so the cloud breaks sample the
  // same world frame the ground does and register with the lit patches below.
  update: (depth: number, camX: number, camZ: number) => void;
}

// sunPos: the scene's DirectionalLight position; its xz bearing is where the warm
// glow sits. cloud: the shared cloud uniforms (texture + drift time) so the dome's
// breaks ride the same field as the ground's lit patches.
export function buildSky(sunPos: THREE.Vector3, cloud: CloudUniforms): Sky {
  const r = Math.hypot(sunPos.x, sunPos.z) || 1;
  const uToSun = { value: new THREE.Vector2(sunPos.x / r, sunPos.z / r) };
  const uDepth = { value: 0 };
  const uCamXZ = { value: new THREE.Vector2(0, 0) };
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
      uCeilLow: { value: CEIL_LOW.clone().convertSRGBToLinear() },
      uCeilTop: { value: CEIL_TOP.clone().convertSRGBToLinear() },
      uBreakHot: { value: BREAK_HOT.clone().convertSRGBToLinear() },
      uHorizonGlow: { value: HORIZON_GLOW.clone().convertSRGBToLinear() },
      uToSun, // world-xz bearing toward the sun (normalized, fixed)
      uCamXZ, // camera world xz, for sampling the cloud field in world space
      uDepth, // player's normalized radial depth into the past
      uHorizonY: { value: HORIZON_Y },
      uSkyTop: { value: SKY_TOP },
      // shared with the ground's cloud shadows: same texture, same drift time, so
      // the sky breaks and the lit dunes are one field. Must be the SAME objects.
      uClouds: cloud.uClouds,
      uCloudTime: cloud.uCloudTime,
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
      uniform vec3 uCeilLow;
      uniform vec3 uCeilTop;
      uniform vec3 uBreakHot;
      uniform vec3 uHorizonGlow;
      uniform vec2 uToSun;
      uniform vec2 uCamXZ;
      uniform float uDepth;
      uniform float uHorizonY;
      uniform float uSkyTop;
      ${SKY_CLOUD_COMMON}
      void main() {
        // elevation measured from the dome's horizon line (below eye level).
        float e = vDir.y - uHorizonY;

        // the OPEN sky revealed through a break: skyline -> zenith, with the warm
        // glow toward the sun and a cool deepening away from it. This is the old
        // dusk dome, now only seen through the holes in the deck.
        vec3 open = mix(uSkyLow, uZenith, pow(smoothstep(0.0, uSkyTop, e), 1.3));
        float az = dot(normalize(vDir.xz + vec2(1e-5)), uToSun);
        float lowBand = 1.0 - smoothstep(0.0, ${LOW_BAND.toFixed(2)}, e);
        float toward = smoothstep(0.0, 1.0, az);
        open = mix(open, uWarm, lowBand * toward * ${WARM_STRENGTH.toFixed(2)});
        float away = smoothstep(0.0, 1.0, -az);
        open = mix(open, uCool, lowBand * away * ${COOL_STRENGTH.toFixed(2)});

        // the near-black cloud deck: a gentle near-neutral gradient spread wide
        // enough that it never saturates inside the visible dome, so there is no
        // hard knee overhead where it would otherwise flatten to a flat pitch.
        vec3 ceiling = mix(uCeilLow, uCeilTop, smoothstep(0.0, ${CEIL_SPREAD.toFixed(2)}, e));

        // cast this dome ray onto the cloud deck at height CLOUD_H and read the same
        // drifting field that lights the ground. vDir.y is floored so rays toward the
        // horizon pierce far out instead of dividing by ~0.
        float t = ${CLOUD_H.toFixed(1)} / max(vDir.y, 0.04);
        vec2 pierce = uCamXZ + vDir.xz * t;
        // sparse, sharp breaks: only the field's brightest peaks open, so the deck
        // stays mostly black with specks of light bursting through (the ground reads
        // the same field with a far wider, softer day/night band, so a break here
        // always sits over already-lit ground). Faded out toward the horizon, where
        // the pierce geometry would smear each break into a vertical streak.
        float openF = smoothstep(${SKY_OPEN_LO.toFixed(2)}, ${SKY_OPEN_HI.toFixed(2)}, skyCover(pierce));
        openF *= smoothstep(${BREAK_RISE_LO.toFixed(2)}, ${BREAK_RISE_HI.toFixed(2)}, vDir.y);

        vec3 col = mix(ceiling, open, openF);
        // a hot silver-warm core in the widest part of a break, like the burst in
        // the reference where the light is brightest at the centre of the opening.
        col += uBreakHot * smoothstep(0.80, 1.0, openF);

        // a dim warm glow hugging the horizon line, just enough to silhouette the far
        // dunes against a sky barely lighter than they are. Tight to the skyline,
        // weak, and a touch brighter toward the sun. This is also what the far land
        // dissolves into below the horizon.
        float strip = 1.0 - smoothstep(0.0, ${HORIZON_BAND.toFixed(2)}, e);
        col = mix(col, uHorizonGlow, strip * ${HORIZON_STRENGTH.toFixed(2)} * (0.55 + 0.45 * toward));

        // the whole dome dims as the player travels into the past, so the light
        // reads as the era you are standing in.
        col = mix(col, col * (1.0 - ${DEPTH_DARKEN.toFixed(2)}), uDepth);

        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
        // dither in display space to break 8-bit banding in the near-black deck,
        // where the quantisation steps are large relative to the colour values. A
        // static per-pixel hash of +/- half a code value, applied after the encode.
        float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
        gl_FragColor.rgb += (dither - 0.5) / 255.0;
      }
    `,
  });

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(RADIUS, 48, 24), mat);
  mesh.frustumCulled = false; // it wraps the camera; never cull it

  function update(depth: number, camX: number, camZ: number): void {
    uDepth.value = depth;
    uCamXZ.value.set(camX, camZ);
  }

  return { mesh, update };
}
