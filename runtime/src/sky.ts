import * as THREE from "three";

// --- sky --------------------------------------------------------------------
// A gradient dome standing in for the open sky. Without it the background is a
// single flat colour, which reads as a wall and gives away the world's finite
// edge. The dome paints two zones split at a horizon line: a ground-haze tone
// below (the far desert the transparent-faded land dissolves into) and the open
// sky above, easing from a pale skyline to a dusty-blue zenith. The ground tone
// is never mixed into the sky, so it cannot brown it.
//
// The land has no opaque edge (it fades to transparent into this dome), so the
// dome itself supplies the horizon the geometry no longer does. It is a single
// inward-facing sphere recentred on the camera each frame, so it sits at a fixed
// apparent distance and never clips. Fog does not apply.

// The sky is the dome's own aesthetic; the ground tone is passed in (= the
// colour the land fades into) so land and sky agree at the line.
const SKY_LOW = new THREE.Color(0xb4bcc0); // pale sky just above the horizon
const ZENITH = new THREE.Color(0x8ea2b4); // dusty blue overhead

// The horizon line sits below eye level on the dome (negative sin-elevation),
// so when you look level you see more sky than ground and the skyline reads as
// dropping away below you rather than cutting the view in half. SKY_TOP is how
// far above that line the sky takes to reach the zenith.
const HORIZON_Y = -0.12; // ~7deg below eye level
const SKY_TOP = 0.7;

// Large enough that the farthest content (rim seen from the opposite rim,
// ~14000u) stays inside it, and inside the camera's far plane (24000u).
const RADIUS = 20000;

export function buildSky(ground: THREE.Color): THREE.Mesh {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false, // pure backdrop; never occludes the world
    fog: false,
    uniforms: {
      // converted to linear working space so that, after the colorspace encode
      // in the fragment shader, the colours render exactly as authored. A raw
      // ShaderMaterial gets none of three's automatic colour management.
      uGround: { value: ground.clone().convertSRGBToLinear() },
      uSkyLow: { value: SKY_LOW.clone().convertSRGBToLinear() },
      uZenith: { value: ZENITH.clone().convertSRGBToLinear() },
      uHorizonY: { value: HORIZON_Y },
      uSkyTop: { value: SKY_TOP },
    },
    vertexShader: /* glsl */ `
      varying vec3 vDir;
      void main() {
        // local position is the direction from the dome centre (= the camera),
        // so its normalized y is the sine of the elevation angle.
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vDir;
      uniform vec3 uGround;
      uniform vec3 uSkyLow;
      uniform vec3 uZenith;
      uniform float uHorizonY;
      uniform float uSkyTop;
      void main() {
        // elevation measured from the dome's horizon line (which sits below eye
        // level): e<0 is ground, e>0 is sky.
        float e = vDir.y - uHorizonY;
        // open sky ABOVE the horizon: pale near the skyline up to a dusty-blue
        // zenith. The ground tone is never mixed in here, so it cannot brown the
        // sky the way it did when the base ran ground -> zenith through the sky.
        vec3 sky = mix(uSkyLow, uZenith, pow(smoothstep(0.0, uSkyTop, e), 1.3));
        // cross the horizon within a couple of degrees: ground tone strictly
        // below, sky strictly above, so the brown stays on the ground side.
        vec3 col = mix(uGround, sky, smoothstep(-0.03, 0.03, e));
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });

  const sky = new THREE.Mesh(new THREE.SphereGeometry(RADIUS, 48, 24), mat);
  sky.frustumCulled = false; // it wraps the camera; never cull it
  return sky;
}
