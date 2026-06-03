import * as THREE from "three";

// --- cloud shadows -----------------------------------------------------------
// The scene's one source of large-scale motion: soft shadows of unseen clouds
// drifting across the whole desert, so a static field of books reads as a living
// landscape under weather. There is no cloud geometry and no sky change (the dome
// carries the time axis, not weather); only the shadows on the ground exist, the
// way image 1's meadow sits under a near-empty sky yet is dappled with light. The
// drama the references share is selective light: a lit patch under a dark ceiling
// (Kuindzhi), bright breaks in a storm. A scrolling coverage mask is exactly that.
//
// The mask is a seamless tiling greyscale texture sampled at the world xz of every
// lit surface (ground, books, heads, stones) and scrolled by uCloudTime, so the
// same shadow falls on a book and the sand it stands in. Two layers at different
// scales and speeds break the tile repeat and read as parallax. Seamless tiling
// needs PERIODIC noise (ordinary fBm doesn't wrap), baked once into a DataTexture;
// it can't be faked cheaply per fragment, which is why this is a texture and not
// in-shader fBm. The shadow is a cool darkening multiply, faded back out into the
// distance dissolve so the far field keeps its clean fade into the dome.

// Texture: power-of-two so it mipmaps (the ground samples it at grazing angles
// where the uv derivatives explode; without mips that aliases into a crawling
// moire). BASE_PERIOD is the base-octave cycles across the tile (low = big soft
// blobs); OCT adds finer structure. The noise is periodic over BASE_PERIOD so the
// tile is seamless.
const RES = 256;
const BASE_PERIOD = 4;
const OCT = 4;

// Two sampling layers in world space: a mid layer that carries the read and a
// larger, slower underlay that de-correlates the tiling. World units per tile.
const SCALE1 = 1700;
const SCALE2 = 3900;
const WEIGHT1 = 0.62; // the two layers' mix (sums to 1)
const WEIGHT2 = 0.38;

// Wind: drift aligned with the dune wind (terrain WIND_ANGLE = 0.7) so the weather
// and the sand agree on a prevailing direction. Speeds in world units/sec; the
// underlay drifts slower for parallax.
const WIND_ANGLE = 0.7;
const WIND1 = 13;
const WIND2 = 7;

// Coverage shaping on the sampled mask (a weighted sum of two [0,1] layers, so
// centred near 0.5): smoothstep(LO, HI) is the fraction in sun; below LO is full
// shadow. Tighten LO..HI for harder cloud edges, widen for hazier ones.
const COVER_LO = 0.46;
const COVER_HI = 0.62;

// Full-shadow multiplier, LINEAR (gl_FragColor is linear before the colorspace
// encode), so authored directly rather than through sRGB: under a cloud the ground
// drops to ~45% brightness and shifts cool, the way the references' shadowed ground
// goes blue-grey rather than just dark. Eyeball knob.
const TINT: [number, number, number] = [0.42, 0.46, 0.56];

export interface CloudUniforms {
  uClouds: { value: THREE.Texture };
  uCloudTime: { value: number };
}

// --- seamless periodic noise (baked once) -----------------------------------
// Periodic Perlin: gradients are hashed from lattice coords WRAPPED to the period,
// so noise(x) == noise(x + period) and the baked tile has no seam. fBm scales each
// octave's period with its frequency, so every octave tiles over the same texture.
const GRAD2 = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [-1, 1],
  [1, -1],
  [-1, -1],
];

function smoother(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function pgrad(ix: number, iy: number, per: number): number[] {
  const wx = ((ix % per) + per) % per;
  const wy = ((iy % per) + per) % per;
  let h = (Math.imul(wx, 374761393) + Math.imul(wy, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return GRAD2[(h >>> 0) & 7];
}

function pnoise(x: number, y: number, per: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;
  const u = smoother(xf);
  const v = smoother(yf);
  const g00 = pgrad(xi, yi, per);
  const g10 = pgrad(xi + 1, yi, per);
  const g01 = pgrad(xi, yi + 1, per);
  const g11 = pgrad(xi + 1, yi + 1, per);
  const n00 = g00[0] * xf + g00[1] * yf;
  const n10 = g10[0] * (xf - 1) + g10[1] * yf;
  const n01 = g01[0] * xf + g01[1] * (yf - 1);
  const n11 = g11[0] * (xf - 1) + g11[1] * (yf - 1);
  const nx0 = n00 + u * (n10 - n00);
  const nx1 = n01 + u * (n11 - n01);
  return nx0 + v * (nx1 - nx0); // ~[-1, 1]
}

function fbm(x: number, y: number): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < OCT; o++) {
    sum += amp * pnoise(x * freq, y * freq, BASE_PERIOD * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

function makeCloudTexture(): THREE.DataTexture {
  const data = new Uint8Array(RES * RES);
  for (let y = 0; y < RES; y++) {
    for (let x = 0; x < RES; x++) {
      // texel -> lattice coord spanning exactly BASE_PERIOD, so the tile wraps.
      const n = fbm((x / RES) * BASE_PERIOD, (y / RES) * BASE_PERIOD);
      const v = n * 0.5 + 0.5; // [-1,1] -> [0,1]
      data[y * RES + x] = Math.max(0, Math.min(255, Math.round(v * 255)));
    }
  }
  const tex = new THREE.DataTexture(data, RES, RES, THREE.RedFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.NoColorSpace; // a data mask, not colour; sample raw
  tex.needsUpdate = true;
  return tex;
}

// --- shader injection --------------------------------------------------------
// Wind as uv-space velocity per layer (direction * speed / scale), baked as GLSL
// literals so the shadow needs only the two shared uniforms (texture + time).
const _dir = [Math.cos(WIND_ANGLE), Math.sin(WIND_ANGLE)];
const _v1 = [(_dir[0] * WIND1) / SCALE1, (_dir[1] * WIND1) / SCALE1];
const _v2 = [(_dir[0] * WIND2) / SCALE2, (_dir[1] * WIND2) / SCALE2];
const TINT_GLSL = `vec3(${TINT[0].toFixed(3)}, ${TINT[1].toFixed(3)}, ${TINT[2].toFixed(3)})`;

// Uniform declarations + the cloudShadow() sampler. Splice into a fragment
// shader's <common>. cloudShadow(worldXZ) returns 1 in full sun, 0 under cloud.
export const CLOUD_FRAG_COMMON = /* glsl */ `
  uniform sampler2D uClouds;
  uniform float uCloudTime;
  float cloudShadow(vec2 wxz) {
    vec2 uv1 = wxz * ${(1 / SCALE1).toFixed(7)} + vec2(${_v1[0].toFixed(7)}, ${_v1[1].toFixed(7)}) * uCloudTime;
    vec2 uv2 = wxz * ${(1 / SCALE2).toFixed(7)} + vec2(${_v2[0].toFixed(7)}, ${_v2[1].toFixed(7)}) * uCloudTime;
    float n = texture2D(uClouds, uv1).r * ${WEIGHT1.toFixed(2)} + texture2D(uClouds, uv2).r * ${WEIGHT2.toFixed(2)};
    return smoothstep(${COVER_LO.toFixed(2)}, ${COVER_HI.toFixed(2)}, n);
  }
`;

// The apply block: darken+cool by the shadow at the given world xz, faded back to
// unshadowed by fadeExpr (the distance dissolve, so the far field stays clean; pass
// "0.0" where there is no fade, e.g. opaque props). Splice after <opaque_fragment>.
export function cloudApplyGLSL(xzExpr: string, fadeExpr: string): string {
  return /* glsl */ `
    {
      float _lit = cloudShadow(${xzExpr});
      _lit = mix(1.0, _lit, 1.0 - (${fadeExpr}));
      gl_FragColor.rgb *= mix(${TINT_GLSL}, vec3(1.0), _lit);
    }`;
}

// Patch a plain material (no other world-xz varying) to take cloud shadow: the
// opaque props, heads (instanced) and teleporter stones (model). Chains onto any
// existing onBeforeCompile. Books and the ground inject inline instead, reusing the
// world-xz varying their glow/relief patches already carry. instanced books/heads
// fold instanceMatrix into the world position; stones use modelMatrix alone. The
// two forms share onBeforeCompile's source text, so they need distinct cache keys
// or three would hand both whichever program compiled first (same defence the book
// and ground materials use).
export function applyCloudShadow(
  mat: THREE.Material,
  cloud: CloudUniforms,
  instanced: boolean,
): void {
  const worldXZ = instanced
    ? "(modelMatrix * instanceMatrix * vec4(transformed, 1.0)).xz"
    : "(modelMatrix * vec4(transformed, 1.0)).xz";
  mat.customProgramCacheKey = () => (instanced ? "cloud:inst" : "cloud:model");
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader, renderer) => {
    if (prev) prev.call(mat, shader, renderer);
    shader.uniforms.uClouds = cloud.uClouds;
    shader.uniforms.uCloudTime = cloud.uCloudTime;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vCloudXZ;",
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>\nvCloudXZ = ${worldXZ};`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vCloudXZ;\n" + CLOUD_FRAG_COMMON,
      )
      .replace(
        "#include <opaque_fragment>",
        "#include <opaque_fragment>\n" + cloudApplyGLSL("vCloudXZ", "0.0"),
      );
  };
}

// Build the cloud system: bake the tiling mask, hand back the shared uniforms and a
// per-frame time advance. uCloudTime wraps at a large value so float precision in
// the scroll never degrades across a long session.
export function buildClouds(): {
  uniforms: CloudUniforms;
  update: (dt: number) => void;
} {
  const uniforms: CloudUniforms = {
    uClouds: { value: makeCloudTexture() },
    uCloudTime: { value: 0 },
  };
  return {
    uniforms,
    update: (dt: number) => {
      let t = uniforms.uCloudTime.value + dt;
      if (t > 1e6) t -= 1e6;
      uniforms.uCloudTime.value = t;
    },
  };
}
