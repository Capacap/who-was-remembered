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
// lit surface (ground, books) and scrolled by uCloudTime, so the
// same shadow falls on a book and the sand it stands in. Two layers at different
// scales and speeds break the tile repeat and read as parallax. Seamless tiling
// needs PERIODIC noise (ordinary fBm doesn't wrap), baked once into a DataTexture;
// it can't be faked cheaply per fragment, which is why this is a texture and not
// in-shader fBm. The mask no longer reads as a shadow but as a day/night terminator:
// full night crushes the surface to a near-black cold blue, full day lifts it bright
// and warm, and the two blend across the mask's soft edge. It is faded back to
// neutral into the distance dissolve so the far field keeps its clean fade into the
// dome.

// Texture: power-of-two so it mipmaps (the ground samples it at grazing angles
// where the uv derivatives explode; without mips that aliases into a crawling
// moire). BASE_PERIOD is the base-octave cycles across the tile (low = big soft
// blobs); OCT adds finer structure. The noise is periodic over BASE_PERIOD so the
// tile is seamless.
const RES = 256;
const BASE_PERIOD = 6;
const OCT = 3;

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
const WIND1 = 25;
const WIND2 = 7;

// Coverage shaping on the sampled mask (a weighted sum of two [0,1] layers, so
// centred near 0.5): smoothstep(LO, HI) is the day fraction; below LO is full night.
// The LO..HI band is the terminator: widen it and night and day blend over a longer
// gradient, tighten it for a starker divide.
// exported so the sky dome can read the field through the EXACT same day/night band
// the ground uses, locking the dome's day fraction to the lit patches below.
export const COVER_LO = 0.50;
export const COVER_HI = 1.0;

// The night/day multipliers, LINEAR (gl_FragColor is linear before the colorspace
// encode), so authored directly rather than through sRGB. This is no longer a cloud
// filtering sunlight; it is night and day themselves drifting across the world and
// blending where they meet. NIGHT crushes the surface to a near-black cold blue; DAY
// lifts it past neutral into a bright warm cast (channels >1 intentionally clip hot
// in the working space). The wide gap between them is the surreal contrast. Eyeball
// knobs.
const NIGHT: [number, number, number] = [0.005, 0.007, 0.016];
const DAY: [number, number, number] = [1.25, 1.08, 0.82];

// Props (heads, stones) carry a fixed sandstone albedo everywhere, but the GROUND
// ramps its OWN albedo from pale at the present to a dark grey in the deep past
// (terrain.groundColor: pale->sand at half-radius, sand->grey at the rim). With the
// props held flat, a drifting DAY patch out in the deep past lifts a bright sandstone
// monument far above the grey ground around it, so it reads as a glowing aura that
// pulses on the cloud's cycle while the ground stays dark. Ramping the prop's
// brightness toward the same deep-past floor by radius removes the differential: a far
// monument darkens with the sand it stands in and only pulses as much as the ground.
// ERA_R mirrors terrain.ERA_GRADIENT_R (= world.json R_MAX); the ramp starts at the
// half-radius (where the ground's own ramp turns sand->grey, so inner props, already
// sand-coloured, are untouched). ERA_FLOOR is grey/sand lightness (~0.55). Eyeball knobs.
const ERA_R = 7100;
const ERA_FLOOR = 0.55;

// Cloud mip from camera DISTANCE, not the GPU's screen-derivative mip. The
// derivative-driven mip is unstable where the cloud uv's screen footprint jumps: a
// near-vertical head/stone face has near-zero uv footprint, so the hardware picks the
// finest, highest-frequency mip, and under the hard night/day contrast each mip flip
// is a visible strobe. Camera distance is stable per object, so deriving the mip from
// it removes that. LOD_REF is the distance still read at full detail (mip 0,
// underfoot); the mask coarsens by one mip per doubling beyond it, up to LOD_MAX at
// the dissolve horizon. Eyeball knobs.
const LOD_REF = 120;
const LOD_MAX = 6;

export interface CloudUniforms {
  uClouds: { value: THREE.Texture };
  // the Voronoi crack web (shared): drives the sky openings AND the ground day/night.
  uCrack: { value: THREE.Texture };
  uCloudTime: { value: number };
  // DIAGNOSTIC kill switch: 1 = full night/day tint, 0 = neutral (effect off).
  // Lets a key disable the whole lighting effect at runtime to localise the flicker.
  uCloudMix: { value: number };
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

// --- crack field (sky only) --------------------------------------------------
// A separate field for the SKY openings, so they stop reading as clouds: instead of
// the soft fbm blobs the ground uses, this is a periodic Voronoi web. The metric is
// the cell-edge distance (F2 - F1, small where two cells are equidistant), which traces
// a network of thin seams. Baked high on the seams, near zero inside the shards, so the
// sky's threshold picks a fracture network and daylight shows through the cracks. Kept
// out of the ground field on purpose (see the sky-vs-ground decision): the dunes keep
// their soft dapple until we decide to extend the cracks down to them.
const CRACK_CELLS = 3; // Voronoi cells across the tile (low = larger shards)
const CRACK_EDGE = 0.1; // seam half-width in cell units; small = thin hairline cracks
//                          (still smootherstep-ramped, so the edge stays soft/diffuse)
const CRACK_KEEP = 0.55; // fraction of cells that keep a point; the rest merge into big
//                          irregular shards (this is what gives shattered-glass variance)
const CRACK_SCALE = 4800; // world units per crack tile (large = big shards, less repeat)
const CRACK_WIND = 25; // seam drift speed, world units/sec (its own slow creep)

// Domain warp applied to the crack lookup, SHARED by the sky and the ground so both read
// the identical wandering web at the same world XZ (the daylight that pours through a sky
// crack lands on the matching ground seam). WARP_SCALE is world xz -> warp-fbm coords
// (smaller = longer, smoother warp); WARP_AMT is the nudge in world units.
const WARP_SCALE = 0.0015;
const WARP_AMT = 500;

// The ground's open band on the crack field: lit (day) along the seams, night in the
// shards. Much wider/softer than the sky's hairline threshold so a seam is a broad diffuse
// wash, not a strip: the low end is what stretches the lit skirt far out from the seam
// core (the "light scatters further" read), the high end where it reaches full day.
const GROUND_OPEN_LO = 0.25;
const GROUND_OPEN_HI = 0.9;
// Extra mip bias on the ground's crack sample, so the seams blur and spread spatially
// (more diffuse, scattering into the surrounding sand) without touching the sky, which
// keeps its own crisp read. Higher = softer, wider, dimmer ground light.
const GROUND_LOD_BIAS = 1.0;

// per-cell hash, periodic over `per` so the tile wraps seamlessly. Returns the jittered
// point (a, b) in [0,1)^2 and a `keep` roll used to drop ~half the points.
function cellPoint(ix: number, iy: number, per: number): { a: number; b: number; keep: number } {
  const wx = ((ix % per) + per) % per;
  const wy = ((iy % per) + per) % per;
  let h = (Math.imul(wx, 374761393) + Math.imul(wy, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  const a = (h % 4096) / 4096;
  const h2 = Math.imul(h ^ (h >>> 15), 1597334677) >>> 0;
  const b = (h2 % 4096) / 4096;
  const h3 = Math.imul(h2 ^ (h2 >>> 13), 951274213) >>> 0;
  return { a, b, keep: (h3 % 4096) / 4096 };
}

function makeCrackTexture(): THREE.DataTexture {
  const data = new Uint8Array(RES * RES);
  const P = CRACK_CELLS;
  for (let y = 0; y < RES; y++) {
    for (let x = 0; x < RES; x++) {
      const gx = (x / RES) * P;
      const gy = (y / RES) * P;
      const cx = Math.floor(gx);
      const cy = Math.floor(gy);
      let f1 = 1e9;
      let f2 = 1e9; // nearest and second-nearest kept-point distances
      // 5x5 search: with ~half the points dropped, the nearest can be two cells away.
      for (let j = -2; j <= 2; j++) {
        for (let i = -2; i <= 2; i++) {
          const p = cellPoint(cx + i, cy + j, P);
          if (p.keep > CRACK_KEEP) continue; // dropped: this cell merges into its neighbours
          const dx = cx + i + p.a - gx;
          const dy = cy + j + p.b - gy;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < f1) {
            f2 = f1;
            f1 = d;
          } else if (d < f2) {
            f2 = d;
          }
        }
      }
      // edge metric: 0 on a Voronoi boundary, growing inward. Ramp to 1 at the seam,
      // 0 by CRACK_EDGE, then smootherstep it so the seam is a soft diffuse gradient
      // (smooth light when this drives the dunes) rather than a hard line. The sky picks
      // the final width with its own threshold, so it retunes without a rebake.
      let crack = 1 - Math.min(1, (f2 - f1) / CRACK_EDGE);
      crack = crack * crack * crack * (crack * (crack * 6 - 15) + 10);
      data[y * RES + x] = Math.round(Math.max(0, Math.min(1, crack)) * 255);
    }
  }
  const tex = new THREE.DataTexture(data, RES, RES, THREE.RedFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

export { makeCrackTexture };

// --- shader injection --------------------------------------------------------
// Wind as uv-space velocity per layer (direction * speed / scale), baked as GLSL
// literals so the shadow needs only the two shared uniforms (texture + time).
const _dir = [Math.cos(WIND_ANGLE), Math.sin(WIND_ANGLE)];
const _v1 = [(_dir[0] * WIND1) / SCALE1, (_dir[1] * WIND1) / SCALE1];
const _v2 = [(_dir[0] * WIND2) / SCALE2, (_dir[1] * WIND2) / SCALE2];
const _cv = [(_dir[0] * CRACK_WIND) / CRACK_SCALE, (_dir[1] * CRACK_WIND) / CRACK_SCALE];

// Shared crack-field reader: declares the crack sampler + clock, a small value-noise fbm
// for the domain warp (uniquely named so it never clashes with a host shader's own
// noise), and crackUV() which warps in world space then scales + drifts into the tile.
// Spliced into BOTH the sky day layer and the grounded materials, so they sample one web.
const _crackHelpers = /* glsl */ `
  uniform sampler2D uCrack;
  uniform float uCloudTime;
  float _cwH(vec2 p){ p = fract(p * vec2(123.34, 345.45)); p += dot(p, p + 34.345); return fract(p.x * p.y); }
  float _cwN(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = _cwH(i), b = _cwH(i + vec2(1,0)), c = _cwH(i + vec2(0,1)), d = _cwH(i + vec2(1,1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }
  float _cwFbm(vec2 p){ float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++){ v += a * _cwN(p); p *= 2.03; a *= 0.5; } return v; }
  vec2 crackUV(vec2 wxz){
    vec2 w = vec2(_cwFbm(wxz * ${WARP_SCALE.toFixed(5)} + 11.3), _cwFbm(wxz * ${WARP_SCALE.toFixed(5)} + 41.7)) - 0.5;
    vec2 wx = wxz + w * ${WARP_AMT.toFixed(1)};
    return wx * ${(1 / CRACK_SCALE).toFixed(7)} + vec2(${_cv[0].toFixed(7)}, ${_cv[1].toFixed(7)}) * uCloudTime;
  }
`;
const NIGHT_GLSL = `vec3(${NIGHT[0].toFixed(3)}, ${NIGHT[1].toFixed(3)}, ${NIGHT[2].toFixed(3)})`;
// exported so the books can cast their sun-reveal in the EXACT daylight colour the
// ground tints to (see main.ts applyProximityGlow), locking the two to one palette.
export const DAY_GLSL = `vec3(${DAY[0].toFixed(3)}, ${DAY[1].toFixed(3)}, ${DAY[2].toFixed(3)})`;

// Uniform declarations + the cloudShadow() sampler. Splice into a fragment shader's
// <common>. cloudShadow(worldXZ, camDist) returns 1 in full day, 0 in full night. The
// day/night driver is now the CRACK web (the same one the sky opens through), so the
// grounded world is mostly night with thin daylight seams drifting across it, matching
// the sky overhead. The mip is taken explicitly from camDist (see the LOD_REF note) via
// texture2DLodEXT (three's WebGL2-safe alias for textureLod) so the thin seams never flip
// on a seam or a face and never alias at grazing distance; far off they mip to their dark
// average and the ground sinks to night to meet the storm dome.
export const CLOUD_FRAG_COMMON = /* glsl */ `
  ${_crackHelpers}
  uniform float uCloudMix;
  float cloudShadow(vec2 wxz, float camDist) {
    float lod = clamp(log2(max(camDist, 1.0) / ${LOD_REF.toFixed(1)}) + ${GROUND_LOD_BIAS.toFixed(1)}, 0.0, ${LOD_MAX.toFixed(1)});
    float c = texture2DLodEXT(uCrack, crackUV(wxz), lod).r;
    return smoothstep(${GROUND_OPEN_LO.toFixed(2)}, ${GROUND_OPEN_HI.toFixed(2)}, c);
  }
`;

// The sky-side reader of the SAME drifting field. Declares the shared uniforms and
// skyCover(worldXZ), the raw weighted coverage [0,1] (NOT run through the ground's
// day/night smoothstep, so the sky can pick its own, much narrower open threshold).
// Splice into the sky dome's fragment <common>; the dome passes shared uClouds and
// uCloudTime so its breaks drift in lockstep with the ground's lit patches. It uses
// auto-mip texture2D rather than the explicit-LOD fetch the ground needs: the dome is
// a smooth surface with no grazing-angle uv blow-up to dodge, and toward the horizon
// the pierce point races outward, where auto-mip coarsening is exactly what keeps the
// far breaks from aliasing. The uv math reuses the same JS-computed scale and wind
// literals as cloudShadow, so the two fields cannot drift apart numerically.
export const SKY_CLOUD_COMMON = /* glsl */ `
  uniform sampler2D uClouds;
  uniform float uCloudTime;
  float skyCover(vec2 wxz) {
    vec2 uv1 = wxz * ${(1 / SCALE1).toFixed(7)} + vec2(${_v1[0].toFixed(7)}, ${_v1[1].toFixed(7)}) * uCloudTime;
    vec2 uv2 = wxz * ${(1 / SCALE2).toFixed(7)} + vec2(${_v2[0].toFixed(7)}, ${_v2[1].toFixed(7)}) * uCloudTime;
    return texture2D(uClouds, uv1).r * ${WEIGHT1.toFixed(2)} + texture2D(uClouds, uv2).r * ${WEIGHT2.toFixed(2)};
  }
`;

// The sky-side reader of the CRACK field: the SAME warped web the ground reads (via the
// shared _crackHelpers / crackUV), so a sky crack and the daylight seam on the ground
// below share one shape and drift. crackField(worldXZ) is the raw seam intensity [0,1]
// (high on a crack), shaped into an opening by the sky's own threshold. Uses auto-mip
// texture2D (not the ground's explicit-LOD fetch): the day plane is smooth and toward the
// horizon the auto-mip coarsening is exactly what the fwidth AA wants. Splice into the
// day layer's fragment <common>.
export const SKY_CRACK_COMMON = /* glsl */ `
  ${_crackHelpers}
  float crackField(vec2 wxz) {
    return texture2D(uCrack, crackUV(wxz)).r;
  }
`;

// The apply block: blend the night/day multiplier by the mask at the given world xz,
// faded toward the NIGHT floor by fadeExpr as the surface recedes into the distance
// dissolve, so the far field sinks into darkness to meet the near-black storm dome at
// the horizon (it used to fade to neutral, which left the far dunes at full albedo,
// glowing bright against the black sky). Fading to a CONSTANT keeps the far field
// free of the cloud dappling's high-frequency flicker either way. Pass "0.0" where
// there is no fade, e.g. opaque props. camDistExpr is the fragment's view-space
// distance, feeding the explicit cloud mip. Splice after <opaque_fragment>.
export function cloudApplyGLSL(
  xzExpr: string,
  fadeExpr: string,
  camDistExpr: string,
): string {
  return /* glsl */ `
    {
      float _lit = cloudShadow(${xzExpr}, ${camDistExpr});
      vec3 _tint = mix(${NIGHT_GLSL}, ${DAY_GLSL}, _lit);
      _tint = mix(_tint, ${NIGHT_GLSL}, ${fadeExpr});
      _tint = mix(vec3(1.0), _tint, uCloudMix); // diagnostic kill switch
      gl_FragColor.rgb *= _tint;
    }`;
}

// Darken a prop toward the deep-past floor by its radius, mirroring the ground's
// own pale->sand->grey era ramp (see the ERA_* note). Ramps in only over the outer
// half-radius, where the ground turns sand->grey; inner props (already sand-coloured)
// keep full brightness. Splice before the cloud apply so the day/night tint multiplies
// the era-correct base, exactly as it does on the ground (which ramps its albedo first).
export function eraDarkenGLSL(xzExpr: string): string {
  return /* glsl */ `
    {
      float _eraT = clamp((length(${xzExpr}) / ${ERA_R.toFixed(1)} - 0.5) * 2.0, 0.0, 1.0);
      gl_FragColor.rgb *= mix(1.0, ${ERA_FLOOR.toFixed(2)}, _eraT);
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
    shader.uniforms.uCrack = cloud.uCrack;
    shader.uniforms.uCloudTime = cloud.uCloudTime;
    shader.uniforms.uCloudMix = cloud.uCloudMix;
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vCloudXZ;\nvarying float vCloudDist;",
      )
      .replace(
        "#include <project_vertex>",
        `#include <project_vertex>\nvCloudXZ = ${worldXZ};\nvCloudDist = length(mvPosition.xyz);`,
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying vec2 vCloudXZ;\nvarying float vCloudDist;\n" +
          CLOUD_FRAG_COMMON,
      )
      .replace(
        "#include <opaque_fragment>",
        "#include <opaque_fragment>\n" +
          eraDarkenGLSL("vCloudXZ") +
          cloudApplyGLSL("vCloudXZ", "0.0", "vCloudDist"),
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
    uCrack: { value: makeCrackTexture() },
    uCloudTime: { value: 0 },
    uCloudMix: { value: 1 },
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
