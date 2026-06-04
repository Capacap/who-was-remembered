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

// Texture resolution: power-of-two so it mipmaps (the ground samples the field at grazing
// angles where the uv derivatives explode; without mips that aliases into a crawling moire).
const RES = 256;

// Wind angle: the prevailing drift direction, aligned with the dune wind (terrain
// WIND_ANGLE = 0.7) so the weather and the sand agree on a direction.
const WIND_ANGLE = 0.7;

// The prevailing wind as a unit vector, exported so the sky's cloud decks drift along the
// EXACT same world axis as the ground shadows (one shared source of truth: they cannot
// disagree on direction). The ground field's own per-layer drift (see _groundLayers below)
// is this same vector scaled by speed; the sky multiplies it by its per-deck speeds.
export const WIND_DIR: [number, number] = [Math.cos(WIND_ANGLE), Math.sin(WIND_ANGLE)];

// The night/day multipliers, LINEAR (gl_FragColor is linear before the colorspace
// encode), so authored directly rather than through sRGB. This is no longer a cloud
// filtering sunlight; it is night and day themselves drifting across the world and
// blending where they meet. NIGHT is a deep COLD tone, not pure black: the storm's shadow
// keeps a blue-green dark (the thin overcast still casts a little ambient), so the
// light-to-dark falloff reads as a continuous gradient instead of dropping into a void.
// DAY lifts past neutral into a warm cast; with the gamma response veiling the peaks it now
// rarely clips, so the lit ground glows golden rather than blowing to white. The gap
// between the warm light and the cold dark is the surreal Kuindzhi contrast. Eyeball knobs.
const NIGHT: [number, number, number] = [0.011, 0.016, 0.032];
const DAY: [number, number, number] = [1.25, 1.08, 0.82];

// Ground mip from camera DISTANCE, not the GPU's screen-derivative mip. The
// derivative-driven mip is unstable where the field's uv screen footprint jumps: a
// near-vertical head/stone face has near-zero uv footprint, so the hardware picks the
// finest, highest-frequency mip, and under the hard night/day contrast each mip flip
// is a visible strobe. Camera distance is stable per object, so deriving the mip from
// it removes that. LOD_REF is the distance still read at full detail (mip 0); the mask
// coarsens by one mip per doubling beyond it, up to LOD_MAX. These are deliberately
// gentle: the holes are broad and low-frequency (they barely alias), and coarsening too
// fast averages the sparse holes into their dark mean within a few hundred units, so a
// daylight pool only resolves once you walk on top of it. A far LOD_REF + a low LOD_MAX
// keep the distant pools readable, so the holes overhead have answering light below.
const LOD_REF = 600;
const LOD_MAX = 3;

export interface CloudUniforms {
  // the diffuse fbm day/night field: drives the ground / books / props day-night.
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

// --- ground day/night field --------------------------------------------------
// Drives the ground / books / props day-night (the sky no longer reads it). The field has
// been a Voronoi crack web, then soft rounded holes; now diffuse fbm OCTAVES. The lit
// ground reads as broad soft swaths of daylight under the dark storm (Kuindzhi's lit
// meadow), not stamped circular pools, and it shares the same noise CHARACTER as the sky's
// fbm cloud decks overhead. The dome can't register positionally to a world-XZ field, but
// the shared octave texture + the shared uCloudTime drift make the storm above and the lit
// breaks below read as one weather system.
const FIELD_PERIOD = 4; // base-octave cycles across the tile (low = large soft swaths)
const FIELD_OCT = 4; // octaves layered onto the base swaths (matches the sky fbm's depth)
const CRACK_SCALE = 3200; // world units per tile for the base layer (smaller = smaller islands)
const CRACK_WIND = 22; // base field drift speed, world units/sec (its own slow creep)

// Ground islands are a SUM of several samples of the one field texture at different scales
// and drift speeds, mirroring the sky's stacked cloud decks: a dominant base layer carries
// the read while finer, faster layers fragment its big islands into smaller, more numerous
// ones and make the whole field evolve (islands morph and split) rather than rigidly slide.
// Each row: [scale x CRACK_SCALE (smaller = finer islands), speed x CRACK_WIND, weight].
// Weights sum to 1 so the summed field stays centred ~0.5 for the day band below. The finer
// layers get an automatic extra mip (their smaller features would otherwise alias far off).
const GROUND_LAYERS: [number, number, number][] = [
  [1.0, 1.0, 0.55], // base: the large islands, slow
  [0.52, 1.5, 0.3], // mid: fragments the base, half the size, faster
  [0.3, 2.2, 0.15], // fine: small satellite islands, fastest
];

// Domain warp applied to the field lookup (crackWarp). fbm is already organic, but a gentle
// warp breaks the tile's grid alignment and gives the swaths turbulent, wind-sheared
// edges. WARP_SCALE is world xz -> warp-fbm coords (smaller = longer, smoother warp);
// WARP_AMT the nudge in world units.
const WARP_SCALE = 0.0015;
const WARP_AMT = 400;

// The ground's day response on the summed field [0,1] (centred ~0.5). NOT a smoothstep
// band: that snapped from black to full day over a short range and flattened at both ends,
// giving hard-edged bright blobs on a flat-black plateau. Instead a WIDE linear ramp from
// LO (night) to HI (full day) shaped by GAMMA, so the light eases out of the dark over a
// long gradient (Kuindzhi's light-to-dark falloff) instead of a defined edge. GAMMA > 1
// feathers the low end (light creeps in slowly) and holds the midtones dim, so the world
// stays mostly dark; because the summed field rarely reaches HI, the brightest islands land
// BELOW full day and read as light filtered through thin cloud, not a clean opening. Eyeball
// knobs: lower LO/HI to light more, widen the gap or raise GAMMA for a softer, more diffuse
// falloff.
const GROUND_OPEN_LO = 0.5;
const GROUND_OPEN_HI = 0.76;
const GROUND_GAMMA = 2.1;
// Extra mip bias on the ground sample, on top of the distance LOD: softens the swath edges
// a touch and lets the field settle toward its mid (dusk) mean far off rather than
// flickering. fbm has energy everywhere, so unlike the sparse holes it never averages to a
// dark void with distance; the far ground just eases to an even half-lit dusk.
const GROUND_LOD_BIAS = 1.0;

// Periodic fbm for the ground field, wrapping over FIELD_PERIOD so the baked tile is
// seamless. Reuses the periodic Perlin lattice (pnoise/pgrad) the cloud mask uses, so the
// ground field and the original mask share one noise basis. Returns ~[-1, 1].
function fieldFbm(x: number, y: number): number {
  let amp = 1;
  let freq = 1;
  let sum = 0;
  let norm = 0;
  for (let o = 0; o < FIELD_OCT; o++) {
    sum += amp * pnoise(x * freq, y * freq, FIELD_PERIOD * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2;
  }
  return sum / norm;
}

function makeCrackTexture(): THREE.DataTexture {
  const data = new Uint8Array(RES * RES);
  for (let y = 0; y < RES; y++) {
    for (let x = 0; x < RES; x++) {
      // texel -> lattice coord spanning exactly FIELD_PERIOD, so the tile wraps.
      const n = fieldFbm((x / RES) * FIELD_PERIOD, (y / RES) * FIELD_PERIOD);
      const v = n * 0.5 + 0.5; // [-1,1] -> [0,1]
      data[y * RES + x] = Math.max(0, Math.min(255, Math.round(v * 255)));
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
const _dir = WIND_DIR;

// Per ground layer: the uv scale (1 / world-units-per-tile), the drift velocity in uv/sec
// (world wind * speed mul, converted to uv by the layer's own scale), the blend weight,
// and a mip bias so a finer layer is sampled one-plus mips coarser (it would otherwise
// alias at distance). Baked to GLSL literals in cloudShadow so the sum needs no per-frame
// work beyond the texture taps.
const _groundLayers = GROUND_LAYERS.map(([sm, vm, w]) => {
  const inv = 1 / (CRACK_SCALE * sm);
  return {
    inv,
    dx: _dir[0] * CRACK_WIND * vm * inv,
    dy: _dir[1] * CRACK_WIND * vm * inv,
    w,
    lodAdj: -Math.log2(sm),
  };
});

// Shared field-reader helpers: declares the field sampler + clock, a small value-noise fbm
// for the domain warp (uniquely named so it never clashes with a host shader's own noise),
// and crackWarp() which warps world xz before the per-layer scale + drift is applied.
// Spliced into the grounded materials' fragment <common>.
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
  // warp world xz once (shared by every ground layer, so they all bend together); the
  // per-layer scale + drift are applied after this.
  vec2 crackWarp(vec2 wxz){
    vec2 w = vec2(_cwFbm(wxz * ${WARP_SCALE.toFixed(5)} + 11.3), _cwFbm(wxz * ${WARP_SCALE.toFixed(5)} + 41.7)) - 0.5;
    return wxz + w * ${WARP_AMT.toFixed(1)};
  }
`;
const NIGHT_GLSL = `vec3(${NIGHT[0].toFixed(3)}, ${NIGHT[1].toFixed(3)}, ${NIGHT[2].toFixed(3)})`;
// exported so the books can cast their sun-reveal in the EXACT daylight colour the
// ground tints to (see main.ts applyProximityGlow), locking the two to one palette.
export const DAY_GLSL = `vec3(${DAY[0].toFixed(3)}, ${DAY[1].toFixed(3)}, ${DAY[2].toFixed(3)})`;

// Uniform declarations + the cloudShadow() sampler. Splice into a fragment shader's
// <common>. cloudShadow(worldXZ, camDist) returns 1 in full day, 0 in full night. The
// day/night driver is the diffuse fbm field, so the grounded world reads as broad soft
// swaths of daylight drifting under the dark storm, echoing the fbm clouds overhead. The
// mip is taken explicitly from camDist (see the LOD_REF note) via texture2DLodEXT (three's
// WebGL2-safe alias for textureLod) so the swaths never flip on a face and never alias at
// grazing distance; far off the field eases to its mid mean and the ground settles to an
// even dusk against the storm dome.
export const CLOUD_FRAG_COMMON = /* glsl */ `
  ${_crackHelpers}
  uniform float uCloudMix;
  float cloudShadow(vec2 wxz, float camDist) {
    float baseLod = clamp(log2(max(camDist, 1.0) / ${LOD_REF.toFixed(1)}) + ${GROUND_LOD_BIAS.toFixed(2)}, 0.0, ${LOD_MAX.toFixed(1)});
    vec2 ww = crackWarp(wxz);
    float c = 0.0;
    ${_groundLayers
      .map(
        (L) =>
          `c += texture2DLodEXT(uCrack, ww * ${L.inv.toFixed(8)} + vec2(${L.dx.toFixed(8)}, ${L.dy.toFixed(8)}) * uCloudTime, clamp(baseLod + ${L.lodAdj.toFixed(3)}, 0.0, ${LOD_MAX.toFixed(1)})).r * ${L.w.toFixed(3)};`,
      )
      .join("\n    ")}
    float t = clamp((c - ${GROUND_OPEN_LO.toFixed(2)}) / ${(GROUND_OPEN_HI - GROUND_OPEN_LO).toFixed(2)}, 0.0, 1.0);
    return pow(t, ${GROUND_GAMMA.toFixed(2)});
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

// Build the cloud system: bake the field texture, hand back the shared uniforms and a
// per-frame time advance. uCloudTime wraps at a large value so float precision in
// the scroll never degrades across a long session.
export function buildClouds(): {
  uniforms: CloudUniforms;
  update: (dt: number) => void;
} {
  const uniforms: CloudUniforms = {
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
