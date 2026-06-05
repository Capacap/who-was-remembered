"""
Stage 9: terrain bake (heightmap). The first step of moving terrain out of the
renderer and into the pipeline.

This is a decoration stage, not a data-truth stage: the vertical axis carries no
data (time and longitude are the horizontal x/y), so nothing here feeds back into
the placement truth. What it produces is a single baked elevation field that will
become the world's one source of ground height. The ground mesh gets decimated
from it, and book/teleporter heights get sampled from it, so everything that sits
on the ground agrees with the surface the eye actually sees. That last point is
the whole reason to bake: a low-poly mesh deviates from any continuous height
function on purpose, so a book seated on the function floats above the chord
mesh; seating it on the baked artifact instead removes the divergence.

For now this only bakes the heightmap and renders it for inspection. Mesh
decimation and book-height seating build on this artifact in later steps.

The world is a spiral. The relief echoes the radial time axis: a central vantage
crater (hill: a hill to spawn on, ringed by a shallow moat) whose bookless basin
is broken up by gentle low-amplitude ripples (center_texture) right out to where
the books start; fine dune TEXTURE that radiates from the
centre out past the rim (dune_spiral: ridged noise sampled in a spiral polar
frame, so the ridges fork and merge organically while running outward); and a few
broad cos spiral arms (spiral_swell) that rise into a central massif of peaks, dig
gaps between them to break the crater rim, and fade to flat dune desert toward the
bounds. The big dunes stay out past the plateau, so the basin reads clean. The
dunes are then run through a thermal avalanche (a sand-slide that moves material
between over-steep neighbouring cells), a neighbour op on the raster that rounds
the sharp crests and fills the toes -- exactly what a pointwise get_height
function can't do and a reason terrain lives in the pipeline. Level plazas are
carved at the teleporter monuments. The runtime catches up by loading the baked
mesh, not by mirroring this code.

Writes cache/heightmap.npz (the raw field + its world mapping) and a shaded-relief
inspection render to cache/plots/heightmap.png.

Run (after the teleporters exist; Stage 7):
    uv run pipeline/stage9_heightmap.py
    uv run pipeline/stage9_heightmap.py --res 4096 --vert-exag 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter

from stage6_place import R_INNER, R_MAX, RADIUS_ALPHA, TIME_SPAN

ROOT = Path(__file__).resolve().parent
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
BOOKS_PATH = ROOT / "cache" / "layout.parquet"
OUT_NPZ = ROOT / "cache" / "heightmap.npz"
OUT_PNG = ROOT / "cache" / "plots" / "heightmap.png"
OUT_DENSITY_PNG = ROOT / "cache" / "plots" / "density.png"

# World extent. The runtime ground mesh is an 18000u square centred on the
# origin (runtime/src/terrain.ts SIZE), so the heightmap covers the same square.
# TODO: once both sides read it from world.json this constant goes away.
WORLD_SIZE = 18000.0

# --- terrain constants ------------------------------------------------------
PEAK_HEIGHT = 60.0  # a whisper of a central rise, not a summit
PLATEAU_R = 700.0  # calm and level here (spawn + plaza + the year-2000 ring)
BASE_R = 5200.0  # the rise has eased to the desert floor (0) by here

# --- central vantage (the spawn crater) -------------------------------------
# The inner disc (r < PLATEAU_R) is otherwise featureless: no dunes, no swells,
# just the flat top of the broad rise. Instead of a barren plateau, dish it into a
# shallow CRATER: the present is a sink, deepest at the centre and easing back to
# the plateau level by PLATEAU_R, so the dune/swell massif beyond reads as the
# crater's outer rim. (An earlier version put a central HILL to spawn on here; it
# wasn't doing its job, so the concavity is flipped -- the present is a hollow the
# accumulated past piles up around, not a summit.) Depth trades against the
# survey-the-ring vantage: deeper sinks the eye below the rim, so keep it shallow.
CRATER_DEPTH = 18.0  # crater depth below the plateau level at the centre (0 = flat)
CRATER_FALLOFF_R = 1400.0  # crater eases back to the plateau level by here. Wider than the
# old PLATEAU_R (700): the broad gentle bowl drops the mid-bowl wall (the lip that occludes
# the view) faster than it drops the books, so the first masses of books just past R_INNER
# clear the lip from the sunken spawn instead of hiding behind it. The deep dunes past here
# are untouched; inside it the crater gently dishes the already-calm inner swell zone.

FBM_OCTAVES = 3  # octaves of the ridged-noise fBm the dunes are built from
NOISE_INNER = PLATEAU_R  # the swell/massif starts past the calm present plateau
NOISE_FULL = 1400.0  # ... at full height by here

# The big spiral dunes are far too tall (SP_AMP) for the basin: from eye height a
# ring of them at a few hundred units becomes a wall that swallows the book circle
# and the horizon. So the centre gets its own, much smaller texture instead: gentle
# low-amplitude ripples across the whole bookless basin (full within
# CENTER_TEX_FULL_R, gone by CENTER_TEX_FADE_R ~ R_INNER), so the empty centre is
# broken up right out to where the books start, with no smooth flatland ring in
# between. It stops at the books so it never competes with them.
# Disabled: isotropic ripples read as static against the radial spiral, and at the
# 70u mesh resolution any wavelength fine enough to feel like texture is below
# Nyquist anyway. The centre is being rethought as a calm vortex eye with the real
# spiral dunes winding in; this knob stays at 0 pending that.
CENTER_TEX_AMP = 0.0  # ripple height, world units (0 = off)
CENTER_TEX_SCALE = 110.0  # ripple wavelength, world units
CENTER_TEX_FULL_R = R_INNER - 120.0  # full strength across the bookless basin
CENTER_TEX_FADE_R = R_INNER + 60.0  # ... faded to nothing where the books start

# --- book-density smoothing -------------------------------------------------
# Books make the busy, well-documented eras; their mass should read as a visible
# mass, not hide behind dunes. So calm the dune height wherever books cluster:
# blur the book positions into a density field and use it to attenuate the dune
# amplitude. Dense recent shelves settle onto readable ground; the sparse deep
# past keeps churning at full height -- the thesis, written into the terrain.
SMOOTH_STRENGTH = 0.85  # max dune attenuation under the densest clusters (0 = off)
SMOOTH_BLUR = 220.0  # density blur radius, world units (~the dune scale)
SMOOTH_PCTL = 98.0  # density percentile mapped to full attenuation
SMOOTH_BLUR2 = 400.0  # post-clip blur, world units: feathers the saturated rim, spreads the mask

# --- central vortex calm ----------------------------------------------------
# The dunes spiral all the way in (a true vortex), but a radial calm flattens
# them toward the centre so spawn stays a clean vantage. Full calm across the
# vantage dome, easing out over a long ramp so the dunes rise gently through the
# eye and book ring rather than walling the horizon. Combined with the book
# density by union (max), never summed, so the two never over-flatten.
CENTER_STRENGTH = 1.0  # max calm at the centre (1 = dead flat on the vantage dome)
CENTER_CALM_R0 = 220.0  # full-calm pocket radius (= the vantage dome foot)
CENTER_CALM_R1 = 1400.0  # ... dunes back to full amplitude by here

# --- radial spiral dunes (the radiating texture) ----------------------------
# The dunes read as TEXTURE that radiates from the centre. They are ridged noise
# (ridges on the zero-set of fBm, which fork, merge and pinch off organically),
# but sampled in a spiral polar frame: the angular coordinate is k*theta (k
# ridges around the circle), the radial coordinate is r stretched by SP_ASPECT so
# the dunes run radially, and a twist of SP_TWIST*ln(r) winds the field into a
# spiral.
#
# Constant *arc* spacing across the radial range can't come from one arm count: a
# fixed count spreads the ridges apart outward (barren at the rim where the books
# are), a radius-varying count tears a ring seam where it steps. So we sum octave
# harmonics whose arm counts double (SP_K0, 2*SP_K0, ...), each faded into the
# annulus where its arc spacing matches SP_SPACE; adjacent octaves crossfade as a
# partition of unity, so the dune scale stays constant and the two scales merge
# where they overlap. The seam is closed by sampling each octave with a PERIODIC
# Perlin whose angular axis wraps at exactly that octave's (integer) arm count.
SP_AMP = 70.0  # radial dune crest height, world units (0 = off)
SP_SPACE = 300.0  # target arc spacing between ridges (world u), held across radius
SP_ASPECT = 4.5  # radial:angular cell ratio (>1 = dunes elongated radially, like real ridges)
SP_TWIST = 0.6  # spiral winding: radians the whole field rotates per ln(r); 0 = radial spokes
SP_R0 = PLATEAU_R  # winding anchor: zero rotation at this radius
SP_K0 = 6  # innermost arm count (integer)
SP_OCTAVES = 6  # doubling harmonics (6,12,24,48,96,192 for K0=6)

# --- spiral swells (large-scale volume: balanced cos arms) -------------------
# The fine dunes carry texture but the big picture stays flat, and the central
# rim reads as a closed crater wall. The swells fix both: a few broad arms that
# rise into peaks and dig BELOW between them (signed), breaking the rim into
# SWELL_ARMS peaks with gaps, the way the original cos arms did. They are cos, not
# the dune's ridged noise, on purpose: at three arms the noise comes out lopsided
# (one peak dominant), while cos gives the balanced N-fold the rim needs. They
# stay coherent with the texture by winding on the SAME SP_TWIST (so the arm
# pitch follows the dune spiral), and a light world-space phase wobble keeps them
# organic rather than ruled.
SWELL_AMP = 60.0  # swell height, world units (0 = off); peaks +AMP, gaps -AMP
SWELL_ARMS = 3  # number of arms / rim peaks (integer for a seamless wrap)
SWELL_WOBBLE = 0.5  # world-space phase wobble (radians), so arms aren't ruled
SWELL_WOBBLE_SCALE = 2600.0  # wavelength of that wobble, world units
# Outer fade: the arms are a central massif, not disc-wide volume. They hold full
# strength out to SWELL_FADE_R0, then ease to nothing by SWELL_FADE_R1, leaving a
# flatter dune desert (texture only) from there to the bounds.
SWELL_FADE_R0 = 2400.0  # arms at full strength within this radius
SWELL_FADE_R1 = 5200.0  # ... faded to nothing by here (= BASE_R, where the hill also ends)

# --- avalanche (raster sand-slide, no runtime equivalent) -------------------
# A thermal sand-slide that moves material between over-steep neighbouring cells.
# Mass-conserving, so it does NOT flatten a uniform slope to repose; it bites at
# curvature, rounding the crests the ridged noise leaves sharp, filling the toes
# and capping the steepest fold cliffs. A naturalising pass, not a repose clamp.
REPOSE_DEG = 33.0  # the avalanche's stable step (talus angle)
AVALANCHE_ITERS = 18  # sand-slide passes: round the crest, fill the toe, cap cliffs
AVALANCHE_RELAX = 0.5  # fraction of the over-steep excess moved each pass


# --- gradient (Perlin) noise ------------------------------------------------
# Faithful port of the runtime's self-contained Perlin so the baked dunes are
# the same field, not merely the same character: the permutation is built with
# the identical LCG-shuffled seed, and the gradient set and fade curve match.
GRADX = np.array([1, -1, 1, -1, 1, -1, 0, 0], dtype=np.float64)
GRADY = np.array([1, 1, -1, -1, 0, 0, 1, -1], dtype=np.float64)


def _build_perm() -> np.ndarray:
    order = list(range(256))
    s = 0x9E3779B1  # fixed seed -> stable dunes
    for i in range(255, 0, -1):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF  # imul + add, masked to u32
        j = s % (i + 1)
        order[i], order[j] = order[j], order[i]
    return np.array([order[i & 255] for i in range(512)], dtype=np.int64)


PERM = _build_perm()


def smootherstep(t: np.ndarray) -> np.ndarray:
    """Zero first and second derivative at both ends; clamps to [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D gradient noise, ~[-1, 1], vectorised over the sample grid."""
    xi = np.floor(x)
    yi = np.floor(y)
    X = xi.astype(np.int64) & 255
    Y = yi.astype(np.int64) & 255
    xf = x - xi
    yf = y - yi
    u = smootherstep(xf)
    v = smootherstep(yf)

    def gdot(h: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        g = PERM[h] & 7
        return GRADX[g] * dx + GRADY[g] * dy

    aa = PERM[X] + Y
    ba = PERM[X + 1] + Y
    x1 = (1 - u) * gdot(aa, xf, yf) + u * gdot(ba, xf - 1, yf)
    x2 = (1 - u) * gdot(aa + 1, xf, yf - 1) + u * gdot(ba + 1, xf - 1, yf - 1)
    return (1 - v) * x1 + v * x2


def perlin_periodic(x: np.ndarray, y: np.ndarray, px: int) -> np.ndarray:
    """Perlin noise periodic in x with integer period px, free in y.

    Same gradient field as perlin(), but the x lattice index is wrapped modulo px
    before the gradient hash, so noise(x + px, y) == noise(x, y) exactly. That is
    what lets a ridge field sampled on an angular coordinate wrap seamlessly
    around the circle (set px to the integer arm count). y is left unwrapped: the
    radial axis is open, not a loop.
    """
    xi = np.floor(x)
    yi = np.floor(y)
    x0 = (np.mod(xi, px).astype(np.int64)) & 255  # lower x corner, wrapped at px
    x1 = (np.mod(xi + 1, px).astype(np.int64)) & 255  # upper x corner, wrapped
    Y = yi.astype(np.int64) & 255
    xf = x - xi
    yf = y - yi
    u = smootherstep(xf)
    v = smootherstep(yf)

    def gdot(h: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        g = PERM[h] & 7
        return GRADX[g] * dx + GRADY[g] * dy

    aa = PERM[x0] + Y
    ba = PERM[x1] + Y
    x1v = (1 - u) * gdot(aa, xf, yf) + u * gdot(ba, xf - 1, yf)
    x2v = (1 - u) * gdot(aa + 1, xf, yf - 1) + u * gdot(ba + 1, xf - 1, yf - 1)
    return (1 - v) * x1v + v * x2v


def fbm_periodic(x: np.ndarray, y: np.ndarray, px: int) -> np.ndarray:
    """fBm built from perlin_periodic, staying seamless in x at period px.

    Each octave doubles the frequency and the period together (px, 2px, 4px ...),
    so every octave wraps on the same circle; the sum does too.
    """
    amp = 1.0
    freq = 1
    out = np.zeros_like(x)
    norm = 0.0
    for _ in range(FBM_OCTAVES):
        out = out + amp * perlin_periodic(x * freq, y * freq, px * freq)
        norm += amp
        amp *= 0.5
        freq *= 2
    return out / norm


# --- height field -----------------------------------------------------------
def hill(r: np.ndarray) -> np.ndarray:
    """The broad radial rise plus the central vantage crater.

    The rise is a whisper at the centre easing to 0 by BASE_R, as before. Onto
    the inner disc we dish a crater: a shallow sink the present sits in, easing
    back to the plateau level by CRATER_FALLOFF_R so the deep dunes beyond are
    untouched.
    """
    rise = PEAK_HEIGHT * (1 - smootherstep((r - PLATEAU_R) / (BASE_R - PLATEAU_R)))
    return rise + vantage_centre(r)


def vantage_centre(r: np.ndarray) -> np.ndarray:
    """Central crater: the present is a shallow sink, not a hill.

    A broad shallow bowl deepest at the centre (-CRATER_DEPTH) easing back to the
    plateau level by CRATER_FALLOFF_R, wide enough that the gentle inner wall lets
    the first masses of books clear the lip from the sunken spawn.
    """
    return -CRATER_DEPTH * (1.0 - smootherstep(r / CRATER_FALLOFF_R))


def _bilinear(F: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Sample F at fractional (row, col), clamping to the edge."""
    nr, nc = F.shape
    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    fr = row - r0
    fc = col - c0
    r0c = np.clip(r0, 0, nr - 1)
    r1c = np.clip(r0 + 1, 0, nr - 1)
    c0c = np.clip(c0, 0, nc - 1)
    c1c = np.clip(c0 + 1, 0, nc - 1)
    f00 = F[r0c, c0c]
    f01 = F[r0c, c1c]
    f10 = F[r1c, c0c]
    f11 = F[r1c, c1c]
    return (f00 * (1 - fr) * (1 - fc) + f01 * (1 - fr) * fc
            + f10 * fr * (1 - fc) + f11 * fr * fc)


def avalanche(D: np.ndarray, texel: float) -> np.ndarray:
    """Thermal sand-slide: move material across over-steep adjacent cells.

    Each pass, for both grid axes, any height difference between adjacent cells
    beyond the stable step (tan(repose) * texel) sheds a fraction of the excess
    to the lower cell. Because it is mass-conserving, a uniform over-steep face
    passes material straight through with no net change; it bites only where the
    slope changes, so in practice it rounds the sharp convex crest the ridged
    noise leaves, fills the concave toe, and caps the steepest fold cliffs. It is
    a naturalising pass, not a repose clamp. The wrap seam is held at zero flow
    so the square's far edges (deep void, past the clipped mesh) don't bleed.
    """
    if AVALANCHE_ITERS <= 0:
        return D
    talus = np.tan(np.radians(REPOSE_DEG)) * texel
    D = D.astype(np.float64).copy()
    for _ in range(AVALANCHE_ITERS):
        for axis in (0, 1):
            diff = D - np.roll(D, -1, axis)  # this cell minus its +1 neighbour
            flow = np.sign(diff) * np.clip(np.abs(diff) - talus, 0.0, None)
            flow *= 0.5 * AVALANCHE_RELAX
            seam = [slice(None), slice(None)]
            seam[axis] = -1  # last cell has no real +1 neighbour (wrap)
            flow[tuple(seam)] = 0.0
            D = D - flow + np.roll(flow, 1, axis)  # +1 neighbour gains the flow
    return D


def _spiral_ridge(ang: np.ndarray, lnr: np.ndarray, k: int,
                  aspect: float) -> np.ndarray:
    """One scale of the spiral ridged field: k organic ridges around the circle.

    The angular axis is k*ang sampled with a period-k Perlin (so it is seamless
    around the circle); the radial axis is ln(r) scaled so cells stay aspect times
    the (radius-growing) arc cell. Returns 1 - |fBm| squared, in [0, 1]: tight
    crests, flat troughs. The shared generator for both the fine dunes and the
    coarse swells, so the two scales spiral identically.
    """
    a = k * ang
    b = k * lnr / (2 * np.pi * aspect)
    ridge = 1.0 - np.abs(fbm_periodic(a, b, int(k)))
    return ridge * ridge


def dune_spiral(x: np.ndarray, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Radial spiral dunes: ridged fBm sampled in a spiral polar frame.

    Each octave is a ridged-noise field (1 - |fBm|, squared for tight crests and
    flat troughs) whose ridges fork, merge and pinch off organically, laid on the
    angular axis k*theta so they radiate from the centre. The angular
    axis is sampled with a PERIODIC Perlin wrapping at the integer arm count k, so
    the field is seamless around the circle; the radial axis r is open and
    stretched by SP_ASPECT so the dunes run radially rather than ring the centre.
    Octaves (k doubling) are weighted into the annulus where their arc spacing ~
    SP_SPACE and crossfade as a partition of unity, holding the dune scale
    constant and letting neighbouring scales merge. The angle is twisted by
    SP_TWIST*ln(r) to wind the field into a spiral (0 = straight radial spokes).
    """
    if SP_AMP == 0.0:
        return np.zeros_like(r)
    rr = np.maximum(r, 1.0)
    theta = np.arctan2(z, x)
    tp = theta - SP_TWIST * np.log(rr / SP_R0)  # spiralled angle, coherent across octaves
    ang = tp / (2 * np.pi)  # turns; * k gives the per-octave angular sample coordinate
    lnr = np.log(rr)
    ln2 = np.log(2.0)
    out = np.zeros_like(r)
    wsum = np.zeros_like(r)
    for n in range(SP_OCTAVES):
        k = SP_K0 * (2 ** n)
        r_n = k * SP_SPACE / (2 * np.pi)  # radius where this octave's arc spacing = SP_SPACE
        u = (lnr - np.log(r_n)) / ln2  # octaves away from this harmonic's home annulus
        w = np.where(np.abs(u) < 1.0, 0.5 * (1 + np.cos(np.pi * u)), 0.0)
        out = out + w * _spiral_ridge(ang, lnr, k, SP_ASPECT)
        wsum = wsum + w
    out = np.where(wsum > 1e-6, out / np.maximum(wsum, 1e-6), out)  # full height at band edges
    return SP_AMP * out  # radial envelope applied later by dune_envelope (unified calm)


def dune_spiral_relief(x: np.ndarray, z: np.ndarray, r: np.ndarray,
                       texel: float) -> np.ndarray:
    """Radial dunes, naturalised by the isotropic avalanche.

    The avalanche is direction-free (it just sheds over-steep slope to lower
    neighbours), so it rounds the sharp crests and fills the toes of the radial
    ridges regardless of which way they run.
    """
    return avalanche(dune_spiral(x, z, r), texel)


def spiral_swell(x: np.ndarray, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Balanced cos spiral arms: large-scale volume, peaks +AMP and gaps -AMP.

    Signed cos of SWELL_ARMS*tp, where tp = theta - SP_TWIST*ln(r) is the same
    twisted angle the dunes ride, so the arms wind with the texture. cos (not the
    dune's ridged noise) keeps the arms balanced N-fold, which is what breaks the
    central rim into evenly spaced peaks rather than one lopsided lump. A
    world-space phase wobble (continuous in x/z, so seam-safe) bends the arms off
    a ruled spiral. Disc-spanning, faded in only past the present plateau.
    """
    if SWELL_AMP == 0.0:
        return np.zeros_like(r)
    inner = smootherstep((r - NOISE_INNER) / (NOISE_FULL - NOISE_INNER))  # fade in past plateau
    outer = 1.0 - smootherstep((r - SWELL_FADE_R0) / (SWELL_FADE_R1 - SWELL_FADE_R0))  # die off to bounds
    env = inner * outer  # a central band: a massif in the middle, flat desert outside
    rr = np.maximum(r, 1.0)
    theta = np.arctan2(z, x)
    tp = theta - SP_TWIST * np.log(rr / SP_R0)  # twisted angle, shared with the dunes
    wob = SWELL_WOBBLE * perlin(x / SWELL_WOBBLE_SCALE, z / SWELL_WOBBLE_SCALE)
    return env * SWELL_AMP * np.cos(SWELL_ARMS * tp + wob)


def center_texture(x: np.ndarray, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Subtle low-amplitude ripples that give the central vantage hill character.

    Two octaves of world-space fBm at small amplitude (CENTER_TEX_AMP) across the
    bookless basin (full within CENTER_TEX_FULL_R, gone by CENTER_TEX_FADE_R ~
    R_INNER) so the empty centre isn't dead-smooth, stopping where the books start
    so it never competes with them. Plain isotropic noise, not the radiating spiral
    dunes, precisely because the dunes were too tall here.
    """
    if CENTER_TEX_AMP == 0.0:
        return np.zeros_like(r)
    n = perlin(x / CENTER_TEX_SCALE, z / CENTER_TEX_SCALE)
    n = n + 0.5 * perlin(x / (CENTER_TEX_SCALE * 0.5), z / (CENTER_TEX_SCALE * 0.5))
    n = n / 1.5
    env = 1.0 - smootherstep((r - CENTER_TEX_FULL_R) / (CENTER_TEX_FADE_R - CENTER_TEX_FULL_R))
    return CENTER_TEX_AMP * env * n


def book_density_field(books: np.ndarray, res: int, texel: float) -> np.ndarray:
    """Smooth, normalised book density on the bake grid, in [0, 1].

    Histogram the book positions into the grid, blur to a smooth falloff (SMOOTH_BLUR
    wide), and normalise against the SMOOTH_PCTL percentile so the densest clusters
    sit near 1 without a few extreme cells dominating. Grid is aligned with the
    height field: row i = world z/y, col j = world x.

    The clip to [0, 1] leaves a flat-topped plateau with a hard rim where the density
    crosses the percentile, which reads as a sharp boundary around the dense core on
    the terrain. A second gaussian blur (SMOOTH_BLUR2) after the clip rounds that rim
    and feathers the mask outward, softening the boundary and spreading the calm zone
    rather than concentrating it.
    """
    if books.size == 0:
        return np.zeros((res, res))
    edges = -WORLD_SIZE / 2 + np.arange(res + 1) * texel
    counts, _, _ = np.histogram2d(books[:, 1], books[:, 0], bins=[edges, edges])
    dens = gaussian_filter(counts, SMOOTH_BLUR / texel)
    norm = np.percentile(dens, SMOOTH_PCTL)
    if norm <= 0:
        return np.zeros((res, res))
    mask = np.clip(dens / norm, 0.0, 1.0)
    return gaussian_filter(mask, SMOOTH_BLUR2 / texel)


def dune_envelope(r: np.ndarray, density: np.ndarray) -> np.ndarray:
    """The single amplitude envelope on the spiral dunes, in [0, 1].

    Two calming sources combine by union (max), never summed: a radial centre
    calm that flattens the dunes toward spawn (full inside CENTER_CALM_R0, easing
    to none by CENTER_CALM_R1) so the vantage stays clean while the spiral still
    reaches inward, and the book-density calm so clusters read as visible mass.
    A point is calmed by whichever source wants it calmer, so the centre pocket
    and a dense cluster on top of it saturate at flat rather than over-flattening.
    """
    center_calm = CENTER_STRENGTH * (
        1.0 - smootherstep((r - CENTER_CALM_R0) / (CENTER_CALM_R1 - CENTER_CALM_R0)))
    calm = np.maximum(center_calm, SMOOTH_STRENGTH * density)
    return 1.0 - np.clip(calm, 0.0, 1.0)


def bake_heightmap(res: int, teleporters: np.ndarray,
                   books: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the height field on a res x res grid over the world square.

    Returns (H, density, envelope). H[i, j] has j indexing world x and i indexing
    world z, both at pixel centres: x = -W/2 + (j + 0.5) * texel, z = -W/2 + (i +
    0.5) * texel. density is the normalised book-density field; envelope is the
    combined dune amplitude scale (centre calm unioned with book calm). Both are
    same-shaped and returned for inspection.
    """
    texel = WORLD_SIZE / res
    axis = -WORLD_SIZE / 2 + (np.arange(res) + 0.5) * texel
    gx, gz = np.meshgrid(axis, axis)  # gx varies along columns, gz along rows
    r = np.hypot(gx, gz)

    # Scale the dunes by the unified envelope: calm toward the centre (clean
    # vantage, dunes spiralling gently inward) unioned with book-density calm
    # (clusters read as mass). The hill, swell and centre are structure, untouched;
    # only the occluding dune texture is attenuated.
    dens = book_density_field(books, res, texel)
    env = dune_envelope(r, dens)
    H = (hill(r) + env * dune_spiral_relief(gx, gz, r, texel)
         + spiral_swell(gx, gz, r) + center_texture(gx, gz, r))

    # Teleporter plazas are no longer flattened. They were levelled so a teleporter
    # MODEL could sit flush on the ground; there is no model now (the gate is a glow
    # baked into the terrain shader plus a beam), so flattening would only punch flat
    # discs into the dunes for no reason. The floor markers drape on the dune surface
    # like everything else. `teleporters` is still loaded for the inspection render.

    return H, dens, env


def bake_window(cx: float, cz: float, size: float, px: int) -> tuple[np.ndarray, float]:
    """Bake a small square window at high pixel density, to inspect crest shape.

    No plaza carving: the window is meant for a patch of open dune field away
    from the monuments, where the full-disc raster is too coarse to judge.
    """
    texel = size / px
    ax = cx - size / 2 + (np.arange(px) + 0.5) * texel
    az = cz - size / 2 + (np.arange(px) + 0.5) * texel
    gx, gz = np.meshgrid(ax, az)
    r = np.hypot(gx, gz)
    return (hill(r) + dune_spiral_relief(gx, gz, r, texel) + spiral_swell(gx, gz, r)
            + center_texture(gx, gz, r)), texel


def report_asymmetry(H: np.ndarray, texel: float, wind: tuple) -> None:
    """Print windward-vs-lee slope stats for a dune patch, as an objective read.

    Projects the gradient onto the wind axis: climbing toward a crest (windward)
    is one sign, dropping down the lee the other. A leaning dune has gentle
    windward slopes and steep lee slopes, so the lee/windward mean-slope ratio
    is the asymmetry number; 1.0 is a symmetric field.
    """
    wind_x, wind_z = wind
    dz, dx = np.gradient(H, texel)  # dH/dz (rows), dH/dx (cols)
    g = dx * wind_x + dz * wind_z  # along-wind directional derivative
    windward = g[g > 0]  # climbing toward the crest as we go downwind
    lee = -g[g < 0]  # dropping off the lee as we continue downwind
    if windward.size and lee.size:
        # Mean climb-vs-drop slope is forced near-equal on any closed transect
        # (total climb = total drop), so the lean lives in the slope-distribution
        # tails: a slip face is a heavy steep tail on the lee side only. Compare
        # the p95 of each, and the share of lee area sitting near the repose
        # angle (a present slip face) vs the same threshold on the windward side.
        wp95 = np.degrees(np.arctan(np.percentile(windward, 95)))
        lp95 = np.degrees(np.arctan(np.percentile(lee, 95)))
        thr = np.tan(np.radians(REPOSE_DEG - 6))  # "near the slip-face angle"
        lee_slip = (lee > thr).mean()
        wind_slip = (windward > thr).mean()
        print(f"  asymmetry: p95 windward~{wp95:.1f}deg lee~{lp95:.1f}deg  "
              f"slip-face area lee {lee_slip:.0%} vs windward {wind_slip:.0%} "
              f"(repose {REPOSE_DEG:g})")


def year_to_radius(year: float) -> float:
    """Base radius for a death year, mirroring stage6's era curve."""
    t = min(max((2000.0 - year) / TIME_SPAN, 0.0), 1.0)
    return R_INNER + (R_MAX - R_INNER) * (t ** RADIUS_ALPHA)


def render(H: np.ndarray, teleporters: np.ndarray, crop: dict, out: Path,
           vert_exag: float, dpi: int, wind: tuple) -> None:
    res = H.shape[0]
    texel = WORLD_SIZE / res
    half = WORLD_SIZE / 2
    extent = [-half, half, -half, half]

    fig = plt.figure(figsize=(33, 20), dpi=dpi)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.93, bottom=0.06,
                        wspace=0.16, hspace=0.18)
    axr = fig.add_subplot(2, 3, 1)
    axe = fig.add_subplot(2, 3, 2)
    axc = fig.add_subplot(2, 3, 3)
    axp = fig.add_subplot(2, 1, 2)

    # Left: shaded relief. The world is ~250x wider than the dunes are tall, so
    # vert_exag lifts the relief into visibility; it only affects the lighting,
    # not the stored heights.
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(H, cmap=plt.cm.copper, blend_mode="soft",
                   vert_exag=vert_exag, dx=texel, dy=texel)
    axr.imshow(rgb, extent=extent, origin="lower")
    axr.set_title(f"shaded relief (vert_exag x{vert_exag:g})")

    # Right: raw elevation with a colourbar, so absolute heights read.
    im = axe.imshow(H, extent=extent, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=axe, fraction=0.046, pad=0.04, label="elevation (u)")
    axe.set_title("elevation")

    # Overlay the world structure on both so the dunes can be read against it:
    # the spawn plaza rim, the content edge, a few era rings, and the plazas.
    theta = np.linspace(0, 2 * np.pi, 400)
    edge_year = 2000 - TIME_SPAN
    for ax in (axr, axe):
        for r, c in ((R_INNER, "#ffd24d"), (R_MAX, "#ff5d5d")):
            ax.plot(r * np.cos(theta), r * np.sin(theta), color=c, lw=0.8,
                    alpha=0.7)
        for yr in (1900, 1700, 1500, 1000, 500, 1):
            if yr <= edge_year:
                continue
            r = year_to_radius(yr)
            ax.plot(r * np.cos(theta), r * np.sin(theta), color="#dddddd",
                    lw=0.4, ls=(0, (4, 4)), alpha=0.4)
        if teleporters.size:
            ax.scatter(teleporters[:, 0], teleporters[:, 1], s=18, marker="o",
                       facecolors="none", edgecolors="#39d0ff", linewidths=0.8)
        ax.scatter([0], [0], marker="+", c="#ffffff", s=70, linewidths=1.0)
        # mark the crop window so it's clear where the zoom comes from
        cx, cz, cs = crop["cx"], crop["cz"], crop["size"]
        ax.plot([cx - cs / 2, cx + cs / 2, cx + cs / 2, cx - cs / 2, cx - cs / 2],
                [cz - cs / 2, cz - cs / 2, cz + cs / 2, cz + cs / 2, cz - cs / 2],
                color="#39ff8c", lw=1.0)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")

    # Right: a high-res crop of open dune field, the view that actually shows
    # whether the crests read as dunes (the full disc is too coarse for that).
    cx, cz, cs = crop["cx"], crop["cz"], crop["size"]
    cext = [cx - cs / 2, cx + cs / 2, cz - cs / 2, cz + cs / 2]
    cls = LightSource(azdeg=315, altdeg=45)
    crgb = cls.shade(crop["H"], cmap=plt.cm.copper, blend_mode="soft",
                     vert_exag=vert_exag, dx=crop["texel"], dy=crop["texel"])
    axc.imshow(crgb, extent=cext, origin="lower")
    axc.set_aspect("equal")
    axc.set_title(f"dune crop @ ({cx:.0f}, {cz:.0f}), {cs:.0f}u wide "
                  f"({crop['texel']:.1f}u/texel)")
    # draw the profile line (a downwind transect through the crop centre)
    wind_x, wind_z = wind
    cH = crop["H"]
    n = cH.shape[0]
    half_n = (n - 1) / 2
    tline = np.arange(n) - half_n
    px = cx + tline * crop["texel"] * wind_x
    pz = cz + tline * crop["texel"] * wind_z
    axc.plot(px, pz, color="#39ff8c", lw=1.2, alpha=0.9)

    # Bottom: an along-wind elevation transect through the crop centre. The wind
    # blows toward +s, so a leaning dune should show a long gentle windward ramp
    # rising to the crest then a short steep drop down the lee face.
    s_world = tline * crop["texel"]
    samp = _bilinear(cH, half_n + tline * wind_z, half_n + tline * wind_x)
    axp.plot(s_world, samp, color="#b5651d", lw=1.6)
    axp.fill_between(s_world, samp.min(), samp, color="#b5651d", alpha=0.18)
    axp.set_xlabel("downwind distance along transect (u)  -->  wind direction")
    axp.set_ylabel("elevation (u)")
    axp.set_title("along-wind profile through the crop centre "
                  "(gentle windward ramp, steep lee drop = asymmetry)")
    axp.grid(True, alpha=0.25)

    fig.suptitle(
        f"Stage 9 heightmap: {res}x{res} ({texel:.1f}u/texel), "
        f"elevation {H.min():.0f}..{H.max():.0f}u. "
        f"Yellow ring = spawn plaza (R_INNER), red = content edge (R_MAX), "
        f"blue = teleporter plazas.",
        fontsize=13,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)


def _overlay_structure(ax, teleporters: np.ndarray) -> None:
    """Draw the world rings, teleporters and centre on a top-down axis."""
    half = WORLD_SIZE / 2
    theta = np.linspace(0, 2 * np.pi, 400)
    for r, c in ((R_INNER, "#ffd24d"), (R_MAX, "#ff5d5d")):
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=c, lw=0.8, alpha=0.7)
    for yr in (1900, 1700, 1500, 1000, 500, 1):
        if yr <= 2000 - TIME_SPAN:
            continue
        r = year_to_radius(yr)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#dddddd", lw=0.4,
                ls=(0, (4, 4)), alpha=0.4)
    if teleporters.size:
        ax.scatter(teleporters[:, 0], teleporters[:, 1], s=18, marker="o",
                   facecolors="none", edgecolors="#39d0ff", linewidths=0.8)
    ax.scatter([0], [0], marker="+", c="#ffffff", s=70, linewidths=1.0)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")


def render_density(density: np.ndarray, envelope: np.ndarray,
                   teleporters: np.ndarray, out: Path, dpi: int) -> None:
    """Inspect the dune smoothing: the book-density field and the combined dune
    envelope (centre calm unioned with book calm), side by side with structure."""
    res = density.shape[0]
    half = WORLD_SIZE / 2
    extent = [-half, half, -half, half]

    fig = plt.figure(figsize=(26, 13), dpi=dpi)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.9, bottom=0.05, wspace=0.16)
    axd = fig.add_subplot(1, 2, 1)
    axa = fig.add_subplot(1, 2, 2)

    im = axd.imshow(density, extent=extent, origin="lower", cmap="magma",
                    vmin=0, vmax=1)
    fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04, label="normalised density")
    axd.set_title(f"book density (blur={SMOOTH_BLUR:g}u, pctl={SMOOTH_PCTL:g})")

    im = axa.imshow(envelope, extent=extent, origin="lower", cmap="viridis",
                    vmin=0, vmax=1)
    fig.colorbar(im, ax=axa, fraction=0.046, pad=0.04, label="dune amplitude x")
    axa.set_title(f"dune envelope: centre calm (R0={CENTER_CALM_R0:g} "
                  f"R1={CENTER_CALM_R1:g}) U book calm (strength={SMOOTH_STRENGTH:g}); "
                  f"dark = flat")

    for ax in (axd, axa):
        _overlay_structure(ax, teleporters)

    fig.suptitle(
        f"Stage 9 book-density smoothing: {res}x{res}. Dense (recent) shelves "
        f"calm the dunes; sparse deep past stays wild. "
        f"Yellow = R_INNER, red = R_MAX.",
        fontsize=13,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleporters", type=Path, default=TELEPORTERS_PATH)
    parser.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--out-density-png", type=Path, default=OUT_DENSITY_PNG)
    parser.add_argument("--res", type=int, default=2048,
                        help="heightmap resolution (square); 18000/res = u/texel")
    parser.add_argument("--vert-exag", type=float, default=2.0,
                        help="relief lighting exaggeration for the render only")
    parser.add_argument("--crop-center", type=float, nargs=2, default=(2600.0, 0.0),
                        metavar=("X", "Z"), help="centre of the high-res dune crop")
    parser.add_argument("--crop-size", type=float, default=2400.0,
                        help="width of the high-res dune crop, world units")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--repose", type=float, default=None,
                        help="override REPOSE_DEG (avalanche talus angle)")
    parser.add_argument("--aval-iters", type=int, default=None,
                        help="override AVALANCHE_ITERS (sand-slide passes)")
    parser.add_argument("--crater-depth", type=float, default=None,
                        help="override CRATER_DEPTH (central crater depth below plateau; 0 = flat)")
    parser.add_argument("--crater-falloff-r", type=float, default=None,
                        help="override CRATER_FALLOFF_R (radius the crater eases back to plateau)")
    parser.add_argument("--center-tex-amp", type=float, default=None,
                        help="override CENTER_TEX_AMP (basin ripple height; 0 = off)")
    parser.add_argument("--center-tex-scale", type=float, default=None,
                        help="override CENTER_TEX_SCALE (ripple wavelength)")
    parser.add_argument("--center-tex-full-r", type=float, default=None,
                        help="override CENTER_TEX_FULL_R (ripple full-strength radius)")
    parser.add_argument("--center-tex-fade-r", type=float, default=None,
                        help="override CENTER_TEX_FADE_R (ripple fade-out radius)")
    parser.add_argument("--sp-amp", type=float, default=None,
                        help="override SP_AMP (radial dune height; 0 = off)")
    parser.add_argument("--sp-space", type=float, default=None,
                        help="override SP_SPACE (target arc spacing between ridges)")
    parser.add_argument("--sp-twist", type=float, default=None,
                        help="override SP_TWIST (spiral winding; 0 = radial spokes)")
    parser.add_argument("--sp-octaves", type=int, default=None,
                        help="override SP_OCTAVES (doubling harmonics)")
    parser.add_argument("--sp-k0", type=int, default=None,
                        help="override SP_K0 (innermost arm count)")
    parser.add_argument("--sp-aspect", type=float, default=None,
                        help="override SP_ASPECT (radial:angular dune cell ratio)")
    parser.add_argument("--swell-amp", type=float, default=None,
                        help="override SWELL_AMP (swell height; peaks +AMP, gaps -AMP; 0 = off)")
    parser.add_argument("--swell-arms", type=int, default=None,
                        help="override SWELL_ARMS (number of arms / rim peaks)")
    parser.add_argument("--swell-wobble", type=float, default=None,
                        help="override SWELL_WOBBLE (arm phase wobble, radians)")
    parser.add_argument("--swell-fade-r0", type=float, default=None,
                        help="override SWELL_FADE_R0 (arms full-strength within this radius)")
    parser.add_argument("--swell-fade-r1", type=float, default=None,
                        help="override SWELL_FADE_R1 (arms faded to nothing by this radius)")
    parser.add_argument("--books", type=Path, default=BOOKS_PATH,
                        help="book layout parquet for density smoothing")
    parser.add_argument("--smooth-strength", type=float, default=None,
                        help="override SMOOTH_STRENGTH (max dune attenuation under books; 0 = off)")
    parser.add_argument("--smooth-blur", type=float, default=None,
                        help="override SMOOTH_BLUR (density blur radius, world units)")
    parser.add_argument("--smooth-pctl", type=float, default=None,
                        help="override SMOOTH_PCTL (density percentile mapped to full attenuation)")
    args = parser.parse_args()

    global REPOSE_DEG, AVALANCHE_ITERS
    global CRATER_DEPTH, CRATER_FALLOFF_R
    global CENTER_TEX_AMP, CENTER_TEX_SCALE, CENTER_TEX_FULL_R, CENTER_TEX_FADE_R
    global SP_AMP, SP_SPACE, SP_TWIST, SP_OCTAVES, SP_K0, SP_ASPECT
    global SWELL_AMP, SWELL_ARMS, SWELL_WOBBLE, SWELL_FADE_R0, SWELL_FADE_R1
    global SMOOTH_STRENGTH, SMOOTH_BLUR, SMOOTH_PCTL
    if args.crater_depth is not None:
        CRATER_DEPTH = args.crater_depth
    if args.crater_falloff_r is not None:
        CRATER_FALLOFF_R = args.crater_falloff_r
    if args.center_tex_amp is not None:
        CENTER_TEX_AMP = args.center_tex_amp
    if args.center_tex_scale is not None:
        CENTER_TEX_SCALE = args.center_tex_scale
    if args.center_tex_full_r is not None:
        CENTER_TEX_FULL_R = args.center_tex_full_r
    if args.center_tex_fade_r is not None:
        CENTER_TEX_FADE_R = args.center_tex_fade_r
    if args.repose is not None:
        REPOSE_DEG = args.repose
    if args.aval_iters is not None:
        AVALANCHE_ITERS = args.aval_iters
    if args.sp_amp is not None:
        SP_AMP = args.sp_amp
    if args.sp_space is not None:
        SP_SPACE = args.sp_space
    if args.sp_twist is not None:
        SP_TWIST = args.sp_twist
    if args.sp_octaves is not None:
        SP_OCTAVES = args.sp_octaves
    if args.sp_k0 is not None:
        SP_K0 = args.sp_k0
    if args.sp_aspect is not None:
        SP_ASPECT = args.sp_aspect
    if args.swell_amp is not None:
        SWELL_AMP = args.swell_amp
    if args.swell_arms is not None:
        SWELL_ARMS = args.swell_arms
    if args.swell_wobble is not None:
        SWELL_WOBBLE = args.swell_wobble
    if args.swell_fade_r0 is not None:
        SWELL_FADE_R0 = args.swell_fade_r0
    if args.swell_fade_r1 is not None:
        SWELL_FADE_R1 = args.swell_fade_r1
    if args.smooth_strength is not None:
        SMOOTH_STRENGTH = args.smooth_strength
    if args.smooth_blur is not None:
        SMOOTH_BLUR = args.smooth_blur
    if args.smooth_pctl is not None:
        SMOOTH_PCTL = args.smooth_pctl
    print(f"avalanche: repose={REPOSE_DEG:g}deg {AVALANCHE_ITERS} passes")
    print(f"radial dunes: amp={SP_AMP:g} space={SP_SPACE:g} twist={SP_TWIST:g} "
          f"k0={SP_K0} octaves={SP_OCTAVES} aspect={SP_ASPECT:g}")
    print(f"swells: amp={SWELL_AMP:g} arms={SWELL_ARMS} wobble={SWELL_WOBBLE:g} "
          f"fade={SWELL_FADE_R0:g}..{SWELL_FADE_R1:g}")

    start = time.perf_counter()

    if args.teleporters.exists():
        tp = pq.read_table(args.teleporters, columns=["x", "y"])
        teleporters = np.column_stack([
            np.asarray(tp["x"].to_pylist(), dtype=np.float64),
            np.asarray(tp["y"].to_pylist(), dtype=np.float64),
        ])
        print(f"loaded {teleporters.shape[0]} teleporter plazas")
    else:
        teleporters = np.empty((0, 2))
        print("no teleporters parquet; baking without plazas")

    if SMOOTH_STRENGTH > 0 and args.books.exists():
        bt = pq.read_table(args.books, columns=["x", "y"])
        books = np.column_stack([
            np.asarray(bt["x"].to_pylist(), dtype=np.float64),
            np.asarray(bt["y"].to_pylist(), dtype=np.float64),
        ])
        print(f"loaded {books.shape[0]} books for density smoothing "
              f"(strength={SMOOTH_STRENGTH:g} blur={SMOOTH_BLUR:g} pctl={SMOOTH_PCTL:g})")
    else:
        books = np.empty((0, 2))
        print("density smoothing off (no books or strength=0)")

    texel = WORLD_SIZE / args.res
    print(f"baking {args.res}x{args.res} heightmap ({texel:.2f}u/texel)...")
    H, density, envelope = bake_heightmap(args.res, teleporters, books)
    print(f"  elevation range {H.min():.1f}..{H.max():.1f}u")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, height=H.astype(np.float32),
                        world_size=np.float64(WORLD_SIZE),
                        resolution=np.int64(args.res))
    print(f"  wrote {args.out_npz}")

    cx, cz = args.crop_center
    cH, ctexel = bake_window(cx, cz, args.crop_size, 1536)
    # transect direction at the crop centre: radial, so it crosses the dune ridges
    cr = np.hypot(cx, cz) or 1.0
    cwind = (cx / cr, cz / cr)
    report_asymmetry(cH, ctexel, cwind)
    crop = {"H": cH, "texel": ctexel, "cx": cx, "cz": cz, "size": args.crop_size}
    render(H, teleporters, crop, args.out_png, args.vert_exag, args.dpi, cwind)
    print(f"  wrote {args.out_png}")

    if books.size:
        render_density(density, envelope, teleporters, args.out_density_png, args.dpi)
        print(f"  wrote {args.out_density_png}")
    print(f"done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
