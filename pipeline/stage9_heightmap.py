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

The height field starts from the runtime's analytic terrain
(runtime/src/terrain.ts): a near-flat desert whose relief is a transverse dune
field (anisotropic ridged Perlin, domain-warped, wind-aligned), a whisper of a
central rise, and level plazas carved at the teleporter monuments. From there it
adds slip-face asymmetry, which the runtime cannot: the symmetric ridged field is
sheared downwind so each crest leans onto a steep lee face, then a thermal
avalanche (a sand-slide that moves material between over-steep neighbouring
cells) rounds the knife-edge crest, fills the toe and caps fold cliffs. Both are
neighbour ops on the raster, which is exactly what a pointwise get_height
function can't do and the reason terrain moved into the pipeline. So the baked
field now deliberately diverges from terrain.ts; the runtime catches up by
loading the baked mesh rather than by mirroring this code.

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

from stage6_place import R_INNER, R_MAX, RADIUS_ALPHA, TIME_SPAN

ROOT = Path(__file__).resolve().parent
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_NPZ = ROOT / "cache" / "heightmap.npz"
OUT_PNG = ROOT / "cache" / "plots" / "heightmap.png"

# World extent. The runtime ground mesh is an 18000u square centred on the
# origin (runtime/src/terrain.ts SIZE), so the heightmap covers the same square.
# TODO: once both sides read it from world.json this constant goes away.
WORLD_SIZE = 18000.0

# --- terrain constants (mirror runtime/src/terrain.ts) ----------------------
PEAK_HEIGHT = 60.0  # a whisper of a central rise, not a summit
PLATEAU_R = 700.0  # calm and level here (spawn + plaza + the year-2000 ring)
BASE_R = 5200.0  # the rise has eased to the desert floor (0) by here

FLATTEN_R = 14.0  # level core of a teleporter plaza
FLATTEN_FALLOFF = 50.0  # ... easing back to the dunes over this

DUNE_AMP = 70.0  # crest height above the trough
DUNE_SPACE = 260.0  # along-wind dune spacing (close)
DUNE_LEN = 900.0  # crosswind ridge length scale (long)
DUNE_OCTAVES = 3
WARP_AMP = 120.0  # how far the crest lines meander off straight
WARP_SCALE = 1100.0  # wavelength of that meander
WIND_ANGLE = 0.7  # prevailing wind bearing, radians
WIND_X = np.cos(WIND_ANGLE)
WIND_Z = np.sin(WIND_ANGLE)
NOISE_INNER = PLATEAU_R  # dunes start past the calm present plateau
NOISE_FULL = 1400.0  # ... at full height by here

# --- slip-face asymmetry (raster ops, no runtime equivalent) ----------------
# The symmetric ridged field reads as sand waves. Real transverse dunes lean
# downwind: a long gentle windward ramp, a short steep lee face. We get that in
# two raster passes the pointwise function couldn't:
#  1. a downwind shear that displaces each point along the wind by an amount
#     proportional to its dune height, so crests migrate over the lee and pile
#     the lee face steep while stretching the windward ramp. This is what creates
#     the asymmetry, and it leaves the lee steeper than the repose angle by design.
#  2. a thermal avalanche (sand-slide) that moves material between over-steep
#     neighbouring cells. It is mass-conserving, so it does NOT flatten the
#     uniform lee face to repose; it bites at curvature, rounding the crest,
#     filling the toe and capping fold cliffs. A naturalising pass, not a clamp.
LEE_SHEAR = 0.7  # downwind crest shift per unit dune height (world u per u)
LEE_SHEAR_ITERS = 6  # fixed-point passes resolving the inverse warp (1 = no lean)
REPOSE_DEG = 33.0  # sets the avalanche's stable step + the slip-face report threshold
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


def fbm_unit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """fBm in unit coordinate space (caller pre-scales), ~[-1, 1]."""
    amp = 1.0
    freq = 1.0
    out = np.zeros_like(x)
    norm = 0.0
    for _ in range(DUNE_OCTAVES):
        out = out + amp * perlin(x * freq, y * freq)
        norm += amp
        amp *= 0.5
        freq *= 2
    return out / norm


# --- height field -----------------------------------------------------------
def hill(r: np.ndarray) -> np.ndarray:
    """The bare radial rise: a whisper at the centre, eased to 0 by BASE_R."""
    return PEAK_HEIGHT * (1 - smootherstep((r - PLATEAU_R) / (BASE_R - PLATEAU_R)))


def dune_symmetric(x: np.ndarray, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Anisotropic ridged dune offset, faded in past the present plateau.

    This is the symmetric field (before slip-face asymmetry). The shear and
    avalanche passes turn it into leaning dunes; see dune_relief.
    """
    env = smootherstep((r - NOISE_INNER) / (NOISE_FULL - NOISE_INNER))
    # meander the crest lines so they aren't ruled straight
    wx = x + WARP_AMP * perlin(x / WARP_SCALE, z / WARP_SCALE)
    wz = z + WARP_AMP * perlin(x / WARP_SCALE + 41.3, z / WARP_SCALE + 17.9)
    # rotate into wind-aligned axes and sample anisotropically
    s = (wx * WIND_X + wz * WIND_Z) / DUNE_SPACE
    t = (-wx * WIND_Z + wz * WIND_X) / DUNE_LEN
    ridge = 1 - np.abs(fbm_unit(s, t))  # crease at the crest
    return env * DUNE_AMP * ridge * ridge  # square: tight crest, flat troughs


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


def lee_shear(D: np.ndarray, texel: float) -> np.ndarray:
    """Shear the field downwind by LEE_SHEAR * height, so the dunes lean.

    Models a forward warp p -> p + L*D(p)*wind: every column of sand slides
    downwind by an amount proportional to its height, so a crest migrates over
    the lee (steepening that face) while the windward ramp stretches gentle. We
    apply it as the inverse map (gap-free, unlike a forward scatter): for each
    output cell q solve p = q - L*D(p)*wind by fixed-point iteration, then read
    D at p. A single pass leaves it symmetric; the iterations are what resolve
    the lean, so LEE_SHEAR_ITERS must stay >= ~4. Columns index world x, rows
    index world z, and (WIND_X, WIND_Z) is the wind in that frame.
    """
    if LEE_SHEAR <= 0 or LEE_SHEAR_ITERS < 1:
        return D
    nr, nc = D.shape
    qc, qr = np.meshgrid(np.arange(nc, dtype=np.float64),
                         np.arange(nr, dtype=np.float64))
    pc, pr = qc.copy(), qr.copy()
    for _ in range(LEE_SHEAR_ITERS):
        shift = LEE_SHEAR * _bilinear(D, pr, pc) / texel  # in texels, at source
        pc = qc - shift * WIND_X
        pr = qr - shift * WIND_Z
    return _bilinear(D, pr, pc)


def avalanche(D: np.ndarray, texel: float) -> np.ndarray:
    """Thermal sand-slide: move material across over-steep adjacent cells.

    Each pass, for both grid axes, any height difference between adjacent cells
    beyond the stable step (tan(repose) * texel) sheds a fraction of the excess
    to the lower cell. Because it is mass-conserving, a uniform over-steep face
    passes material straight through with no net change; it bites only where the
    slope changes, so in practice it rounds the convex crest the shear leaves
    knife-edged, fills the concave toe, and caps the steepest fold cliffs. It is
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


def dune_relief(x: np.ndarray, z: np.ndarray, r: np.ndarray,
                texel: float) -> np.ndarray:
    """Asymmetric dune relief: symmetric field, sheared downwind, avalanched."""
    D = dune_symmetric(x, z, r)
    D = lee_shear(D, texel)
    D = avalanche(D, texel)
    return D


def bake_heightmap(res: int, teleporters: np.ndarray) -> np.ndarray:
    """Sample the height field on a res x res grid over the world square.

    Returns H[i, j] where j indexes world x and i indexes world z, both at
    pixel centres: x = -W/2 + (j + 0.5) * texel, z = -W/2 + (i + 0.5) * texel.
    """
    texel = WORLD_SIZE / res
    axis = -WORLD_SIZE / 2 + (np.arange(res) + 0.5) * texel
    gx, gz = np.meshgrid(axis, axis)  # gx varies along columns, gz along rows
    r = np.hypot(gx, gz)

    H = hill(r) + dune_relief(gx, gz, r, texel)

    # Carve a level plaza at each teleporter: blend the field toward the
    # monument's own local ground height inside FLATTEN_R, easing back to the
    # dunes over FLATTEN_FALLOFF. The target height is read from the baked
    # surface at the teleporter's cell (the asymmetric dunes are a raster, no
    # longer a pointwise function), so the plaza sits flush with the sand around
    # it instead of on a mesa. Sampled before any carving so overlapping plazas
    # can't drift (the monuments are far enough apart that they don't overlap).
    if teleporters.size:
        tx, ty = teleporters[:, 0], teleporters[:, 1]
        ci = np.clip(np.round((tx + WORLD_SIZE / 2) / texel - 0.5)
                     .astype(np.int64), 0, res - 1)
        cj = np.clip(np.round((ty + WORLD_SIZE / 2) / texel - 0.5)
                     .astype(np.int64), 0, res - 1)
        th = H[cj, ci]  # surface height at each monument's cell
        for k in range(tx.size):
            d = np.hypot(gx - tx[k], gz - ty[k])
            blend = 1 - smootherstep((d - FLATTEN_R) / FLATTEN_FALLOFF)
            H += (th[k] - H) * blend

    return H


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
    return hill(r) + dune_relief(gx, gz, r, texel), texel


def report_asymmetry(H: np.ndarray, texel: float) -> None:
    """Print windward-vs-lee slope stats for a dune patch, as an objective read.

    Projects the gradient onto the wind axis: climbing toward a crest (windward)
    is one sign, dropping down the lee the other. A leaning dune has gentle
    windward slopes and steep lee slopes, so the lee/windward mean-slope ratio
    is the asymmetry number; 1.0 is a symmetric field.
    """
    dz, dx = np.gradient(H, texel)  # dH/dz (rows), dH/dx (cols)
    g = dx * WIND_X + dz * WIND_Z  # along-wind directional derivative
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
           vert_exag: float, dpi: int) -> None:
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
    cH = crop["H"]
    n = cH.shape[0]
    half_n = (n - 1) / 2
    tline = np.arange(n) - half_n
    px = cx + tline * crop["texel"] * WIND_X
    pz = cz + tline * crop["texel"] * WIND_Z
    axc.plot(px, pz, color="#39ff8c", lw=1.2, alpha=0.9)

    # Bottom: an along-wind elevation transect through the crop centre. The wind
    # blows toward +s, so a leaning dune should show a long gentle windward ramp
    # rising to the crest then a short steep drop down the lee face.
    s_world = tline * crop["texel"]
    samp = _bilinear(cH, half_n + tline * WIND_Z, half_n + tline * WIND_X)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleporters", type=Path, default=TELEPORTERS_PATH)
    parser.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--res", type=int, default=2048,
                        help="heightmap resolution (square); 18000/res = u/texel")
    parser.add_argument("--vert-exag", type=float, default=2.0,
                        help="relief lighting exaggeration for the render only")
    parser.add_argument("--crop-center", type=float, nargs=2, default=(2600.0, 0.0),
                        metavar=("X", "Z"), help="centre of the high-res dune crop")
    parser.add_argument("--crop-size", type=float, default=2400.0,
                        help="width of the high-res dune crop, world units")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--lean", type=float, default=None,
                        help="override LEE_SHEAR (downwind crest lean)")
    parser.add_argument("--repose", type=float, default=None,
                        help="override REPOSE_DEG (lee-face stable angle)")
    parser.add_argument("--aval-iters", type=int, default=None,
                        help="override AVALANCHE_ITERS (sand-slide passes)")
    args = parser.parse_args()

    global LEE_SHEAR, REPOSE_DEG, AVALANCHE_ITERS
    if args.lean is not None:
        LEE_SHEAR = args.lean
    if args.repose is not None:
        REPOSE_DEG = args.repose
    if args.aval_iters is not None:
        AVALANCHE_ITERS = args.aval_iters
    print(f"asymmetry: lean={LEE_SHEAR:g} repose={REPOSE_DEG:g}deg "
          f"avalanche={AVALANCHE_ITERS} passes")

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

    texel = WORLD_SIZE / args.res
    print(f"baking {args.res}x{args.res} heightmap ({texel:.2f}u/texel)...")
    H = bake_heightmap(args.res, teleporters)
    print(f"  elevation range {H.min():.1f}..{H.max():.1f}u")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, height=H.astype(np.float32),
                        world_size=np.float64(WORLD_SIZE),
                        resolution=np.int64(args.res))
    print(f"  wrote {args.out_npz}")

    cx, cz = args.crop_center
    cH, ctexel = bake_window(cx, cz, args.crop_size, 1536)
    report_asymmetry(cH, ctexel)
    crop = {"H": cH, "texel": ctexel, "cx": cx, "cz": cz, "size": args.crop_size}
    render(H, teleporters, crop, args.out_png, args.vert_exag, args.dpi)
    print(f"  wrote {args.out_png}")
    print(f"done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
