"""
Stage 9: terrain bake (mesh). The inspection bench for Step 2 of the terrain
bake plan, built before the real decimator so we can judge a triangulation by
eye and by number instead of by faith.

A low-poly mesh is a chord surface: each triangle is a flat plane stretched
between three samples of the heightmap, so it cuts under convex crests and
bridges over concave troughs. That deviation is not an error to minimise, it is
the look we are buying (big facets on the flats, the chord visibly cutting the
crests). But we only want it where we want it, so we have to see it. This module
is the tool for that.

It does three things, all dependency-free (numpy + matplotlib):
  1. build a mesh from cache/heightmap.npz. For now the only builder is a uniform
     grid at a chosen stride, disc-clipped. That is the dumb baseline a real
     feature-aligned decimator has to beat; when the decimator lands it produces
     the same (verts, faces) and flows through the same render.
  2. rasterize an arbitrary triangle mesh back onto the heightmap grid by
     barycentric interpolation, recovering the exact chord of *that* triangulation
     (not a Delaunay guess). This is what lets us diff mesh against source.
  3. render a comparison: source relief vs chord relief vs the signed deviation
     field, an oblique faceted view of a crop, and an along-wind profile showing
     the chord cutting the crests.

Run (after the heightmap exists; stage9_heightmap.py):
    uv run pipeline/stage9_mesh.py
    uv run pipeline/stage9_mesh.py --stride 4 --crop-center 2600 0 --crop-size 1800
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import fast_simplification as fs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stage6_place import R_INNER, R_MAX, TIME_SPAN
from stage9_heightmap import WIND_X, WIND_Z, _bilinear, year_to_radius

ROOT = Path(__file__).resolve().parent
HEIGHTMAP_PATH = ROOT / "cache" / "heightmap.npz"
OUT_PNG = ROOT / "cache" / "plots" / "mesh.png"
OUT_BIN = ROOT / "cache" / "ground.bin"
# the runtime serves from here; --serve drops the bake straight in so a budget
# tweak is one command + a browser refresh. Step 5 moves this into export_runtime.
SERVE_BIN = ROOT.parent / "runtime" / "public" / "ground.bin"
# the heightmap itself, shipped for the camera-following LOD patch: the runtime
# tessellates a fine near-field grid from it (and samples the player's walk
# height) so the near ground is crisp without a multi-million-tri static mesh.
SERVE_HEIGHTMAP = ROOT.parent / "runtime" / "public" / "heightmap.bin"

# Disc to keep. The world square is 18000u but the corners are pure faded void;
# the real mesh clips to roughly R_MAX + a fade margin, so the baseline does too.
CLIP_RADIUS = 8000.0

# Provisional tri budget for the baked mesh. 30k keeps the transverse dunes
# distinct while reading bold-faceted (15k starts merging crests, 60k softens to
# near-smooth); chosen against the bench's deviation-vs-budget curve. Performance
# is not the constraint (a static 30k-tri ground is trivial), so this is an
# aesthetic dial the runtime gets the final say on. See TERRAIN_BAKE.md Step 2.
DEFAULT_BAKE_TRIS = 30000


# --- loading -----------------------------------------------------------------
def load_heightmap(path: Path) -> tuple[np.ndarray, float, int]:
    """Return (H, world_size, resolution) from the stage-9 npz."""
    data = np.load(path)
    H = data["height"].astype(np.float64)
    world_size = float(data["world_size"])
    res = int(data["resolution"])
    return H, world_size, res


# --- mesh building -----------------------------------------------------------
def grid_mesh(H: np.ndarray, world_size: float, stride: int,
              clip_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a uniform grid mesh from the heightmap, disc-clipped.

    Vertices are world-space (x, y, z) with y the height (Three.js convention:
    x and z horizontal, y up). Every stride-th texel becomes a vertex; each cell
    of the coarse grid is two triangles. A quad is dropped if its centre falls
    outside clip_radius, so the result is a disc, not the full square.

    This is the no-intelligence baseline: facets are uniform everywhere, so the
    flats waste triangles and the crests are starved of them. A feature-aligned
    decimator has to do better than this on both counts at the same tri budget.
    """
    res = H.shape[0]
    texel = world_size / res
    half = world_size / 2

    rows = np.arange(0, res, stride)
    cols = np.arange(0, res, stride)
    gj, gi = np.meshgrid(cols, rows)  # gj indexes x (cols), gi indexes z (rows)
    X = -half + (gj + 0.5) * texel
    Z = -half + (gi + 0.5) * texel
    Y = H[gi, gj]

    nr, nc = rows.size, cols.size
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nr * nc).reshape(nr, nc)

    # quad centre radius, vectorised, to clip to the disc
    qx = (X[:-1, :-1] + X[1:, 1:]) * 0.5
    qz = (Z[:-1, :-1] + Z[1:, 1:]) * 0.5
    keep = np.hypot(qx, qz) <= clip_radius

    v00 = idx[:-1, :-1][keep]
    v01 = idx[:-1, 1:][keep]
    v10 = idx[1:, :-1][keep]
    v11 = idx[1:, 1:][keep]
    # two triangles per kept quad, wound consistently
    faces = np.concatenate([
        np.stack([v00, v10, v11], axis=1),
        np.stack([v00, v11, v01], axis=1),
    ], axis=0).astype(np.int64)
    return verts, faces


def decimate_mesh(verts: np.ndarray, faces: np.ndarray, target_tris: int,
                  agg: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Quadric edge-collapse decimation down to target_tris (fast-simplification).

    Fed a dense indexed grid mesh, this collapses edges in order of least
    quadric error, which on a near-flat dune field means it spends triangles
    where the surface bends (crests, fold cliffs) and strips them off the flats:
    feature-aligned, the thing the uniform grid can't do. `agg` is the library's
    aggressiveness (higher = more willing to collapse for the same budget).

    Returns an indexed (verts, faces); topology stays smooth/welded here. The
    unindexed-flat split for per-face shading and colour is a later concern (it
    only triples the vertex count and changes nothing about the silhouette), so
    we keep the mesh welded while judging the decimation itself.
    """
    v, f = fs.simplify(verts.astype(np.float64), faces.astype(np.int64),
                       target_count=int(target_tris), agg=agg)
    return v, f.astype(np.int64)


# --- export (ground.bin) -----------------------------------------------------
# ground.bin layout (little-endian, mirrors the positions.bin idiom in
# export_runtime.py: a Uint32 count header then Float32 payload):
#   uint32   vertexCount   (= 3 * triangle count; the mesh is unindexed)
#   float32  positions[vertexCount * 3]   x, y, z per vertex, triangle soup
#
# Unindexed triangle soup, not an indexed mesh: the runtime flat-shades the
# ground (per-face normal from position derivatives) and Step 3 will bake a
# per-face colour, both of which want each triangle to own its three vertices.
# No normals are stored (flatShading derives them) and no colour yet (the
# runtime still computes the radial tint from position; Step 3 moves it here).
def to_triangle_soup(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unweld an indexed mesh into a flat (3*tris, 3) triangle soup."""
    return verts[faces].reshape(-1, 3).astype(np.float32)


def write_ground_bin(path: Path, soup: np.ndarray) -> None:
    """Write the triangle soup as ground.bin (uint32 count + float32 xyz)."""
    n = soup.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(np.uint32(n).tobytes())
        fh.write(np.ascontiguousarray(soup, dtype="<f4").tobytes())


# heightmap.bin layout (little-endian), the raster shipped for the runtime LOD
# patch. Row-major, matching the npz: row indexes world z, col indexes world x,
# at pixel centres (x = -W/2 + (col+0.5)*texel, same for z). The runtime reads it
# back with the identical mapping and bilinear-samples it.
#   uint32   resolution            (square, res*res cells)
#   float32  world_size            (world units across the square)
#   float32  height[res * res]     (row-major: i*res + j -> world (col j, row i))
def write_heightmap_bin(path: Path, H: np.ndarray, world_size: float) -> None:
    """Write the heightmap raster as heightmap.bin (res + world_size + grid)."""
    res = H.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(np.uint32(res).tobytes())
        fh.write(np.float32(world_size).tobytes())
        fh.write(np.ascontiguousarray(H, dtype="<f4").tobytes())


def read_ground_bin(path: Path) -> np.ndarray:
    """Read ground.bin back into a (vertexCount, 3) float32 soup (round-trip)."""
    raw = path.read_bytes()
    n = int(np.frombuffer(raw, dtype="<u4", count=1)[0])
    pos = np.frombuffer(raw, dtype="<f4", count=n * 3, offset=4)
    return pos.reshape(n, 3)


# --- rasterizing a triangulation back to the grid ---------------------------
def rasterize_mesh(verts: np.ndarray, faces: np.ndarray, res: int,
                   world_size: float) -> np.ndarray:
    """Sample the mesh's chord surface onto a res x res grid; NaN off-mesh.

    For each triangle we walk its raster bounding box and, for the pixels whose
    barycentric coordinates are all non-negative, write the plane-interpolated
    height. This reconstructs the exact piecewise-flat chord of the given
    triangulation, which is what the renderer will show and what books would
    float above. Pure numpy, one Python loop over faces (fine for inspection-
    scale meshes; warns past a budget).
    """
    texel = world_size / res
    half = world_size / 2
    col = (verts[:, 0] + half) / texel - 0.5
    row = (verts[:, 2] + half) / texel - 0.5
    val = verts[:, 1]

    if faces.shape[0] > 600_000:
        print(f"  rasterize: {faces.shape[0]} faces is a lot for the Python "
              f"loop; this will be slow")

    S = np.full((res, res), np.nan, dtype=np.float64)
    c = col[faces]  # (F, 3)
    r = row[faces]
    v = val[faces]
    cmin = np.clip(np.floor(c.min(1)).astype(np.int64), 0, res - 1)
    cmax = np.clip(np.ceil(c.max(1)).astype(np.int64), 0, res - 1)
    rmin = np.clip(np.floor(r.min(1)).astype(np.int64), 0, res - 1)
    rmax = np.clip(np.ceil(r.max(1)).astype(np.int64), 0, res - 1)

    for f in range(faces.shape[0]):
        c0, c1, c2 = c[f]
        r0, r1, r2 = r[f]
        denom = (r1 - r2) * (c0 - c2) + (c2 - c1) * (r0 - r2)
        if denom == 0.0:
            continue
        cc, rg = np.meshgrid(np.arange(cmin[f], cmax[f] + 1),
                             np.arange(rmin[f], rmax[f] + 1))
        a = ((r1 - r2) * (cc - c2) + (c2 - c1) * (rg - r2)) / denom
        b = ((r2 - r0) * (cc - c2) + (c0 - c2) * (rg - r2)) / denom
        g = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (g >= 0)
        if not inside.any():
            continue
        z = a * v[f, 0] + b * v[f, 1] + g * v[f, 2]
        sub = S[rmin[f]:rmax[f] + 1, cmin[f]:cmax[f] + 1]
        sub[inside] = z[inside]
    return S


# --- metrics -----------------------------------------------------------------
def deviation_stats(H: np.ndarray, S: np.ndarray, world_size: float) -> dict:
    """Chord-minus-source deviation, overall and split by radius band.

    Negative deviation means the chord sits below the source (cutting under a
    crest); positive means it bridges above (over a trough). The flats should be
    near zero, the crests carry the big negatives: that is the signature we want.
    """
    mask = ~np.isnan(S)
    d = S[mask] - H[mask]
    res = H.shape[0]
    texel = world_size / res
    half = world_size / 2
    ax = -half + (np.arange(res) + 0.5) * texel
    gx, gz = np.meshgrid(ax, ax)
    rad = np.hypot(gx, gz)[mask]
    absd = np.abs(d)
    bands = []
    for lo, hi in ((0, R_INNER), (R_INNER, 2000), (2000, 4000), (4000, R_MAX)):
        m = (rad >= lo) & (rad < hi)
        if m.any():
            bands.append((lo, hi, np.sqrt(np.mean(d[m] ** 2)),
                          np.percentile(absd[m], 99)))
    return {
        "coverage": mask.mean(),
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "p95": float(np.percentile(absd, 95)),
        "p99": float(np.percentile(absd, 99)),
        "max_abs": float(absd.max()),
        "signed_mean": float(np.mean(d)),
        "bands": bands,
    }


# --- rendering ---------------------------------------------------------------
def _facet_view(ax, verts: np.ndarray, faces: np.ndarray, crop: dict,
                vert_exag: float) -> int:
    """Draw the crop's triangles as an oblique flat-shaded facet collection.

    mpl 3D axes are z-up, so we map (worldX, worldZ, height*exag) -> (x, y, z)
    and the vertical axis shows the (exaggerated) relief. Faces are flat-shaded
    by a cheap lambert against a fixed light so the facet structure reads.
    """
    cx, cz, cs = crop["cx"], crop["cz"], crop["size"]
    h = cs / 2
    cen = verts[faces].mean(axis=1)  # (F, 3) centroids in world space
    sel = ((np.abs(cen[:, 0] - cx) <= h) & (np.abs(cen[:, 2] - cz) <= h))
    fsel = faces[sel]
    tri = verts[fsel]  # (k, 3, 3) world (x, y, z)

    # map to mpl 3D space and exaggerate height
    pts = np.stack([tri[:, :, 0], tri[:, :, 2], tri[:, :, 1] * vert_exag], axis=2)

    # flat lambert shade, in the exaggerated space so steep facets read steep
    e1 = pts[:, 1] - pts[:, 0]
    e2 = pts[:, 2] - pts[:, 0]
    nrm = np.cross(e1, e2)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.clip(nlen, 1e-9, None)
    light = np.array([0.4, 0.3, 0.85])
    light = light / np.linalg.norm(light)
    lam = np.clip(np.abs(nrm @ light), 0.0, 1.0)
    base = np.array([0.80, 0.66, 0.42])  # sand
    colors = base[None, :] * (0.45 + 0.55 * lam)[:, None]
    colors = np.clip(colors, 0, 1)

    poly = Poly3DCollection(pts, facecolors=colors, edgecolors=(0, 0, 0, 0.18),
                            linewidths=0.2)
    ax.add_collection3d(poly)
    ax.set_xlim(cx - h, cx + h)
    ax.set_ylim(cz - h, cz + h)
    zmax = max(pts[:, :, 2].max(), 1.0)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect((1, 1, 0.35))
    ax.view_init(elev=32, azim=-60)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"facets @ ({cx:.0f}, {cz:.0f}), {cs:.0f}u  "
                 f"({fsel.shape[0]} tris, height x{vert_exag:g})")
    return fsel.shape[0]


def render_compare(H: np.ndarray, S: np.ndarray, verts: np.ndarray,
                   faces: np.ndarray, world_size: float, stats: dict,
                   crop: dict, out: Path, vert_exag: float, dpi: int) -> None:
    res = H.shape[0]
    texel = world_size / res
    half = world_size / 2
    extent = [-half, half, -half, half]
    ls = LightSource(azdeg=315, altdeg=45)

    fig = plt.figure(figsize=(33, 20), dpi=dpi)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.06,
                        wspace=0.2, hspace=0.22)
    ax_src = fig.add_subplot(2, 3, 1)
    ax_chd = fig.add_subplot(2, 3, 2)
    ax_dif = fig.add_subplot(2, 3, 3)
    ax_3d = fig.add_subplot(2, 2, 3, projection="3d")
    ax_prof = fig.add_subplot(2, 2, 4)

    # top-left: source heightmap relief
    rgb = ls.shade(H, cmap=plt.cm.copper, blend_mode="soft",
                   vert_exag=vert_exag, dx=texel, dy=texel)
    ax_src.imshow(rgb, extent=extent, origin="lower")
    ax_src.set_title("source heightmap")

    # top-middle: chord relief (mesh rasterized back to the grid). The off-mesh
    # void is filled with the source height before shading (NaNs poison the
    # hillshade gradient and wash the whole panel out), then greyed back so the
    # disc clip still reads.
    Sm = np.ma.masked_invalid(S)
    void = np.isnan(S)
    Sfill = np.where(void, H, S)
    crgb = ls.shade(Sfill, cmap=plt.cm.copper, blend_mode="soft",
                    vert_exag=vert_exag, dx=texel, dy=texel)
    crgb[void] = 0.5
    ax_chd.imshow(crgb, extent=extent, origin="lower")
    ax_chd.set_title(f"chord ({faces.shape[0]} tris, {verts.shape[0]} verts)")

    # top-right: signed deviation, diverging, symmetric scale
    d = Sm - H
    vmax = max(stats["p99"], 1.0)
    im = ax_dif.imshow(d, extent=extent, origin="lower", cmap="RdBu",
                       vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax_dif, fraction=0.046, pad=0.04,
                 label="chord - source (u)")
    ax_dif.set_title(f"deviation (rms {stats['rms']:.2f}u, "
                     f"p99 {stats['p99']:.1f}u, max {stats['max_abs']:.1f}u)")

    # world-structure overlay on the three maps
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax in (ax_src, ax_chd, ax_dif):
        for rr, col in ((R_INNER, "#ffd24d"), (R_MAX, "#ff5d5d"),
                        (CLIP_RADIUS, "#39ff8c")):
            ax.plot(rr * np.cos(theta), rr * np.sin(theta), color=col, lw=0.7,
                    alpha=0.6)
        ax.scatter([0], [0], marker="+", c="#000000", s=60, linewidths=1.0)
        cx, cz, cs = crop["cx"], crop["cz"], crop["size"]
        ax.plot([cx - cs / 2, cx + cs / 2, cx + cs / 2, cx - cs / 2, cx - cs / 2],
                [cz - cs / 2, cz - cs / 2, cz + cs / 2, cz + cs / 2, cz - cs / 2],
                color="#39ff8c", lw=1.0)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")

    # bottom-left: oblique faceted view of the crop
    n3d = _facet_view(ax_3d, verts, faces, crop, vert_exag)

    # bottom-right: along-wind profile, source vs chord, through the crop centre.
    # This is where the chord cutting the crests is unmistakable.
    cx, cz, cs = crop["cx"], crop["cz"], crop["size"]
    n = 800
    s = np.linspace(-cs / 2, cs / 2, n)
    px = cx + s * WIND_X
    pz = cz + s * WIND_Z
    col = (px + half) / texel - 0.5
    rowi = (pz + half) / texel - 0.5
    src_line = _bilinear(H, rowi, col)
    chd_line = _bilinear(np.nan_to_num(S, nan=0.0), rowi, col)
    ax_prof.plot(s, src_line, color="#b5651d", lw=1.6, label="source heightmap")
    ax_prof.plot(s, chd_line, color="#1f6feb", lw=1.4, label="mesh chord")
    ax_prof.fill_between(s, src_line, chd_line,
                         where=src_line >= chd_line, color="#1f6feb",
                         alpha=0.18, label="chord cuts under")
    ax_prof.set_xlabel("along-wind distance through crop centre (u)")
    ax_prof.set_ylabel("elevation (u)")
    ax_prof.set_title("source vs chord profile")
    ax_prof.legend(loc="upper right", fontsize=9)
    ax_prof.grid(True, alpha=0.25)

    band_txt = "  ".join(f"[{lo:.0f}-{hi:.0f}]rms{rms:.2f}"
                         for lo, hi, rms, _ in stats["bands"])
    fig.suptitle(
        f"Stage 9 mesh: {faces.shape[0]} tris, coverage {stats['coverage']:.0%}, "
        f"deviation rms {stats['rms']:.2f}u p99 {stats['p99']:.1f}u. "
        f"By radius band (u): {band_txt}. "
        f"Yellow=spawn rim, red=content edge, green=clip disc + crop.",
        fontsize=13,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return n3d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heightmap", type=Path, default=HEIGHTMAP_PATH)
    parser.add_argument("--out-png", type=Path, default=OUT_PNG)
    parser.add_argument("--stride", type=int, default=8,
                        help="grid-mesh stride in texels (baseline decimation)")
    parser.add_argument("--target-tris", type=int, default=None,
                        help="if set, decimate a dense grid to this tri count "
                             "(fast-simplification) instead of the stride grid")
    parser.add_argument("--dense-stride", type=int, default=2,
                        help="stride of the dense grid fed to the decimator")
    parser.add_argument("--agg", type=int, default=7,
                        help="decimation aggressiveness (fast-simplification)")
    parser.add_argument("--bake", action="store_true",
                        help="write the decimated mesh to ground.bin (unindexed "
                             "triangle soup); decimates to --target-tris (default "
                             f"{DEFAULT_BAKE_TRIS} when baking)")
    parser.add_argument("--out-bin", type=Path, default=OUT_BIN)
    parser.add_argument("--serve", action="store_true",
                        help=f"also copy the bake to {SERVE_BIN} (runtime serving dir)")
    parser.add_argument("--serve-heightmap", action="store_true",
                        help=f"write the raster to {SERVE_HEIGHTMAP} for the runtime "
                             f"LOD patch (independent of the mesh bake)")
    parser.add_argument("--no-inspect", action="store_true",
                        help="skip the rasterize + comparison render (the Python "
                             "rasterizer is O(faces); skip it for big-budget bakes)")
    parser.add_argument("--clip-radius", type=float, default=CLIP_RADIUS)
    parser.add_argument("--crop-center", type=float, nargs=2, default=(2600.0, 0.0),
                        metavar=("X", "Z"), help="centre of the faceted crop")
    parser.add_argument("--crop-size", type=float, default=1800.0,
                        help="width of the faceted crop, world units")
    parser.add_argument("--vert-exag", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()

    start = time.perf_counter()
    H, world_size, res = load_heightmap(args.heightmap)
    texel = world_size / res
    print(f"loaded heightmap {res}x{res} ({texel:.2f}u/texel), "
          f"elevation {H.min():.1f}..{H.max():.1f}u")

    if args.serve_heightmap:
        write_heightmap_bin(SERVE_HEIGHTMAP, H, world_size)
        size_mb = SERVE_HEIGHTMAP.stat().st_size / 1e6
        print(f"  served heightmap {SERVE_HEIGHTMAP} ({size_mb:.2f} MB)")

    target_tris = args.target_tris
    if args.bake and target_tris is None:
        target_tris = DEFAULT_BAKE_TRIS

    if target_tris is not None:
        print(f"building dense grid (stride {args.dense_stride} "
              f"= {args.dense_stride * texel:.1f}u quads, clip r<={args.clip_radius:.0f})...")
        dense_v, dense_f = grid_mesh(H, world_size, args.dense_stride,
                                     args.clip_radius)
        print(f"  dense: {dense_v.shape[0]} verts, {dense_f.shape[0]} tris")
        print(f"decimating to ~{target_tris} tris (agg {args.agg})...")
        verts, faces = decimate_mesh(dense_v, dense_f, target_tris, args.agg)
        print(f"  decimated: {verts.shape[0]} verts, {faces.shape[0]} tris "
              f"({faces.shape[0] / dense_f.shape[0]:.1%} of dense)")
    else:
        print(f"building grid mesh (stride {args.stride} = {args.stride * texel:.0f}u "
              f"quads, clip r<={args.clip_radius:.0f})...")
        verts, faces = grid_mesh(H, world_size, args.stride, args.clip_radius)
        print(f"  {verts.shape[0]} verts, {faces.shape[0]} tris")

    if args.bake:
        soup = to_triangle_soup(verts, faces)
        write_ground_bin(args.out_bin, soup)
        size_mb = args.out_bin.stat().st_size / 1e6
        # round-trip read-back as a self-check that the file is well-formed
        back = read_ground_bin(args.out_bin)
        ok = back.shape == soup.shape and np.allclose(back, soup)
        print(f"  wrote {args.out_bin} ({soup.shape[0]} verts = {faces.shape[0]} "
              f"tris, {size_mb:.2f} MB); round-trip {'ok' if ok else 'MISMATCH'}")
        if args.serve:
            write_ground_bin(SERVE_BIN, soup)
            print(f"  served {SERVE_BIN}")

    if args.no_inspect:
        print(f"done in {time.perf_counter() - start:.1f}s (skipped inspect render)")
        return

    print("rasterizing chord back to the grid...")
    S = rasterize_mesh(verts, faces, res, world_size)
    stats = deviation_stats(H, S, world_size)
    print(f"  coverage {stats['coverage']:.1%}, deviation rms {stats['rms']:.2f}u, "
          f"p95 {stats['p95']:.2f}u, p99 {stats['p99']:.2f}u, "
          f"max {stats['max_abs']:.1f}u, signed mean {stats['signed_mean']:+.2f}u")
    for lo, hi, rms, p99 in stats["bands"]:
        print(f"    band [{lo:.0f}-{hi:.0f}]u: rms {rms:.2f}u, p99 {p99:.2f}u")

    cx, cz = args.crop_center
    crop = {"cx": cx, "cz": cz, "size": args.crop_size}
    n3d = render_compare(H, S, verts, faces, world_size, stats, crop,
                         args.out_png, args.vert_exag, args.dpi)
    print(f"  facet view drew {n3d} tris in the crop")
    print(f"  wrote {args.out_png}")
    print(f"done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
