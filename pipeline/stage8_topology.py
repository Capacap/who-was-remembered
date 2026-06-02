"""
Stage 8: topology. Massage the ground-truth placement into something the game
can actually be walked through, without lying about what the data says.

Stages 6 and 7 produce truth: every book sits where its era (radius) and
longitude (angle) put it, and every teleporter sits at its cluster's outward
gate. But truth clumps. The angular jitter in Stage 6 spreads the dense modern
core with independent random offsets, and random offsets are a Poisson process,
so neighbours land on top of one another no matter the sigma; books clip and you
cannot target them. Worse, books now vary in size (book_scale, from article
length), so a big book can wholly swallow a small neighbour whose centre falls
inside its footprint: the small book vanishes visually and the look-at picker,
which grabs the nearest centre, reads the wrong figure. This stage is the seam
between "where the data says" and "what reads on screen": it nudges books for
legibility and nothing else.

Two operations, run together as one relaxation:

- Separation. A few iterations of size-aware pair separation push apart only the
  pairs whose centres fall within SUBSUME_K * max(R_i, R_j) (see constants), so a
  smaller book's centre always clears a larger book's footprint. Edges may still
  overlap (SUBSUME_K < 1): the goal is not even spacing but an end to
  subsumption. Isolated books and loose clumps keep their placement untouched;
  only collisions move. Each book is tethered to its placement position
  (RELAX_MAX_DRIFT), so the era and longitude a book encodes survive. Where the
  field is genuinely too dense to separate within the tether (the Western modern
  apex) books pile to the tether and stay a crush, which is honest, not an
  artefact; they pile but no longer subsume.

- Teleporter clearings. Each monument needs a small empty plaza around it (for
  the waypoint's visual, and so it does not bury books). The clear-zones act as
  obstacles inside the same relaxation: books within TP_CLEAR_RADIUS are pushed
  radially out of the zone, and because the spacing relaxation runs in the same
  loop, the evicted books disperse into the surrounding gaps instead of piling
  into a dense ring at the boundary. A final hard projection guarantees the plaza
  is empty.

Writes layout.parquet: the full placement table with x/y (and the derived
radius/angle) replaced by the relaxed positions. placement.parquet stays the
pristine ground truth; the runtime export reads this.

Run (after Stages 6 and 7):
    uv run pipeline/stage8_topology.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
PLACEMENT_PATH = ROOT / "cache" / "placement.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_PATH = ROOT / "cache" / "layout.parquet"

# Size-aware separation (see module docstring). Each book's footprint is a disc
# of radius R = 0.5 * BOOK_LEN * book_scale (BOOK_LEN mirrors the runtime's
# BOOK_LENGTH; book_scale comes from Stage 6, larger for longer articles). The
# separation target for a pair is SUBSUME_K * max(R_i, R_j): closer than that and
# the SMALLER book's centre sits inside the LARGER footprint, where it is both
# visually swallowed and stolen by the look-at picker (which grabs the nearest
# centre). SUBSUME_K < 1 so edges may still overlap and the dense apex stays a
# crush; it just can no longer subsume. SEP_DAMP splits each correction between
# the pair (0.5 resolves an isolated overlap in one step; lower damps the churn
# from books caught in many overlaps at once). DRIFT caps how far the pass can
# move a book from its placement, so legibility never overrides the data by more
# than a sub-decade nudge; books too crushed to separate within it stay slightly
# overlapped, which is honest. At this corpus the field clears all subsumption
# well within the tether (max move ~2u), so RELAX_ITERS is set where residual
# subsumption reaches zero rather than where movement plateaus.
RELAX_ITERS = 30        # converges to zero residual subsumption at this corpus
BOOK_LEN = 0.58         # world units; must match runtime BOOK_LENGTH
SUBSUME_K = 0.9         # fraction of the larger radius centres must clear
SEP_DAMP = 0.9          # share of each pair's overlap corrected per iteration
RELAX_MAX_DRIFT = 20.0  # world units

# Teleporter plaza. The clear radius around each monument: empty ground for the
# waypoint visual (a stone circle or similar). TP_PUSH is how hard a book is
# shoved out per iteration; < 1 lets the spacing relaxation disperse the evicted
# books before the final projection so they do not bunch at the rim.
TP_CLEAR_RADIUS = 10.0  # world units
TP_PUSH = 0.6


def relax(x: np.ndarray, y: np.ndarray, R: np.ndarray,
          mx: np.ndarray, my: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64).copy()
    y = y.astype(np.float64).copy()
    ax, ay = x.copy(), y.copy()  # tether anchors: the placement positions
    # A pair can only violate when its centres are within SUBSUME_K * max(R_i,
    # R_j) <= SUBSUME_K * R.max(); query that radius and test the per-pair need.
    query_r = SUBSUME_K * float(R.max())

    def push_out_of_zones(scale: float) -> None:
        # push books inside any clear-zone radially outward from that monument
        for j in range(mx.size):
            dx, dy = x - mx[j], y - my[j]
            dist = np.hypot(dx, dy)
            inside = dist < TP_CLEAR_RADIUS
            if not inside.any():
                continue
            d = dist[inside] + 1e-9
            pen = TP_CLEAR_RADIUS - dist[inside]
            x[inside] += scale * pen * dx[inside] / d
            y[inside] += scale * pen * dy[inside] / d

    for _ in range(RELAX_ITERS):
        # book-book: size-aware separation. Only pairs whose centres fall inside
        # SUBSUME_K * max(R_i, R_j) are touched, so isolated books and loose
        # clumps keep their placement; the relaxer removes subsumption, it does
        # not re-grid the field. Most of the disc is sparse, so the pair set is
        # small outside the dense modern wedge.
        tree = cKDTree(np.column_stack((x, y)))
        pairs = tree.query_pairs(r=query_r, output_type="ndarray")
        if pairs.size:
            i, j = pairs[:, 0], pairs[:, 1]
            need = SUBSUME_K * np.maximum(R[i], R[j])
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            d = np.hypot(dx, dy)
            viol = d < need
            if viol.any():
                i, j = i[viol], j[viol]
                dd = d[viol] + 1e-9
                pen = (need[viol] - dd) * SEP_DAMP
                ux, uy = dx[viol] / dd, dy[viol] / dd
                dispx = np.zeros_like(x)
                dispy = np.zeros_like(y)
                np.add.at(dispx, i, -ux * pen)
                np.add.at(dispy, i, -uy * pen)
                np.add.at(dispx, j, ux * pen)
                np.add.at(dispy, j, uy * pen)
                x += dispx
                y += dispy

        # teleporter clearings, dispersed by the same relaxation
        push_out_of_zones(TP_PUSH)

        # tether: clamp drift from placement so nothing wanders off its data
        ddx, ddy = x - ax, y - ay
        drift = np.hypot(ddx, ddy)
        over = drift > RELAX_MAX_DRIFT
        scale = RELAX_MAX_DRIFT / drift[over]
        x[over] = ax[over] + ddx[over] * scale
        y[over] = ay[over] + ddy[over] * scale

    # final hard projection: guarantee no book remains inside a plaza
    push_out_of_zones(1.0)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", type=Path, default=PLACEMENT_PATH)
    parser.add_argument("--teleporters", type=Path, default=TELEPORTERS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    start = time.perf_counter()

    table = pq.read_table(args.placement)
    n = table.num_rows
    x = np.asarray(table["x"].to_pylist(), dtype=np.float64)
    y = np.asarray(table["y"].to_pylist(), dtype=np.float64)
    # footprint radius per book: half the drawn long axis (BOOK_LEN * book_scale).
    scale = np.asarray(table["book_scale"].to_pylist(), dtype=np.float64)
    R = 0.5 * BOOK_LEN * scale
    print(f"loaded {n:,} placed figures from {args.placement.name}")
    print(f"  book radius R: {R.min():.3f}..{R.max():.3f}u (from book_scale)")

    tp = pq.read_table(args.teleporters, columns=["x", "y"])
    mx = np.asarray(tp["x"].to_pylist(), dtype=np.float64)
    my = np.asarray(tp["y"].to_pylist(), dtype=np.float64)
    print(f"  {mx.size} teleporter clearings (r={TP_CLEAR_RADIUS:.0f}u)")

    x2, y2 = relax(x, y, R, mx, my)

    moved = np.hypot(x2 - x, y2 - y)
    print(f"  relaxed: median move {np.median(moved):.2f}u, "
          f"max {moved.max():.1f}u, {(moved > 0.01).sum():,} books nudged")

    # residual subsumption: centres still inside a larger book's footprint after
    # the tether-capped relaxation (the apex crush too dense to fully separate).
    tree = cKDTree(np.column_stack((x2, y2)))
    pairs = tree.query_pairs(r=SUBSUME_K * float(R.max()), output_type="ndarray")
    subsumed = 0
    if pairs.size:
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.hypot(x2[j] - x2[i], y2[j] - y2[i])
        subsumed = int((d < SUBSUME_K * np.maximum(R[i], R[j])).sum())
    print(f"  residual subsumed pairs: {subsumed:,} ({subsumed / n * 100:.2f}% of books)")

    radius = np.hypot(x2, y2)
    angle = np.arctan2(y2, x2) % (2 * math.pi)
    for name, arr in (("x", x2), ("y", y2), ("radius", radius), ("angle", angle)):
        i = table.schema.get_field_index(name)
        table = table.set_column(i, name, pa.array(arr, type=pa.float64()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {n:,} figures -> {args.out} in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
