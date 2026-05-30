"""
Stage 8: topology. Massage the ground-truth placement into something the game
can actually be walked through, without lying about what the data says.

Stages 6 and 7 produce truth: every book sits where its era (radius) and
longitude (angle) put it, and every teleporter sits at its cluster's outward
gate. But truth clumps. The angular jitter in Stage 6 spreads the dense modern
core with independent random offsets, and random offsets are a Poisson process,
so neighbours land on top of one another no matter the sigma; books clip and you
cannot target them. This stage is the seam between "where the data says" and
"what reads on screen": it nudges books for legibility and nothing else.

Two operations, run together as one relaxation:

- Spacing. A few iterations of Lloyd-style relaxation push books that share a
  grid cell away from the cell's centroid, so random clumps spread into nearby
  gaps and the field converges to even, organic spacing. Each book is tethered
  to its placement position (RELAX_MAX_DRIFT), so this removes collisions but
  never re-places: the era and longitude a book encodes survive. Where the field
  is genuinely too dense to separate (the Western modern apex) books pack to the
  tether and stay a crush, which is honest, not an artefact.

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

ROOT = Path(__file__).resolve().parent
PLACEMENT_PATH = ROOT / "cache" / "placement.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_PATH = ROOT / "cache" / "layout.parquet"

# Spacing relaxation (see module docstring). SEP is the target min spacing, set
# near the modern mean (~0.9u, books ~0.5u wide). STEP is the fraction of a cell
# a crowded book moves per iteration; DRIFT caps how far the whole pass can move
# a book from its placement, so legibility never overrides the data by more than
# a sub-decade nudge. 8 iterations is where this relaxer plateaus; more churns.
RELAX_ITERS = 8
RELAX_SEP = 0.9
RELAX_STEP = 0.35
RELAX_MAX_DRIFT = 20.0  # world units

# Teleporter plaza. The clear radius around each monument: empty ground for the
# waypoint visual (a stone circle or similar). TP_PUSH is how hard a book is
# shoved out per iteration; < 1 lets the spacing relaxation disperse the evicted
# books before the final projection so they do not bunch at the rim.
TP_CLEAR_RADIUS = 10.0  # world units
TP_PUSH = 0.6


def relax(x: np.ndarray, y: np.ndarray, mx: np.ndarray, my: np.ndarray,
          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64).copy()
    y = y.astype(np.float64).copy()
    ax, ay = x.copy(), y.copy()  # tether anchors: the placement positions
    sep = RELAX_SEP
    OFF = 1_000_000  # keep cell indices positive so the packed id stays unique

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
        # book-book: push apart from the local cell centroid (Lloyd-lite)
        ox, oy = rng.uniform(0, sep), rng.uniform(0, sep)
        ix = np.floor((x + ox) / sep).astype(np.int64) + OFF
        iy = np.floor((y + oy) / sep).astype(np.int64) + OFF
        cid = ix * (2 * OFF) + iy
        _, inv, counts = np.unique(cid, return_inverse=True, return_counts=True)
        sumx = np.zeros(counts.size)
        sumy = np.zeros(counts.size)
        np.add.at(sumx, inv, x)
        np.add.at(sumy, inv, y)
        dx = x - (sumx / counts)[inv]
        dy = y - (sumy / counts)[inv]
        norm = np.hypot(dx, dy)
        moving = (counts[inv] > 1) & (norm > 1e-9)
        step = np.zeros_like(x)
        step[moving] = RELAX_STEP * sep / norm[moving]
        x += step * dx
        y += step * dy

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
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args()

    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    table = pq.read_table(args.placement)
    n = table.num_rows
    x = np.asarray(table["x"].to_pylist(), dtype=np.float64)
    y = np.asarray(table["y"].to_pylist(), dtype=np.float64)
    print(f"loaded {n:,} placed figures from {args.placement.name}")

    tp = pq.read_table(args.teleporters, columns=["x", "y"])
    mx = np.asarray(tp["x"].to_pylist(), dtype=np.float64)
    my = np.asarray(tp["y"].to_pylist(), dtype=np.float64)
    print(f"  {mx.size} teleporter clearings (r={TP_CLEAR_RADIUS:.0f}u)")

    x2, y2 = relax(x, y, mx, my, rng)

    moved = np.hypot(x2 - x, y2 - y)
    print(f"  relaxed: median move {np.median(moved):.2f}u, "
          f"max {moved.max():.1f}u, {(moved > 0.01).sum():,} books nudged")

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
