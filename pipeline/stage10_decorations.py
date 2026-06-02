"""
Stage 10: decoration scatter. Place surreal stone heads in the desert without
disturbing the books or teleporters.

The heads are pure decoration: three sculpted variants dropped half-buried in the
sand, faces to the sky, to thicken the dream-logic of the place. They carry no
data and must not push a single book aside. So rather than carve clearings the
way Stage 8 does for teleporters, this stage reads the FINISHED book layout and
finds the gaps already in it: spots far enough from every book footprint and
every teleporter plaza to seat a head in open sand.

That constraint does the aesthetic work on its own. The dense modern wedge has no
gaps, so no head lands in the crush; the vast empty deep-past desert is almost all
gap, so that is where the heads collect, half-sunk and staring up out of an
emptiness that is itself the point of the piece. The inner plaza (r < R_INNER) is
left clear on purpose: the player spawns there and surveys the ring across it, so
a head in it would break that first read.

Each placement gets size, yaw and sink jitter so three assets read as many. The
runtime seats them on the baked terrain (it already samples height for the books),
so this stage emits only the ground position and the per-head jitter.

Writes runtime/public/decorations.json: a small array of
{x, y, v (variant 0-2), s (scale), rot (yaw), sink (buried fraction)}.

Run (after Stages 7 and 8):
    uv run pipeline/stage10_decorations.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from stage6_place import R_INNER, R_MAX
from stage8_topology import BOOK_LEN, TP_CLEAR_RADIUS

ROOT = Path(__file__).resolve().parent
LAYOUT_PATH = ROOT / "cache" / "layout.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_PATH = ROOT.parent / "runtime" / "public" / "decorations.json"

# How many heads to scatter, and how big a patch of open sand each one needs. A
# head lies on its back, so its ground footprint is roughly its sculpted height;
# HEAD_RADIUS is that half-extent at scale 1, plus a margin so books never touch
# even the largest head. Candidates are also kept HEAD_SPACING apart so the heads
# never clump into a pile. All eyeball knobs; the stage runs in seconds.
N_HEADS = 10000
HEAD_RADIUS = 1.2       # world units: open-sand radius a scale-1 head needs (~half
                        # HEAD_HEIGHT=2.0, the laid head's ground half-length)
HEAD_SPACING = 35.0     # world units: minimum gap between two heads. Small heads
                        # (~2u) read as findable company only at high density; this
                        # is well above their own size, so they never pile, but the
                        # disc is vast (area-uniform NN ~60u at this count).
MARGIN = 1.0            # extra clearance from book footprints

# Per-head jitter ranges (uniform). Scale spreads the apparent size; yaw is a free
# spin so no two face the same way; sink is the fraction of the head buried below
# the surface, so some are chins-deep and others barely breach the sand.
SCALE_MIN, SCALE_MAX = 0.7, 1.4
SINK_MIN, SINK_MAX = 0.30, 0.55
N_VARIANTS = 3

# Book footprint, mirrored from Stage 8: a book's disc radius is 0.5 * BOOK_LEN *
# book_scale. We reject a candidate whose head disc overlaps any book disc.
BOOK_R_MAX_GUESS = 0.5 * BOOK_LEN * 3.0  # generous upper bound for the coarse cull


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    parser.add_argument("--teleporters", type=Path, default=TELEPORTERS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--n", type=int, default=N_HEADS)
    args = parser.parse_args()

    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    table = pq.read_table(args.layout, columns=["x", "y", "book_scale"])
    bx = np.asarray(table["x"].to_pylist(), dtype=np.float64)
    by = np.asarray(table["y"].to_pylist(), dtype=np.float64)
    book_r = 0.5 * BOOK_LEN * np.asarray(table["book_scale"].to_pylist(), dtype=np.float64)
    book_tree = cKDTree(np.column_stack((bx, by)))
    print(f"loaded {len(bx):,} book footprints from {args.layout.name}")

    tp = pq.read_table(args.teleporters, columns=["x", "y"])
    tx = np.asarray(tp["x"].to_pylist(), dtype=np.float64)
    ty = np.asarray(tp["y"].to_pylist(), dtype=np.float64)
    tp_tree = cKDTree(np.column_stack((tx, ty))) if tx.size else None
    print(f"  {tx.size} teleporter plazas to avoid (r={TP_CLEAR_RADIUS:.0f}u)")

    # Candidate spots, area-uniform over the disc between the spawn plaza and the
    # rim: draw r with sqrt so density is even per unit area (not per unit radius),
    # which puts proportionally more candidates in the big empty outer desert. We
    # oversample (the accept rate near the dense core is low) and greedily keep
    # those that clear books, teleporters, and already-placed heads.
    target = args.n
    accepted: list[dict] = []
    # accepted head coords in preallocated arrays, so the inter-head spacing test is
    # a vectorized O(count) min over a slice rather than rebuilding an array from a
    # python list each candidate (that O(n^2) rebuild is the bottleneck at N=2000).
    acc_x = np.empty(target, dtype=np.float64)
    acc_y = np.empty(target, dtype=np.float64)
    count = 0
    max_head_r = HEAD_RADIUS * SCALE_MAX
    book_cull = max_head_r + BOOK_R_MAX_GUESS + MARGIN

    batches = 0
    while count < target and batches < 400:
        batches += 1
        m = max(target * 8, 20000)
        rr = np.sqrt(rng.uniform(R_INNER**2, R_MAX**2, size=m))
        th = rng.uniform(0.0, 2 * math.pi, size=m)
        cx = rr * np.cos(th)
        cy = rr * np.sin(th)
        cs = rng.uniform(SCALE_MIN, SCALE_MAX, size=m)
        for k in range(m):
            if count >= target:
                break
            hx, hy, hs = cx[k], cy[k], cs[k]
            hr = HEAD_RADIUS * hs
            # books: any footprint disc overlapping the head disc rejects it.
            near = book_tree.query_ball_point((hx, hy), hr + book_cull)
            if near:
                d = np.hypot(bx[near] - hx, by[near] - hy)
                if np.any(d < hr + book_r[near] + MARGIN):
                    continue
            # teleporters: stay outside the plaza plus the head's own radius.
            if tp_tree is not None:
                td, _ = tp_tree.query((hx, hy))
                if td < TP_CLEAR_RADIUS + hr:
                    continue
            # other heads: keep them HEAD_SPACING apart so they never pile up.
            if count and np.min(
                np.hypot(acc_x[:count] - hx, acc_y[:count] - hy)
            ) < HEAD_SPACING:
                continue
            acc_x[count] = hx
            acc_y[count] = hy
            count += 1
            accepted.append(
                {
                    "x": round(float(hx), 1),
                    "y": round(float(hy), 1),
                    "v": int(rng.integers(0, N_VARIANTS)),
                    "s": round(float(hs), 3),
                    "rot": round(float(rng.uniform(0, 2 * math.pi)), 3),
                    "sink": round(float(rng.uniform(SINK_MIN, SINK_MAX)), 3),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(accepted, ensure_ascii=False, indent=0))

    rad = np.hypot(
        np.array([d["x"] for d in accepted]), np.array([d["y"] for d in accepted])
    ) if accepted else np.array([0.0])
    print(
        f"placed {len(accepted)} heads in {time.perf_counter() - start:.1f}s "
        f"(radius {rad.min():.0f}..{rad.max():.0f}u) -> {args.out}"
    )
    if len(accepted) < target:
        print(f"  note: wanted {target}, the layout had room for {len(accepted)}")


if __name__ == "__main__":
    main()
