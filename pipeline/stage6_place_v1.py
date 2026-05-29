"""
Stage 6 (v1): minimal radial placement for inspection.

The point of this script is to get pixels on screen so the inspection
loop can begin, not to be a defensible final algorithm. v1 uses:

- radius from death_year via a power-law compression to keep the
  recent-heavy centre from drowning out antiquity. r = r_max * t^alpha
  with t = (2000 - death_year) / 2800 clipped to [0, 1], alpha = 0.75.
- angle from the figure's birth-place longitude (or death-place,
  whichever resolves first) mapped to [0, 2π). Figures with neither
  geographic anchor get a random angle from a hash of their QID.
- per-figure random jitter on both radius and angle to keep things
  loose; we want a general next to a shoemaker, not crisp partitions.

There is no link-graph term yet; that comes in v2 once we know what
the geographic-only placement looks like.

Run:
    uv run pipeline/stage6_place_v1.py
    uv run pipeline/inspect_placement.py --in pipeline/cache/placement_v1.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
FIGURES_PATH = ROOT / "cache" / "wikidata_figures_quality.parquet"
PLACES_PATH = ROOT / "cache" / "places.parquet"
OUT_PATH = ROOT / "cache" / "placement_v1.parquet"

R_MAX = 1000.0
TIME_SPAN = 2800.0  # years; -800 BC to year 2000 roughly covers the corpus
RADIUS_ALPHA = 0.75
RADIUS_JITTER = 0.012  # fraction of r_max
ANGLE_JITTER = 0.18    # radians


def hash_qid(qid: str) -> int:
    """Deterministic per-qid integer for seeding jitter."""
    return int.from_bytes(hashlib.blake2b(qid.encode(), digest_size=8).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=FIGURES_PATH)
    parser.add_argument("--places", type=Path, default=PLACES_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    start = time.perf_counter()

    figures = pq.read_table(args.figures)
    n = figures.num_rows
    print(f"loaded {n:,} figures from {args.figures.name}")

    places = pq.read_table(
        args.places, columns=["qid", "lat", "lon", "country_qid"]
    )
    place_lon = {}
    place_country = {}
    for q, lat, lon, cc in zip(
        places["qid"].to_pylist(),
        places["lat"].to_pylist(),
        places["lon"].to_pylist(),
        places["country_qid"].to_pylist(),
    ):
        if lat is not None and lon is not None:
            place_lon[q] = lon
        if cc is not None:
            place_country[q] = cc
    print(f"  geo lookup has {len(place_lon):,} place coords")

    qids = figures["qid"].to_pylist()
    years = figures["death_year"].to_pylist()
    birth_qs = figures["birth_place_qid"].to_pylist()
    death_qs = figures["death_place_qid"].to_pylist()

    rng = np.random.default_rng(args.seed)
    radius_jitter = rng.normal(0.0, RADIUS_JITTER, size=n) * R_MAX
    angle_jitter = rng.normal(0.0, ANGLE_JITTER, size=n)

    radii = np.empty(n, dtype=np.float64)
    angles = np.empty(n, dtype=np.float64)
    geo_source = np.empty(n, dtype=object)
    countries = np.empty(n, dtype=object)

    geo_hits = 0
    random_angles = 0

    for i in range(n):
        y = years[i] if years[i] is not None else 2000
        t = (2000 - y) / TIME_SPAN
        t = max(0.0, min(1.0, t))
        r = R_MAX * (t ** RADIUS_ALPHA) + radius_jitter[i]
        radii[i] = max(0.0, r)

        b = birth_qs[i]
        d = death_qs[i]
        lon = place_lon.get(b) if b in place_lon else place_lon.get(d)
        if lon is not None:
            base = math.radians(lon) % (2 * math.pi)
            geo_source[i] = "birth" if b in place_lon else "death"
            geo_hits += 1
        else:
            base = (hash_qid(qids[i]) % 10_000_000) / 10_000_000 * 2 * math.pi
            geo_source[i] = None
            random_angles += 1
        angles[i] = (base + angle_jitter[i]) % (2 * math.pi)

        countries[i] = (
            place_country.get(b)
            or place_country.get(d)
        )

    x = radii * np.cos(angles)
    y_coord = radii * np.sin(angles)

    out = figures.append_column("radius", pa.array(radii, type=pa.float64()))
    out = out.append_column("angle", pa.array(angles, type=pa.float64()))
    out = out.append_column("x", pa.array(x, type=pa.float64()))
    out = out.append_column("y", pa.array(y_coord, type=pa.float64()))
    out = out.append_column(
        "geo_source", pa.array(geo_source.tolist(), type=pa.string())
    )
    out = out.append_column(
        "country_qid", pa.array(countries.tolist(), type=pa.string())
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out, compression="zstd")

    elapsed = time.perf_counter() - start
    print(f"placed {n:,} figures in {elapsed:.1f}s")
    print(f"  geo-anchored: {geo_hits:,} ({geo_hits/n*100:.1f}%)")
    print(f"  random-angle: {random_angles:,} ({random_angles/n*100:.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
