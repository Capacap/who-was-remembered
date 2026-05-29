"""
Stage 6: radial-time placement with population-equalized longitude.

Each figure gets a polar position the runtime renders as a book in the desert:

- radius = era. r = R_MAX * t^alpha, t = (2000 - death_year) / TIME_SPAN clipped
  to [0, 1]. The player stands at the origin (year 2000) and walks outward into
  the past. alpha < 1 compresses the recent, dense centuries so the modern crowd
  stays legible; that crowding is the declared subject (recency bias), so it is
  left honest rather than spread out.

- angle = population CDF of birth (else death) longitude, not raw longitude.

The reconnaissance behind the CDF choice: raw longitude makes angle a
fame/social-class proxy. Fame in this corpus concentrates in Western Europe,
Western Europe is a narrow longitude band (~0-15 deg E), so raw longitude
compresses most of the mass and nearly all the famous into a thin wedge and
leaves the disc lopsided. Replacing longitude with its population CDF,

    angle = 2*pi * (#figures with longitude <= this) / (#geo-anchored figures)

stretches dense longitude bands across a wide arc and shrinks sparse ones to
slivers, so every angular direction carries comparable population. The transform
is monotonic, so east-west ordering and regional adjacency survive (China stays
"up", the Americas "down-left"), and it is global (one CDF for the whole corpus)
so a region keeps a stable angle across every era ring. This trades away the
visibility of the West's overrepresentation (which raw longitude rendered as a
bright wedge) for a navigable field; the primary subject, recency, lives on the
radial axis and stays honest in every version.

Geo-less figures (~26%, no resolvable birth/death place) take a deterministic
per-QID uniform-random angle, which is already equalized. Prominence never
enters position; it drives book thickness in the runtime, a separate axis.

Run:
    uv run pipeline/stage6_place.py
    uv run pipeline/inspect_placement.py --in pipeline/cache/placement.parquet
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
OUT_PATH = ROOT / "cache" / "placement.parquet"

R_MAX = 1000.0
TIME_SPAN = 2800.0
RADIUS_ALPHA = 0.75
RADIUS_JITTER = 0.012  # fraction of r_max
ANGLE_JITTER = 0.10    # radians


def hash_qid(qid: str) -> int:
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

    places = pq.read_table(args.places, columns=["qid", "lat", "lon", "country_qid"])
    place_lon: dict[str, float] = {}
    place_country: dict[str, str] = {}
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

    # --- radius from era ---
    radius_jitter = rng.normal(0.0, RADIUS_JITTER, size=n) * R_MAX
    yr = np.array([y if y is not None else 2000 for y in years], dtype=np.float64)
    t = np.clip((2000.0 - yr) / TIME_SPAN, 0.0, 1.0)
    radii = np.maximum(0.0, R_MAX * (t ** RADIUS_ALPHA) + radius_jitter)

    # --- resolve a longitude per figure (birth, else death) ---
    lon_per = np.full(n, np.nan)
    geo_source = np.empty(n, dtype=object)
    countries = np.empty(n, dtype=object)
    for i in range(n):
        b, d = birth_qs[i], death_qs[i]
        if b in place_lon:
            lon_per[i] = place_lon[b]
            geo_source[i] = "birth"
        elif d in place_lon:
            lon_per[i] = place_lon[d]
            geo_source[i] = "death"
        else:
            geo_source[i] = None
        countries[i] = place_country.get(b) or place_country.get(d)

    geo_mask = ~np.isnan(lon_per)
    n_geo = int(geo_mask.sum())
    print(f"  geo-anchored: {n_geo:,} ({n_geo / n * 100:.1f}%)")

    # --- population CDF of longitude -> equalized angle ---
    sorted_lons = np.sort(lon_per[geo_mask])
    base = np.empty(n, dtype=np.float64)
    cdf = np.searchsorted(sorted_lons, lon_per[geo_mask], side="right") / n_geo
    base[geo_mask] = 2 * math.pi * cdf
    # geo-less: deterministic per-qid uniform random (already equalized)
    missing = np.where(~geo_mask)[0]
    base[missing] = np.array(
        [(hash_qid(qids[i]) % 10_000_000) / 10_000_000 * 2 * math.pi for i in missing]
    )

    angle = (base + rng.normal(0.0, ANGLE_JITTER, size=n)) % (2 * math.pi)
    x = radii * np.cos(angle)
    y_coord = radii * np.sin(angle)

    out = figures.append_column("radius", pa.array(radii, type=pa.float64()))
    out = out.append_column("angle", pa.array(angle, type=pa.float64()))
    out = out.append_column("x", pa.array(x, type=pa.float64()))
    out = out.append_column("y", pa.array(y_coord, type=pa.float64()))
    out = out.append_column("geo_source", pa.array(geo_source.tolist(), type=pa.string()))
    out = out.append_column("country_qid", pa.array(countries.tolist(), type=pa.string()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out, compression="zstd")

    elapsed = time.perf_counter() - start
    print(f"placed {n:,} figures in {elapsed:.1f}s (population-equalized longitude)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
