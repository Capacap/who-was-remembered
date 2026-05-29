"""
Stage 6: radial-time placement with population-equalized longitude.

Each figure gets a polar position the runtime renders as a book in the desert:

- radius = era. r = R_INNER + (R_MAX - R_INNER) * t^alpha, t = (2000 -
  death_year) / TIME_SPAN clipped to [0, 1]. The player spawns on an empty
  landing pad of radius R_INNER (year 2000) and walks outward into the past;
  the most recent figures ring the pad. alpha < 1 compresses the recent, dense
  centuries so the modern crowd stays legible; that crowding is the declared
  subject (recency bias), so it is left honest rather than spread out. Uncertain
  death years (round-number estimates, placeholders, anything deep in antiquity)
  get extra radial scatter proportional to that uncertainty, so vague dates land
  in a vague band rather than on a false-precise ring; see UNC_* constants. The
  per-figure score is also emitted as date_uncertainty for the renderer.

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

That thickness axis is captured here as landmark_tier (major / minor /
ordinary). A landmark exists purely for the PLAYER's benefit: it is a figure
the player is more likely to recognize, so a taller book gives them a reference
point to steer by and orient against while crossing the desert. Recognizability
has two independent sources. The GLOBAL one is an absolute sitelink_count floor:
cross-lingual coverage means real recognizability wherever a figure sits, which
is what makes every Roman emperor a landmark. The LOCAL one fills directions and
eras where nobody clears the floor (the deep-past Egyptian, Persian, Mesopotamian
cells): the most-covered geo-anchored figure in each region-and-era cell becomes
a reference point even when globally obscure, because it is the thing worth
walking toward in that part of the map. The continuous magnitude stays in
sitelink_count; landmark_tier is the categorical "is this a reference point"
flag the renderer reads to choose a book's asset and labelling.

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
R_INNER = 30.0         # empty landing pad: player spawns here, books start beyond it
TIME_SPAN = 2800.0
RADIUS_ALPHA = 0.75
RADIUS_JITTER = 0.012  # fraction of r_max
ANGLE_JITTER = 0.10    # radians

# Date-uncertainty radial scatter. Deep-past death years are mostly estimates:
# round-number guesses (a death snapped to -500) and birth==death placeholders.
# Placing them on a crisp radius claims a precision the record does not have and
# renders the antiquity edge as an artificial circle. Instead we scatter radius
# by how uncertain the date is, so vague dates land in a vague band and the rim
# dissolves into an honest diffuse frontier. The signal is a proxy (Wikidata's
# real date-precision qualifier was not captured in stage 1); year roundness
# plus a floor that distrusts all deep dates stands in for it.
UNC_RAMP_START = 1000.0   # death_year >= this is trusted (modern dates, no scatter)
UNC_RAMP_SPAN = 1000.0    # years over which uncertainty ramps to full (full by year 0)
UNC_FLOOR = 0.3           # ancient dates are uncertain even when they look precise
UNC_MAX_SCATTER = 55.0    # world units: radial sigma at full uncertainty

# Notability / landmark tier. A landmark is a figure the player is more likely
# to recognize, so it can serve as a reference point while exploring. Two
# sources feed it (see module docstring); both are tunable knobs frozen into the
# output at bake time, which is cheap to redo since this stage runs in seconds.
NOTABILITY_FLOOR = 80  # sitelink_count >= this => globally recognizable "major"
TIER_SECTORS = 24      # local-coverage grid: angular sectors
TIER_RINGS = 14        # local-coverage grid: radial / era rings


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
    birth_years = figures["birth_year"].to_pylist()
    birth_qs = figures["birth_place_qid"].to_pylist()
    death_qs = figures["death_place_qid"].to_pylist()

    rng = np.random.default_rng(args.seed)

    # --- radius from era ---
    # The era curve maps into [R_INNER, R_MAX], leaving an empty landing pad of
    # radius R_INNER around the origin where the player spawns. Jitter that would
    # carry a book into the pad is reflected back out rather than clamped to the
    # edge: a clamp piled every near-origin figure onto one coordinate (year-1999
    # deaths land at t~0), reflection keeps the pad clear and the inner edge from
    # becoming a new pile-up ring.
    radius_jitter = rng.normal(0.0, RADIUS_JITTER, size=n) * R_MAX
    yr = np.array([y if y is not None else 2000 for y in years], dtype=np.float64)
    t = np.clip((2000.0 - yr) / TIME_SPAN, 0.0, 1.0)
    radii = R_INNER + (R_MAX - R_INNER) * (t ** RADIUS_ALPHA) + radius_jitter

    # --- date uncertainty -> extra radial scatter ---
    # uncertainty in [0, 1]: roundness of the death year (a proxy for how
    # estimated it is), floored so even precise-looking deep dates stay vague,
    # and ramped to zero for modern dates we trust. by == dy is a placeholder.
    by = np.array([b if b is not None else 2000 for b in birth_years], dtype=np.float64)
    ady = np.abs(yr)
    round_u = np.full(n, 0.1)
    round_u[ady % 10 == 0] = 0.3
    round_u[ady % 100 == 0] = 0.7
    round_u[(ady % 1000 == 0) & (yr != 0)] = 1.0
    round_u[by == yr] = 1.0  # birth==death placeholder
    ramp = np.clip((UNC_RAMP_START - yr) / UNC_RAMP_SPAN, 0.0, 1.0)
    uncertainty = ramp * np.maximum(round_u, UNC_FLOOR)
    radii = radii + rng.normal(0.0, 1.0, size=n) * (UNC_MAX_SCATTER * uncertainty)

    below = radii < R_INNER
    radii[below] = 2 * R_INNER - radii[below]

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

    # --- landmark tier: which books the player can use as reference points ---
    sitelinks = np.array(
        [s if s is not None else 0 for s in figures["sitelink_count"].to_pylist()]
    )
    tier = np.full(n, "ordinary", dtype=object)
    # global source: absolute recognizability, wherever the figure sits.
    is_major = sitelinks >= NOTABILITY_FLOOR
    tier[is_major] = "major"
    # local source: in each region-and-era cell with no major to anchor it, the
    # most-covered geo-anchored figure becomes the local reference point. Only
    # geo-anchored figures qualify: a minor landmark is noteworthy in the
    # context of its position, and a geo-less figure's angle is just a hash.
    sec = (angle % (2 * math.pi)) // (2 * math.pi / TIER_SECTORS)
    rmax = radii.max() or 1.0
    ring = np.clip((radii / rmax * TIER_RINGS).astype(int), 0, TIER_RINGS - 1)
    minor_n = 0
    for s in range(TIER_SECTORS):
        for rr in range(TIER_RINGS):
            cell = np.where(geo_mask & (sec == s) & (ring == rr))[0]
            if len(cell) == 0:
                continue
            best = cell[np.argmax(sitelinks[cell])]
            if not is_major[best]:  # cell has no major; promote its local best
                tier[best] = "minor"
                minor_n += 1
    print(
        f"  landmark tiers: {int(is_major.sum()):,} major, {minor_n:,} minor, "
        f"{n - int(is_major.sum()) - minor_n:,} ordinary"
    )

    out = figures.append_column("radius", pa.array(radii, type=pa.float64()))
    out = out.append_column("angle", pa.array(angle, type=pa.float64()))
    out = out.append_column("x", pa.array(x, type=pa.float64()))
    out = out.append_column("y", pa.array(y_coord, type=pa.float64()))
    out = out.append_column("geo_source", pa.array(geo_source.tolist(), type=pa.string()))
    out = out.append_column("country_qid", pa.array(countries.tolist(), type=pa.string()))
    out = out.append_column("landmark_tier", pa.array(tier.tolist(), type=pa.string()))
    out = out.append_column("date_uncertainty", pa.array(uncertainty, type=pa.float64()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.out, compression="zstd")

    elapsed = time.perf_counter() - start
    print(f"placed {n:,} figures in {elapsed:.1f}s (population-equalized longitude)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
