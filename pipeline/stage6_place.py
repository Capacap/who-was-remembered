"""
Stage 6: radial-time placement with population-equalized longitude.

Each figure gets a polar position the runtime renders as a book in the desert:

- radius = era. r = R_INNER + (R_MAX - R_INNER) * t^alpha, t = (2000 -
  death_year) / TIME_SPAN clipped to [0, 1]. The player spawns on a landing pad
  of radius R_INNER (year 2000) and walks outward into the past. alpha = 1:
  radius is linear in time, so every century gets equal radial width. This is
  deliberate and load-bearing. Figure density is ~100x higher in the modern
  centuries than in antiquity; a mapping that equalized book spacing would
  compress the sparse ancient centuries into a thin ring and imply the past is
  better recorded than it is. Linear time instead lets the few ancient figures
  sit in vast near-empty rings, so the emptiness reads as the missing record it
  is. The cost lands at the centre: the modern crowd is too dense to spread, so
  the player spawns inside an overlapping thicket of the recently-dead that
  thins underfoot as they walk back. That crowding is the declared subject
  (recency bias), left honest rather than spread out. The span is so large
  (R_MAX in the thousands of world units) that deep time is unwalkable by
  design; it is a void to be crossed once flight is unlocked, not strolled.
  Uncertain
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

A quarter of figures have no birth/death place. Rather than scatter them at
random, they fall through a ladder that reads only recorded data: P27
citizenship, then a hand-curated gazetteer over the English description
(demonyms, historical polities, regions; see gazetteer.json). A resolved
country does not become a centroid; the figure borrows a real longitude
sampled from an anchored compatriot, so it spreads across that country's true
arc and folds into the same population CDF. What stays unresolved is genuinely
place-less in the record and keeps the deterministic per-QID random angle, with
geo_source left None so the runtime can mark it adrift rather than pretend it
is anchored. Prominence never enters position; it drives book thickness in the
runtime, a separate axis.

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
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
FIGURES_PATH = ROOT / "cache" / "wikidata_figures_quality.parquet"
PLACES_PATH = ROOT / "cache" / "places.parquet"
GAZETTEER_PATH = ROOT / "gazetteer.json"
OUT_PATH = ROOT / "cache" / "placement.parquet"

# Scale is derived, not chosen: linear time (alpha=1) over a 100:1 density
# gradient forces a large world. R_MAX is set so the bulk modern band (~1850 on)
# clears a ~1-second walk between books (~1.4 u, 1 unit ~= 1 m); the spawn decade
# stays an overlapping thicket on purpose. See /tmp world-sizing math and the
# project memory. R_INNER is a wide pad so the densest recent years have
# circumference to ring rather than collapse onto the origin.
R_MAX = 7100.0
R_INNER = 200.0        # landing pad: player spawns at its rim, inside the modern crowd
TIME_SPAN = 2800.0     # edge = year -800; the pre-800 tail scatters beyond as a frontier
RADIUS_ALPHA = 1.0     # linear time: equal radial width per century (honest sparsity)
RADIUS_JITTER = 2.5    # world units: ~one year of radial width, softens the year-rings
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
# Under linear time, radius IS date, so radial uncertainty should equal date
# uncertainty in years x radial-units-per-year (~2.46 u/yr at this scale). Full
# uncertainty (a millennium-rounded date or birth==death placeholder) stands for
# a ~century-scale error, so the sigma is hundreds of units: the antiquity rim
# dissolves into a genuinely diffuse frontier instead of a crisp circle.
UNC_MAX_SCATTER = 350.0   # world units: radial sigma at full uncertainty

# Notability / landmark tier. A landmark is a figure the player is more likely
# to recognize, so it can serve as a reference point while exploring. Two
# sources feed it (see module docstring); both are tunable knobs frozen into the
# output at bake time, which is cheap to redo since this stage runs in seconds.
NOTABILITY_FLOOR = 80  # sitelink_count >= this => globally recognizable "major"
TIER_SECTORS = 24      # local-coverage grid: angular sectors
TIER_RINGS = 14        # local-coverage grid: radial / era rings


def hash_qid(qid: str) -> int:
    return int.from_bytes(hashlib.blake2b(qid.encode(), digest_size=8).digest(), "big")


def load_gazetteer(path: Path) -> tuple[dict, dict, re.Pattern, set, int]:
    """Load the hand-curated geography map and compile its description matcher.

    Returns (citizenship_aliases, regions, pattern, ignore). The pattern matches
    any region key or ignore phrase on word boundaries, longest phrase first so
    'ancient greek' wins over 'greek' and 'roman catholic' is caught before the
    bare 'roman' polity. min_pool lives on the dict for the caller to read.
    """
    data = json.loads(path.read_text())
    regions = {k.lower(): v for k, v in data["regions"].items()}
    ignore = {k.lower() for k in data.get("_ignore", [])}
    aliases = data["citizenship_aliases"]
    keys = sorted(set(regions) | ignore, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return aliases, regions, pattern, ignore, data.get("min_pool", 30)


def match_region(desc: str | None, pattern: re.Pattern, regions: dict, ignore: set) -> str | None:
    """First region QID named in a description, skipping ignore-list collisions."""
    if not desc:
        return None
    for m in pattern.finditer(desc.lower()):
        key = m.group(1)
        if key in ignore:
            continue
        return regions[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=FIGURES_PATH)
    parser.add_argument("--places", type=Path, default=PLACES_PATH)
    parser.add_argument("--gazetteer", type=Path, default=GAZETTEER_PATH)
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
    descriptions = figures["description"].to_pylist()
    citizenships = figures["citizenships"].to_pylist()

    aliases, regions, gaz_re, gaz_ignore, min_pool = load_gazetteer(args.gazetteer)
    print(f"  gazetteer: {len(regions):,} region terms, {len(aliases):,} citizenship aliases")

    rng = np.random.default_rng(args.seed)

    # --- radius from era ---
    # The era curve maps into [R_INNER, R_MAX], leaving an empty landing pad of
    # radius R_INNER around the origin where the player spawns. Jitter that would
    # carry a book into the pad is reflected back out rather than clamped to the
    # edge: a clamp piled every near-origin figure onto one coordinate (year-1999
    # deaths land at t~0), reflection keeps the pad clear and the inner edge from
    # becoming a new pile-up ring.
    radius_jitter = rng.normal(0.0, RADIUS_JITTER, size=n)
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

    # --- tier 1: longitude from a real birth (else death) place ---
    # The strongest anchor: an actual coordinate. Everything below borrows from
    # the spread of these figures rather than inventing a point.
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

    anchored = ~np.isnan(lon_per)
    n_anchored = int(anchored.sum())
    print(f"  tier 1 birth/death place: {n_anchored:,} ({n_anchored / n * 100:.1f}%)")

    # Per-country pools of anchored longitudes. A geo-less figure we can assign a
    # country (via citizenship or description) borrows a real longitude from a
    # compatriot, so it spreads across that country's true arc instead of piling
    # on a single centroid. Pools below min_pool are too thin to sample honestly.
    pools: dict[str, np.ndarray] = {}
    for cc in set(c for c in countries[anchored] if c):
        lons = lon_per[anchored & (countries == cc)]
        if len(lons) >= min_pool:
            pools[cc] = lons

    def pool_for(cc: str | None) -> str | None:
        """Resolve a citizenship/region QID to a country with a sampleable pool."""
        if cc is None:
            return None
        cc = aliases.get(cc, cc)
        return cc if cc in pools else None

    # --- tier 2: citizenship (P27), tier 3: description gazetteer ---
    # Both read recorded data, then sample a longitude from the country's pool.
    # Precedence: structured citizenship beats parsed text. Whatever stays
    # unresolved is genuinely place-less in the record and kept as residue.
    n_cit = n_gaz = 0
    for i in np.where(~anchored)[0]:
        cc = None
        cit = citizenships[i]
        if cit:
            cc = pool_for(cit[0])
            if cc is not None:
                geo_source[i] = "citizenship"
                n_cit += 1
        if cc is None:
            cc = pool_for(match_region(descriptions[i], gaz_re, regions, gaz_ignore))
            if cc is not None:
                geo_source[i] = "gazetteer"
                n_gaz += 1
        if cc is not None:
            lon_per[i] = rng.choice(pools[cc])
            countries[i] = cc

    located = ~np.isnan(lon_per)
    n_located = int(located.sum())
    residue = n - n_located
    print(
        f"  tier 2 citizenship: {n_cit:,}   tier 3 gazetteer: {n_gaz:,}   "
        f"residue (unplaceable): {residue:,} ({residue / n * 100:.1f}%)"
    )
    print(f"  located from recorded data: {n_located:,} ({n_located / n * 100:.1f}%)")

    # --- population CDF of longitude -> equalized angle ---
    # Built over everyone we could locate (tiers 1-3). Because tiers 2-3 sampled
    # from the anchored distribution, adding them barely moves the CDF for tier-1
    # figures while spreading the borrowed ones across their countries' arcs.
    sorted_lons = np.sort(lon_per[located])
    base = np.empty(n, dtype=np.float64)
    cdf = np.searchsorted(sorted_lons, lon_per[located], side="right") / n_located
    base[located] = 2 * math.pi * cdf
    # residue: deterministic per-qid uniform random. Unknown location is left
    # unknown (geo_source is None) for the renderer to mark, not faked precise.
    missing = np.where(~located)[0]
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
    # most-covered located figure becomes the local reference point. Only located
    # figures qualify (tiers 1-3): a minor landmark is noteworthy in the context
    # of its position, and a residue figure's angle is just a hash.
    sec = (angle % (2 * math.pi)) // (2 * math.pi / TIER_SECTORS)
    rmax = radii.max() or 1.0
    ring = np.clip((radii / rmax * TIER_RINGS).astype(int), 0, TIER_RINGS - 1)
    minor_n = 0
    for s in range(TIER_SECTORS):
        for rr in range(TIER_RINGS):
            cell = np.where(located & (sec == s) & (ring == rr))[0]
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
