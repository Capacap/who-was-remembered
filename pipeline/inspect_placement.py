"""
Inspect placement: render PNG views of a figure-positioning parquet.

Reads a parquet that has the Stage 4 schema plus either (x, y) Cartesian
columns or (radius, angle) polar columns, and produces a set of standard
plots for eyeballing how the placement algorithm distributes the corpus.

The piece's geometry has the player at the origin (year 2000) walking
outward back in time, so polar input is converted to Cartesian on read
with the convention x = r*cos(angle), y = r*sin(angle).

Views:
  era       — points colored by death-century (purple → yellow, recent → old)
  country   — points colored by country_qid (top-N countries highlighted,
              everything else grey)
  density   — 2D histogram heatmap, exposes over- and under-crowded regions
  ring      — points within --ring-years of --ring-center, plotted at full
              size so local angular neighborhoods are inspectable

Run:
    uv run pipeline/inspect_placement.py --in pipeline/cache/<placement>.parquet \\
        --out-dir pipeline/cache/plots
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (x, y, metadata_table_as_pylists) from a placement parquet."""
    table = pq.read_table(path)
    cols = set(table.column_names)
    if {"x", "y"}.issubset(cols):
        x = np.asarray(table["x"].to_pylist(), dtype=np.float64)
        y = np.asarray(table["y"].to_pylist(), dtype=np.float64)
    elif {"radius", "angle"}.issubset(cols):
        r = np.asarray(table["radius"].to_pylist(), dtype=np.float64)
        a = np.asarray(table["angle"].to_pylist(), dtype=np.float64)
        x = r * np.cos(a)
        y = r * np.sin(a)
    else:
        raise SystemExit(
            "placement parquet must have either (x,y) or (radius,angle) columns"
        )
    meta = {
        "death_year": table["death_year"].to_pylist() if "death_year" in cols else None,
        "country_qid": (
            table["country_qid"].to_pylist() if "country_qid" in cols else None
        ),
        "title": table["title"].to_pylist() if "title" in cols else None,
        "sitelink_count": (
            table["sitelink_count"].to_pylist() if "sitelink_count" in cols else None
        ),
    }
    return x, y, meta


def plot_era(x: np.ndarray, y: np.ndarray, years: list[int | None], out: Path) -> None:
    valid = np.array([yy is not None for yy in years])
    yrs = np.array([yy if yy is not None else 2000 for yy in years], dtype=np.float64)
    # Most of the corpus lives in the last few centuries; a linear colormap
    # gets dominated by the few ancient outliers. log(years-before-2000)
    # gives the recent dense cluster real contrast while still ordering
    # antiquity correctly.
    age = np.log1p(np.clip(2000 - yrs, 0, None))
    fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
    sc = ax.scatter(
        x[valid],
        y[valid],
        c=age[valid],
        s=0.4,
        alpha=0.25,
        cmap="viridis_r",
        linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("log(years before 2000)")
    ax.set_aspect("equal")
    ax.set_title(f"Placement by era ({valid.sum():,} figures)")
    ax.set_facecolor("#111")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_country(
    x: np.ndarray, y: np.ndarray, countries: list[str | None], out: Path, top_n: int = 12
) -> None:
    counts = Counter(c for c in countries if c)
    top = [c for c, _ in counts.most_common(top_n)]
    palette = plt.get_cmap("tab20", top_n)
    color_lookup = {c: palette(i) for i, c in enumerate(top)}
    fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
    grey = np.array([c not in color_lookup for c in countries])
    ax.scatter(x[grey], y[grey], c="#444", s=0.2, alpha=0.1, linewidths=0)
    for c, col in color_lookup.items():
        mask = np.array([cc == c for cc in countries])
        ax.scatter(
            x[mask],
            y[mask],
            c=[col],
            s=0.5,
            alpha=0.4,
            label=f"{c} ({counts[c]:,})",
            linewidths=0,
        )
    ax.set_aspect("equal")
    ax.set_title(f"Placement by country (top {top_n}; grey = other/unknown)")
    ax.set_facecolor("#111")
    ax.legend(loc="lower right", fontsize=7, markerscale=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_density(x: np.ndarray, y: np.ndarray, out: Path, bins: int = 800) -> None:
    fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
    h, xe, ye = np.histogram2d(x, y, bins=bins)
    im = ax.imshow(
        np.log1p(h.T),
        origin="lower",
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        cmap="magma",
        aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("log(1 + count) per bin")
    ax.set_title(f"Density heatmap ({len(x):,} figures, {bins}x{bins} bins)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_ring(
    x: np.ndarray,
    y: np.ndarray,
    years: list[int | None],
    countries: list[str | None],
    center_year: int,
    half_window: int,
    out: Path,
    top_n: int = 8,
) -> None:
    in_ring = np.array(
        [yy is not None and abs(yy - center_year) <= half_window for yy in years]
    )
    ring_countries = [countries[i] for i in range(len(countries)) if in_ring[i]]
    counts = Counter(c for c in ring_countries if c)
    top = [c for c, _ in counts.most_common(top_n)]
    palette = plt.get_cmap("tab10", top_n)
    color_lookup = {c: palette(i) for i, c in enumerate(top)}

    fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
    x_r = x[in_ring]
    y_r = y[in_ring]
    grey_mask = np.array([c not in color_lookup for c in ring_countries])
    ax.scatter(x_r[grey_mask], y_r[grey_mask], c="#666", s=3, alpha=0.25, linewidths=0)
    for c, col in color_lookup.items():
        mask = np.array([cc == c for cc in ring_countries])
        ax.scatter(
            x_r[mask], y_r[mask], c=[col], s=6, alpha=0.6,
            label=f"{c} ({counts[c]:,})", linewidths=0,
        )
    ax.set_aspect("equal")
    ax.set_title(
        f"Ring at year {center_year} ± {half_window} ({in_ring.sum():,} figures)"
    )
    ax.set_facecolor("#111")
    ax.legend(loc="lower right", fontsize=8, markerscale=3, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def select_landmarks(
    titles: list[str | None],
    sitelinks: list[int | None] | None,
    angles: np.ndarray,
    floor: int,
    top_n: int,
    mode: str,
    sectors: int,
    override_titles: list[str] | None,
) -> list[int]:
    """Major beacons: the GLOBAL notability source.

    Notability is its own axis, separate from placement, and it has two
    independent feeds. This is the global one: a figure is a recognizable
    milestone if its sitelink_count (cross-lingual coverage) clears an absolute
    floor, wherever it sits. That is what makes every Roman emperor a beacon.

    The earlier per-sector rank selection was the wrong shape: it promoted only
    the best-in-direction and pooled all eras, so Caesar and Augustus captured
    the Roman wedge while Nero, Caligula, Tiberius, Cato and Mark Antony, all
    with 90+ sitelinks, stayed plain books and Caesar read as lonely. An
    absolute floor sees their recognizability directly. It is Western/recent
    dense in raw COUNT, which is honest about where global recognizability
    actually concentrates; the local source (select_minor_landmarks) is what
    keeps sparse directions and the deep past from going dark.

    mode="floor" (default): sitelink_count >= floor.
    mode="global": top-N by sitelink (fixed budget, for comparison).
    mode="local": top-(N/sectors) per angular sector (the old behaviour).
    An explicit title list overrides everything, for ad-hoc comparison.
    """
    if override_titles:
        want = {s.lower() for s in override_titles}
        return [i for i, tt in enumerate(titles or []) if tt and tt.lower() in want]
    if sitelinks is None:
        return []
    sl = np.array([s if s is not None else 0 for s in sitelinks])
    if mode == "floor":
        return list(np.where(sl >= floor)[0])
    if mode == "global":
        return list(np.argsort(-sl)[:top_n])
    per = max(1, top_n // sectors)
    sec = (angles % (2 * math.pi)) // (2 * math.pi / sectors)
    hits: list[int] = []
    for s in range(sectors):
        members = np.where(sec == s)[0]
        if len(members) == 0:
            continue
        order = members[np.argsort(-sl[members])][:per]
        hits.extend(int(i) for i in order)
    return hits


def select_minor_landmarks(
    sitelinks: list[int | None] | None,
    radius: np.ndarray,
    angles: np.ndarray,
    sectors: int,
    rings: int,
    exclude: set[int],
) -> list[int]:
    """Minor beacons: the LOCAL notability source.

    The global floor (select_landmarks) catches everyone with absolute
    recognizability, but it goes dark in directions and eras where nobody clears
    the bar, the deep-past Persian, Egyptian, Mesopotamian, mid-Pacific cells.
    This feed takes the top figure by sitelink in each (angular sector x radial
    ring) cell that has no global major, so every populated patch has at least
    one target to walk toward (Djoser, Narmer, the Achaemenid kings) even when
    that figure is globally obscure. Noteworthy in the context of its position,
    which is exactly the point. The two feeds together are the notability axis.
    """
    if sitelinks is None:
        return []
    sl = np.array([s if s is not None else 0 for s in sitelinks])
    sec = (angles % (2 * math.pi)) // (2 * math.pi / sectors)
    rmax = radius.max() or 1.0
    ring = np.clip((radius / rmax * rings).astype(int), 0, rings - 1)
    hits: list[int] = []
    for s in range(sectors):
        for r in range(rings):
            cell = np.where((sec == s) & (ring == r))[0]
            if len(cell) == 0:
                continue
            best = cell[np.argmax(sl[cell])]
            if int(best) not in exclude:
                hits.append(int(best))
    return hits


def plot_landmarks(
    x: np.ndarray,
    y: np.ndarray,
    years: list[int | None],
    titles: list[str | None],
    sitelinks: list[int | None] | None,
    hits: list[int],
    out: Path,
    minor_hits: list[int] | None = None,
) -> None:
    """Whole corpus faint grey; major landmarks sized by sitelink, plus an
    optional dimmer tier of locally-significant minor landmarks.

    Dot area mirrors the book-thickness mechanic, so this is a direct preview:
    if the beacons spread across the disc, they lead the player outward in every
    direction; if they clump into one sector, most of the desert is irrelevant
    and the milestone mechanic fails.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=120)
    ax.scatter(x, y, c="#333", s=0.2, alpha=0.08, linewidths=0)

    if minor_hits:
        mx = np.array([x[i] for i in minor_hits])
        my = np.array([y[i] for i in minor_hits])
        ax.scatter(mx, my, c="#3a86c8", s=14, alpha=0.7, linewidths=0, zorder=2)

    hits = sorted(hits, key=lambda i: years[i] if years[i] is not None else 0)
    lx = np.array([x[i] for i in hits])
    ly = np.array([y[i] for i in hits])
    if sitelinks is not None:
        sl = np.array([sitelinks[i] or 0 for i in hits], dtype=float)
        sizes = 20 + (sl / sl.max()) * 160 if sl.max() else np.full(len(hits), 60)
    else:
        sizes = np.full(len(hits), 60)
    ax.scatter(lx, ly, c="#ff7a18", s=sizes, alpha=0.9, edgecolors="white",
               linewidths=0.5, zorder=3)
    # label only the most notable few dozen majors to keep the plot readable
    label_cut = np.sort(sizes)[-40] if len(sizes) > 40 else (sizes.min() if len(sizes) else 0)
    for i, sz in zip(hits, sizes):
        if sz < label_cut:
            continue
        yr = years[i]
        tag = f"{titles[i]} ({yr})" if yr is not None else titles[i]
        ax.annotate(tag, (x[i], y[i]), fontsize=7, color="#ffd9b0",
                    xytext=(4, 4), textcoords="offset points", zorder=4)
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0a0a")
    minor_n = len(minor_hits) if minor_hits else 0
    ax.set_title(
        f"Landmarks: {len(hits)} major (orange, size=sitelinks), "
        f"{minor_n} minor (blue, locally significant)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    if len(hits) >= 2:
        ang = np.arctan2(ly, lx)
        r = math.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        print(f"  landmark angular dispersion: {1 - r:.3f} "
              f"(0=one direction, ~1=spread); resultant={r:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline/cache/plots"))
    parser.add_argument("--ring-center", type=int, default=1700)
    parser.add_argument("--ring-window", type=int, default=25)
    parser.add_argument("--bins", type=int, default=800)
    parser.add_argument(
        "--top-landmarks", type=int, default=96,
        help="Budget for --landmark-mode global/local (by sitelink_count).",
    )
    parser.add_argument(
        "--landmark-floor", type=int, default=80,
        help="Sitelink floor for the global notability source "
             "(--landmark-mode floor): a figure is a major beacon if its "
             "sitelink_count clears this, wherever it sits.",
    )
    parser.add_argument(
        "--landmark-mode", choices=["floor", "global", "local"], default="floor",
        help="floor = absolute sitelink floor (global source, default); "
             "global = top-N corpus-wide; local = top per angular sector.",
    )
    parser.add_argument("--landmark-sectors", type=int, default=12)
    parser.add_argument(
        "--minor-landmarks", action="store_true",
        help="Overlay a dimmer tier of locally-significant beacons "
             "(top per sector x ring cell), to give sparse regions targets.",
    )
    parser.add_argument("--minor-sectors", type=int, default=24)
    parser.add_argument("--minor-rings", type=int, default=14)
    parser.add_argument(
        "--landmarks", type=Path, default=None,
        help="Optional JSON with a 'figures' title list to override the auto selection.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x, y, meta = load_xy(args.in_path)
    print(f"loaded {len(x):,} positions from {args.in_path}")

    if meta["death_year"] is not None:
        plot_era(x, y, meta["death_year"], args.out_dir / "era.png")
        print(f"  wrote {args.out_dir / 'era.png'}")
    if meta["country_qid"] is not None:
        plot_country(x, y, meta["country_qid"], args.out_dir / "country.png")
        print(f"  wrote {args.out_dir / 'country.png'}")
    plot_density(x, y, args.out_dir / "density.png", bins=args.bins)
    print(f"  wrote {args.out_dir / 'density.png'}")
    if meta["death_year"] is not None and meta["country_qid"] is not None:
        plot_ring(
            x, y, meta["death_year"], meta["country_qid"],
            args.ring_center, args.ring_window,
            args.out_dir / f"ring_{args.ring_center}.png",
        )
        print(f"  wrote {args.out_dir / f'ring_{args.ring_center}.png'}")
    if meta["title"] is not None:
        override = None
        if args.landmarks is not None:
            import json
            override = json.loads(args.landmarks.read_text()).get("figures", [])
        angles = np.arctan2(y, x)
        hits = select_landmarks(
            meta["title"], meta["sitelink_count"], angles,
            args.landmark_floor, args.top_landmarks, args.landmark_mode,
            args.landmark_sectors, override,
        )
        minor = None
        if args.minor_landmarks:
            radius = np.hypot(x, y)
            minor = select_minor_landmarks(
                meta["sitelink_count"], radius, angles,
                args.minor_sectors, args.minor_rings, set(hits),
            )
            print(f"  minor landmarks: {len(minor)} locally-significant beacons")
        plot_landmarks(
            x, y, meta["death_year"], meta["title"], meta["sitelink_count"],
            hits, args.out_dir / "landmarks.png", minor_hits=minor,
        )
        print(f"  wrote {args.out_dir / 'landmarks.png'}")


if __name__ == "__main__":
    main()
