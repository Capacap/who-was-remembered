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
    }
    return x, y, meta


def plot_era(x: np.ndarray, y: np.ndarray, years: list[int | None], out: Path) -> None:
    valid = np.array([y is not None for y in years])
    yrs = np.array([yy if yy is not None else 0 for yy in years], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=110)
    sc = ax.scatter(
        x[valid],
        y[valid],
        c=yrs[valid],
        s=0.4,
        alpha=0.25,
        cmap="viridis",
        linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("death_year")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("pipeline/cache/plots"))
    parser.add_argument("--ring-center", type=int, default=1700)
    parser.add_argument("--ring-window", type=int, default=25)
    parser.add_argument("--bins", type=int, default=800)
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


if __name__ == "__main__":
    main()
