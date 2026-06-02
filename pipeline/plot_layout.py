"""
Plot the final layout with the teleporter network marked.

Unlike inspect_placement.py (the placement-tuning bench, which renders single
diagnostic views over one parquet), this joins the two finished artifacts the
runtime actually walks on: the topology-adjusted corpus (layout.parquet, Stage
8's relaxed-and-cleared placement) and the fast-travel anchors
(teleporters.parquet). It answers the question those stages can't on their own:
does each monument sit on populated ground, and do they spread across the disc
or pile into one wedge? (Stage 8's sub-unit spacing nudges and the small
teleporter plazas are local effects, invisible at full-disc scale; this view is
for the macro structure, not for judging spacing.)

The whole corpus is drawn faint and colored by era (recent bright, ancient
dark), so the time gradient and the honest voids both read. On top of it the
teleporters are gold stars labelled with their place name; their radius already
encodes their era, so the faint concentric year rings give a quantitative read
of how far back each monument sits. The player spawns at the origin (year 2000)
and surveys the field across the empty R_INNER plaza.

The radius->year mapping (and so the ring placement) is imported from
stage6_place, not duplicated, so retuning the world geometry keeps these
rings honest.

Run:
    uv run pipeline/plot_layout.py
    uv run pipeline/plot_layout.py --out /tmp/layout.png --dpi 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from stage6_place import R_INNER, R_MAX, RADIUS_ALPHA, TIME_SPAN

ROOT = Path(__file__).resolve().parent
PLACEMENT_PATH = ROOT / "cache" / "layout.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
DECORATIONS_PATH = ROOT.parent / "runtime" / "public" / "decorations.json"
OUT_PATH = ROOT / "cache" / "plots" / "layout.png"

# Stage 10's three head variants, coloured distinctly so the plot shows both the
# spatial spread (do they really only collect in the outer void?) and the variant
# mix. Magenta-ish so they pop off the viridis era ramp.
DECO_COLORS = ["#ff5fa2", "#ffa83f", "#7cf0ff"]

# Years to attempt a guide ring for. Drawn only where the era curve has not
# yet clipped to R_MAX (year - TIME_SPAN), i.e. inside the linear regime;
# the pre-edge tail is a date-uncertainty frontier, not a clean ring.
RING_YEARS = [2000, 1900, 1800, 1700, 1500, 1300, 1000, 500, 1, -500]


def year_to_radius(year: int) -> float:
    """Base radius for a death year, mirroring stage6's era curve (no jitter)."""
    t = min(max((2000.0 - year) / TIME_SPAN, 0.0), 1.0)
    return R_INNER + (R_MAX - R_INNER) * (t ** RADIUS_ALPHA)


def year_label(year: int) -> str:
    if year > 0:
        return f"{year} CE" if year < 1000 else str(year)
    return f"{abs(year)} BCE"


def load_corpus(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["x", "y", "death_year"])
    x = np.asarray(table["x"].to_pylist(), dtype=np.float64)
    y = np.asarray(table["y"].to_pylist(), dtype=np.float64)
    years = np.array(
        [yy if yy is not None else 2000 for yy in table["death_year"].to_pylist()],
        dtype=np.float64,
    )
    return x, y, years


def load_teleporters(path: Path) -> list[dict]:
    table = pq.read_table(
        path, columns=["label", "x", "y", "era", "n_members", "seat_title"]
    )
    return table.to_pylist()


def load_decorations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def plot(
    cx: np.ndarray,
    cy: np.ndarray,
    cyears: np.ndarray,
    anchors: list[dict],
    decos: list[dict],
    out: Path,
    dpi: int,
) -> None:
    # Corpus colored by era. log(years-before-2000) gives the recent dense
    # cluster contrast while still ordering antiquity (same idiom as plot_era).
    age = np.log1p(np.clip(2000 - cyears, 0, None))

    # Wide figure: the disc on the left, a chronological legend in the right
    # margin. A full-disc view can't label 30 anchors inline (the recent
    # European cluster collapses into a knot near the origin), so the stars
    # carry only a number and the legend maps number -> place.
    fig, ax = plt.subplots(figsize=(19, 15), dpi=dpi)
    fig.subplots_adjust(left=0.03, right=0.76, top=0.95, bottom=0.03)
    ax.set_facecolor("#0a0a0a")
    ax.scatter(cx, cy, c=age, s=0.5, alpha=0.16, cmap="viridis_r", linewidths=0,
               zorder=1)

    # Faint guide rings labelled by year. Skip any that have clipped to the
    # R_MAX edge (everything at/older than year - TIME_SPAN sits there).
    edge_year = 2000 - TIME_SPAN
    theta = np.linspace(0, 2 * np.pi, 400)
    for yr in RING_YEARS:
        if yr <= edge_year:
            continue
        r = year_to_radius(yr)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="#555", lw=0.5,
                ls=(0, (4, 4)), alpha=0.5, zorder=2)
        ax.text(0, r, year_label(yr), color="#9a9a9a", fontsize=7, ha="center",
                va="bottom", zorder=2,
                bbox=dict(facecolor="#0a0a0a", edgecolor="none", alpha=0.6, pad=1))

    # Landing pad rim (year 2000, where the player spawns) and the origin.
    ax.plot(R_INNER * np.cos(theta), R_INNER * np.sin(theta), color="#c8a23a",
            lw=0.6, alpha=0.5, zorder=2)
    ax.scatter([0], [0], marker="+", c="#e8e8e8", s=80, linewidths=1.0, zorder=5)

    # Teleporters numbered from the center outward (1 = most recent, last =
    # deepest past), so the numbering itself reads as a walk back in time and
    # the legend is chronological. Radius encodes era, so sort by it.
    anchors = sorted(anchors, key=lambda a: a["x"] ** 2 + a["y"] ** 2)
    tx = np.array([a["x"] for a in anchors])
    ty = np.array([a["y"] for a in anchors])
    members = np.array([a["n_members"] for a in anchors], dtype=float)
    sizes = 90 + (members - members.min()) / max(np.ptp(members), 1) * 170
    ax.scatter(tx, ty, marker="*", s=sizes, c="#ffcf40", edgecolors="white",
               linewidths=0.7, alpha=0.95, zorder=6)
    stroke = [pe.withStroke(linewidth=1.6, foreground="#0a0a0a")]
    for i, a in enumerate(anchors, start=1):
        ax.annotate(
            str(i), (a["x"], a["y"]), fontsize=8, color="#ffffff", weight="bold",
            ha="center", va="center", xytext=(8, 6), textcoords="offset points",
            zorder=7, path_effects=stroke,
        )

    # Decoration heads (Stage 10). Plotted last so they sit above the corpus, with
    # the marker sized by each head's scale jitter, coloured by variant. This is the
    # read the user wants: how sparse the scatter is and whether they really avoid
    # the dense core and gather in the outer void.
    if decos:
        dx = np.array([d["x"] for d in decos])
        dy = np.array([d["y"] for d in decos])
        dv = np.array([d["v"] for d in decos])
        ds = np.array([d["s"] for d in decos])
        dcol = [DECO_COLORS[v % len(DECO_COLORS)] for v in dv]
        ax.scatter(dx, dy, s=18 + ds * 26, c=dcol, marker="o",
                   edgecolors="#0a0a0a", linewidths=0.5, alpha=0.95, zorder=8)

    lim = float(np.max(np.hypot(cx, cy))) * 1.02
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(
        f"Final layout: {len(cx):,} figures (color = era), "
        f"{len(anchors)} teleporters (gold stars), {len(decos)} heads (dots). "
        f"Angle = longitude, radius = time; origin = year 2000."
    )

    # Chronological legend in the right margin: number -> place (era).
    fig.text(0.775, 0.95, "teleporters (recent → ancient)", color="#fff1cf",
             fontsize=11, weight="bold", va="top")
    for i, a in enumerate(anchors, start=1):
        y = 0.915 - (i - 1) * 0.0295
        fig.text(0.775, y, f"{i:>2}.  {a['label']}  ({a['era']})",
                 color="#d8d8d8", fontsize=8.5, va="top")

    fig.savefig(out, dpi=dpi, facecolor="#0a0a0a")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", type=Path, default=PLACEMENT_PATH)
    parser.add_argument("--teleporters", type=Path, default=TELEPORTERS_PATH)
    parser.add_argument("--decorations", type=Path, default=DECORATIONS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()

    if not args.placement.exists():
        raise SystemExit(f"placement parquet not found: {args.placement}")
    if not args.teleporters.exists():
        raise SystemExit(f"teleporters parquet not found: {args.teleporters}")

    cx, cy, cyears = load_corpus(args.placement)
    anchors = load_teleporters(args.teleporters)
    decos = load_decorations(args.decorations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(cx, cy, cyears, anchors, decos, args.out, args.dpi)
    print(
        f"plotted {len(cx):,} figures + {len(anchors)} teleporters "
        f"+ {len(decos)} heads -> {args.out}"
    )


if __name__ == "__main__":
    main()
