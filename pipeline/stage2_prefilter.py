"""
Stage 2: recency and stub pre-filter.

Reads the Stage 1 parquet, applies the death-year cut and a conservative
stub pre-filter, and writes the surviving rows to a new parquet for the
Stage 3 article fetch to consume.

This is NOT a notability ranking. The cuts are:

- ``death_year < 2000`` (exclusive). The player spawn point sits at year
  2000; post-2000 figures are also where still-contested contemporary
  politics concentrate. The cut keeps every WWII-era figure and is dialable
  via --death-year-max.
- Wikidata description is non-null. This is the only Stage 1 signal that is
  a clean stub indicator rather than a notability proxy in disguise; rows
  with no en description are the entries Wikidata editors haven't even
  written a one-liner for. About 1% of Stage 1 rows. Disable with
  --keep-missing-description.

Article-quality filtering proper happens in Stage 3 once we have the actual
Wikipedia article content (word count, lead completeness). The point of this
stage is just to make the Stage 3 fetch a bit cheaper without prejudging
who deserves to be in the piece.

Run:
    uv run pipeline/stage2_prefilter.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "cache" / "wikidata_figures.parquet"
OUT_PATH = ROOT / "cache" / "wikidata_figures_prefiltered.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=IN_PATH,
        help="Stage 1 output parquet.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="Stage 2 output parquet.",
    )
    parser.add_argument(
        "--death-year-max",
        type=int,
        default=2000,
        help="Exclusive upper bound on death_year. Default 2000.",
    )
    parser.add_argument(
        "--keep-missing-description",
        action="store_true",
        help="Keep rows whose Wikidata description is null. Off by default.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Stage 1 output not found: {args.in_path}")

    start = time.perf_counter()
    table = pq.read_table(args.in_path)
    total = table.num_rows

    death_mask = pc.less(table["death_year"], args.death_year_max)
    recency_kept = pc.sum(pc.cast(death_mask, pa.int64())).as_py()

    desc_valid = pc.is_valid(table["description"])
    stub_dropped_after_recency = pc.sum(
        pc.cast(pc.and_(death_mask, pc.invert(desc_valid)), pa.int64())
    ).as_py()

    if args.keep_missing_description:
        mask = death_mask
    else:
        mask = pc.and_(death_mask, desc_valid)

    filtered = table.filter(mask)
    kept = filtered.num_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, args.out, compression="zstd")

    elapsed = time.perf_counter() - start
    recency_dropped = total - recency_kept
    pct = (kept / total * 100) if total else 0.0
    print(f"read {total:,} rows from {args.in_path.name} in {elapsed:.1f}s")
    print(
        f"  death_year >= {args.death_year_max}: -{recency_dropped:,} "
        f"({recency_dropped / total * 100:.1f}%)"
    )
    if not args.keep_missing_description:
        print(
            f"  missing description (after recency cut): "
            f"-{stub_dropped_after_recency:,} "
            f"({stub_dropped_after_recency / total * 100:.2f}%)"
        )
    print(f"kept {kept:,} rows ({pct:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
