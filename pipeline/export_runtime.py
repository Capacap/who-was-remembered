"""
Export a compact binary of book positions for the runtime tracer.

Reads placement.parquet and writes runtime/public/positions.bin: just the
fields the first visualization needs (position, landmark tier, geo source),
packed little-endian as

    uint32   count N
    float32  x[N]
    float32  y[N]
    uint8    tier[N]    (0 ordinary, 1 minor, 2 major)
    uint8    geo[N]     (0 birth, 1 death, 2 citizenship, 3 gazetteer, 4 residue)

geo lets the renderer express placement confidence: 0-1 are real coordinates,
2-3 are sampled from a country (coarse), 4 is genuinely place-less and should
read as adrift rather than confidently positioned. ~10 bytes per figure, ~4 MB
for the full corpus, fetched as one ArrayBuffer.
This is deliberately not the shipping format: no titles, no text, no tiling.
It exists to get the field on screen so we can judge density, the landing pad,
the antiquity edge, and whether landmarks read as beacons. Regenerate after any
stage 6 change:

    uv run pipeline/export_runtime.py

Also writes runtime/public/teleporters.json: the 26 fast-travel monuments as a
small array of {label, x, y, era, seat, n}. Tiny enough to ship as plain JSON
rather than packed into the binary, and the labels/era are wanted for UI later.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PLACEMENT_PATH = ROOT / "cache" / "placement.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_PATH = ROOT.parent / "runtime" / "public" / "positions.bin"
TELEPORTERS_OUT = ROOT.parent / "runtime" / "public" / "teleporters.json"

TIER_CODE = {"ordinary": 0, "minor": 1, "major": 2}
GEO_CODE = {"birth": 0, "death": 1, "citizenship": 2, "gazetteer": 3}  # None -> 4 residue


def export_teleporters() -> int:
    t = pq.read_table(
        TELEPORTERS_PATH, columns=["label", "x", "y", "era", "seat_title", "n_members"]
    )
    rows = [
        {
            "label": r["label"],
            "x": round(r["x"], 1),
            "y": round(r["y"], 1),
            "era": r["era"],
            "seat": r["seat_title"],
            "n": r["n_members"],
        }
        for r in t.to_pylist()
    ]
    TELEPORTERS_OUT.parent.mkdir(parents=True, exist_ok=True)
    TELEPORTERS_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=0))
    return len(rows)


def main() -> None:
    t = pq.read_table(PLACEMENT_PATH, columns=["x", "y", "landmark_tier", "geo_source"])
    n = t.num_rows

    x = np.asarray(t["x"].to_pylist(), dtype=np.float32)
    y = np.asarray(t["y"].to_pylist(), dtype=np.float32)
    tier = np.array(
        [TIER_CODE.get(v, 0) for v in t["landmark_tier"].to_pylist()], dtype=np.uint8
    )
    geo = np.array(
        [GEO_CODE.get(v, 4) for v in t["geo_source"].to_pylist()], dtype=np.uint8
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(tier.tobytes())
        f.write(geo.tobytes())

    size_mb = OUT_PATH.stat().st_size / 1e6
    majors = int((tier == 2).sum())
    minors = int((tier == 1).sum())
    residue = int((geo == 4).sum())
    print(f"wrote {n:,} figures ({majors:,} major, {minors:,} minor) -> {OUT_PATH}")
    print(f"  {residue:,} residue (adrift) · {size_mb:.2f} MB")

    n_tp = export_teleporters()
    print(f"wrote {n_tp} teleporters -> {TELEPORTERS_OUT}")


if __name__ == "__main__":
    main()
