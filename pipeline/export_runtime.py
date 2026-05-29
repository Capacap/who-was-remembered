"""
Export a compact binary of book positions for the runtime tracer.

Reads placement.parquet and writes runtime/public/positions.bin: just the
fields the first visualization needs (position and landmark tier), packed
little-endian as

    uint32   count N
    float32  x[N]
    float32  y[N]
    uint8    tier[N]    (0 ordinary, 1 minor, 2 major)

~9 bytes per figure, ~3.8 MB for the full corpus, fetched as one ArrayBuffer.
This is deliberately not the shipping format: no titles, no text, no tiling.
It exists to get the field on screen so we can judge density, the landing pad,
the antiquity edge, and whether landmarks read as beacons. Regenerate after any
stage 6 change:

    uv run pipeline/export_runtime.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PLACEMENT_PATH = ROOT / "cache" / "placement.parquet"
OUT_PATH = ROOT.parent / "runtime" / "public" / "positions.bin"

TIER_CODE = {"ordinary": 0, "minor": 1, "major": 2}


def main() -> None:
    t = pq.read_table(PLACEMENT_PATH, columns=["x", "y", "landmark_tier"])
    n = t.num_rows

    x = np.asarray(t["x"].to_pylist(), dtype=np.float32)
    y = np.asarray(t["y"].to_pylist(), dtype=np.float32)
    tier = np.array(
        [TIER_CODE.get(v, 0) for v in t["landmark_tier"].to_pylist()], dtype=np.uint8
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(tier.tobytes())

    size_mb = OUT_PATH.stat().st_size / 1e6
    majors = int((tier == 2).sum())
    minors = int((tier == 1).sum())
    print(f"wrote {n:,} figures ({majors:,} major, {minors:,} minor) -> {OUT_PATH}")
    print(f"  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
