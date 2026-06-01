"""
Export a compact binary of book positions for the runtime tracer.

Reads layout.parquet (Stage 8's topology-adjusted placement) and writes
runtime/public/positions.bin: just the fields the first visualization needs
(position, landmark tier, geo source), packed little-endian as

    uint32   count N
    float32  x[N]
    float32  y[N]
    uint8    tier[N]    (0 ordinary, 1 minor, 2 major)
    uint8    geo[N]     (0 birth, 1 death, 2 citizenship, 3 gazetteer, 4 residue)
    uint8    lon[N]     canonical longitude angle, 0..255 around the disc

geo lets the renderer express placement confidence: 0-1 are real coordinates,
2-3 are sampled from a country (coarse), 4 is genuinely place-less and should
read as adrift rather than confidently positioned. lon is the PRE-jitter angle
(stage6 base_angle) quantized to a byte: the renderer hues each book by its home
region, not its scattered position, and desaturates residue (geo==4) whose angle
is only a hash. ~11 bytes per figure, ~6 MB for the full corpus, one ArrayBuffer.
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

from stage6_place import R_INNER, R_MAX, RADIUS_ALPHA, TIME_SPAN

ROOT = Path(__file__).resolve().parent
# layout.parquet is Stage 8's topology-adjusted placement (relaxed spacing,
# teleporter clearings). placement.parquet is the pristine ground truth; the
# runtime wants what the player actually walks through, so it reads the layout.
PLACEMENT_PATH = ROOT / "cache" / "layout.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_PATH = ROOT.parent / "runtime" / "public" / "positions.bin"
TELEPORTERS_OUT = ROOT.parent / "runtime" / "public" / "teleporters.json"
META_OUT = ROOT.parent / "runtime" / "public" / "meta.bin"
WORLD_OUT = ROOT.parent / "runtime" / "public" / "world.json"

REF_YEAR = 2000  # the spawn year at radius R_INNER; matches stage6's (2000 - death)

TIER_CODE = {"ordinary": 0, "minor": 1, "major": 2}
GEO_CODE = {"birth": 0, "death": 1, "citizenship": 2, "gazetteer": 3}  # None -> 4 residue

YEAR_MISSING = -32768  # int16 sentinel; below any real year in the corpus


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


def export_meta() -> int:
    """Per-figure label data for the look-at glance and inspect overlay, in the
    SAME row order as positions.bin so the renderer's instanceId indexes it
    directly. Packed binary, not JSON: the string blobs are decoded lazily (only
    the book under the crosshair), so the ~30 MB never hits a JSON.parse. The
    Wikipedia URL is derived client-side from the title. lead_text is omitted on
    purpose; it is large and belongs to per-tile delivery, not this prototype.

    Layout (little-endian), ordered so every typed section is naturally aligned
    (uint32 sections on 4-byte bounds, int16 on 2-byte) and the byte blobs trail:

        uint32  N
        uint32  name_blob_len
        uint32  desc_blob_len
        uint32  name_off[N+1]   byte offsets into name_blob
        uint32  desc_off[N+1]   byte offsets into desc_blob
        int16   birth_year[N]   (YEAR_MISSING if unknown)
        int16   death_year[N]
        bytes   name_blob       concatenated UTF-8 titles
        bytes   desc_blob       concatenated UTF-8 descriptions
    """
    t = pq.read_table(
        PLACEMENT_PATH, columns=["title", "description", "birth_year", "death_year"]
    )
    n = t.num_rows
    titles = t["title"].to_pylist()
    descs = t["description"].to_pylist()

    def to_year(v: object) -> int:
        return YEAR_MISSING if v is None else int(v)

    birth = np.array([to_year(v) for v in t["birth_year"].to_pylist()], dtype=np.int16)
    death = np.array([to_year(v) for v in t["death_year"].to_pylist()], dtype=np.int16)

    def pack(strs: list) -> tuple[np.ndarray, bytes]:
        blob = bytearray()
        offs = np.zeros(n + 1, dtype=np.uint32)
        for i, s in enumerate(strs):
            if s:
                blob += s.encode("utf-8")
            offs[i + 1] = len(blob)
        return offs, bytes(blob)

    name_off, name_blob = pack(titles)
    desc_off, desc_blob = pack(descs)

    with open(META_OUT, "wb") as f:
        f.write(struct.pack("<III", n, len(name_blob), len(desc_blob)))
        f.write(name_off.tobytes())
        f.write(desc_off.tobytes())
        f.write(birth.tobytes())
        f.write(death.tobytes())
        f.write(name_blob)
        f.write(desc_blob)
    return n


def export_world() -> None:
    """The placement constants the runtime needs to invert radius -> era for the
    compass readouts. Imported from stage6_place so they cannot drift from the
    actual placement; the runtime should never hard-code them.
    """
    WORLD_OUT.write_text(
        json.dumps(
            {
                "R_INNER": R_INNER,
                "R_MAX": R_MAX,
                "TIME_SPAN": TIME_SPAN,
                "RADIUS_ALPHA": RADIUS_ALPHA,
                "REF_YEAR": REF_YEAR,
            }
        )
    )


def main() -> None:
    t = pq.read_table(
        PLACEMENT_PATH, columns=["x", "y", "landmark_tier", "geo_source", "base_angle"]
    )
    n = t.num_rows

    x = np.asarray(t["x"].to_pylist(), dtype=np.float32)
    y = np.asarray(t["y"].to_pylist(), dtype=np.float32)
    tier = np.array(
        [TIER_CODE.get(v, 0) for v in t["landmark_tier"].to_pylist()], dtype=np.uint8
    )
    geo = np.array(
        [GEO_CODE.get(v, 4) for v in t["geo_source"].to_pylist()], dtype=np.uint8
    )
    # canonical pre-jitter angle -> a byte around the colour wheel (0..255). The
    # renderer hues each book by this, so scatter mixes home-region colours rather
    # than painting a clean gradient (see base_angle in stage6).
    base = np.asarray(t["base_angle"].to_pylist(), dtype=np.float64)
    lon = np.mod(np.round(base / (2 * np.pi) * 256.0), 256.0).astype(np.uint8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(tier.tobytes())
        f.write(geo.tobytes())
        f.write(lon.tobytes())

    size_mb = OUT_PATH.stat().st_size / 1e6
    majors = int((tier == 2).sum())
    minors = int((tier == 1).sum())
    residue = int((geo == 4).sum())
    print(f"wrote {n:,} figures ({majors:,} major, {minors:,} minor) -> {OUT_PATH}")
    print(f"  {residue:,} residue (adrift) · {size_mb:.2f} MB")

    n_tp = export_teleporters()
    print(f"wrote {n_tp} teleporters -> {TELEPORTERS_OUT}")

    export_meta()
    meta_mb = META_OUT.stat().st_size / 1e6
    print(f"wrote {n:,} meta records · {meta_mb:.2f} MB -> {META_OUT}")

    export_world()
    print(f"wrote world constants -> {WORLD_OUT}")


if __name__ == "__main__":
    main()
