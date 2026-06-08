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
    uint8    scale[N]   book_scale quantized over [BOOK_SCALE_MIN, BOOK_SCALE_MAX]

geo lets the renderer express placement confidence: 0-1 are real coordinates,
2-3 are sampled from a country (coarse), 4 is genuinely place-less and should
read as adrift rather than confidently positioned. lon is the PRE-jitter angle
(stage6 base_angle) quantized to a byte: the renderer hues each book by its home
region, not its scattered position, and desaturates residue (geo==4) whose angle
is only a hash. scale is the per-book size (article length); the renderer maps
the byte back through the bounds in world.json and multiplies its baked book
length, so size is one continuous axis and tier no longer touches it. ~12 bytes
per figure, ~7 MB for the full corpus, one ArrayBuffer.
This is deliberately not the shipping format: no titles, no text, no tiling.
It exists to get the field on screen so we can judge density, the landing pad,
the antiquity edge, and whether landmarks read as beacons. Regenerate after any
stage 6 change:

    uv run pipeline/export_runtime.py

Also writes runtime/public/teleporters.json: the 26 fast-travel monuments as a
small array of {label, x, y, era, seat, n}. Tiny enough to ship as plain JSON
rather than packed into the binary, and the labels/era are wanted for UI later.

And runtime/public/spawn_anchors.json: a curated set of recent landmark figures
the opening vista is anchored to. The runtime spawns the player at the inner rim
on one anchor's bearing, facing outward, so the first thing seen is a recognisable
name a few steps ahead rather than the empty plaza. Anchored to QIDs (not raw
coordinates) so the set survives any re-layout, exactly like the teleporters.
"""

from __future__ import annotations

import gzip
import io
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from stage6_place import (
    BOOK_SCALE_MAX,
    BOOK_SCALE_MIN,
    R_INNER,
    R_MAX,
    RADIUS_ALPHA,
    TIME_SPAN,
)

ROOT = Path(__file__).resolve().parent
# layout.parquet is Stage 8's topology-adjusted placement (relaxed spacing,
# teleporter clearings). placement.parquet is the pristine ground truth; the
# runtime wants what the player actually walks through, so it reads the layout.
PLACEMENT_PATH = ROOT / "cache" / "layout.parquet"
TELEPORTERS_PATH = ROOT / "cache" / "teleporters.parquet"
OUT_PATH = ROOT.parent / "runtime" / "public" / "positions.bin"
TELEPORTERS_OUT = ROOT.parent / "runtime" / "public" / "teleporters.json"
SPAWN_ANCHORS_OUT = ROOT.parent / "runtime" / "public" / "spawn_anchors.json"
META_OUT = ROOT.parent / "runtime" / "public" / "meta.gz.bin"
WORLD_OUT = ROOT.parent / "runtime" / "public" / "world.json"

# Curated opening-vista anchors: globally recognisable figures who died before
# 2000 (so they sit near the rim), spread across the angular sectors so the
# opening varies by session. The list is intentionally Western-heavy -- that
# mirrors both the corpus bias the piece is about and the likely audience -- with
# the rarer non-Western names kept in as the quieter draw. No dictators: the
# tier's top-by-sitelink slots are full of them, so this is hand-picked for tone,
# favouring artists, writers, scientists and humane icons. Anchored by QID; the
# runtime resolves each to its current layout position.
SPAWN_ANCHOR_QIDS = [
    # Americas (sectors 2-4)
    "Q4616",   # Marilyn Monroe
    "Q8704",   # Walt Disney
    "Q7245",   # Mark Twain
    "Q1779",   # Louis Armstrong
    "Q5588",   # Frida Kahlo
    "Q23434",  # Ernest Hemingway
    "Q303",    # Elvis Presley
    "Q409",    # Bob Marley
    "Q5603",   # Andy Warhol
    "Q8027",   # Martin Luther King Jr.
    "Q160534", # Jack Kerouac
    "Q909",    # Jorge Luis Borges
    # Europe (sectors 5-6)
    "Q882",    # Charlie Chaplin
    "Q937",    # Albert Einstein
    "Q5577",   # Salvador Dalí
    "Q35064",  # Agatha Christie
    "Q9036",   # Nikola Tesla
    "Q30875",  # Oscar Wilde
    "Q7251",   # Alan Turing
    "Q5593",   # Pablo Picasso
    "Q905",    # Franz Kafka
    "Q7186",   # Marie Curie
    # East of there (sectors 7-10)
    "Q892",    # J. R. R. Tolkien
    "Q5685",   # Anton Chekhov
    "Q1001",   # Mahatma Gandhi
    "Q8873",   # Satyajit Ray
    "Q3335",   # George Orwell
    "Q7241",   # Rabindranath Tagore
    "Q8006",   # Akira Kurosawa
]

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


def export_spawn_anchors() -> int:
    """Resolve the curated QID list to its current layout positions and write
    spawn_anchors.json. Order follows SPAWN_ANCHOR_QIDS; a QID that no longer
    survives the filter is dropped with a warning rather than failing the export
    (the corpus can shift under a re-run). The runtime derives bearing and spawn
    radius from x/y, so only position + title (for the dev cycle HUD) ship.
    """
    t = pq.read_table(PLACEMENT_PATH, columns=["qid", "title", "x", "y"])
    by_qid = {r["qid"]: r for r in t.to_pylist()}
    rows = []
    for qid in SPAWN_ANCHOR_QIDS:
        r = by_qid.get(qid)
        if r is None:
            print(f"  WARNING: spawn anchor {qid} not in layout -- dropped")
            continue
        rows.append(
            {
                "qid": qid,
                "title": r["title"],
                "x": round(r["x"], 1),
                "y": round(r["y"], 1),
            }
        )
    SPAWN_ANCHORS_OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=0))
    return len(rows)


def export_meta() -> int:
    """Per-figure label data for the look-at glance and inspect overlay, in the
    SAME row order as positions.bin so the renderer's instanceId indexes it
    directly. Packed binary, not JSON: the string blobs are decoded lazily (only
    the book under the crosshair), so the payload never hits a JSON.parse. The
    file is gzipped at rest (the name/desc blobs are ~3x compressible text):
    ~34 MB raw -> ~12 MB on disk and over the wire, which both speeds the cold
    load on every host and keeps the artifact under Cloudflare Pages' 25 MiB
    per-file cap. The runtime inflates it with DecompressionStream. Named
    `meta.gz.bin`, not `meta.bin.gz`: a `.gz` extension makes many hosts send
    Content-Encoding: gzip (browser auto-inflates -> the runtime would then
    double-inflate); ending in `.bin` keeps it an opaque octet-stream so the
    single inflate always happens client-side, deterministically. The
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

    buf = io.BytesIO()
    buf.write(struct.pack("<III", n, len(name_blob), len(desc_blob)))
    buf.write(name_off.tobytes())
    buf.write(desc_off.tobytes())
    buf.write(birth.tobytes())
    buf.write(death.tobytes())
    buf.write(name_blob)
    buf.write(desc_blob)
    # mtime=0 keeps the gzip header byte-stable across runs, so an unchanged
    # corpus produces an identical artifact (clean diffs / cache hits).
    with gzip.GzipFile(META_OUT, "wb", compresslevel=9, mtime=0) as f:
        f.write(buf.getvalue())
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
                "BOOK_SCALE_MIN": BOOK_SCALE_MIN,
                "BOOK_SCALE_MAX": BOOK_SCALE_MAX,
            }
        )
    )


def main() -> None:
    t = pq.read_table(
        PLACEMENT_PATH,
        columns=["x", "y", "landmark_tier", "geo_source", "base_angle", "book_scale"],
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
    # book_scale -> byte over its known bounds; the runtime inverts with the same
    # bounds shipped in world.json. Already in-range by construction; clip is safety.
    bs = np.asarray(t["book_scale"].to_pylist(), dtype=np.float64)
    frac = (bs - BOOK_SCALE_MIN) / (BOOK_SCALE_MAX - BOOK_SCALE_MIN)
    scale = np.clip(np.round(frac * 255.0), 0, 255).astype(np.uint8)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(tier.tobytes())
        f.write(geo.tobytes())
        f.write(lon.tobytes())
        f.write(scale.tobytes())

    size_mb = OUT_PATH.stat().st_size / 1e6
    majors = int((tier == 2).sum())
    minors = int((tier == 1).sum())
    residue = int((geo == 4).sum())
    print(f"wrote {n:,} figures ({majors:,} major, {minors:,} minor) -> {OUT_PATH}")
    print(f"  {residue:,} residue (adrift) · {size_mb:.2f} MB")

    n_tp = export_teleporters()
    print(f"wrote {n_tp} teleporters -> {TELEPORTERS_OUT}")

    n_sa = export_spawn_anchors()
    print(f"wrote {n_sa} spawn anchors -> {SPAWN_ANCHORS_OUT}")

    export_meta()
    meta_mb = META_OUT.stat().st_size / 1e6
    print(f"wrote {n:,} meta records · {meta_mb:.2f} MB -> {META_OUT}")

    export_world()
    print(f"wrote world constants -> {WORLD_OUT}")


if __name__ == "__main__":
    main()
