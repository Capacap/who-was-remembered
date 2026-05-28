"""
Stage 5: Wikidata place extraction.

Second pass over the Wikidata dump, this time pulling coordinates for
the place QIDs referenced by Stage 4 figures' birth_place_qid and
death_place_qid. Output is a small parquet keyed by place QID and used
by Stage 6 to give each figure a geographic anchor for angular
placement (alongside the link graph it already has).

Most entities in the 95 GB dump are not in our place set, so the inner
loop sniffs the QID out of the line bytes before paying for full JSON
parsing. Lines whose QID is not in the needed set get skipped without
ever hitting orjson, which makes this pass much faster than Stage 1
even though the same file is being streamed.

Captured per place:

- qid, label (English),
- coordinate location (P625) → lat, lon when present,
- instance_of (P31) for category context,
- located in administrative entity (P131), country (P17), for an
  eventual parent-lookup fallback if direct coordinate coverage is poor.

Run:
    uv run pipeline/stage5_extract_places.py
"""

from __future__ import annotations

import argparse
import bz2
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DUMP_PATH = ROOT / "data" / "latest-all.json.bz2"
IN_PATH = ROOT / "cache" / "wikidata_figures_quality.parquet"
OUT_PATH = ROOT / "cache" / "places.parquet"

BATCH_SIZE = 5_000

ITEM_PREFIX = b'{"type":"item","id":"'

SCHEMA = pa.schema(
    [
        ("qid", pa.string()),
        ("label", pa.string()),
        ("lat", pa.float64()),
        ("lon", pa.float64()),
        ("instance_of_qids", pa.list_(pa.string())),
        ("admin_parent_qid", pa.string()),
        ("country_qid", pa.string()),
    ]
)


@contextmanager
def open_dump(path: Path) -> Iterator[IO[bytes]]:
    """Stream the bz2 dump with a compressed-bytes progress bar.

    Same lbzip2-or-fallback pattern as Stage 1; duplicated rather than
    extracted to keep the two streaming passes independently runnable.
    """
    decompressor = shutil.which("lbzip2") or shutil.which("pbzip2")
    if decompressor:
        file_size = path.stat().st_size
        pbar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="dump",
        )
        proc = subprocess.Popen(
            [decompressor, "-dc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=-1,
        )
        assert proc.stdin is not None and proc.stdout is not None

        def feed() -> None:
            try:
                with open(path, "rb") as fh:
                    while True:
                        chunk = fh.read(1 << 20)
                        if not chunk:
                            break
                        proc.stdin.write(chunk)
                        pbar.update(len(chunk))
            except BrokenPipeError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass

        feeder = threading.Thread(target=feed, daemon=True)
        feeder.start()
        try:
            yield proc.stdout
        finally:
            proc.stdout.close()
            proc.terminate()
            proc.wait()
            feeder.join(timeout=2)
            pbar.close()
    else:
        with bz2.open(str(path), "rb") as f:
            yield f


def usable_claims(claims: list[dict] | None) -> Iterator[dict]:
    """Yield value-snak claims in rank order, dropping deprecated and
    somevalue/novalue. Same shape as Stage 1's helper.
    """
    if not claims:
        return
    preferred: list[dict] = []
    normal: list[dict] = []
    for c in claims:
        rank = c.get("rank", "normal")
        if rank == "deprecated":
            continue
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        if rank == "preferred":
            preferred.append(c)
        else:
            normal.append(c)
    yield from preferred
    yield from normal


def first_qid(claims: list[dict] | None) -> str | None:
    for c in usable_claims(claims):
        qid = c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
        if qid:
            return qid
    return None


def qid_values(claims: list[dict] | None) -> list[str]:
    out: list[str] = []
    for c in usable_claims(claims):
        qid = c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
        if qid:
            out.append(qid)
    return out


def first_coord(claims: list[dict] | None) -> tuple[float, float] | None:
    """Return (lat, lon) of the first usable P625 coordinate-location claim."""
    for c in usable_claims(claims):
        v = c["mainsnak"].get("datavalue", {}).get("value", {})
        lat = v.get("latitude")
        lon = v.get("longitude")
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except (TypeError, ValueError):
                continue
    return None


def extract_qid_fast(line: bytes) -> str | None:
    """Pull the QID out of an entity line without parsing it as JSON.

    The Wikidata dump emits one entity per line in a strict layout that
    starts with ``{"type":"item","id":"Q...","..."``. We match the prefix
    and read up to the closing quote of the id field. Returns None for
    non-item lines (properties, array brackets, the lonesome ``[`` and
    ``]`` framers).
    """
    if not line.startswith(ITEM_PREFIX):
        return None
    end = line.find(b'"', len(ITEM_PREFIX))
    if end == -1:
        return None
    return line[len(ITEM_PREFIX):end].decode("ascii", errors="replace")


def extract_row(entity: dict) -> dict:
    claims = entity.get("claims", {})
    coord = first_coord(claims.get("P625"))
    lat, lon = (coord if coord is not None else (None, None))
    label = entity.get("labels", {}).get("en", {}).get("value")
    return {
        "qid": entity["id"],
        "label": label,
        "lat": lat,
        "lon": lon,
        "instance_of_qids": qid_values(claims.get("P31")),
        "admin_parent_qid": first_qid(claims.get("P131")),
        "country_qid": first_qid(claims.get("P17")),
    }


def load_needed_qids(in_path: Path) -> set[str]:
    table = pq.read_table(in_path, columns=["birth_place_qid", "death_place_qid"])
    needed: set[str] = set()
    for col in ("birth_place_qid", "death_place_qid"):
        for q in table[col].to_pylist():
            if q:
                needed.add(q)
    return needed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=IN_PATH,
        help="Stage 4 parquet to read place QIDs from.",
    )
    parser.add_argument("--dump", type=Path, default=DUMP_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many entities. Useful for smoke tests.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Stage 4 output not found: {args.in_path}")
    if not args.dump.exists():
        raise SystemExit(f"Wikidata dump not found: {args.dump}")

    print(f"loading needed place QIDs from {args.in_path.name}...")
    needed = load_needed_qids(args.in_path)
    print(f"  {len(needed):,} distinct place QIDs to resolve")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    batch: list[dict] = []
    start = time.perf_counter()

    with pq.ParquetWriter(args.out, schema=SCHEMA, compression="zstd") as writer:
        with open_dump(args.dump) as stream:
            for raw in stream:
                if args.limit is not None and total >= args.limit:
                    break
                total += 1
                qid = extract_qid_fast(raw)
                if qid is None or qid not in needed:
                    continue
                line = raw.rstrip(b"\r\n").rstrip(b",")
                try:
                    entity = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                row = extract_row(entity)
                batch.append(row)
                matched += 1
                if len(batch) >= BATCH_SIZE:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))

    elapsed = time.perf_counter() - start
    rate = total / elapsed if elapsed else 0.0
    coverage = matched / len(needed) * 100 if needed else 0.0
    print(
        f"processed {total:,} entities in {elapsed:.1f}s ({rate:,.0f} ent/s)"
    )
    print(
        f"matched {matched:,} places of {len(needed):,} needed "
        f"({coverage:.1f}%) -> {args.out}"
    )


if __name__ == "__main__":
    main()
