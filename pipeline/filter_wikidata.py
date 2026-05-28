"""
Stage 1: Wikidata filter.

Stream-parse the Wikidata JSON dump and emit one row per historical figure
(human with date of death and English Wikipedia sitelink) to a Parquet file.

Setup (from the project root):
    uv sync
    sudo dnf install lbzip2  # parallel bz2 decompression; single-threaded bz2 is the bottleneck without it

pbzip2 cannot parallelize files compressed with standard bzip2 (which the
Wikidata dump is), so it falls back to single-threaded mode. lbzip2 actually
does multi-core decompression of standard bzip2 streams and is the right tool.

Download the dump (~100 GB) into pipeline/data/ before running. Use aria2c for
parallel segments; single-stream curl tops out at ~5 MB/s while aria2c reaches
15+ MB/s. Three connections is Wikimedia's per-IP cap for this server (more
just earns 429s in the log).

    sudo dnf install aria2  # if not already installed
    mkdir -p pipeline/data
    aria2c -x 3 -s 3 -k 50M -c -d pipeline/data -o latest-all.json.bz2 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2

Then:
    uv run pipeline/filter_wikidata.py

Output: pipeline/cache/wikidata_figures.parquet
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
OUT_PATH = ROOT / "cache" / "wikidata_figures.parquet"

BATCH_SIZE = 50_000

SCHEMA = pa.schema(
    [
        ("qid", pa.string()),
        ("title", pa.string()),
        ("description", pa.string()),
        ("birth_year", pa.int32()),
        ("death_year", pa.int32()),
        ("sitelink_count", pa.int32()),
        ("claim_count", pa.int32()),
        ("identifier_count", pa.int32()),
        ("has_image", pa.bool_()),
        ("gender", pa.string()),
        ("citizenships", pa.list_(pa.string())),
        ("birth_place_qid", pa.string()),
        ("death_place_qid", pa.string()),
        ("occupations", pa.list_(pa.string())),
        ("instance_of_qids", pa.list_(pa.string())),
    ]
)


@contextmanager
def open_dump(path: Path) -> Iterator[IO[bytes]]:
    """Open the bz2 dump for streaming, with a compressed-bytes progress bar.

    Prefers lbzip2 (genuine multi-core decompression of standard bzip2 streams).
    Falls back to pbzip2 (cannot parallelize standard bzip2 output, but still
    runs in a separate process so decompression overlaps with Python parsing),
    then to Python's single-threaded bz2.open as a last resort.

    When a subprocess decompressor is used, the file is read in Python and
    piped to the decompressor's stdin. This lets us count compressed bytes
    consumed for a true percentage progress bar against the known file size.
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
                        chunk = fh.read(1 << 20)  # 1 MiB
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


def iter_entities(stream: IO[bytes]) -> Iterator[dict]:
    """Yield one parsed entity dict per line, skipping the array brackets."""
    for raw in stream:
        line = raw.rstrip(b"\r\n").rstrip(b",")
        if not line or line in (b"[", b"]"):
            continue
        yield orjson.loads(line)


def parse_year(time_str: str | None) -> int | None:
    """Parse a Wikidata time string like '+1815-05-05T00:00:00Z' to a signed year integer."""
    if not time_str:
        return None
    sign = -1 if time_str[0] == "-" else 1
    try:
        end = time_str.index("-", 1)
    except ValueError:
        return None
    try:
        return sign * int(time_str[1:end])
    except ValueError:
        return None


def usable_claims(claims: list[dict] | None) -> Iterator[dict]:
    """Yield value-snak claims in rank order (preferred first, then normal).

    Skips deprecated claims and any whose mainsnak is somevalue/novalue. This is
    how the extractor functions below navigate Wikidata's per-statement rank
    model so that disputed values prefer the editorially chosen one.
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


def first_year(claims: list[dict] | None) -> int | None:
    """Return the year of the first usable time claim, rank-aware."""
    for c in usable_claims(claims):
        value = c["mainsnak"].get("datavalue", {}).get("value", {})
        year = parse_year(value.get("time"))
        if year is not None:
            return year
    return None


def is_human(claims: list[dict] | None) -> bool:
    """True if any non-deprecated P31 claim is Q5 (human)."""
    for c in usable_claims(claims):
        if c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") == "Q5":
            return True
    return False


def has_value_claim(claims: list[dict] | None) -> bool:
    """True if at least one non-deprecated claim carries a real value."""
    for _ in usable_claims(claims):
        return True
    return False


def qid_values(claims: list[dict] | None) -> list[str]:
    """Collect QID values from claims, rank-aware (preferred first, deprecated dropped)."""
    out: list[str] = []
    for c in usable_claims(claims):
        qid = c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
        if qid:
            out.append(qid)
    return out


def first_qid(claims: list[dict] | None) -> str | None:
    """Return the first usable QID value, rank-aware."""
    for c in usable_claims(claims):
        qid = c["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
        if qid:
            return qid
    return None


def count_total_claims(claims_dict: dict) -> int:
    """Sum the number of claims across all properties on an entity."""
    return sum(len(v) for v in claims_dict.values())


def count_external_id_claims(claims_dict: dict) -> int:
    """Count claims whose mainsnak datatype is 'external-id' (VIAF, GND, LoC, ...)."""
    total = 0
    for claim_list in claims_dict.values():
        for claim in claim_list:
            if claim.get("mainsnak", {}).get("datatype") == "external-id":
                total += 1
    return total


def extract_row(entity: dict) -> dict | None:
    """Return a row dict if the entity is a historical figure with an enwiki sitelink, else None."""
    claims = entity.get("claims", {})
    if not is_human(claims.get("P31")):
        return None
    death_year = first_year(claims.get("P570"))
    if death_year is None:
        return None
    sitelinks = entity.get("sitelinks", {})
    enwiki = sitelinks.get("enwiki")
    if not enwiki:
        return None
    description = entity.get("descriptions", {}).get("en", {}).get("value")
    return {
        "qid": entity["id"],
        "title": enwiki["title"],
        "description": description,
        "birth_year": first_year(claims.get("P569")),
        "death_year": death_year,
        "sitelink_count": len(sitelinks),
        "claim_count": count_total_claims(claims),
        "identifier_count": count_external_id_claims(claims),
        "has_image": has_value_claim(claims.get("P18")),
        "gender": first_qid(claims.get("P21")),
        "citizenships": qid_values(claims.get("P27")),
        "birth_place_qid": first_qid(claims.get("P19")),
        "death_place_qid": first_qid(claims.get("P20")),
        "occupations": qid_values(claims.get("P106")),
        "instance_of_qids": qid_values(claims.get("P31")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many entities. Useful for benchmarking.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to wikidata_figures.parquet, "
        "or wikidata_figures.trial.parquet when --limit is set.",
    )
    args = parser.parse_args()

    if not DUMP_PATH.exists():
        raise SystemExit(
            f"Dump not found: {DUMP_PATH}\n"
            f"Download it first (see this file's docstring for the URL)."
        )

    out_path = args.out or (
        OUT_PATH
        if args.limit is None
        else OUT_PATH.with_name(f"{OUT_PATH.stem}.trial.parquet")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    batch: list[dict] = []
    start = time.perf_counter()

    with pq.ParquetWriter(out_path, schema=SCHEMA, compression="zstd") as writer:
        with open_dump(DUMP_PATH) as stream:
            for entity in iter_entities(stream):
                if args.limit is not None and total >= args.limit:
                    break
                total += 1
                row = extract_row(entity)
                if row is None:
                    continue
                batch.append(row)
                matched += 1
                if len(batch) >= BATCH_SIZE:
                    writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))

    elapsed = time.perf_counter() - start
    rate = total / elapsed if elapsed else 0.0
    pct = (matched / total * 100) if total else 0.0
    print(
        f"processed {total:,} entities in {elapsed:.1f}s ({rate:,.0f} ent/s)"
    )
    print(f"matched {matched:,} figures ({pct:.2f}%) -> {out_path}")


if __name__ == "__main__":
    main()
