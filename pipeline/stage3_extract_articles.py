"""
Stage 3: Wikipedia article extraction.

Streams the English Wikipedia pages-articles dump, finds the article for
each surviving figure from Stage 2, and pulls:

- the lead section text (everything before the first level-2 heading,
  with templates / refs / comments stripped),
- the {{short description|...}} template value if the article has one,
- the article's outgoing wikilinks, resolved through redirects and
  restricted to QIDs that are themselves figures in our set.

Pipeline shape:

    Stage 2 parquet -----.
                          \\
    enwiki dump -------> [ this stage ] ---> Stage 3 parquet
                          /
            mwparserfromhell + redirect map

The dump pass is linear with lbzip2 decompression (same pattern as
Stage 1). Wikitext parsing is shipped to a multiprocessing pool because
mwparserfromhell is pure Python and the per-article cost is non-trivial.

Setup (from the project root):
    uv sync

Download (~24 GB; same aria2c pattern as Stage 1, plays nice with the
3-connection per-IP cap):

    aria2c -x 3 -s 3 -k 50M -c -d pipeline/data \\
      -o enwiki-latest-pages-articles-multistream.xml.bz2 \\
      https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2

Run:
    uv run pipeline/stage3_extract_articles.py

Output: pipeline/cache/wikidata_figures_with_articles.parquet
"""

from __future__ import annotations

import argparse
import bz2
import html
import multiprocessing as mp
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

import mwparserfromhell as mw
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DUMP_PATH = ROOT / "data" / "enwiki-latest-pages-articles-multistream.xml.bz2"
IN_PATH = ROOT / "cache" / "wikidata_figures_prefiltered.parquet"
OUT_PATH = ROOT / "cache" / "wikidata_figures_with_articles.parquet"

WRITE_BATCH = 25_000

# MediaWiki built-in namespaces whose link targets are not article-space
# links and should be excluded from the outgoing-link graph. Match is
# case-insensitive against the prefix before the first colon.
SKIP_NAMESPACES = frozenset(
    {
        "talk",
        "user",
        "user talk",
        "wikipedia",
        "wp",
        "wikipedia talk",
        "file",
        "image",
        "file talk",
        "mediawiki",
        "mediawiki talk",
        "template",
        "template talk",
        "help",
        "help talk",
        "category",
        "category talk",
        "portal",
        "portal talk",
        "draft",
        "draft talk",
        "timedtext",
        "timedtext talk",
        "module",
        "module talk",
        "book",
        "book talk",
        "special",
    }
)

# Inline formatting templates whose stripped form would lose meaningful
# text (date dashes, non-breaking spaces, etc.). Wikitext leans heavily on
# {{snd}} and {{ndash}} for life-span dates, so dropping them mangles every
# modern figure's birth/death line.
INLINE_TEMPLATE_TEXT = {
    "snd": " – ",
    "spaced ndash": " – ",
    "ndash": "–",
    "endash": "–",
    "en dash": "–",
    "mdash": "—",
    "emdash": "—",
    "em dash": "—",
    "nbsp": " ",
    "'": "'",
    "=": "=",
}
CIRCA_TEMPLATES = frozenset({"circa", "c.", "ca.", "ca", "c"})
FLORUIT_TEMPLATES = frozenset({"floruit", "fl.", "fl"})

TITLE_RE = re.compile(rb"<title>([^<]*)</title>")
NS_RE = re.compile(rb"<ns>(-?\d+)</ns>")
REDIRECT_RE = re.compile(rb'<redirect title="([^"]*)"')
TEXT_RE = re.compile(rb"<text[^>]*>(.*?)</text>", re.DOTALL)


@contextmanager
def open_dump(path: Path) -> Iterator[IO[bytes]]:
    """Open a bz2 dump for streaming with a compressed-bytes progress bar.

    Prefers lbzip2 (real multi-core decompression of standard bzip2),
    then pbzip2 (decompression in a separate process, still helpful),
    then Python's single-threaded bz2.open. Same machinery as Stage 1.
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


def iter_page_chunks(stream: IO[bytes]) -> Iterator[bytes]:
    """Yield the raw bytes of each <page>...</page> block from the dump.

    Wikipedia dumps emit each XML tag on its own line with content lines in
    between for the multi-line <text> element. We just buffer lines from
    <page> to </page> and yield the joined chunk.
    """
    buf: list[bytes] = []
    in_page = False
    for line in stream:
        if in_page:
            buf.append(line)
            if b"</page>" in line:
                yield b"".join(buf)
                buf = []
                in_page = False
        else:
            idx = line.find(b"<page>")
            if idx >= 0:
                in_page = True
                buf.append(line[idx:])
                if b"</page>" in line:
                    yield b"".join(buf)
                    buf = []
                    in_page = False


def parse_page_meta(chunk: bytes) -> tuple[str, str | None, str | None] | None:
    """Extract (title, redirect_target, wikitext) from a <page> chunk.

    Returns None for non-mainspace pages (we only care about ns=0).
    For a redirect page, ``redirect_target`` is set and ``wikitext`` is None.
    For a content page, ``redirect_target`` is None and ``wikitext`` is the
    page's wikitext (XML-unescaped).
    """
    ns_match = NS_RE.search(chunk)
    if not ns_match or ns_match.group(1) != b"0":
        return None
    title_match = TITLE_RE.search(chunk)
    if not title_match:
        return None
    title = html.unescape(title_match.group(1).decode("utf-8", "replace"))

    redirect_match = REDIRECT_RE.search(chunk)
    if redirect_match:
        target = html.unescape(redirect_match.group(1).decode("utf-8", "replace"))
        return title, target, None

    text_match = TEXT_RE.search(chunk)
    if not text_match:
        return None
    wikitext = html.unescape(text_match.group(1).decode("utf-8", "replace"))
    return title, None, wikitext


def normalize_title(target: str) -> str:
    """Canonicalize a wikilink target the way MediaWiki does.

    Strips a leading colon (link-as-link escape), drops any section
    anchor, swaps underscores for spaces, trims whitespace, and
    uppercases the first character. The result is comparable against
    Wikidata sitelink titles.
    """
    target = target.strip()
    if target.startswith(":"):
        target = target[1:].lstrip()
    hash_idx = target.find("#")
    if hash_idx >= 0:
        target = target[:hash_idx]
    target = target.replace("_", " ").strip()
    if not target:
        return ""
    return target[0].upper() + target[1:]


def is_skip_namespace(title: str) -> bool:
    """True if the title's namespace prefix is one we exclude from links."""
    colon = title.find(":")
    if colon <= 0:
        return False
    prefix = title[:colon].strip().lower()
    return prefix in SKIP_NAMESPACES


def parse_article(item: tuple[str, str]) -> tuple[str, dict | None]:
    """Worker: parse one article's wikitext into the fields we store.

    Returns ``(title, fields)`` where fields is a dict, or ``(title, None)``
    on parse failure. Designed to be sent through a multiprocessing pool,
    so it's a module-level function and all inputs / outputs are picklable.
    """
    title, wikitext = item
    try:
        code = mw.parse(wikitext)
    except Exception:
        return title, None

    short_desc: str | None = None
    for tpl in code.filter_templates(recursive=False):
        name = str(tpl.name).strip().lower()
        if name == "short description" and tpl.params:
            short_desc = str(tpl.params[0].value).strip()
            break

    seen: set[str] = set()
    outgoing_titles: list[str] = []
    for link in code.filter_wikilinks():
        try:
            target = str(link.title)
        except Exception:
            continue
        target = normalize_title(target)
        if not target:
            continue
        if is_skip_namespace(target):
            continue
        if target in seen:
            continue
        seen.add(target)
        outgoing_titles.append(target)

    # Lead: everything up to the first level-2 heading. get_sections gives
    # us a clean split; the lead is sections[0] when include_lead is True.
    try:
        sections = code.get_sections(
            levels=[2], include_lead=True, include_headings=False, flat=True
        )
        lead_code = mw.parse(str(sections[0])) if sections else mw.parse(str(code))
    except Exception:
        lead_code = mw.parse(str(code))

    for tpl in list(lead_code.filter_templates()):
        try:
            name = str(tpl.name).strip().lower()
        except Exception:
            name = ""
        if name in INLINE_TEMPLATE_TEXT:
            replacement = INLINE_TEMPLATE_TEXT[name]
        elif name in CIRCA_TEMPLATES:
            year = ""
            if tpl.params:
                try:
                    year = str(tpl.params[0].value).strip()
                except Exception:
                    year = ""
            replacement = f"c. {year}" if year else "c."
        elif name in FLORUIT_TEMPLATES:
            year = ""
            if tpl.params:
                try:
                    year = str(tpl.params[0].value).strip()
                except Exception:
                    year = ""
            replacement = f"fl. {year}" if year else "fl."
        else:
            replacement = None
        try:
            if replacement is None:
                lead_code.remove(tpl)
            else:
                lead_code.replace(tpl, replacement)
        except ValueError:
            pass
    for node in list(lead_code.filter_tags(matches="ref")):
        try:
            lead_code.remove(node)
        except ValueError:
            pass
    for node in list(lead_code.filter_tags(matches="gallery")):
        try:
            lead_code.remove(node)
        except ValueError:
            pass
    for node in list(lead_code.filter_comments()):
        try:
            lead_code.remove(node)
        except ValueError:
            pass
    for node in list(lead_code.filter_wikilinks()):
        try:
            target = normalize_title(str(node.title))
        except Exception:
            continue
        if is_skip_namespace(target):
            try:
                lead_code.remove(node)
            except ValueError:
                pass

    lead_text = str(lead_code.strip_code()).strip()
    # Collapse leftover artifacts from removed templates and skipped wikilinks.
    # Removed inline templates like {{IPA|...}} leave punctuation residue
    # inside the surrounding parens; we trim leading/trailing punctuation
    # within parens, drop fully empty parens, then tidy whitespace around
    # punctuation and collapse whitespace runs.
    lead_text = re.sub(r"\(\s*[;,:\s]+", "(", lead_text)
    lead_text = re.sub(r"[;,:\s]+\)", ")", lead_text)
    lead_text = re.sub(r"\(\s*\)", "", lead_text)
    lead_text = re.sub(r"\s+([,.;:])", r"\1", lead_text)
    lead_text = re.sub(r"\n{3,}", "\n\n", lead_text)
    lead_text = re.sub(r"[ \t]+", " ", lead_text)
    lead_word_count = len(lead_text.split())

    return title, {
        "lead_text": lead_text,
        "lead_word_count": lead_word_count,
        "short_description": short_desc,
        "outgoing_titles": outgoing_titles,
    }


def iter_figure_pages(
    stream: IO[bytes],
    figure_titles: set[str],
    redirects: dict[str, str],
    limit: int | None,
) -> Iterator[tuple[str, str]]:
    """Stream the dump, yield (title, wikitext) for figure pages, and
    populate ``redirects`` for redirect pages whose target is a figure.

    Side effect on ``redirects`` is the point: we do a single dump pass and
    accumulate both content articles (yielded to the worker pool) and the
    redirect map (built up directly).
    """
    seen_pages = 0
    for chunk in iter_page_chunks(stream):
        if limit is not None and seen_pages >= limit:
            break
        seen_pages += 1
        parsed = parse_page_meta(chunk)
        if parsed is None:
            continue
        title, redirect_target, wikitext = parsed
        if redirect_target is not None:
            target = normalize_title(redirect_target)
            if target in figure_titles:
                redirects[title] = target
            continue
        if title in figure_titles and wikitext is not None:
            yield title, wikitext


def resolve_outgoing_qids(
    outgoing_titles: list[str],
    redirects: dict[str, str],
    title_to_qid: dict[str, str],
) -> list[str]:
    """Resolve raw link targets through one redirect hop and look up QIDs.

    Returns the deduplicated list of QIDs for targets that resolve into our
    figure set. Single-hop is enough in practice; MediaWiki's no-double-
    redirect policy plus cleanup bots keep chains rare.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in outgoing_titles:
        resolved = redirects.get(raw, raw)
        qid = title_to_qid.get(resolved)
        if qid is None:
            continue
        if qid in seen:
            continue
        seen.add(qid)
        out.append(qid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=IN_PATH,
        help="Stage 2 prefiltered parquet (input).",
    )
    parser.add_argument(
        "--dump",
        dest="dump_path",
        type=Path,
        default=DUMP_PATH,
        help="enwiki pages-articles-multistream bz2 dump.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_PATH,
        help="Stage 3 output parquet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many <page> elements. Useful for a smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes for mwparserfromhell. Defaults to cpu_count.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Stage 2 output not found: {args.in_path}")
    if not args.dump_path.exists():
        raise SystemExit(
            f"Wikipedia dump not found: {args.dump_path}\n"
            f"Download it first (see this file's docstring for the URL)."
        )

    in_table = pq.read_table(args.in_path)
    titles_col = in_table.column("title").to_pylist()
    qids_col = in_table.column("qid").to_pylist()
    title_to_qid: dict[str, str] = dict(zip(titles_col, qids_col))
    figure_titles: set[str] = set(titles_col)
    print(f"loaded {len(figure_titles):,} figures from {args.in_path.name}")

    parsed_results: dict[str, dict] = {}
    redirects: dict[str, str] = {}
    workers = args.workers or mp.cpu_count()
    start = time.perf_counter()

    parse_pbar = tqdm(
        total=len(figure_titles),
        desc="articles",
        unit="art",
    )
    with mp.Pool(processes=workers) as pool:
        with open_dump(args.dump_path) as stream:
            page_iter = iter_figure_pages(
                stream, figure_titles, redirects, args.limit
            )
            for title, fields in pool.imap_unordered(
                parse_article, page_iter, chunksize=64
            ):
                if fields is not None:
                    parsed_results[title] = fields
                parse_pbar.update(1)
    parse_pbar.close()

    dump_elapsed = time.perf_counter() - start
    print(
        f"dump pass: {len(parsed_results):,} articles parsed, "
        f"{len(redirects):,} relevant redirects, in {dump_elapsed:.1f}s"
    )

    join_start = time.perf_counter()
    n = in_table.num_rows
    lead_text_out: list[str | None] = [None] * n
    lead_words_out: list[int | None] = [None] * n
    short_desc_out: list[str | None] = [None] * n
    outgoing_qids_out: list[list[str]] = [[] for _ in range(n)]

    for i, title in enumerate(titles_col):
        fields = parsed_results.get(title)
        if fields is None:
            continue
        lead_text_out[i] = fields["lead_text"]
        lead_words_out[i] = fields["lead_word_count"]
        short_desc_out[i] = fields["short_description"]
        outgoing_qids_out[i] = resolve_outgoing_qids(
            fields["outgoing_titles"], redirects, title_to_qid
        )

    out_table = in_table.append_column(
        "lead_text", pa.array(lead_text_out, type=pa.string())
    )
    out_table = out_table.append_column(
        "lead_word_count", pa.array(lead_words_out, type=pa.int32())
    )
    out_table = out_table.append_column(
        "short_description", pa.array(short_desc_out, type=pa.string())
    )
    out_table = out_table.append_column(
        "outgoing_qids", pa.array(outgoing_qids_out, type=pa.list_(pa.string()))
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, args.out, compression="zstd")
    join_elapsed = time.perf_counter() - join_start

    matched = sum(1 for x in lead_text_out if x is not None)
    pct = matched / n * 100 if n else 0.0
    print(
        f"joined and wrote {n:,} rows ({matched:,} with article = "
        f"{pct:.1f}%) in {join_elapsed:.1f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
