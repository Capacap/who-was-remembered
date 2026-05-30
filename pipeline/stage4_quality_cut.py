"""
Stage 4: article quality cut.

Drop database stubs (bare one-line entries) while keeping ordinary people
with a genuine, if short, article. The criterion is article substance, not
fame and not the lead's prose style.

A row survives when its Wikipedia article body has at least
--min-article-words words (default 100). That body count comes from Stage 3,
which strip_code's the whole article (templates, infoboxes and the citation
templates inside ref tags are dropped), so it measures real prose rather than
markup.

This replaces an earlier rule that gated on the LEAD shape (two sentences and
30 words). Lead shape measured the first paragraph's style, not whether an
article exists: it cut thousands of substantive figures whose lead happens to
be a single dense sentence (Atahualpa, whose article runs past 4000 words; a
wall of Nobel laureates) while passing thin bodies that merely opened with two
sentences. Measuring the body fixes both errors.

The floor is deliberately low. At 100 words the cut removes the genuine
one-fact stubs ("X was a Y." plus at most a trailing fragment, almost all
under ~75 words) and keeps the minimal-but-real bios that are the point of the
piece (a sprinter who ran one relay, an obscure 17th-century MP). The corpus's
25th percentile is ~200 article words, so this is a stub gate, not a
notability bar.

Also runs a magic-word cleanup pass (__NOTOC__ and similar markers that Stage
3's strip_code missed) on the lead text and recomputes lead_word_count and
lead_sentence_count. The lead is now a display and landmark-labelling column,
no longer the cut.

Run:
    uv run pipeline/stage4_quality_cut.py
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
IN_PATH = ROOT / "cache" / "wikidata_figures_with_articles.parquet"
OUT_PATH = ROOT / "cache" / "wikidata_figures_quality.parquet"

# MediaWiki magic words sometimes leak through strip_code as raw tokens
# (e.g. __NOTOC__, __FORCETOC__). They surface as visible text in the lead
# without this pass.
MAGIC_WORD_RE = re.compile(r"__[A-Z_]+__")

# A "second sentence" is detected as terminal punctuation, whitespace, and
# a capital letter. This intentionally tolerates abbreviations like "c."
# and "Dr." that lack a following capital, and ignores the closing period
# of the final sentence. Approximate but the cut is robust to noise at
# the ~30-word threshold.
SENT_BOUNDARY_RE = re.compile(r"[.!?]\s+[A-Z]")

WS_RE = re.compile(r"\s+")


def clean_lead(text: str) -> tuple[str, int, int]:
    """Strip leftover magic words, tidy whitespace, and report word and
    approximate sentence counts. Sentence count is 1 plus the number of
    detected boundaries; empty text returns (text, 0, 0).
    """
    text = MAGIC_WORD_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    if not text:
        return text, 0, 0
    wc = len(text.split())
    sc = 1 + len(SENT_BOUNDARY_RE.findall(text))
    return text, wc, sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, default=IN_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--min-article-words",
        type=int,
        default=100,
        help="Minimum article body word count. Default 100.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Stage 3 output not found: {args.in_path}")

    start = time.perf_counter()
    table = pq.read_table(args.in_path)
    total = table.num_rows

    # The cut: article body length. None means Stage 3 found no article.
    article_wc = table["article_word_count"].to_pylist()
    keep_mask = [
        (a is not None and a >= args.min_article_words) for a in article_wc
    ]

    # Tidy the lead text (a display / landmark column now, not the gate):
    # strip leftover magic words, collapse whitespace, recompute its counts.
    leads = table["lead_text"].to_pylist()
    cleaned: list[str | None] = []
    word_counts: list[int | None] = []
    sent_counts: list[int | None] = []
    for raw in leads:
        if raw is None:
            cleaned.append(None)
            word_counts.append(None)
            sent_counts.append(None)
            continue
        c, w, s = clean_lead(raw)
        cleaned.append(c)
        word_counts.append(w)
        sent_counts.append(s)

    table = table.set_column(
        table.schema.get_field_index("lead_text"),
        "lead_text",
        pa.array(cleaned, type=pa.string()),
    )
    table = table.set_column(
        table.schema.get_field_index("lead_word_count"),
        "lead_word_count",
        pa.array(word_counts, type=pa.int32()),
    )
    table = table.append_column(
        "lead_sentence_count", pa.array(sent_counts, type=pa.int32())
    )

    filtered = table.filter(pa.array(keep_mask, type=pa.bool_()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, args.out, compression="zstd")

    no_article = sum(1 for a in article_wc if a is None)
    below_floor = sum(
        1 for a in article_wc if a is not None and a < args.min_article_words
    )
    kept = filtered.num_rows
    elapsed = time.perf_counter() - start
    print(f"read {total:,} rows from {args.in_path.name} in {elapsed:.1f}s")
    print(f"  no parsed article: -{no_article:,}")
    print(f"  article body under {args.min_article_words} words: -{below_floor:,}")
    print(f"kept {kept:,} rows ({kept / total * 100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
