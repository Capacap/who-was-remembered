"""
Stage 4: article quality cut.

Filter Stage 3 figures down to the ones whose Wikipedia lead carries at
least a bit of narrative past the opening (name, dates, role) sentence.
The criterion is "is there enough story to make the figure feel real",
not "is this person famous".

Concretely, a row survives when:

- lead_text is non-null (Stage 3 successfully fetched and parsed the
  article),
- the cleaned lead contains a sentence boundary, i.e. terminal punctuation
  followed by whitespace and a capital letter, taken as evidence of a real
  second sentence rather than a label with an abbreviation, and
- the cleaned lead has at least --min-words words (default 30, chosen
  empirically: below ~25 words the corpus is dominated by single-sentence
  "X (1810-1890) was a Y" labels; from ~30 words two-sentence leads with
  biographical content become the norm).

Also runs a magic-word cleanup pass (__NOTOC__ and similar markers that
Stage 3 strip_code missed) and recomputes lead_word_count from the cleaned
text. Adds a lead_sentence_count column for downstream stages.

Run:
    uv run pipeline/stage4_quality_cut.py
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
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
        "--min-words",
        type=int,
        default=30,
        help="Minimum lead word count after cleanup. Default 30.",
    )
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=2,
        help="Minimum lead sentence count. Default 2.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        raise SystemExit(f"Stage 3 output not found: {args.in_path}")

    start = time.perf_counter()
    table = pq.read_table(args.in_path)
    total = table.num_rows

    has_lead = pc.is_valid(table["lead_text"])
    no_article = total - pc.sum(pc.cast(has_lead, pa.int64())).as_py()
    table = table.filter(has_lead)

    leads = table["lead_text"].to_pylist()
    cleaned: list[str] = []
    word_counts: list[int] = []
    sent_counts: list[int] = []
    for raw in leads:
        c, w, s = clean_lead(raw)
        cleaned.append(c)
        word_counts.append(w)
        sent_counts.append(s)

    keep_mask = [
        (w >= args.min_words) and (s >= args.min_sentences)
        for w, s in zip(word_counts, sent_counts)
    ]

    new_lead = pa.array(cleaned, type=pa.string())
    new_wc = pa.array(word_counts, type=pa.int32())
    new_sc = pa.array(sent_counts, type=pa.int32())

    table = table.set_column(
        table.schema.get_field_index("lead_text"), "lead_text", new_lead
    )
    table = table.set_column(
        table.schema.get_field_index("lead_word_count"),
        "lead_word_count",
        new_wc,
    )
    table = table.append_column("lead_sentence_count", new_sc)

    filtered = table.filter(pa.array(keep_mask, type=pa.bool_()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, args.out, compression="zstd")

    short_wc = sum(
        1 for w, s, k in zip(word_counts, sent_counts, keep_mask)
        if not k and w < args.min_words
    )
    single_sent = sum(
        1 for w, s, k in zip(word_counts, sent_counts, keep_mask)
        if not k and w >= args.min_words and s < args.min_sentences
    )
    kept = filtered.num_rows
    elapsed = time.perf_counter() - start
    print(f"read {total:,} rows from {args.in_path.name} in {elapsed:.1f}s")
    print(f"  no parsed article: -{no_article:,}")
    print(f"  lead under {args.min_words} words: -{short_wc:,}")
    print(
        f"  single-sentence lead (after word cut): -{single_sent:,}"
    )
    print(f"kept {kept:,} rows ({kept / total * 100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
