# Wikipedia Exploration

A browser-based 3D art piece. The player walks through a low-poly desert scattered with books, one per Wikipedia article about a historical figure. Starting at year 2000 in the centre, walking outward moves back in time, with book density thinning as recorded history grows sparser. Wikipedia's recency bias is the explicit subject, not a flaw to be corrected.

See [DESIGN.md](DESIGN.md) for the full design document and the rationale behind the major choices.

## Status

Preprocessing pipeline in progress. Stages 1 and 2 run end-to-end on the 2026-05 dumps. Stage 3 is written and unit-tested but waiting on the English Wikipedia dump download to validate against real data. The runtime (Three.js scene, embedding placement, terrain) is not yet implemented.

## Pipeline

Each stage consumes the previous stage's output from `pipeline/cache/` and writes its own. Dumps go into `pipeline/data/`. Both directories are git-ignored.

| Stage | Script | Reads | Writes | What it does |
|---|---|---|---|---|
| 1 | `pipeline/stage1_filter_wikidata.py` | `latest-all.json.bz2` (Wikidata, ~95 GB compressed) | `wikidata_figures.parquet` | Streams the Wikidata dump and emits one row per `instance of: human` entity with a date of death and an English Wikipedia sitelink. Captures gender, citizenships, places of birth and death, occupations, the full P31 list, and stub-detection signals. About 914k rows. |
| 2 | `pipeline/stage2_prefilter.py` | `wikidata_figures.parquet` | `wikidata_figures_prefiltered.parquet` | Drops post-2000 figures (year 2000 is the spawn point, and contested contemporary politics concentrate there) and the clearest Wikidata stubs (no description). About 638k rows. |
| 3 | `pipeline/stage3_extract_articles.py` | English Wikipedia dump (`enwiki-latest-pages-articles-multistream.xml.bz2`, ~24 GB compressed) plus the Stage 2 parquet | `wikidata_figures_with_articles.parquet` | Streams the Wikipedia dump, parses each figure's article with mwparserfromhell, captures the lead text, the `{{short description}}` template, and outgoing wikilinks (resolved through redirects, restricted to QIDs that are themselves figures). |

Later stages (article-quality cut, embedding, placement, terrain, runtime bundle) are described in DESIGN.md but not yet implemented.

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/), `lbzip2` for parallel bz2 decompression, `aria2` for the dump downloads, and the Python development headers for `mwparserfromhell`'s C tokenizer.

On Fedora-like systems:

```sh
sudo dnf install lbzip2 python3-devel aria2
uv sync
```

## Running

Each stage script's docstring documents its inputs, the `aria2c` command to fetch the dump it needs, and rough runtimes. The order on a fresh checkout:

```sh
# 1. Wikidata dump (~95 GB compressed). Download, then:
uv run pipeline/stage1_filter_wikidata.py    # ~2.5 hours
uv run pipeline/stage2_prefilter.py          # under a second

# 2. English Wikipedia dump (~24 GB compressed). Download, then:
uv run pipeline/stage3_extract_articles.py   # ~20 minutes
```
