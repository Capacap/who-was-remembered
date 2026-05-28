# Wikipedia Exploration

A browser-based 3D art piece. The player walks through a low-poly desert scattered with books, one per Wikipedia article about a historical figure. Starting at year 2000 in the centre, walking outward moves back in time, with book density thinning as recorded history grows sparser. Wikipedia's recency bias is the explicit subject, not a flaw to be corrected.

See [DESIGN.md](DESIGN.md) for the full design document and the rationale behind the major choices.

## Status

Preprocessing pipeline in progress. Stages 1 and 2 run end-to-end on the 2026-05 dumps. Stage 3 is written and unit-tested but waiting on the English Wikipedia dump download to validate against real data. The runtime (Three.js scene, embedding placement, terrain) is not yet implemented.

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/), `lbzip2` for parallel bz2 decompression, `aria2` for the dump downloads, and the Python development headers for `mwparserfromhell`'s C tokenizer.

On Fedora-like systems:

```sh
sudo dnf install lbzip2 python3-devel aria2
uv sync
```

## Pipeline

Each stage consumes the previous stage's output from `pipeline/cache/` and writes its own. Dumps go into `pipeline/data/`. Both directories are git-ignored.

### Stage 1: Wikidata filter

`pipeline/stage1_filter_wikidata.py` streams `latest-all.json.bz2` (Wikidata, ~95 GB compressed) and writes `wikidata_figures.parquet` (about 914k rows). For each entity that is `instance of: human` with a date of death and an English Wikipedia sitelink, it captures gender, citizenships, places of birth and death, occupations, the full P31 list, and stub-detection signals.

Download (3 connections is Wikimedia's per-IP cap; more just earns 429s):

```sh
aria2c -x 3 -s 3 -k 50M -c -d pipeline/data \
  -o latest-all.json.bz2 \
  https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
```

Run (about 2.5 hours on 8 cores):

```sh
uv run pipeline/stage1_filter_wikidata.py
```

### Stage 2: Recency and stub pre-filter

`pipeline/stage2_prefilter.py` reads `wikidata_figures.parquet` and writes `wikidata_figures_prefiltered.parquet` (about 638k rows). It drops post-2000 figures (year 2000 is the player spawn point, and contested contemporary politics concentrate there) and the clearest Wikidata stubs (no description). No download needed.

Run (under a second):

```sh
uv run pipeline/stage2_prefilter.py
```

### Stage 3: Wikipedia article extraction

`pipeline/stage3_extract_articles.py` reads `enwiki-latest-pages-articles-multistream.xml.bz2` (about 24 GB compressed) plus the Stage 2 parquet, and writes `wikidata_figures_with_articles.parquet`. It streams the dump, parses each figure's article with mwparserfromhell, and captures the lead text, the `{{short description}}` template, and outgoing wikilinks (resolved through redirects and restricted to QIDs that are themselves figures).

Download:

```sh
aria2c -x 3 -s 3 -k 50M -c -d pipeline/data \
  -o enwiki-latest-pages-articles-multistream.xml.bz2 \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
```

Run (about 20 minutes):

```sh
uv run pipeline/stage3_extract_articles.py
```

### Later stages

Article-quality cut, embedding, placement, terrain, and the runtime bundle are described in DESIGN.md but not yet implemented.
