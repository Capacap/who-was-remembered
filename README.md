# Wikipedia Exploration

A browser-based 3D art piece. The player walks through a low-poly desert scattered with books, one per Wikipedia article about a historical figure. Starting at year 2000 in the centre, walking outward moves back in time, with book density thinning as recorded history grows sparser. Wikipedia's recency bias is the explicit subject, not a flaw to be corrected.

See [DESIGN.md](DESIGN.md) for the full design document and the rationale behind the major choices.

## Status

Preprocessing pipeline in progress. Stages 1 through 4 run end-to-end on the 2026-05 dumps: Stage 1 emits about 914k figures, Stage 2 cuts to 638k on the recency and stub pre-filter, Stage 3 attaches a Wikipedia lead and outgoing link graph to 99.1% of those, and Stage 4 keeps the 419k whose leads carry narrative past the opener. Density runs from a few hundred figures per century in deep antiquity to 279k in the 20th century, which is the temporal gradient the desert framing wants. Stages 5 and 6 then resolve birth/death coordinates and assign each figure a radial-time position (era as radius, population-equalized longitude as angle). The runtime (Three.js scene, terrain, tile loading) is not yet implemented.

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

### Stage 4: Article quality cut

`pipeline/stage4_quality_cut.py` reads `wikidata_figures_with_articles.parquet` and writes `wikidata_figures_quality.parquet` (about 419k rows). The criterion is "the lead carries narrative past the opener", not fame: a row survives when its cleaned lead has at least two sentences and at least 30 words. Also strips leftover `__NOTOC__`-style magic words that the wikitext parser missed and recomputes the word count from the cleaned text. No download needed.

Run (under a minute):

```sh
uv run pipeline/stage4_quality_cut.py
```

### Stage 5: Place extraction

`pipeline/stage5_extract_places.py` reads the Stage 4 parquet, collects every birth and death place QID, and resolves their coordinates and country from Wikidata, writing `places.parquet` (qid, label, lat, lon, instance_of, admin parent, country). This is the geographic lookup Stage 6 needs to turn a birthplace into an angle.

### Stage 6: Placement

`pipeline/stage6_place.py` reads the Stage 4 figures plus `places.parquet` and writes `placement.parquet`, adding polar coordinates (radius, angle), their Cartesian projection (x, y), and a `landmark_tier`. Radius is era only: `R_MAX * t^0.75` where `t = (2000 - death_year) / 2800`, so the player at the origin walks outward into the past and the dense recent centuries stay compressed and legible. Angle is the population CDF of the figure's longitude rather than raw longitude: raw longitude makes the angular axis a fame proxy (fame concentrates in the narrow Western-European longitude band and piles into a thin wedge), whereas the CDF gives every direction comparable population while preserving east-west order and regional adjacency. About 73% of figures are geo-anchored; the rest take a deterministic per-QID random angle. Prominence never enters position.

`landmark_tier` (major / minor / ordinary) is the notability axis, kept separate from position and meant only as a player navigation aid: a landmark is a figure the player is likelier to recognize, so a taller book gives a reference point to steer by. It has two sources. The global source is an absolute `sitelink_count` floor (cross-lingual coverage), which makes a recognizable figure a major wherever it sits (about 1,600 figures). The local source fills directions and eras where nobody clears the floor: the most-covered geo-anchored figure in each region-and-era grid cell with no major becomes a minor reference point even when globally obscure (Moctezuma II, the early-dynasty pharaohs). `sitelink_count` carries the continuous magnitude for the renderer to modulate book height within a tier.

`pipeline/inspect_placement.py` renders diagnostic PNGs (era, country, density, ring, landmarks) from a placement parquet, and is the tuning lens for the floor and grid before re-baking.

Run (a few seconds each):

```sh
uv run pipeline/stage5_extract_places.py
uv run pipeline/stage6_place.py
```

### Later stages

Embedding, terrain, and the runtime bundle are described in DESIGN.md but not yet implemented. The runtime will need spatial tile-based loading rather than a single bundle, because 419k books exceeds the ~30-50k DESIGN.md called the single-bundle limit.
