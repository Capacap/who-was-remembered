# Wikipedia Exploration Game — Design

Status: living design doc, updated as the build settles. Two pivots from the original plan. First, from figurative statues placed by raw UMAP to books with time as radial distance. Then, from embedding-driven angular placement to geography: angle is the population-equalized longitude of birthplace, adopted after the Stage 1-4 data review showed that both semantic and link-graph embeddings encode social class and would herd the famous into one wedge. A notability axis was also added, a per-figure landmark tier derived from cross-lingual coverage, which selects the book mesh and serves as a player navigation aid; it never touches position. The runtime architecture (Three.js scene, tiling, terrain, lighting) is unchanged from the original plan and not yet built.

## Concept

A browser-based 3D world the user walks through. The world is a low-poly desert scattered with books, each one representing a Wikipedia article about a historical figure. Approaching a book surfaces the article title and lead text and offers a link to read the full article on Wikipedia.

The player spawns at year 2000 in the center of the world. Radial distance from the center represents distance into the past: walking outward moves the player back in time, and book density thins out with depth, expressing the gaps in collectively recorded knowledge. Walking sideways at constant distance threads through contemporaries grouped by geographic region, since angular position is set by birthplace longitude.

There is no gameplay loop. It is an art project, not a game. The aesthetic and emotional goal is the feeling of walking back into emptier and emptier deep time, with Wikipedia's recency bias as the explicit subject rather than a flaw to be corrected. Pre-history is genuinely sparse and the desert is genuinely vast; that is the piece.

The book metaphor is deliberate. A book represents knowledge *about* a person, not a representation *of* them. This sidesteps the iconography questions a figurative-statue framing would raise (depictions of Muhammad, the Buddha, and so on are functionally equivalent to the Wikipedia articles themselves, which exist without controversy) and lets the figure set include ordinary people without inadvertently claiming they were "notable" in any monument-erecting sense. The unifying metaphor: lost knowledge is buried, and the further back you walk the more of it the desert has reclaimed.

## Scope and non-goals

In scope:

- Static, client-side experience. Loads from a static host with no backend.
- One precomputed JSON of book data shipped with the build (or chunked spatially).
- Desktop-first, mobile-supported through a quality profile and swappable input controller.
- Two or three book meshes (open, closed, stacked), instanced.
- Procedural low-poly desert with vertex colors and atmospheric lighting.

Out of scope:

- Gameplay, scoring, progression, inventory.
- Multiplayer, persistence, accounts.
- Inline article rendering. Book interaction links out to Wikipedia.
- PBR materials, physically based atmospherics, complex shaders.
- A real physics engine or AI.

## Preprocessing pipeline

Run once, offline. Output is a static data file consumed by the runtime.

### Data source

Full English Wikipedia as the article set, joined to Wikidata for entity-level filtering and metadata. The pipeline synthesizes the world from raw dumps locally, not from live APIs or SPARQL endpoints, so it is fully reproducible and rate-limit-free.

Three dumps:

- **Wikidata** — `latest-all.json.bz2` (~100 GB compressed; ~101 GB as of 2026-05-27) from `dumps.wikimedia.org/wikidatawiki/entities/`. Stream-parsed (qwikidata or manual JSON-lines). Source of the human-with-date-of-death filter, sitelink counts, birth and death dates, and occupation labels.
- **English Wikipedia** — `enwiki-latest-pages-articles-multistream.xml.bz2` (~22 GB compressed) plus its index file from `dumps.wikimedia.org/enwiki/latest/`. The multistream variant supports random-access seeking to specific articles via the index, so we read only articles in our filtered set rather than scanning the entire dump. Source of lead paragraphs and outgoing-link extraction.
- **Pantheon** (held in reserve) — MIT Media Lab curated dataset of ~88k notable historical figures with precomputed popularity index. Substitute for the dump pipeline if processing the dumps proves intractable on local hardware.

Cold-start cost: roughly 2-3 hours on a 12-thread CPU, dominated by `lbzip2` decompression of the Wikidata dump (first full run on the 2026-05 dump took 2h 29m and produced 913,830 figures). Subsequent runs read from the intermediate cache (next section) and never touch the raw dumps.

### Acquisition and intermediate cache

Between raw-dump parsing and the layout pipeline sits a series of Parquet files in `pipeline/cache/`, one per stage, each reading the previous stage's output. Only the dump-parse stages (1 and 3) touch the `.bz2` files; every downstream cut, the geo pass, and placement read Parquet. The schema below describes the figure columns carried through; later stages append to them (Stage 5 emits a separate `places.parquet`, Stage 6 appends the placement and landmark columns).

Schema:

- `qid` (string) — Wikidata QID, primary key
- `title` (string) — canonical English Wikipedia article title after redirect resolution
- `description` (string, nullable) — Wikidata one-line description, e.g., "French general and emperor (1769-1821)". Used for the placard tagline.
- `short_description` (string, nullable) — Wikipedia `{{Short description}}` template content if the article defines one. One layer more detail than `description`.
- `lead_text` (string) — Wikipedia first paragraph as plain text. The fullest summary we keep without leaving the page.
- `birth_year` (int32) — signed; negative for BCE
- `death_year` (int32)
- `sitelink_count` (int32) — count of language Wikipedias the figure appears on; cross-lingual coverage signal. Never a position coordinate, but it does drive the separate notability axis (landmark tier and book thickness; see Visual variation), read as player-facing recognizability rather than a fame ranking.
- `claim_count` (int32) — total Wikidata statements on the entity; used as a stub-detection signal in Stage 2 pre-filter, not as a fame rank.
- `identifier_count` (int32) — number of external authority IDs (VIAF, GND, LoC, ...). Cross-database presence complements `sitelink_count`.
- `has_image` (bool) — Wikidata P18 is set. Cheap article-elaboration signal; stub-quality figures typically lack a primary image.
- `gender` (string, nullable) — Wikidata P21 QID
- `citizenships` (list<string>) — Wikidata P27 QIDs; geographic affiliation, often multiple
- `birth_place_qid` (string, nullable) — Wikidata P19 QID; resolved to coordinates by the Stage 5 geo pass (`places.parquet`) and used as the primary longitude for angular placement
- `death_place_qid` (string, nullable) — Wikidata P20 QID
- `occupations` (list<string>) — Wikidata P106 QIDs, useful for sanity-checking clusters
- `instance_of_qids` (list<string>) — full P31 list. A figure may carry Q5 (human) alongside markers like Q21070568 (legendary character), letting the runtime visually distinguish documented from semi-legendary figures.
- `outgoing_qids` (list<string>) — figures-only adjacency list; contains only QIDs that also appear in this file

The runtime uses the three summary fields as UI tiers: `description` for the placard tagline, `short_description` (when present) when the user lingers, `lead_text` for inspect mode, and a click-through link for the full article.

Format choice: Parquet over SQLite. The workload is read-heavy and batch-oriented, never updates individual rows, and rewrites the whole file on every regeneration. The nested list column for the link graph fits naturally as a Parquet `list<string>`, avoiding a separate edges table. Polars reads it fast; DuckDB queries it directly with SQL when ad-hoc exploration during pipeline development is useful.

Pipeline stages:

1. **Wikidata stream-filter.** Read the bz2-decompressed JSON-lines through `orjson`, emit one row per entity matching `P31=Q5 AND has(P570) AND has(enwiki sitelink)`. Capture `qid`, `title`, `description`, `birth_year`, `death_year`, `sitelink_count`, `claim_count`, `identifier_count`, `has_image` (P18 presence), `gender` (P21), `citizenships` (P27), `birth_place_qid` (P19), `death_place_qid` (P20), `occupations` (P106), and `instance_of_qids` (full P31 list, for legendary/fictional flagging). All claim extraction is rank-aware: preferred values are picked over normal, deprecated values are skipped. Output on the 2026-05 dump: 913,830 rows.
2. **Recency and pre-filter cut.** Drop figures with `death_year >= 2000` (the player spawn point sits at year 2000; post-2000 figures are also where still-contested politics concentrate). Drop clearly empty Stage-1 rows (no description, low identifier_count) as a cheap pre-filter before the expensive Stage 3 fetch. This is NOT a notability ranking; it is a stub pre-filter. Final article-quality filtering happens in Stage 3 once article content is available.
3. **Wikipedia article extraction.** For each surviving title, use the multistream index to seek to the article in the XML dump and pull its wikitext. Parse with `mwparserfromhell`: extract the lead paragraph and the list of `[[link]]` targets. Resolve redirects against a redirects table extracted from the same dump pass.
4. **Article quality filter.** Drop articles below a word-count or lead-completeness threshold. The cut criterion is "is this a real article" not "is this person famous." Target output: small enough to ship (see Target scale) while preserving the ordinary-people texture that makes the piece honest about Wikipedia coverage.
5. **Place extraction.** Collect every birth and death place QID and resolve its coordinates and country from Wikidata, writing `places.parquet`. This is the geo-enrichment pass that turns a birthplace QID into a longitude for angular placement.
6. **Placement.** Assign each figure a radial-time position (era as radius, population-equalized longitude as angle; see Placement and density) and derive its landmark tier. Writes `placement.parquet`.

(Link-graph restriction is folded into Stage 3, where outgoing links are resolved through redirects and filtered to in-corpus targets as each article is parsed.)

Stages 1 and 3 are the slow ones (bz2-bound). Stages 2, 4, 5, 6 are seconds to minutes. The Parquet file is the boundary that lets us iterate freely on everything downstream.

Wikidata-based filter:

1. Restrict to articles whose Wikidata entity has `instance of (P31) = human (Q5)`.
2. Require `date of death (P570)` to be present (excludes living people, ensures "historical").
3. Require `death_year < 2000` for the recency cut.

Rationale: Wikidata gives a clean, principled filter and yields useful derived properties (lifespan, occupation, era, nationality) for free. Category-based filtering inside Wikipedia is messy and inconsistent. Notability ranking is explicitly avoided: the piece is meant to be an honest map of who Wikipedia covers, including the long tail of ordinary figures.

### Angular placement

The original plan set angular position from an embedding (sentence-transformers on the lead, or node2vec on the people-link subgraph) projected through UMAP, grouping contemporaries by semantic neighborhood. The Stage 1-4 data review killed this. Both signals encode social class: the people-link graph is assortative by prominence (edges with both endpoints in the top 10% are ~22x over-represented), and even a prominence-blind version that down-weights hub links by `1/sqrt(degree)` did not fix it, because fame in this corpus is fundamentally geographic. Recognizable figures concentrate in a narrow Western-European longitude band, so any axis correlated with prominence collapses them into a single wedge. Grouping by social class is the one thing the piece must avoid.

So angle is geography, made honest by equalization. Each figure's birthplace (else death place) longitude is mapped through its population CDF:

    angle = 2*pi * (#figures with longitude <= this) / (#geo-anchored figures)

Dense longitude bands stretch across a wide arc and sparse ones shrink to slivers, so every direction carries comparable population while east-west order and regional adjacency survive (China stays "up", the Americas "down-left"). The transform is global, so a region keeps a stable angle across every era ring. Raw longitude was rejected: it leaves the disc lopsided and renders the Western overrepresentation as a literal bright wedge. Equalization trades that visibility for a navigable field, and the primary subject, recency, stays honest on the radial axis regardless.

About 73% of figures carry a real birth/death coordinate. The quarter without one are not scattered at random; they fall through a ladder that reads only recorded data, never an inference from the name. First P27 citizenship, then a hand-curated gazetteer (`gazetteer.json`) over the English description, which already names a demonym, polity, or region for most ("German nobleman", "ancient Greek physician"). A resolved country is not turned into a centroid, which would stack thousands of figures on one longitude as a hard national spoke. Instead the figure borrows a real longitude sampled from an anchored compatriot, so it spreads across that country's true arc (US citizenship figures span 99 degrees, matching the 99 of anchored US figures) and folds into the same population CDF. That lifts coverage from 73% to about 95%. The embedding idea resurfaced here too and was declined again for the same reason it was for angle: a name predicts ethnolinguistic origin, not citizenship, fails hardest on the anglophone majority it would most need to split, and would fabricate the very data whose absence the piece is about. The residual ~5% are genuinely place-less in the record; they keep the deterministic per-QID random angle and carry `geo_source = null`, which the renderer reads to mark them adrift rather than confidently positioned, the spatial counterpart to the date-uncertainty haze. The people-link graph still ships from Stage 3 in case a future iteration wants it, but nothing in placement consumes it, and there is no embedding or UMAP step in the pipeline.

### Placement and density

The piece pivots on **time as radial distance from origin.** The player spawns at year 2000 on an empty landing pad in the center; walking outward moves the player back in time, and book density thins out with depth, making the gaps in collective recorded knowledge the explicit subject of the piece. Radius is `R_INNER + (R_MAX - R_INNER) * t^0.75` with `t = (2000 - death_year) / 2800`, the sub-linear exponent compressing the dense recent centuries so the modern crowd stays legible rather than smearing into a thin core. `R_INNER` (30 units) keeps a clear spawn clearing the player stands in; the most recent figures ring its edge, and jitter that would carry a book into the pad is reflected back out rather than clamped to the origin. Angular position around each year-ring is geographic (see Angular placement), so books at the same radius are contemporaries grouped by region: a sweep at fixed radius takes the player through similar-era figures from neighbouring parts of the world, and walking inward or outward threads a slice of geography through dense modernity into sparse antiquity.

Radial distance is enforced explicitly from death_year and angle comes from geography, so the layout is computed directly rather than projected. The earlier open question of whether radial-time should emerge from a pure UMAP-2D layout is moot now that there is no embedding to project.

Hard overlaps resolved with minimal jitter or a few iterations of pairwise repulsion using a spatial hash. Lloyd (Voronoi) relaxation is the heavier alternative if needed.

The density distribution is exported as a 2D field that shapes the terrain (see Terrain generation): dunes rise in the sparse outer regions of deep history, flattening into clearings where modern figures cluster. Layout and terrain walk hand in hand: regenerating the layout means regenerating the terrain.

### Visual variation

Two channels carry orthogonal signals:

- **Mesh variant: landmark tier.** The `landmark_tier` column (major / minor / ordinary) selects the base book mesh. Majors read as grander books (a folio or monument silhouette) so a recognizable figure is legible as a reference point from a distance; minors and ordinary figures take the plainer open/closed/stacked meshes, with the specific variant randomized per instance for texture. This replaces the original k-means-cluster basis, which is gone with the embedding and would no longer reinforce anything spatial. Book color and a per-tier scale multiplier are candidate extra channels, deferred until there is a scene to tune them against.
- **Thickness: notability magnitude.** Book thickness scales with `sitelink_count` on a gentle (log) curve, so the modern crowd keeps internal texture and a Confucius or Shakespeare stands taller than a mid-tier major. Surface condition (well-bound to falling-apart, dust accumulation) can still encode age. Read as player-facing recognizability and a navigation cue, not a fame ranking: the books framing already disclaims any monument-erecting notability claim (see Concept).

Era is already encoded by radial position, so a separate color-by-era channel is largely redundant; a subtle sun-faded tint at the deep end may still help legibility at distance.

Per-instance jitter on top: small random rotation, slight tilt, partial burial in sand. Cheap, adds life, reinforces the half-reclaimed feel.

### Terrain generation

Heightmap precomputed from the book distribution. Three steps:

1. **Density field.** Gaussian kernel density estimate over book `(x, z)` coordinates, sampled on a grid (1024×1024 covering world bounds is the working target). Produces a continuous 2D density function.
2. **Dune field.** Anisotropic ridged fBm on the same grid. Sample coordinates are stretched along a chosen wind direction so ridges align with it. The ridge transform (`1 - |2 * fbm - 1|`) makes peaks sharp and troughs flat; asymmetric bias optional. Produces dune-shaped dunes rather than amorphous hills.
3. **Composite.** Multiply the dune field by `(1 - density_mask)`. Dunes rise where books are sparse, which under the radial-time layout means dunes rise dramatically in the outer regions of deep history while clearings flatten around modern clusters near the center. Books sit in basins or half-buried in flanking sand. The metaphor: lost knowledge is buried, and the further back you walk the more of it the desert has reclaimed.

Output: a single heightmap, sampled by both the renderer (for terrain vertices) and the player controller (for player Y). Bilinear sampling in both cases.

Optional: precomputed per-vertex terrain colors (sand tones varying with elevation and density), shipped as a second texture. Fallback is per-vertex color computed from height in-shader.

### Output format

The pipeline produces a versioned world dataset, treated as a single regenerable bundle:

- `books.json.gz`: array of per-book records.
- `heightmap.bin` (or `.png`): precomputed terrain heightmap.
- `metadata.json`: world bounds, heightmap dimensions and world-units-per-pixel, dataset version, content hash of inputs for cache busting.

Per-book record (target ~500 bytes uncompressed):

- Wikidata QID or stable ID
- Title
- Lead sentence
- World coordinates `(x, z)` (from placement: radial-time distance and geographic angle; Y derived from heightmap at runtime)
- Landmark tier (major / minor / ordinary; selects the book mesh)
- Death year (drives radial position; also surfaced in placard)
- Birth year (surfaced in placard)
- `sitelink_count` (drives book thickness on a log curve; the notability magnitude)
- Random seed for per-instance jitter
- Wikipedia URL

Heightmap format: raw `Float32Array` dumped to a binary file is the simplest path (4MB at 1024×1024, gzips well since heightfields are smooth). Alternative: 16-bit PNG, smaller but needs careful decoding. 8-bit PNG loses too much precision at this world scale.

At the original ~10k target a single bundle was viable (~5MB compressed JSON plus ~2-3MB compressed heightmap). At the actual ~419k scale the data ships chunked per spatial tile instead; the tile grid (see Runtime architecture) doubles as the chunk boundary.

## Target scale

The ~10k figure was the original target, when the plan was an aggressive notability-style cut. That changed. The Stage 4 quality cut keys on whether an article carries real narrative past its opener, not on fame, to preserve the ordinary-people texture that makes the piece honest about Wikipedia coverage. After the death_year < 2000 and quality cuts the corpus is ~419k figures, and the world ships at that scale. This makes spatial tile chunking of the data required rather than optional (see Output format and Runtime architecture); a single bundle is viable only up to ~30-50k.

## Runtime architecture

### Stack

- Three.js for rendering.
- Vite for dev server and build.
- TypeScript.
- WebGL2 backend (WebGPU later, when Three.js's backend stabilises).

Rejected alternatives: Godot (large WASM payload, threading complications on web, editor advantage irrelevant for data-driven placement), Bevy (rough WASM dev loop, ECS doesn't earn its cost in a near-static scene, large bundle).

### Spatial structure

Uniform grid tiling of the world. Each tile owns its own `InstancedMesh` per book mesh type, containing only the books whose computed `(x, z)` position lands in that tile.

The same grid serves three purposes:

1. Rendering: only tiles within view distance are added to the scene; each tile's InstancedMesh frustum-culls as one unit.
2. Proximity queries: "what books are near the player" is a range query against the player's cell plus neighbors.
3. Streaming boundary: if data ever chunks per tile, the same grid defines the chunks.

Cell size roughly matches the largest query radius (proximity check or view-distance unit, whichever is larger). Note that the radial layout means tile density varies dramatically from center to edge; an angular-radial tiling scheme is an alternative to consider if uniform-grid streaming wastes effort on near-empty outer tiles.

### Terrain

Terrain is precomputed (see Preprocessing / Terrain generation) and shipped as a heightmap. Runtime samples it, never computes it.

Heightmap loaded once at startup. Kept in memory as a `Float32Array` for CPU-side sampling, and uploaded to a `DataTexture` for GPU-side sampling. Bilinear interpolation in both.

Terrain geometry generated per-tile to match the book tile grid. Each tile builds a displaced plane from its region of the heightmap, with the same culling and streaming behavior as the book tiles. Tile boundaries align with heightmap samples to prevent seams.

Player Y derives from sampling the heightmap at the player's `(x, z)` plus eye-height offset. No raycast against terrain geometry needed. Renderer and player controller share a data source, so terrain and movement stay consistent.

### Book rendering

2-3 GLTF base meshes (open, closed, stacked), loaded once. One `InstancedMesh` per (mesh type, tile) pair. Per-instance transform matrix carries position, rotation, scale variation; thickness is encoded as per-instance scale on the Z axis to avoid needing distinct meshes per word-count bucket. Optional per-instance color via `InstancedBufferAttribute` for the sun-faded depth tint.

### Lighting

- One directional light at low angle (sun). Long shadows are most of the mood.
- Hemisphere light for sky/ground ambient.
- Cascaded Shadow Maps (CSM) for directional shadows, 2-3 cascades with logarithmic split. Fallback to baked blob decals under each book if CSM performance wobbles on the lower quality profile.

### Atmospherics

- `FogExp2` with warm sand color, density tuned to hide tile-load boundary.
- Procedural sky: fragment shader gradient between zenith and horizon colors based on view direction Y. No skybox texture.
- Optional sparse dust particles. Defer until base scene is working.

### Movement and input

- Player controller behind an interface. `DesktopController` uses `PointerLockControls`, WASD plus mouse look. `TouchController` (later) uses virtual joystick plus drag-to-look.
- Euler-angle FPS camera, pitch clamped to ±89°.
- Walk speed slow (around 2-3 units/sec) to suit the pacing. Optional hold-to-run.
- No physics. Player Y snaps to terrain height plus eye offset each frame.

### Interaction

Each frame, query player's cell plus 8 neighbors against the spatial hash. Pick the closest book within trigger radius (say 4 units). If a book is in range, show a DOM overlay with title, dates, lead sentence, and a link to Wikipedia.

DOM overlay rather than canvas text: easier to style as a book placard or open-page motif, accessible, and trivially supports clicking through to Wikipedia.

## Quality profile system

A single `QualityProfile` config threaded through the rendering systems from day one. Two presets initially: `desktop` and `mobile`. Selectable, with room for auto-detect or adaptive tiering later.

Knobs:

- View distance (drives fog density and tile load radius)
- Tile load radius
- Maximum instanced books rendered per frame
- Shadow map enabled, cascade count, cascade range
- Terrain plane subdivision count
- Particle/dust count
- `renderer.setPixelRatio` cap (critical on mobile, where devicePixelRatio of 2-3 tanks fill rate)
- Anti-aliasing on/off
- Target frame rate

Threaded as a config object, not constants scattered across files. Retrofitting after the fact is the failure mode to avoid.

## Algorithm inventory

Named algorithms in use, listed by subsystem so they are easy to look up.

Spatial and culling:

- Uniform grid / spatial hashing for tiles and proximity queries.
- AABB-vs-frustum culling per tile (built into Three.js).
- Distance culling at the tile level.

Terrain (preprocessing):

- Gaussian kernel density estimate (KDE) over book positions for the density mask.
- Simplex noise as the noise primitive (preferred over classic Perlin).
- Fractional Brownian motion stacking 4-6 octaves of simplex.
- Anisotropic sampling: stretch sample coordinates along a wind direction for directional ridges.
- Ridged transform (`1 - |2 * fbm - 1|`) for sharp peaks and flat troughs.
- Density-mask multiplication to suppress dunes around clusters.

Terrain (runtime):

- Bilinear sampling of the precomputed heightmap, CPU-side (player Y) and GPU-side (vertices).

Book placement:

- Radial-time layout: death_year mapped to radial distance via `R_INNER + (R_MAX - R_INNER) * t^0.75`, with an empty inner landing pad and jitter reflected at its edge.
- Population-CDF (quantile) transform of birthplace longitude for the angular coordinate (see Angular placement).
- Per-QID hash (blake2b) for deterministic jitter and for the random angle of the place-less residue (figures with no coordinate, citizenship, or gazetteer hit).
- Landmark tiering: absolute sitelink floor for the global tier, top-per-(sector x era) cell for the local tier (see Visual variation).
- Iterative pairwise repulsion via spatial hash for overlap resolution, if hard overlaps need it.
- Lloyd (Voronoi) relaxation as a heavier alternative if needed.

Lighting and shadows:

- Cascaded Shadow Maps for directional light shadows.
- Hemisphere lighting for sky/ground ambient.
- Lambertian diffuse, optionally half-Lambert for softer unlit-side falloff.

Atmospherics:

- Exponential squared fog (`FogExp2`).
- Procedural sky gradient via fragment shader.

Movement and interaction:

- Euler-angle first-person camera.
- Kinematic player on heightmap (no physics).
- Range query on the spatial hash for nearby books.

Rendering:

- GPU instancing via `InstancedMesh`.
- Instanced vertex attributes for per-instance variation.

Explicitly not building: BVH/octree, pathfinding, physics engine, PBR materials.

## Open decisions

These are deferred until they need to be made:

1. **Final book count. (Resolved: ~419k.)** The quality cut keys on article narrative, not fame, so the world ships at ~419k figures rather than the original ~10k. Density is handled by the radial layout and tiling, not by a notability cut.
2. **Layout signal: text vs graph embedding. (Resolved: geography, no embedding.)** Both embedding options were dropped. The data review showed text and link-graph signals both encode social class and would cluster the famous into one wedge. Angular position is the population-equalized longitude of birthplace instead (see Angular placement).
3. **Radial-time enforcement: emergent vs explicit. (Resolved: explicit.)** Radius is computed directly from death_year; there is no projection for it to emerge from.
4. **Date resolution and uncertainty.** Many ancient figures have invented or wide date ranges (the Buddha listed as -500 to -500). Options: snap to published `death_year` and accept false precision, or render uncertain figures as a smudge, range, or dimmer marker. A piece *about* gaps in knowledge probably wants to express date uncertainty honestly.
5. **Birth vs death year for radial placement.** Working default is `death_year` (when the record closes). Reconsider if it produces visual artifacts (long-lived figures with influential early careers showing up "later" than expected).
6. **Heightmap format.** Raw `Float32Array` binary is the working plan. Switch to 16-bit PNG if bundle size matters more than load simplicity.
7. **Shadow strategy.** CSM with fallback to baked blob decals. Final call depends on mobile profile performance.
8. **Mobile rollout timing.** Desktop-first; mobile after the core experience is stable. Quality profile and input interface are in from day one so the cost of adding mobile is low.
