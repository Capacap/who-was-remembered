# Terrain plan

The ground started as a runtime analytic function and is now a baked **heightmap**
that the runtime tessellates live as a geometry clipmap. This is the living plan
for that work; update status markers as steps land.

The big change from the original version of this plan: we are **not** baking a
low-poly mesh any more. The pipeline bakes the height *field* (a raster); the
runtime builds the visible surface from it every frame (`runtime/src/terrain.ts:
createGround`). That single decision upends the parts of the old plan that assumed
a static `ground.bin` to seat props on, which is most of it. What survives is the
heightmap generation (Step 1) and the colour intent (Step 3). The seating problem
that pivot created (Step 4) is now solved: props seat on the facet the clipmap
actually draws, reconstructed at load, rather than on any baked surface. Colour
(Step 3) is the remaining open work.

## Why a baked field at all

The analytic `getGroundHeight` (pointwise Perlin dunes) can't express generation
that needs neighbour info: slip-face asymmetry, and later erosion-class passes. So
the height field becomes a baked artifact produced by the pipeline, where a raster
pass has every neighbour to hand. The runtime ships and reads that raster.

Why a runtime clipmap instead of a baked mesh (the old plan): a static low-poly
mesh has to choose one triangle budget for the whole disc, which is either coarse
underfoot or ruinous to ship, and a decimated disc is coarsest exactly where you
stand. A camera-following clipmap keeps on-screen facet size bounded everywhere for
a trivial vertex count, and it tessellates the same shipped raster the player walks
on, so the walk height and the visible ground agree by construction. The decimated
mesh (Step 2 below) was built and benched before this pivot and is retired.

This whole concern stays decoration (the vertical axis carries no data): it lives
downstream of the data-truth and legibility stages and never feeds back into
placement. It reads the layout, it does not change it.

## Artifact chain

```
stage7 teleporters.parquet ─┐
stage8 layout.parquet ──────┼─> stage9 heightmap.npz ─> heightmap.bin (SHIPPED)
                            │                          └─> colour raster (Step 3, TBD)
                            └─> (book/teleporter x,z into the export)

runtime: heightmap.bin ─> geometry clipmap (terrain.ts) ─> visible ground
                       ├─> sampleHeight ─> player feet
                       └─> facetHeight ─> prop seating (the drawn facet)
```

The heightmap is the one elevation source: the clipmap tessellates it, the player's
feet read it, and props seat on `facetHeight` (the facet the clipmap draws, built
from the same field), so they agree. No mesh ships. Colour is still computed in-shader
from world position today; Step 3 may add a baked colour raster
beside the heightmap.

## Step 1 — Heightmap (`stage9_mesh.py`, `--serve-heightmap`)  [shape done; resolution open]

A numpy port of the dune field sampled to a raster, with a shaded-relief +
high-res-crop inspection render so we tune by sight and number, not by guessing.

Done: uniform wind-aligned transverse field; calm centre plateau; plazas carved at
the 26 teleporters; `heightmap.npz` holds the float32 field plus its world mapping
(size + resolution); `--serve-heightmap` writes `runtime/public/heightmap.bin`
(uint32 res, float32 world_size, float32 height[res*res] row-major).

Done: slip-face asymmetry, two raster passes the pointwise runtime can't do:
- `lee_shear`: a downwind warp p -> p + L*D(p)*wind, every sand column sliding
  downwind in proportion to its height so crests migrate over the lee. Applied as
  the gap-free inverse map (fixed-point solve, LEE_SHEAR_ITERS >= 4; a single pass
  stays symmetric, the trap the first attempt fell into). At LEE_SHEAR=0.7 dunes
  read windward ~24deg, lee ~45deg, slip-face terrain over ~20% of lee area vs ~2%
  windward.
- `avalanche`: a mass-conserving thermal sand-slide. It does NOT clamp uniform
  over-steep faces to repose (material passes straight through a constant slope);
  it bites at curvature, rounding the knife-edge crest, filling the toe, capping
  fold cliffs. A naturalising pass, not a repose clamp. (`report_asymmetry` measures
  the windward/lee slope tails so this is tuned by number.)

Still open:
- [ ] Ridge continuity / scale variation if the crests feel too uniform.
- [ ] Trough floor (bottoms at ~8u, reads fine) and the raised ring at r~700-1400
  where the hill and dune envelopes stack.
- [ ] Lock the raster resolution. Currently 2048 (8.8u/texel). The fidelity bound is
  no longer "what the decimator needs", nor the prop float (Step 4 now seats on the
  drawn facet, so float is handled at any resolution); it is just how crisp the dune
  crests should read against the finest clipmap cell (8u, ~1 texel). 2048 makes the
  finest facet about one texel; 4096 would sharpen it. A look-and-decide call.

Output: `cache/heightmap.npz`, `runtime/public/heightmap.bin`.

## Step 2 — Decimated mesh  [RETIRED — superseded by the runtime clipmap]

The original plan built a dense grid from the heightmap and quadric-decimated it to
a feature-following low-poly mesh (`fast-simplification`, ~30k tris, unindexed-flat
`ground.bin`). It worked and benched well (~2-3x lower deviation than a uniform grid
at equal budget), but the clipmap replaces it: same faceted look, bounded facet size
at every distance, no ship cost, and it seats the player on the surface it draws.

`ground.bin` is no longer fetched and is git-ignored. The bench in `stage9_mesh.py`
(builder + chord-deviation render + `fast-simplification` wiring) is kept only as the
heightmap inspection tool and the host of the `--serve-heightmap` exporter; the
`--bake`/`--target-tris` mesh path is dead for production but left as a measuring
instrument.

## Step 3 — Colouring  [intent stands; delivery reframed]

Colour is a composite of three layers, dominant to subtle. The runtime currently
does layer 1 only, in-shader from world position (`groundColor`). With no mesh,
"bake per-vertex colour" no longer applies; the two delivery options are:
- compute all three layers in the clipmap shader from world position (cheap, no new
  asset, but form/biome both want data the shader doesn't have handy), or
- bake a low-res **colour raster** beside the heightmap and sample it in the shader
  (the natural parallel to the height raster; form and biome are pipeline-side data,
  so this is the likely path).

The three layers, unchanged in intent:
1. **Time (radial). Keep.** Pale present -> sand -> grey deep-past void. The
   meaningful axis (recency lighting the map); stays the dominant read.
2. **Form (height / aspect). New.** Wind-scoured pale crests, darker cooler troughs,
   a tint on the lee face. Driven by height relative to the local mean and
   slope/aspect, both already computed in the heightmap passes. **Bake material, not
   lighting:** the runtime already shades facets directionally (Lambert + flat
   normals + sun), so the baked tint says what the sand *is*, not where the sun hits,
   or it double-shades and breaks when the light moves.
3. **Biome (layout-driven). New.** Each teleporter region a gentle, distinct
   hue/sat shift, as a smooth low-frequency hue field with the 26 anchors as
   soft-blended control points (not 26 hard Voronoi cells, not 26 separate hues which
   would read garish). Open: group anchors into ~5-7 families, and what drives the
   offset (palette, longitude, era). Keep saturation low so time + desert dominate.

Composite: time base, hue/sat nudged by biome, lightness modulated by form.

## Step 4 — Seating (books, props, player)  [LANDED — seat on the drawn facet]

Solved by candidate 1 below: books seat on the facet the clipmap actually draws,
reconstructed at runtime load. No pipeline change, no exported columns, no bias.

The problem the clipmap created: there is **no single rendered surface** to seat on.
What the ground is at a world point depends on the camera, both which level covers it
(cell 8..128 by distance) and the camera-relative morph in each level's outer band.
And the float itself has two sources on a tight convex crest: the flat facet chords
*below* the smooth `sampleHeight` field (deviation ~ facet^2 x curvature), and the
in-plane jitter shoves the drawn corners sideways (up to GROUND_JIT_FRAC*cell), which
on a steep face becomes a vertical offset of a couple of units. Seating on
`sampleHeight` floats over both; seating on the un-jittered chord still floats over
the jitter.

The key realisation: seat to the drawn chord, not the smooth field. The mesh is what
the player sees, so it is the seat truth. And the finest level's surface near the
player is **computable without the camera**: the jitter is world-anchored (keyed to
the world cell hash) and the finest level carries no morph within ~456u, so its
facets are a deterministic function of world position. That turns the feared
mesh-raycast into a closed-form lookup, no mesh and no ray.

What landed (`terrain.ts: facetHeight`, used by `main.ts: buildField`):
- For each book, find the finest-level (cell 8) jittered triangle its (x,z) lands in
  and return that triangle's plane height. `finestVertex` reproduces a drawn vertex
  exactly (the same disk-jitter hash, height sampled at the jittered position);
  `baryHeight` does the point-in-triangle test and plane interpolation in one pass.
- Jitter can pull the containing triangle into a neighbour, so the search covers the
  3x3 cell block around the point (jitter < cell, so one ring suffices). A fast path
  tests the book's own cell first (4 vertices, no lattice) and only falls back to the
  full 4x4 lattice on a cross-edge miss. ~540ms for the full 576k field at load.
- Tilt still follows the smooth `sampleNormal`; the float was a height problem, and
  the facet's own normal would only add per-book tilt jumps a book this small does
  not need. The existing `lift` still settles the underside a hair into the sand.

Residual, by design: a static seat is exact only where the finest level draws (the
on-foot view that matters). A distant book under a coarser morphing level breathes as
the camera moves; that is unavoidable for any fixed seat, shows only when flying, and
is faded out by distance. Verified clean on foot, including dune crests.

Player walk height stays `sampleHeight` per frame: the player is a point, so faceting
float does not apply to the feet.

## Step 5 — Runtime integration  [LANDED]

The clipmap is in (`terrain.ts: createGround`, committed):
- 5 levels, cells 8/16/32/64/128u, each a 128-cell grid snapped to its own cell so
  facets never swim; built entirely from `heightmap.bin`.
- Seams: geomorph each level's outer band to the coarser level's exact triangulated
  chord (C0), jitter relaxed to zero at the rim and around the hole so both sides
  meet on a plain lattice; a per-level depth bias breaks the coplanar tie; an
  explicit `renderOrder` (finest first) keeps the transparent levels from
  double-blending their overlap rings (the band/flicker fix).
- Distance dissolve: per-level camera-distance opacity fade to the dome, so the world
  dissolves circularly and no square footprint edge ever shows.
- Books seat on `facetHeight` (the drawn facet, Step 4); pillars, ring and player
  feet read `sampleHeight`/`sampleNormal` (flat plazas and a point, so no faceting
  float). The analytic dune field stays only as the pre-heightmap-load fallback.

Remaining runtime work: Step 3 colour (raster or in-shader). World dims already live
in `world.json`, shared by both sides.

## Sequencing

Step 4 (seating) has landed, so Step 3 (colour) is the live frontier. Lock the
heightmap resolution (Step 1) by eye against dune crispness, not a mesh budget or the
now-handled prop float. Step 3 reads the final field whenever it lands.
