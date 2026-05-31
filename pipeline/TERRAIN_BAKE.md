# Terrain plan

The ground started as a runtime analytic function and is now a baked **heightmap**
that the runtime tessellates live as a geometry clipmap. This is the living plan
for that work; update status markers as steps land.

The big change from the original version of this plan: we are **not** baking a
low-poly mesh any more. The pipeline bakes the height *field* (a raster); the
runtime builds the visible surface from it every frame (`runtime/src/terrain.ts:
createGround`). That single decision upends the parts of the old plan that assumed
a static `ground.bin` to seat props on, which is most of it. What survives is the
heightmap generation (Step 1) and the colour intent (Step 3); what changes is that
there is no longer one rendered surface to bake against, which is the heart of the
open seating problem (Step 4).

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
                       └─> sampleHeight ─> player feet + prop seating
```

The heightmap is the one elevation source: the clipmap tessellates it and the
player's feet and props read it, so they agree. No mesh ships. Colour is still
computed in-shader from world position today; Step 3 may add a baked colour raster
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
  no longer "what the decimator needs"; it is the finest clipmap cell (8u, ~1 texel)
  and the heightmap-curvature term that drives prop float (Step 4). 2048 means the
  finest facet is about one texel, so up-close chord error is already small;
  4096 would halve it. Decide against the Step 4 float, not against a mesh budget.

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

## Step 4 — Seating (books, props, player)  [OPEN — the float-on-convex problem]

This is the unresolved one, and the clipmap is what makes it hard. The old plan
seated everything on the baked mesh so props and ground agreed by construction. The
clipmap removes that anchor: **there is no single rendered surface.** What the ground
is at a world point depends on the camera, both which level covers it (cell 8..128 by
distance) and the camera-relative morph in each level's outer band. A prop baked to
one height cannot match a surface that changes as you move.

The float itself: a flat facet chords the smooth height field. On convex curvature
(crests) the chord sits *below* `sampleHeight`, so a book seated at `sampleHeight`
floats; on concave (troughs) it sinks. The deviation grows with facet_size^2 x
curvature, so it is worst on tight crests and big (far) facets. Books seat on
`sampleHeight` today, so they float over convex curvature, which is what we are
trying to fix.

What is fixed vs camera-relative, which decides what is even seatable:
- `sampleHeight` (bilinear height field) is fixed. Stable, but it is the smooth
  surface, not the drawn chord, hence the float.
- The in-plane jitter is **world-anchored** (keyed to the world cell hash, not the
  camera), and the finest level carries **no morph** within ~450u of the camera. So
  the finest level's facets near the player are a deterministic function of world
  position: jitter the cell corners, sample height there, split the quad on the
  fixed diagonal. That triangle is computable without knowing the camera.
- The morph and the choice of level are camera-relative, so the far-field surface
  under a fixed point breathes as the camera moves.

So the float has a static part (faceting; present on any low-poly surface and partly
the intended "settled into the sand" look) and a dynamic part (the surface under a
distant point shifts with the camera, so far books breathe/pop). The dynamic part
only shows when flying; on foot, the books you are near are on the finest,
un-morphed, world-anchored facets, which is the only surface worth seating to.

Candidate approaches, none committed:
1. **Seat on the finest-level facet, at runtime load.** The runtime already owns the
   jitter hash and `sampleHeight`, so for each book it can find the cell-8 triangle
   the book lands in and drop it onto that plane (height + a seating normal, or a
   least-squares plane over the footprint so a book spanning facets doesn't see-saw).
   O(1) per book, so 576k at load is a few ms, not the heavy mesh-raycast the old
   plan feared (there is no mesh to raycast). Exact for the close-up view, which is
   the view that matters. Residual: distant books still breathe across level handoffs
   (small, faded, flight-only). **Leaning toward this:** it needs no pipeline change,
   no exported columns, and no pipeline/runtime jitter-constant coupling, because the
   runtime that draws the facet is the one that seats on it.
2. **Bake the same seat in the pipeline.** Replicate the finest-level tessellation in
   numpy and write per-book height/normal columns on the export. Same result as (1),
   but couples the pipeline to the runtime's jitter constants and grid (a coupling to
   police on every change). Only worth it if load-time cost ever bites, which (1)
   suggests it won't.
3. **Curvature-biased bilinear seat.** Keep `sampleHeight`, subtract an offset
   proportional to local convexity (the heightmap Laplacian) x cell^2 so crests sink
   the book onto roughly where the chord sits. Cheap, no jitter at all, approximate
   (one facet scale, ignores the jitter displacement). A single global lever to tune
   by eye; a fallback if (1) is fiddly.
4. **Accept the static float as aesthetic** and only treat the dynamic pop if it
   bothers (today it only shows when flying). The current `lift` (settle the book
   into the sand by part of its spine) is the crude version of this.

Player walk height is already `sampleHeight` per frame: cheap, and the player is a
point so faceting float does not apply to the feet. Leave it.

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
- Props (books, pillars, ring) and player feet read `sampleHeight`/`sampleNormal`.
  The analytic dune field stays only as the pre-heightmap-load fallback.

Remaining runtime work: Step 3 colour (raster or in-shader) and Step 4 seating.
World dims already live in `world.json`, shared by both sides.

## Sequencing

Lock the heightmap resolution (Step 1) against the Step 4 float and the finest cell,
not against a mesh budget. Step 3 (colour) and Step 4 (seating) are independent and
can land in either order; both read the final field. Step 4 is the one with an open
design question, so it is the next real decision.
