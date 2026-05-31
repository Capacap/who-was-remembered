# Terrain bake plan

Moving the ground from a runtime analytic function to a baked pipeline asset.
This is the living plan for that work; update status markers as steps land.

## Why

The runtime today generates terrain from `getGroundHeight` (analytic Perlin
dunes) and seats books, the player and teleporter bases on that same function,
so they agree by construction. Two things break that:

1. We want generation that a pointwise function can't express (slip-face
   asymmetry needs neighbour info; later, erosion-class passes).
2. We want a low-poly mesh, and a low-poly mesh deviates from any continuous
   surface *on purpose*. A book seated on the function then floats above the
   chord mesh, worst exactly where the facets are biggest, which is the look we
   want. The float gets worse as the mesh gets more stylised, not better.

So the height field becomes a **baked artifact** and everything that sits on the
ground seats on the **rendered mesh**, not on the function. The function retires
to being the generator that feeds the bake.

This stays a decoration concern (the vertical axis carries no data), so it lives
downstream of the data-truth and legibility stages and never feeds back into
placement. It reads the layout, it does not change it.

## Artifact chain

```
stage7 teleporters.parquet ─┐
stage8 layout.parquet ──────┼─> stage9 heightmap.npz ─> mesh (ground.bin)
                            │                          ├─> per-vertex colours
                            │                          └─> seated book y+normal
                            └────────────────────────────> (into the export)
```

Runtime loads the mesh; books read their baked height; the player raycasts the
mesh for walk height. No analytic terrain and no heightmap ship to the browser.

## Step 1 — Heightmap (`stage9_heightmap.py`)  [dune shape done; resolution lock deferred to Step 2]

A faithful numpy port of the runtime dune field, sampled to a raster, with a
shaded-relief + high-res-crop inspection render so we stop guessing.

Done: port matches the approved runtime shape; uniform wind-aligned transverse
field; calm centre plateau; plazas carved at the 26 teleporters; `heightmap.npz`
holds the float32 field plus its world mapping (size + resolution).

Done: slip-face asymmetry. The symmetric ridged field now goes through two
raster passes (see `lee_shear`, `avalanche`) that the pointwise runtime can't:
- `lee_shear`: a downwind warp p -> p + L*D(p)*wind, every sand column sliding
  downwind in proportion to its height so crests migrate over the lee. Applied
  as the gap-free inverse map (fixed-point solve, LEE_SHEAR_ITERS >= 4; a single
  pass stays symmetric, which is the trap the first attempt fell into). At
  LEE_SHEAR=0.7 the dunes read windward ~24deg, lee ~45deg, slip-face-angle
  terrain over ~20% of lee area vs ~2% windward. The forward scatter gives the
  same shape but is a Python loop, too slow for the full raster.
- `avalanche`: a mass-conserving thermal sand-slide. It does NOT clamp uniform
  over-steep faces to repose (material passes straight through a constant
  slope); it bites at curvature, rounding the knife-edge crest, filling the toe,
  capping fold cliffs. A naturalising pass, not a repose clamp. (`report_asymmetry`
  in the script measures windward/lee slope tails so this is tuned by number,
  not by eye; the render now also draws an along-wind profile transect.)

This is the deliberate divergence from terrain.ts: the runtime keeps its
symmetric analytic dunes until it loads the baked mesh; it does not mirror these
two passes. The asymmetry is the first thing the bake buys that the runtime
couldn't express.

Still open in Step 1:
- [ ] Ridge continuity / scale variation if the crests feel too uniform (they
  currently read varied enough; revisit after seeing them meshed).
- [ ] Decide the trough floor (bottoms at ~8u, not 0; reads fine as uneven sand)
  and the raised ring at r~700-1400 where the hill and dune envelopes stack.
- [ ] Lock the final raster resolution (currently 2048 = 8.8u/texel; the mesh
  fidelity is bounded by this, so it likely wants 4096+). Defer to Step 2: the
  right resolution is the one the decimation needs, judged against facet reads.

Output: `cache/heightmap.npz`.

## Step 2 — Mesh (decimation)

Build a dense grid mesh from the heightmap, then decimate to a low-poly mesh
whose triangulation follows features: big facets on the flats (the Vane look),
tight facets on the crests (the resolution dunes need). One pass serves both the
aesthetic and the fidelity, which a uniform grid can't.

Decisions:
- **Tool / dependency.** Start with `fast-simplification` (small, quadric edge
  collapse, returns an indexed mesh, preserves features). Fall back to headless
  Blender's decimate (planar mode, edge-angle control) if quadric output reads
  too noisy for the faceted look.
- **Budget.** Target a tri count, not a fixed ratio; tune against how the facets
  read. Clip the mesh to a disc of radius ~R_MAX + fade margin (~8000u) rather
  than the full 18000 square; the corners are pure void and always faded out.
- **Topology.** Indexed-smooth vs unindexed-flat. The low-poly look wants flat
  per-face shading, and per-*face* colour (every facet one tone) needs unwelded
  triangles. Lean unindexed-flat with baked per-face normals. Re-evaluate the
  facet character after the first decimation; it won't match the current
  jitter-driven look.
- **Format.** Custom `ground.bin` (Float32 positions, Uint32 index or unindexed,
  colour buffer) to match the project's existing `.bin` idiom and avoid a
  GLTF/Draco runtime dependency. GLB only if we want the tooling.

## Step 3 — Colouring

Colour is baked per vertex/face on the *final* decimated mesh, as a composite of
three layers, dominant to subtle. The current radial gradient is layer 1 only.

1. **Time (radial). Keep.** Pale present -> sand -> grey deep-past void. This is
   the meaningful axis (recency lighting the map) and must stay the dominant
   read.
2. **Form (height / aspect). New.** Give the dunes material variation: wind-
   scoured pale crests, darker cooler troughs, a tint on the lee face. Driven by
   height relative to the local mean (works at any radius) and slope/aspect.
   **Bake material, not lighting.** The runtime already shades facets
   directionally (Lambert + flat normals + sun). If we bake directional shading
   into the colour we double-shade and it breaks when the light changes. So the
   baked tint says what the sand *is*, not where the sun hits.
3. **Biome (layout-driven). New.** Each teleporter region gets a gentle, distinct
   hue/saturation shift so parts of the world are identifiable without breaking
   the desert. Drive it as a smooth low-frequency hue field with the 26 anchors
   as control points (soft-blended), *not* 26 hard Voronoi cells, and *not* 26
   separate hues (they would collide and read garish). Open: whether to group
   anchors into ~5-7 biome families and what drives the hue offset (arbitrary
   palette, longitude, era). Keep saturation low so time + desert still dominate.

Composite: time base, hue/sat nudged by biome, lightness modulated by form.

## Step 4 — Seating (books, props, player)

The consistency fix. Sample the **final decimated mesh** (the rendered surface,
not the heightmap, which the decimation deviates from) at each book's (x, y),
and bake the resulting height and seating normal.

- Books / static props: ray-cast straight down onto the mesh; write `y` (and a
  seating normal, or a least-squares plane over the footprint so a book spanning
  several facets doesn't see-saw). Tool: `trimesh` + embree, or a 2D triangle
  grid index. Land it as columns on the export (or a sidecar the export reads).
- Player walk height: the runtime raycasts the loaded ground mesh under the
  player each frame (one ray, trivial), so the feet agree with the visible
  ground and nothing needs the heightmap at runtime.

## Step 5 — Runtime integration

- Load `ground.bin` (mesh + colours) instead of `buildGroundMesh`; retire the
  analytic generation. Keep the distance-fade material injection.
- Books instance at their baked `y`; teleporter bases likewise.
- Player height from a mesh raycast.
- Move the world dimensions (currently `SIZE = 18000` duplicated in `terrain.ts`
  and `WORLD_SIZE` in `stage9`) into `world.json` so both sides read one source.

## Cross-cutting decisions still open

- Single mesh vs tiles. Start single (disc-clipped); tile only if load/frame
  budget demands it. The memory expects tiling eventually.
- Mesh format: custom bin (recommended) vs GLB.
- Topology: unindexed-flat (recommended) vs indexed-smooth.
- Biome hue source and count (Step 3).
- Final heightmap resolution and mesh tri budget.

## Sequencing

Iterate the field cheaply (Step 1) before meshing, because the field is free to
change now and expensive to re-derive once a mesh and seated books depend on it.
Lock resolution and the shared world dimensions before Step 4. Colour and seating
both run on the final mesh, so they come after decimation, not before.

Resume point: Step 1 dune shape is in good shape (slip-face asymmetry landed).
Next is Step 2 (mesh decimation), unless we want to revisit the open Step 1 items
above once the dunes are seen as a meshed surface rather than a heightmap render.
