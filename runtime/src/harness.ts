// Golden-image regression harness (dev-only, gated by the ?harness URL param).
//
// The renderer is the project's most sensitive component: "correct" is a visual
// judgement and there is no automated test that catches "it looks subtly wrong
// now". Before any cleanup touches the render path, this turns that into a diff.
//
// How it stays deterministic: a frame's only animation is daylight.uDriftTime
// (the drifting daylight mask, which the sky's cloud decks also ride -- see
// sky.ts), and the field placement is seeded (mulberry32, fixed PERM). So if we
// (a) pin the camera to a fixed pose instead of running the controller, and
// (b) freeze uDriftTime to a constant, the same code renders the same pixels.
// A capture tool reads renderer.domElement.toDataURL() per pose (canvas only --
// the DOM HUD/reticle/glance are NOT in that image, which is exactly what we
// want: a pure render-output baseline); scripts/diff.mjs compares to goldens.
//
// Inert unless ?harness is present. When active it never alters the render path
// beyond pinning the camera and the drift clock; the renderer, materials,
// shaders, draw order and uniforms are untouched.

import * as THREE from "three";
import type { DaylightUniforms } from "./daylight";

export const HARNESS_ON =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).has("harness");

// A fixed point on the drift clock. Any constant works for regression (goldens
// and re-captures use the same value); this one lands on a representative mix of
// lit and shadowed ground rather than full glare or full night.
const FROZEN_DRIFT = 137.0;

// A teleporter as shipped in teleporters.json. Typed locally to avoid importing
// from main.ts (which imports this module) and creating a cycle.
interface Tp {
  label: string;
  x: number;
  y: number;
}

interface HarnessCtx {
  camera: THREE.PerspectiveCamera;
  daylight: DaylightUniforms; // we freeze .uDriftTime.value
  teleporters: Tp[];
  sampleHeight: (x: number, z: number) => number; // ground height (world units)
  eyeHeight: number; // EYE_HEIGHT: camera sits this far above the ground
  world: { R_MAX: number };
  renderer: THREE.WebGLRenderer;
  // Supersampled grab: render one frame at an absolute pixel ratio and read it back
  // (see captureAt in main.ts). Used by shot(scale) for crisp promo stills.
  capture: (scale: number) => string;
}

interface Pose {
  name: string;
  pos: THREE.Vector3;
  look: THREE.Vector3;
}

export interface Harness {
  active: boolean;
  // Re-assert the active pose every frame (replaces the controller's update);
  // returns a MoveMode-compatible "walk" so the caller's downstream is unchanged.
  applyPose(): "walk";
  // Pin the drift clock. Call after daylight.update(dt) so it overrides the advance.
  freeze(): void;
}

// Build the pose list from the live world. Teleporters are sorted by radius --
// which IS the era axis (radius is linear in time), so the nearest-centre monument
// is the most recent and the outermost is the most ancient. Anchoring poses on
// monuments guarantees populated ground (monuments sit on real clusters).
function buildPoses(ctx: HarnessCtx): Pose[] {
  const { teleporters, sampleHeight, eyeHeight, world } = ctx;

  const eyeAt = (x: number, z: number) =>
    new THREE.Vector3(x, sampleHeight(x, z) + eyeHeight, z);
  // A look target at roughly the teleporter sphere's centre (it sits ~0.3u proud
  // of the sand) so the ball is framed, not the dirt in front of it.
  const ballAt = (tp: Tp) =>
    new THREE.Vector3(tp.x, sampleHeight(tp.x, tp.y) + 0.3, tp.y);

  const byRadius = [...teleporters].sort(
    (a, b) => Math.hypot(a.x, a.y) - Math.hypot(b.x, b.y),
  );
  const recent = byRadius[0];
  const mid = byRadius[Math.floor(byRadius.length / 2)];
  const ancient = byRadius[byRadius.length - 1];

  // Unit outward (centre -> tp) bearing, and its tangent.
  const radial = (tp: Tp): [number, number] => {
    const r = Math.hypot(tp.x, tp.y) || 1;
    return [tp.x / r, tp.y / r];
  };

  const poses: Pose[] = [];

  // 1. The canonical opening view: stand at the centre plaza and gaze outward
  //    across the empty ring into the field. Tests spawn lighting + the void.
  {
    const p = eyeAt(0, 0);
    poses.push({ name: "spawn", pos: p, look: new THREE.Vector3(300, p.y, 0) });
  }

  // 2. Nose-to-node on the most recent monument: a few units back, looking at the
  //    ball. Tests the ball material (Lambert + flatShading + fresnel rim) and the
  //    dense modern thicket behind it at full near-LOD.
  {
    const [dx, dz] = radial(recent);
    const p = eyeAt(recent.x - dx * 6, recent.y - dz * 6);
    poses.push({ name: "tp-recent-near", pos: p, look: ballAt(recent) });
  }

  // 3. Standing well back from the same monument, looking at it across the dense
  //    field. Tests the near -> mid -> far points LOD handoff in a populated view
  //    (the spot most likely to expose a fade/draw-order regression).
  {
    const [dx, dz] = radial(recent);
    const p = eyeAt(recent.x - dx * 150, recent.y - dz * 150);
    poses.push({ name: "tp-recent-vista", pos: p, look: ballAt(recent) });
  }

  // 4. At a mid-era monument, looking tangentially (along the ring) so the view
  //    threads contemporaries of one region rather than crossing eras. Tests the
  //    geo-hue colouring and book forms mid-field.
  {
    const [dx, dz] = radial(mid);
    const [tx, tz] = [-dz, dx]; // tangent
    const p = eyeAt(mid.x, mid.y);
    poses.push({
      name: "tp-mid-tangent",
      pos: p,
      look: new THREE.Vector3(p.x + tx * 200, p.y, p.z + tz * 200),
    });
  }

  // 5. Out at the most ancient monument, looking back toward the centre across the
  //    sparse deep-past field. Tests thin density + landmark beacons reading
  //    against the void, and the long recession toward the lit core.
  {
    const [dx, dz] = radial(ancient);
    const p = eyeAt(ancient.x + dx * 90, ancient.y + dz * 90);
    poses.push({ name: "tp-ancient-vista", pos: p, look: ballAt(ancient) });
  }

  // 6. Past the outermost books, facing further out: the honest empty frontier
  //    where the field has dissolved and only the sky horizon remains.
  {
    const [dx, dz] = radial(ancient);
    const r = Math.min(Math.hypot(ancient.x, ancient.y) + 600, world.R_MAX);
    const p = eyeAt(dx * r, dz * r);
    poses.push({
      name: "deep-void",
      pos: p,
      look: new THREE.Vector3(dx * (r + 400), p.y - 8, dz * (r + 400)),
    });
  }

  // 7. From the centre, tilt up into the storm dome. Tests sky.ts in isolation:
  //    the gradient and the layered drifting cloud decks (frozen by the clock).
  {
    const p = eyeAt(0, 0);
    poses.push({
      name: "sky-up",
      pos: p,
      look: new THREE.Vector3(120, p.y + 160, 0),
    });
  }

  return poses;
}

export function setupHarness(ctx: HarnessCtx): Harness {
  if (!HARNESS_ON) {
    return { active: false, applyPose: () => "walk", freeze: () => {} };
  }

  const { camera, daylight, renderer } = ctx;
  const poses = buildPoses(ctx);
  const byName = new Map(poses.map((p) => [p.name, p]));
  let current = poses[0];
  let frame = 0;

  const applyPose = (): "walk" => {
    camera.position.copy(current.pos);
    camera.lookAt(current.look);
    frame++;
    return "walk";
  };

  const freeze = (): void => {
    daylight.uDriftTime.value = FROZEN_DRIFT;
  };

  // Console / automation API. Capture flow: set a pose, wait for the field LOD to
  // settle (poll frame()), then read shot().
  (window as unknown as { __HARNESS: unknown }).__HARNESS = {
    poses: () => poses.map((p) => p.name),
    pose: (name: string) => {
      const p = byName.get(name);
      if (!p) throw new Error(`unknown pose: ${name}`);
      current = p;
      frame = 0; // reset so callers can wait for the LOD refill to settle
    },
    frame: () => frame,
    // scale=1: read the live buffer as-is. scale>1: supersample via the host's
    // captureAt (e.g. shot(2) -> 2x the CSS window, display-independent) for crisp
    // promo stills; restores the live scale before returning.
    shot: (scale = 1) =>
      scale === 1
        ? renderer.domElement.toDataURL("image/png")
        : ctx.capture(scale),
  };

  // eslint-disable-next-line no-console
  console.log(
    `[harness] active. poses: ${poses.map((p) => p.name).join(", ")}. ` +
      `Drive via window.__HARNESS.`,
  );

  return { active: true, applyPose, freeze };
}
