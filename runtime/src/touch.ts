// Touch controls for coarse-pointer devices. There is no pointer lock and no
// keyboard here, so this module is the whole input surface: the LEFT half drags a
// floating stick to move (analog — deflection is speed), the RIGHT half drags to
// look, a quick tap on the right inspects whatever the centre reticle is over, and
// two chips toggle skate / open the pause menu. It owns only the DOM and the
// pointer bookkeeping; the callbacks feed the same controller + scene flow the
// keyboard/mouse path uses, so grounded physics, skate and the world bound are shared.
//
// The element ids it binds live in index.html (#touch and children). It never reads
// game state — main() drives setEnabled() as overlays/pause take over, and forwards
// the callbacks into the controller. Desktop never instantiates this.

export interface TouchOpts {
  // analog move axis, each component -1..1; the vector's magnitude is the stick
  // deflection (so a half-pushed stick walks at half speed).
  onMove(x: number, y: number): void;
  // look drag delta in px since the last move event (fed to the controller as the
  // mouse movementX/Y equivalent).
  onLook(dx: number, dy: number): void;
  onSkateToggle(on: boolean): void;
  // a tap (a quick press that didn't become a drag) in the look zone — inspect.
  onTap(): void;
  onPause(): void;
}

export interface TouchControls {
  // show/hide + reset the controls. Off when an overlay or the pause menu owns the
  // screen, so taps reach the card instead of steering the camera.
  setEnabled(on: boolean): void;
  isSkating(): boolean;
}

export function createTouchControls(opts: TouchOpts): TouchControls {
  const root = document.getElementById("touch") as HTMLDivElement;
  const stick = document.getElementById("touch-stick") as HTMLDivElement;
  const thumb = document.getElementById("touch-thumb") as HTMLDivElement;
  const skateBtn = document.getElementById("touch-skate") as HTMLButtonElement;
  const pauseBtn = document.getElementById("touch-pause") as HTMLButtonElement;

  const STICK_R = 56; // px to full deflection (half the 112px base)
  const TAP_MOVE = 12; // px; a press that strays less than this (and is quick) is a tap
  const TAP_MS = 250;

  let enabled = false;
  let skating = false;

  // the move pointer lives in the left half; its origin is wherever it touched down
  // (a FLOATING stick, so you never hunt for a fixed pad).
  let moveId = -1;
  let moveOX = 0;
  let moveOY = 0;

  // the look pointer lives in the right half; we track its last position for the
  // per-event delta and its start/age to tell a tap from a drag.
  let lookId = -1;
  let lookLX = 0;
  let lookLY = 0;
  let lookSX = 0;
  let lookSY = 0;
  let lookT0 = 0;
  let lookMoved = 0; // peak distance from the touch-down point

  const resetThumb = (): void => {
    thumb.style.transform = "translate(-50%, -50%)";
  };

  const onDown = (e: PointerEvent): void => {
    if (!enabled) return;
    const half = window.innerWidth / 2;
    if (e.clientX < half && moveId === -1) {
      // left half → spawn the floating stick under the finger
      moveId = e.pointerId;
      moveOX = e.clientX;
      moveOY = e.clientY;
      stick.style.left = `${moveOX}px`;
      stick.style.top = `${moveOY}px`;
      stick.style.display = "block";
      resetThumb();
    } else if (lookId === -1) {
      // anything else (right half, or a second left touch) → look / tap candidate
      lookId = e.pointerId;
      lookLX = lookSX = e.clientX;
      lookLY = lookSY = e.clientY;
      lookT0 = performance.now();
      lookMoved = 0;
    } else {
      return;
    }
    // capture so the rest of this gesture keeps reaching us even if the finger
    // slides over a button or off the element.
    root.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onMove = (e: PointerEvent): void => {
    if (e.pointerId === moveId) {
      let dx = e.clientX - moveOX;
      let dy = e.clientY - moveOY;
      const d = Math.hypot(dx, dy);
      if (d > STICK_R) {
        dx *= STICK_R / d;
        dy *= STICK_R / d;
      }
      thumb.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
      // screen-up is forward (-dy); screen-right is strafe-right (+dx).
      opts.onMove(dx / STICK_R, -dy / STICK_R);
    } else if (e.pointerId === lookId) {
      const dx = e.clientX - lookLX;
      const dy = e.clientY - lookLY;
      lookLX = e.clientX;
      lookLY = e.clientY;
      lookMoved = Math.max(lookMoved, Math.hypot(e.clientX - lookSX, e.clientY - lookSY));
      opts.onLook(dx, dy);
    }
  };

  const endPointer = (e: PointerEvent): void => {
    if (e.pointerId === moveId) {
      moveId = -1;
      stick.style.display = "none";
      opts.onMove(0, 0); // release → coast to rest under the controller's decel
    } else if (e.pointerId === lookId) {
      const quick = performance.now() - lookT0 < TAP_MS;
      if (quick && lookMoved < TAP_MOVE) opts.onTap();
      lookId = -1;
    }
  };

  root.addEventListener("pointerdown", onDown);
  root.addEventListener("pointermove", onMove);
  root.addEventListener("pointerup", endPointer);
  root.addEventListener("pointercancel", endPointer);

  // buttons sit inside #touch; swallow their pointerdown so it never starts a
  // move/look drag, and act on the click.
  const swallow = (e: Event): void => e.stopPropagation();
  skateBtn.addEventListener("pointerdown", swallow);
  pauseBtn.addEventListener("pointerdown", swallow);
  skateBtn.addEventListener("click", () => {
    skating = !skating;
    skateBtn.classList.toggle("on", skating);
    opts.onSkateToggle(skating);
  });
  pauseBtn.addEventListener("click", () => opts.onPause());

  const setEnabled = (on: boolean): void => {
    enabled = on;
    root.style.display = on ? "block" : "none";
    if (!on) {
      // an overlay/pause took the screen: drop any in-progress gesture + visuals so
      // we don't resume mid-drag, and disengage skate so a resume never silently coasts.
      moveId = -1;
      lookId = -1;
      stick.style.display = "none";
      opts.onMove(0, 0);
      if (skating) {
        skating = false;
        skateBtn.classList.remove("on");
        opts.onSkateToggle(false);
      }
    }
  };

  return { setEnabled, isSkating: () => skating };
}
