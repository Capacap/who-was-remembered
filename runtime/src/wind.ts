// Procedurally-synthesised desert wind. Zero payload: nothing is served — the whole
// bed is Web Audio noise + filters, so the download is unchanged and the sound never
// loops audibly. Three layers:
//   bed  - a low brown-noise rumble through a lowpass; the constant desert floor.
//   gust - band-passed pink-noise hiss swelling on a slow LFO; a wind that never sits
//          still, independent of the player.
//   rush - a brighter band of pink hiss driven by the player's speed; silent at a walk,
//          rising as you skate across the voids (quadratic, so it stays out of the way
//          until you are genuinely fast).
// Kept near-subliminal (MASTER_CEIL) to match the Kuindzhi restraint of the visuals.
// Everything is lazy: the AudioContext is not touched until the first resume() (a user
// gesture), so we never fight the autoplay policy or spin up audio for a player who
// never enters.

export interface Wind {
  /** Start/unlock the AudioContext. Must be called inside a user-gesture handler. */
  resume(): void;
  /** Master volume 0..1 (the pause slider). 0 fades out and suspends the context. */
  setVolume(v: number): void;
  /** Normalised player speed 0..1 for the rush coupling. */
  setSpeed(s: number): void;
  /** Per-frame smoothing of volume + speed toward their targets. */
  update(dt: number): void;
}

const MASTER_CEIL = 0.55; // slider max maps here, not to 1.0 — the wind is a presence,
//                           not a foreground sound.

export function createWind(): Wind {
  let ctx: AudioContext | null = null;

  // smoothed control state (targets set by setVolume/setSpeed, eased in update())
  let volTarget = 0;
  let volCur = 0;
  let speedTarget = 0;
  let speedCur = 0;

  // nodes modulated every frame; assigned in build()
  let master: GainNode;
  let bedFilter: BiquadFilterNode;
  let rushGain: GainNode;

  // Fill an AudioBuffer with coloured noise. A long buffer (seconds) looped under the
  // moving LFOs/filters reads as continuous, never as a loop.
  function noiseBuffer(kind: "brown" | "pink", seconds: number): AudioBuffer {
    const len = Math.floor(ctx!.sampleRate * seconds);
    const buf = ctx!.createBuffer(1, len, ctx!.sampleRate);
    const d = buf.getChannelData(0);
    if (kind === "brown") {
      // brown = integrated white, leaked back toward 0 so it can't wander off-scale.
      let last = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        last = (last + 0.02 * w) / 1.02;
        d[i] = last * 3.5; // make up the gain the leak/integration loses
      }
    } else {
      // pink via Paul Kellet's economical filter bank (1/f, warmer than white).
      let b0 = 0,
        b1 = 0,
        b2 = 0,
        b3 = 0,
        b4 = 0,
        b5 = 0,
        b6 = 0;
      for (let i = 0; i < len; i++) {
        const w = Math.random() * 2 - 1;
        b0 = 0.99886 * b0 + w * 0.0555179;
        b1 = 0.99332 * b1 + w * 0.0750759;
        b2 = 0.969 * b2 + w * 0.153852;
        b3 = 0.8665 * b3 + w * 0.3104856;
        b4 = 0.55 * b4 + w * 0.5329522;
        b5 = -0.7616 * b5 - w * 0.016898;
        d[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
        b6 = w * 0.115926;
      }
    }
    return buf;
  }

  function source(buf: AudioBuffer): AudioBufferSourceNode {
    const s = ctx!.createBufferSource();
    s.buffer = buf;
    s.loop = true;
    return s;
  }

  function build(): void {
    master = ctx!.createGain();
    master.gain.value = 0; // fades up in update() toward volCur
    master.connect(ctx!.destination);

    const brown = source(noiseBuffer("brown", 6));
    const pink1 = source(noiseBuffer("pink", 7));
    const pink2 = source(noiseBuffer("pink", 5));

    // bed: low brown rumble — the steady floor. Its lowpass opens a touch with speed.
    bedFilter = ctx!.createBiquadFilter();
    bedFilter.type = "lowpass";
    bedFilter.frequency.value = 360;
    const bedGain = ctx!.createGain();
    bedGain.gain.value = 0.32; // subdued: the rumble is a floor, not a presence of its own
    brown.connect(bedFilter).connect(bedGain).connect(master);

    // gust: band-passed pink hiss that breathes on two slow LFOs — one swelling its
    // level, one sweeping the band centre so each gust shifts colour.
    const gustFilter = ctx!.createBiquadFilter();
    gustFilter.type = "bandpass";
    gustFilter.frequency.value = 700;
    gustFilter.Q.value = 0.7;
    const gustGain = ctx!.createGain();
    gustGain.gain.value = 0.18;
    pink1.connect(gustFilter).connect(gustGain).connect(master);

    const lfoLevel = ctx!.createOscillator();
    lfoLevel.frequency.value = 0.08; // ~12s breath
    const lfoLevelGain = ctx!.createGain();
    lfoLevelGain.gain.value = 0.12;
    lfoLevel.connect(lfoLevelGain).connect(gustGain.gain);

    const lfoSweep = ctx!.createOscillator();
    lfoSweep.frequency.value = 0.17;
    const lfoSweepGain = ctx!.createGain();
    lfoSweepGain.gain.value = 220;
    lfoSweep.connect(lfoSweepGain).connect(gustFilter.frequency);

    // rush: brighter band, gain driven by player speed in update(). Silent at rest.
    const rushFilter = ctx!.createBiquadFilter();
    rushFilter.type = "bandpass";
    rushFilter.frequency.value = 820; // near the gust band (was 1500, a brighter hiss):
    rushFilter.Q.value = 0.7; //         the rush now reads as the same wind, just louder
    rushGain = ctx!.createGain();
    rushGain.gain.value = 0;
    pink2.connect(rushFilter).connect(rushGain).connect(master);

    brown.start();
    pink1.start();
    pink2.start();
    lfoLevel.start();
    lfoSweep.start();
  }

  function resume(): void {
    if (!ctx) {
      const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      ctx = new AC();
      build();
    }
    if (ctx.state === "suspended") void ctx.resume();
  }

  function setVolume(v: number): void {
    volTarget = Math.max(0, Math.min(1, v));
  }

  function setSpeed(s: number): void {
    speedTarget = Math.max(0, Math.min(1, s));
  }

  function update(dt: number): void {
    if (!ctx) return;
    // ease toward targets — volume slow for a soft fade, speed a little snappier so
    // the rush tracks a skate without lag but still without a click.
    volCur += (volTarget - volCur) * (1 - Math.exp(-2.5 * dt));
    speedCur += (speedTarget - speedCur) * (1 - Math.exp(-1.5 * dt));

    master.gain.value = volCur * MASTER_CEIL;
    bedFilter.frequency.value = 360 + speedCur * 340; // speed opens the rumble slightly
    rushGain.gain.value = speedCur * speedCur * 0.5; // quadratic: quiet until truly fast

    // when the player has set the slider to 0 and the fade is done, suspend the context
    // to stop burning CPU synthesising noise nobody hears. Gated on volTarget (not the
    // eased volCur) so a fade-down to a low-but-nonzero level never thrashes suspend.
    if (volTarget > 0 && ctx.state === "suspended") void ctx.resume();
    else if (volTarget === 0 && volCur < 0.0008 && ctx.state === "running") void ctx.suspend();
  }

  return { resume, setVolume, setSpeed, update };
}
