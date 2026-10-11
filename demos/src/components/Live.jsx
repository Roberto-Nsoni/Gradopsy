import { useCallback, useEffect, useRef, useState } from "react";

import { liveOpen, liveStep, liveClose } from "../engine";
import { INK, CUT_NOW, cutPast, cutPath, strokeCut, fitCanvas, paintField, drawPoints, frame }
  from "../world/section";

/*
 * The live section.
 *
 * Every other drawing on this site replays a run that was computed whole. This one has no
 * recording at all: two clouds of observations circle each other, a real MLP takes three
 * gradient steps inside every animation frame, and the cut you are looking at is the one the
 * network holds right now. There is nothing to scrub, because nothing has happened yet.
 *
 * Pull an observation across the cut and it stays where you put it — it keeps travelling with
 * its own cloud from there — so the network has to carve a pocket around it. That is the whole
 * argument of the project in a gesture: the mathematics is running, in this tab, against you.
 */

const LOBES = 6;                 // six lobes, alternating class: opposite lobes disagree, so no
const PER_LOBE = 3;              // straight line and no stripe can do it — only a real rosette
const CLASSES = 2;
const GRID = 80;                 // 9.5 ms a frame with three steps, measured in this browser
const STEPS_PER_FRAME = 3;
const WAKE = 4;                  // how many earlier cuts stay behind the live one
const WAKE_EVERY = 5;            // frames between wake samples, so the trail spans a real second

const ORBIT = 0.56;              // how far each lobe sits from the middle
const SPREAD = 0.15;             // how loosely an observation sits inside its lobe
const SPIN = 0.1;                // radians a second: slow enough to read, never still

/**
 * Where a lobe's centre is at time t.
 *
 * Six lobes sixty degrees apart with the classes alternating, which is chosen and not decorative.
 * Two clouds side by side are cut by a straight line, and a straight line is a thing this engine
 * did not need to exist to draw. Four lobes at right angles are still solved by a single stripe,
 * because opposite lobes then share a class. At six, opposite lobes disagree: nothing but a real
 * rosette separates them, so the first mark anyone sees is the network doing something no linear
 * model can. Turning it slowly turns the rosette with it.
 */
const lobeAt = (lobe, t) => {
  const a = t * SPIN + (lobe * 2 * Math.PI) / LOBES;
  return [Math.cos(a) * ORBIT, Math.sin(a) * ORBIT];
};

function seedPoints() {
  // Deterministic: the page opens on the same arrangement every time, so the first thing a
  // sceptic sees is not a different picture on every reload.
  const rng = (i) => Math.sin(i * 12.9898) * 43758.5453;
  const frac = (v) => v - Math.floor(v);

  return Array.from({ length: LOBES * PER_LOBE }, (_, i) => {
    const lobe = i % LOBES;
    const a = frac(rng(i + 1)) * Math.PI * 2;
    const r = Math.sqrt(frac(rng(i + 97))) * SPREAD;

    return { lobe, cls: lobe % 2, ox: Math.cos(a) * r, oy: Math.sin(a) * r,
             wobble: frac(rng(i + 41)) * Math.PI * 2 };
  });
}

const clamp = (v) => Math.max(-0.94, Math.min(0.94, v));

export default function LiveField({ size, onReadout }) {
  const canvas = useRef(null);
  const seeds = useRef(seedPoints());
  const held = useRef(null);
  const [holding, setHolding] = useState(false);

  // Everything the loop touches lives in refs: sixty state updates a second would spend the
  // whole frame budget in React and none of it on the mathematics.
  const loop = useRef({ positions: [], wake: [], tick: 0 });

  const place = useCallback((t) => {
    return seeds.current.map(({ lobe, ox, oy, wobble }) => {
      const [cx, cy] = lobeAt(lobe, t);
      const w = t * 0.6 + wobble;
      return [clamp(cx + ox + Math.cos(w) * 0.02), clamp(cy + oy + Math.sin(w * 1.3) * 0.02)];
    });
  }, []);

  useEffect(() => {
    if (!size) return;

    const labels = seeds.current.map((s) => s.cls);
    const key = "hero";
    let opened = false;
    let raf = 0;
    let readoutAt = 0;
    const started = performance.now();

    try {
      liveOpen(key, { points: place(0), labels, hidden: [8, 8], activation: "tanh",
                      optimizer: "adam", lr: 0.08, gridN: GRID, seed: 0 });
      opened = true;
    } catch (e) {
      console.error("live section could not open a session", e);
      return;
    }

    const state = loop.current;

    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      const t = (now - started) / 1000;

      state.lastT = t;
      const drifting = place(t);
      const grab = held.current;
      if (grab) drifting[grab.index] = grab.at;

      state.positions = drifting;

      let out;
      try {
        out = liveStep(key, STEPS_PER_FRAME, drifting, null);
      } catch (e) {
        console.error("live section stopped", e);
        cancelAnimationFrame(raf);
        return;
      }

      if (state.tick++ % WAKE_EVERY === 0) {
        state.wake.push(cutPath(out.field, out.confidence, GRID, 0, CLASSES, size));
        if (state.wake.length > WAKE) state.wake.shift();
      }

      draw(canvas.current, size, state, out, labels);

      if (now - readoutAt > 140) {
        readoutAt = now;
        onReadout?.({ loss: out.loss, accuracy: out.accuracy, step: out.k });
      }
    };

    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      if (opened) liveClose(key);
    };
  }, [size, place, onReadout]);

  /** Which observation the pointer is over, in the drawing's own coordinates. */
  const at = (event) => {
    const box = canvas.current.getBoundingClientRect();
    return [
      ((event.clientX - box.left) / box.width) * 2 - 1,
      1 - ((event.clientY - box.top) / box.height) * 2,
    ];
  };

  const down = (event) => {
    const [x, y] = at(event);
    const positions = loop.current.positions;
    let best = -1;
    let near = 0.12;

    positions.forEach(([px, py], i) => {
      const d = Math.hypot(px - x, py - y);
      if (d < near) { near = d; best = i; }
    });
    if (best < 0) return;

    // Re-seat the seed on release: the offset it keeps is the one you dropped it at
    held.current = { index: best, at: [clamp(x), clamp(y)] };
    setHolding(true);
    canvas.current.setPointerCapture(event.pointerId);
  };

  const move = (event) => {
    if (!held.current) return;
    const [x, y] = at(event);
    held.current.at = [clamp(x), clamp(y)];
  };

  // Dropping re-seats the observation against its own cloud, so where you left it is where it
  // stays: it goes on travelling from there and the network has to keep accounting for it.
  const up = () => {
    const grab = held.current;
    if (grab) {
      const seed = seeds.current[grab.index];
      const [cx, cy] = lobeAt(seed.lobe, loop.current.lastT ?? 0);
      seed.ox = grab.at[0] - cx;
      seed.oy = grab.at[1] - cy;
    }
    held.current = null;
    setHolding(false);
  };

  return (
    <canvas
      ref={canvas}
      className="live-canvas"
      /* pan-y, not none: this drawing is the top of the page on a phone, and taking the whole
         gesture would trap the scroll on the first thing anyone touches. Vertical swipes belong
         to the page; what is left over moves an observation. */
      style={{ cursor: holding ? "grabbing" : "grab", touchAction: "pan-y" }}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      aria-label="Two moving clouds of observations with a decision boundary a network is
                  redrawing between them, live. Drag an observation to move it."
    />
  );
}

function draw(node, size, state, out, labels) {
  if (!node) return;
  const ctx = fitCanvas(node, size, size);

  ctx.fillStyle = INK.ground;
  ctx.fillRect(0, 0, size, size);
  paintField(ctx, { field: out.field, confidence: out.confidence, n: GRID, offset: 0 },
             size, CLASSES);

  // The cut, with the cuts it held a moment ago receding behind it. On a plate that trail says
  // where training came from; here it says the network is chasing something that will not hold
  // still, which is the same sentence about a different kind of time.
  const wake = state.wake;
  wake.forEach((path, i) => strokeCut(ctx, path, cutPast(wake.length - i, wake.length)));

  const here = cutPath(out.field, out.confidence, GRID, 0, CLASSES, size);
  for (const ink of CUT_NOW) strokeCut(ctx, here, ink);

  const toPixel = (x, y) => [((x + 1) / 2) * size, size - ((y + 1) / 2) * size];
  drawPoints(ctx, state.positions, labels, toPixel, { radius: size > 520 ? 7 : 5.5 });

  frame(ctx, size);
}
