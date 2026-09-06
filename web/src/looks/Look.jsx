import { useEffect, useRef, useState } from "react";

import { liveOpen, liveStep, liveClose } from "../engine";
import { CLASSES } from "../world/lobes";
import { MOTIONS, motionById, randomMotion, system } from "./motions";
import { SCOPE, paintGround, paintTrace } from "./scope";
import "./looks.css";

/*
 * The first viewport: a phosphor screen with a network on it.
 *
 * A different choreography every time the page loads, because the shape of the problem is the
 * one thing that should not be a fixed image. Six lobes force a rosette, a ring inside a ring
 * forces a circle, a braid forces a wave — and the network has to find each of them live.
 *
 * The learning rate is very low, and the reason is worth writing down because two rounds of
 * lowering it changed nothing. Measured while an observation is dragged across the plane, the
 * fraction of the drag during which the boundary is still behind it:
 *
 *   adam 0.028   0%      adam 0.003    0%      sgd 0.015   92% but never converges
 *   adam 0.006   0%      adam 0.0015  12%      momentum 0.008   0%
 *
 * Adam normalises its step by the gradient, so one misplaced observation always buys a full
 * step and the network is never behind at any rate worth using. Only below about 0.002 does it
 * fall behind at all while still solving the problem at rest. Plain SGD lags beautifully and
 * never converges, which is worse than arriving too fast.
 *
 * Drawn on a phosphor that decays rather than a canvas that clears, that small lag reads as a
 * cut bending, trailing and catching up.
 */

const GRID = 72;
const STEPS = 1;

/**
 * Three rates out of one run.
 *
 * WARM finds the first boundary from a random start, and it is the part worth watching on
 * arrival: a few seconds of a cut groping its way into place. CRUISE is what keeps it correct
 * while the choreography moves underneath — slower than warm, but it cannot be much slower or
 * the network simply loses the problem and the screen reads twenty-seven per cent.
 *
 * FOCUS is the interesting one: the moment an observation is picked up, and for a few seconds
 * after it is dropped, the rate halves. That is the whole point of being able to touch this
 * thing. Measured as the fraction of a drag during which the boundary is still behind the
 * observation being dragged:
 *
 *   adam 0.028   0%     adam 0.003    0%     momentum 0.008    0%
 *   adam 0.006   0%     adam 0.0015  12%     sgd 0.015        92%, and never converges
 *
 * Adam normalises its step by the gradient, so one misplaced observation always buys a full
 * step: the rate was never the lever at any value anyone would train at. Only far below 0.002
 * does it fall behind at all, and only there does dragging a point look like work rather than
 * like painting the ground.
 */
const WARM = 0.0011;
const CRUISE = 0.0015;
const FOCUS = 0.00075;
const WARM_FOR = 6;              // seconds of arrival before the run settles into cruising
const FOCUS_AFTER = 4.5;         // seconds the slow rate is held after an observation is dropped

/**
 * The drawing is a square cropped to cover the viewport, so a wide screen throws away the top
 * and bottom of it. This is the half-extent of the world that survives the crop — the region a
 * choreography is allowed to use. It depends on the viewport's shape, never on its size, so
 * zooming the page leaves the animation exactly where it was.
 */
const boundsFor = (w, h) => ({ ex: w >= h ? 1 : w / h, ey: h >= w ? 1 : h / w });

/**
 * The backing store is a fixed square, stretched by CSS to cover whatever the viewport is.
 *
 * Sizing it to the viewport meant a canvas of eleven thousand pixels a side at the lowest zoom
 * — half a gigabyte of texture for a field that is only seventy-two cells wide. Fixed, the
 * drawing is identical at every zoom level and costs the same.
 */
const CANVAS = 1400;

const FADE = 0.2;                // how much of the phosphor is lost per frame

/** `#/look/wave?lr=0.0004&h=8&opt=sgd` — for sweeping the feel without editing code. */
function tuning() {
  const q = new URLSearchParams(window.location.hash.split("?")[1] || "");
  const h = q.get("h");
  return {
    warm: q.get("warm") ? Number(q.get("warm")) : null,
    cruise: q.get("cruise") ? Number(q.get("cruise")) : CRUISE,
    focus: q.get("focus") ? Number(q.get("focus")) : FOCUS,
    hidden: h ? h.split(",").map(Number) : [12, 12],
    steps: q.get("steps") ? Number(q.get("steps")) : STEPS,
    opt: q.get("opt") || "adam",
  };
}

export default function Look({ id }) {
  const canvas = useRef(null);
  const glow = useRef(null);
  const box = useRef({ w: 1440, h: 900 });
  const [motion] = useState(() => (id && motionById(id)) || randomMotion());
  const [readout, setReadout] = useState(null);
  const [grabbed, setGrabbed] = useState(null);
  const world = useRef(null);
  const [, force] = useState(0);

  useEffect(() => {
    const onResize = () => {
      box.current = { w: window.innerWidth, h: window.innerHeight };
      world.current?.resize(boundsFor(box.current.w, box.current.h));
      force((n) => n + 1);
    };
    onResize();
    window.addEventListener("resize", onResize);

    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    const tune = tuning();
    // a choreography whose cut is harder to find may ask for a faster arrival than the default
    const warm = tune.warm ?? motion.warm ?? WARM;
    const sys = system(motion, boundsFor(box.current.w, box.current.h));
    world.current = sys;
    if (typeof window !== "undefined") window.__sys = sys;   // a hook for measuring the motion

    const key = `scope-${motion.id}`;
    let raf = 0;
    let readAt = 0;
    let last = performance.now();
    const t0 = last;
    let rate = null;
    let touchedAt = null;

    try {
      liveOpen(key, { points: sys.advance(0, 0), labels: sys.labels, hidden: tune.hidden,
                      activation: "tanh", optimizer: tune.opt, lr: warm, gridN: GRID, seed: 0 });
    } catch (e) {
      console.error("the screen could not start", e);
      return;
    }

    if (!glow.current) glow.current = document.createElement("canvas");

    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const age = (now - t0) / 1000;

      const pts = sys.advance(age, dt);

      // fast enough to find the first cut, and an order of magnitude slower the moment anyone
      // touches it, so a dragged observation is something the network is visibly working at
      const touching = sys.heldIndex != null || age - (touchedAt ?? -99) < FOCUS_AFTER;
      if (sys.heldIndex != null) touchedAt = age;
      const want = touching ? tune.focus : age < WARM_FOR ? warm : tune.cruise;
      const send = want === rate ? undefined : (rate = want);

      let out;
      try { out = liveStep(key, tune.steps, pts, null, send); }
      catch (e) { console.error("the screen stopped", e); cancelAnimationFrame(raf); return; }

      const node = canvas.current;
      if (node) {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const px = Math.round(CANVAS * dpr);
        if (node.width !== px) {
          node.width = node.height = px;
          glow.current.width = glow.current.height = px;
        }
        // the backing store is fixed; CSS stretches it to whatever has to be covered
        const cover = Math.max(box.current.w, box.current.h);
        node.style.width = node.style.height = `${cover}px`;

        const run = { field: out.field, confidence: out.confidence, n: GRID, classes: CLASSES };

        const ctx = node.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        paintGround(ctx, run, CANVAS);

        // the phosphor: faded, never cleared, so the cut leaves the wake it is chasing through
        const g = glow.current.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.globalCompositeOperation = "destination-out";
        g.fillStyle = `rgba(0, 0, 0, ${FADE})`;
        g.fillRect(0, 0, CANVAS, CANVAS);
        g.globalCompositeOperation = "source-over";
        paintTrace(g, run, CANVAS, pts, sys.labels, sys.heldIndex);

        ctx.globalCompositeOperation = "lighter";
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(glow.current, 0, 0);
        ctx.globalCompositeOperation = "source-over";
      }

      if (typeof window !== "undefined") window.__last = { field: out.field, confidence: out.confidence, n: GRID };

      if (now - readAt > 170) {
        readAt = now;
        setReadout({ loss: out.loss, accuracy: out.accuracy });
      }
    };

    raf = requestAnimationFrame(tick);

    return () => { cancelAnimationFrame(raf); try { liveClose(key); } catch { /* gone */ } };
  }, [motion]);

  const at = (event) => {
    const b = canvas.current.getBoundingClientRect();
    return [((event.clientX - b.left) / b.width) * 2 - 1, 1 - ((event.clientY - b.top) / b.height) * 2];
  };

  const drag = useRef({ last: null, vx: 0, vy: 0 });

  const down = (event) => {
    event.preventDefault();
    const sys = world.current;
    const [x, y] = at(event);
    const i = sys.pick(x, y);
    if (i < 0) return;
    sys.hold(i, [x, y]);
    drag.current = { last: [x, y, performance.now()], vx: 0, vy: 0 };
    setGrabbed(i);
    canvas.current.setPointerCapture(event.pointerId);
  };

  const move = (event) => {
    const sys = world.current;
    if (!sys || sys.heldIndex == null) return;
    const [x, y] = at(event);
    sys.moveTo([x, y]);

    // the speed the pointer is carrying, so letting go can throw
    const prev = drag.current.last;
    const now = performance.now();
    if (prev) {
      const dt = Math.max(0.008, (now - prev[2]) / 1000);
      drag.current.vx = drag.current.vx * 0.6 + ((x - prev[0]) / dt) * 0.4;
      drag.current.vy = drag.current.vy * 0.6 + ((y - prev[1]) / dt) * 0.4;
    }
    drag.current.last = [x, y, now];
  };

  const up = () => {
    const sys = world.current;
    if (!sys || sys.heldIndex == null) return;
    sys.release(drag.current.vx, drag.current.vy);
    drag.current = { last: null, vx: 0, vy: 0 };
    setGrabbed(null);
  };

  const style = {
    "--ground": SCOPE.ground, "--ink": SCOPE.ink,
    "--quiet": SCOPE.quiet, "--accent": SCOPE.accent,
  };
  const cover = Math.max(box.current.w, box.current.h);

  return (
    <div className="look" style={style}>
      <canvas
        ref={canvas} className="look-canvas"
        style={{
          left: (box.current.w - cover) / 2, top: (box.current.h - cover) / 2,
          cursor: grabbed == null ? "grab" : "grabbing", touchAction: "pan-y",
        }}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
        aria-label="A live decision boundary you can rearrange by dragging an observation."
      />

      <header className="look-head">
        <a className="look-mark" href="#/">visualML</a>
        <nav className="look-nav">
          <a href="#/boundary">boundary</a>
          <a href="#/overfit">capacity</a>
          <a href="#/lenses">two lenses</a>
          <a href="#/descent">descent</a>
        </nav>
      </header>

      <div className="look-say">
        <h1 className="look-title">Machine learning<br />with nothing<br />underneath it</h1>
        <p className="look-lede">
          An autograd engine, seven optimizers and four plates that run the mathematics
          in front of you. No PyTorch. No server.
        </p>
        <div className="look-acts">
          <a className="look-act" href="#/boundary">Open the plates</a>
          <a className="look-act look-act-quiet" href="https://github.com/Roberto-Nsoni/visualML"
             target="_blank" rel="noreferrer">Read the source</a>
        </div>
      </div>

      <footer className="look-foot">
        <div className="look-band">
          <span><i>loss</i>{readout ? readout.loss.toFixed(4) : "—"}</span>
          <span><i>accuracy</i>{readout ? (readout.accuracy * 100).toFixed(1) + "%" : "—"}</span>
        </div>
        <p className="look-name">
          <a href="#/looks">{motion.name}</a> · {motion.note}
        </p>
      </footer>
    </div>
  );
}

/** Every choreography, so they can be compared without reloading nine times. */
export function LooksIndex() {
  return (
    <div className="looks-index">
      <h1>Coreografías</h1>
      <p>
        La portada elige una al azar en cada recarga. El mundo es el mismo — la pantalla de
        fósforo — y lo que cambia es la forma del problema que la red tiene que resolver en vivo.
      </p>
      <ol>
        {MOTIONS.map((m, i) => (
          <li key={m.id}>
            <a href={`#/look/${m.id}`}>
              <b>{String(i + 1).padStart(2, "0")} · {m.name}</b>
              <span>{m.note}</span>
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}
