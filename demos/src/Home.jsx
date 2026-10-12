/*
 * The Daylight Section home page. No longer routed: the cover took the front door.
 *
 * Kept because it is the only place three pieces of real evidence are written down — the test
 * count, the engine's line count, and the plain statement that nothing underneath is borrowed.
 * None of that belongs on the cover, and none of it should be lost while the rest of the site
 * is converted. Delete this file once that evidence has a surface of its own.
 */
import { useCallback, useEffect, useRef, useState } from "react";

import { boundaryRun, descentRun, lensesRun, overfitRun, preset, presetCurve, surfaceInfo } from "./engine";
import { MetricBand, Leader, Foot, Arrow } from "./components/Chrome";
import LiveField from "./components/Live";
import { useMeasure, useViewport } from "./components/Section";
import { INK, fitCanvas, paintField, drawPoints, frame } from "./world/section";
import { paintDescent } from "./demos/Descent";
import { paintLandscape } from "./demos/Lenses";

const toPixel = (size) => (x, y) => [((x + 1) / 2) * size, size - ((y + 1) / 2) * size];

export default function Home() {
  return (
    <>
      <Hero />
      <Plates />
      <Evidence />
      <Foot />
    </>
  );
}

/**
 * The first viewport, and the only drawing on the site with no recording behind it.
 *
 * Two clouds of observations circle each other and a real MLP re-cuts the plane between them
 * inside every animation frame. Nothing here was computed in advance, which is why there is no
 * scrubber: there is no run to walk through, only the step the network is on right now. Take
 * hold of an observation and drag it across the cut and the network has to answer for it —
 * which is the project's claim, made in a gesture rather than a sentence.
 */
function Hero() {
  const [stageRef, width] = useMeasure(760);
  const viewport = useViewport();
  const [readout, setReadout] = useState(null);

  // The caption and the measured band are part of the drawing, and the band is the only proof
  // on the page that the numbers are moving, so the whole plate is sized to clear the fold.
  const size = Math.max(280, Math.min(width || 420, 760, viewport - 300));

  const onReadout = useCallback((r) => setReadout(r), []);

  return (
    <section className="hero">
      <div className="hero-say">
        <p className="display hero-designation">Decision<br />field</p>
        <p className="prose hero-prose">
          These eighteen observations are moving, and a network is re-cutting the plane between
          them as you read this — <em className="hero-em">every frame, in this tab</em>, by an
          autograd engine written from scratch in NumPy. Nothing is precomputed and nothing is a
          video. <strong className="hero-strong">Take hold of a point and drag it across the
          cut</strong>, and watch what the network has to do about it.
        </p>

        <div className="leaders">
          <Leader label="network" value="2 · 8 · 8 · 2" />
          <Leader label="optimizer" value="adam · lr 0.08" />
          <Leader label="rate" value="3 steps a frame" />
          <Leader label="engine" value="numpy, in this tab" />
        </div>
      </div>

      <div className="hero-stage" ref={stageRef}>
        <div className="live-plate" style={{ maxWidth: size || undefined }}>
          <LiveField size={size} onReadout={onReadout} />
          <p className="section-hint overline">drag an observation · the cut in gold is now</p>

          <MetricBand items={[
            { label: "step", value: readout ? readout.step : "—" },
            { label: "loss", value: readout ? readout.loss.toFixed(4) : "—" },
            { label: "accuracy", value: readout ? (readout.accuracy * 100).toFixed(1) : "—",
              unit: "%", lit: true },
            { label: "state", value: readout ? "learning" : "starting" },
          ]} />
        </div>
      </div>
    </section>
  );
}

const PLATES = [
  {
    index: 1, href: "#/boundary", title: "Boundary",
    line: "Put points on a plane by hand and watch a network find the shape that separates them. Then pull one hidden unit out and see the single cut it contributes.",
  },
  {
    index: 2, href: "#/overfit", title: "Capacity",
    line: "The same eighteen points, fitted by networks of growing size. Somewhere along the slider the curve stops describing the trend and starts threading the noise.",
  },
  {
    index: 3, href: "#/lenses", title: "Two lenses",
    line: "One run, reported twice on the same step: what the network believes about the plane, and the loss around the path its parameters actually took.",
  },
  {
    index: 4, href: "#/descent", title: "Descent",
    line: "Seven optimizers from one starting point over the same terrain, first order against BFGS and Newton. Newton settles onto a saddle, because a saddle is a stationary point too.",
  },
];

function Plates() {
  const [stills, setStills] = useState({});

  // Real output, computed once. A placeholder image of a plot would be the one lie this page
  // cannot afford, so each plate waits for its own run instead.
  useEffect(() => {
    let cancelled = false;

    const work = [
      ["1", () => {
        const d = preset("moons");
        return { kind: "field", run: boundaryRun({ points: d.points, labels: d.labels, hidden: [8, 8],
          activation: "tanh", optimizer: "adam", lr: 0.05, steps: 160, gridN: 96, seed: 0 }), data: d };
      }],
      ["2", () => {
        const c = presetCurve();
        return { kind: "curve", run: overfitRun(c.points, [48, 48], 1200, 0.015), data: c };
      }],
      ["3", () => {
        const d = preset("circles");
        // The parameter-space half. The input-space half is already the picture on plate 1.
        return { kind: "landscape", run: lensesRun({ points: d.points, labels: d.labels, hidden: [8, 8],
          lr: 0.05, steps: 160, gridN: 40, surfaceN: 46, seed: 0 }) };
      }],
      ["4", () => {
        const info = surfaceInfo("himmelblau", 300);
        const active = ["sgd", "adam", "bfgs", "newton"];
        return { kind: "descent", info, active,
                 runs: descentRun("himmelblau", [
                   { key: "sgd", lr: 0.002 }, { key: "adam", lr: 0.05 },
                   { key: "bfgs", lr: 0 }, { key: "newton", lr: 0 },
                 ], 200, null) };
      }],
    ];

    (async () => {
      for (const [id, make] of work) {
        await new Promise((r) => setTimeout(r, 24));
        if (cancelled) return;
        setStills((s) => ({ ...s, [id]: make() }));
      }
    })();

    return () => { cancelled = true; };
  }, []);

  return (
    <section className="plates" id="plates">
      <div className="plates-head">
        <h2 className="headline">Four plates</h2>
        <p className="prose">
          Each one is the same engine under a different lens. Everything below is live: open a
          plate and the numbers move because the mathematics is running, not replaying.
        </p>
      </div>

      <ol className="plate-list">
        {PLATES.map((p) => (
          <li key={p.index}>
            <a className="plate-card" href={p.href}>
              <Still still={stills[String(p.index)]} />
              <div className="plate-card-say">
                <h3 className="plate-card-title">{p.title}</h3>
                <p className="prose plate-card-line">{p.line}</p>
                <span className="plate-card-go overline">open the plate <Arrow /></span>
              </div>
            </a>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Still({ still }) {
  const [ref, width] = useMeasure();
  const node = useRef(null);
  const size = Math.max(0, width);

  useEffect(() => {
    if (!node.current || !size) return;
    const h = Math.round(size * 0.78);
    const ctx = fitCanvas(node.current, size, h);

    ctx.fillStyle = INK.raised;
    ctx.fillRect(0, 0, size, h);

    if (!still) {
      ctx.strokeStyle = INK.graticule;
      ctx.setLineDash([4, 5]);
      ctx.strokeRect(8.5, 8.5, size - 17, h - 17);
      ctx.setLineDash([]);
      return;
    }

    // The drawing fills its box: a square plate floating in a wide frame reads as an accident.
    const square = Math.max(size, h);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, size, h);
    ctx.clip();
    ctx.translate((size - square) / 2, (h - square) / 2);

    if (still.kind === "field") {
      const { run, data } = still;
      paintField(ctx, { field: run.field, confidence: run.confidence, n: run.n,
                        offset: (run.frames - 1) * run.n * run.n }, square, run.classes);
      drawPoints(ctx, data.points, data.labels, toPixel(square), { radius: 3 });
    } else if (still.kind === "landscape") {
      paintLandscape(ctx, { run: still.run, frameIndex: still.run.frames - 1, size: square });
    } else if (still.kind === "descent") {
      // Unlabelled: seven names on leaders close a drawing up at a third of its plate size.
      paintDescent(ctx, { info: still.info, runs: still.runs, active: still.active,
                          step: Infinity, size: square, labelled: false });
    } else {
      const { run, data } = still;
      ctx.fillStyle = INK.ground;
      ctx.fillRect(0, 0, square, square);
      const at = (x, y) => [((x + 1) / 2) * square, square - ((y + 1.15) / 2.3) * square];
      const n = run.curve_n;
      const last = (run.frames - 1) * n;

      ctx.beginPath();
      for (let k = 0; k < n; k++) {
        const [px, py] = at(-1 + (2 * k) / (n - 1), run.curve[last + k]);
        k === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.strokeStyle = INK.ink;
      ctx.lineWidth = 2;
      ctx.stroke();

      data.points.forEach(([x, y]) => {
        const [px, py] = at(x, y);
        ctx.beginPath(); ctx.arc(px, py, 3, 0, 2 * Math.PI);
        ctx.fillStyle = INK.accent; ctx.fill();
      });
    }

    frame(ctx, square);
    ctx.restore();
  }, [still, size]);

  return (
    <div className="still" ref={ref}>
      <canvas ref={node} aria-hidden="true" />
      {!still && <span className="still-wait overline">drawing</span>}
    </div>
  );
}

function Evidence() {
  // Counted, not estimated: `pytest -q` and a line count over src/gradopsy. PRODUCT.md forbids
  // putting a figure on this page that nobody measured.
  const facts = [
    ["121", "tests, green", "every operation checked against a numerical gradient"],
    ["1,367", "lines of engine", "autograd, losses, optimizers; no framework underneath"],
    ["6", "optimizers", "BFGS and a damped Newton with a finite-difference Hessian among them"],
    ["0", "servers", "the mathematics runs in this tab, through Pyodide"],
  ];

  return (
    <section className="evidence">
      <hr className="rule" />
      <div className="evidence-head">
        <h2 className="headline">What is underneath</h2>
        <p className="prose">
          The interesting part of this project is not the page, it is that nothing here is
          borrowed. There is no PyTorch and no TensorFlow below this: the backward pass, the
          line search and the Hessian are all written out.
        </p>
      </div>

      <div className="leaders evidence-leaders">
        {facts.map(([value, label, note]) => (
          <Leader key={label} label={label} value={value} note={note} />
        ))}
      </div>
    </section>
  );
}
