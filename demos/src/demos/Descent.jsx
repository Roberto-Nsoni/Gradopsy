import { useEffect, useRef, useState } from "react";

import { surfaceInfo } from "../engine";
import { MetricBand, Plate } from "../components/Chrome";
import Scrubber from "../components/Scrubber";
import { useMeasure, usePlayback } from "../components/Section";
import useRun from "../components/Run";
import { INK, fitCanvas, frame, paintTerrain } from "../world/section";

/*
 * Tone and dash, never hue alone. Seven series separated only by colour is exactly the failure
 * this project has had open since the first build; a dash pattern survives any kind of vision.
 *
 * Three phosphor drives carry the three families, and the drives are chosen by measurement: the
 * three are 2.5, 5.3 and 2.1 apart from each other, where the pairing this replaced put two
 * families 1.48 apart — close enough that two series read as one.
 *
 * Which family gets which drive is not arbitrary either. BFGS and Newton are what this
 * project is *for* — a damped Newton with a finite-difference Hessian is the thing a gallery of
 * SGD-and-Adam demos does not have — so they get the brightest phosphor, and the first-order
 * baselines they are being compared against sit at the dimmest. Every trail is stroked over a
 * casing of ground first, so a dim drive stays readable across the brightest terrain.
 */
export const RUNNERS = [
  { key: "sgd", name: "SGD", ink: INK.quiet, dash: [], width: 2.2, mark: "circle", order: 1, lr: 0.002 },
  { key: "momentum", name: "Momentum", ink: INK.quiet, dash: [9, 5], width: 2.2, mark: "square", order: 1, lr: 0.002 },
  { key: "rmsprop", name: "RMSProp", ink: INK.fieldB, dash: [], width: 2.2, mark: "diamond", order: 1, lr: 0.02 },
  { key: "adam", name: "Adam", ink: INK.fieldB, dash: [9, 5], width: 2.2, mark: "triangle", order: 1, lr: 0.05 },
  { key: "bfgs", name: "BFGS", ink: INK.ink, dash: [], width: 3, mark: "cross", order: 2 },
  // Not the strike. The strike is this world's one accent and it means "the step you are looking
  // at"; a runner wearing it made a seventh of the drawing claim to be the present moment.
  { key: "newton", name: "Newton", ink: INK.ink, dash: [2.5, 4], width: 3, mark: "ring", order: 2 },
];
/**
 * The terminal marks. Shape is a channel that survives any kind of vision, unlike hue.
 *
 * `cross` is the one mark with no interior, so it is stroked where the others are filled;
 * `ring` punches its centre by winding the inner circle the other way, which is the only thing
 * that stops the non-zero fill rule from handing back a plain disc — and a ring that fills
 * solid is indistinguishable from SGD's circle, which costs the drawing a whole channel.
 */
function drawMark(ctx, kind, x, y, r) {
  ctx.beginPath();
  if (kind === "circle") ctx.arc(x, y, r, 0, 2 * Math.PI);
  else if (kind === "square") ctx.rect(x - r, y - r, 2 * r, 2 * r);
  else if (kind === "diamond") { ctx.moveTo(x, y - r * 1.3); ctx.lineTo(x + r * 1.3, y); ctx.lineTo(x, y + r * 1.3); ctx.lineTo(x - r * 1.3, y); ctx.closePath(); }
  else if (kind === "triangle") { ctx.moveTo(x, y - r * 1.35); ctx.lineTo(x + r * 1.2, y + r); ctx.lineTo(x - r * 1.2, y + r); ctx.closePath(); }
  else if (kind === "cross") { ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); }
  else if (kind === "bar") ctx.rect(x - r * 1.4, y - r * 0.55, r * 2.8, r * 1.1);
  else { ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.moveTo(x + r * 0.45, y); ctx.arc(x, y, r * 0.45, 0, 2 * Math.PI, true); }
}

const BY_KEY = Object.fromEntries(RUNNERS.map((r) => [r.key, r]));

const NOTES = {
  double_well: "A saddle at the origin and real minima either side. Everything escapes except Newton, which settles onto the saddle: it solves for a stationary point, and a saddle is one.",
  rosenbrock: "A long curved valley. BFGS and Newton walk it; first order crawls.",
  narrow_valley: "One direction fifty times steeper than the other.",
  beale: "Flat plateaus pressed against steep walls.",
  himmelblau: "Four minima. Where you land depends only on where you began.",
  goldstein_price: "Values from 3 to a million, which is why the ground is mapped by percentile.",
  saddle: "Unbounded below: this surface has no minimum, so divergence is the correct answer.",
};

export default function Descent() {
  const [surface, setSurface] = useState("double_well");
  const [info, setInfo] = useState(null);
  const [active, setActive] = useState(["sgd", "momentum", "adam", "bfgs", "newton"]);
  const [start, setStart] = useState(null);
  const [stageRef, size] = useMeasure(900);

  useEffect(() => { setInfo(surfaceInfo(surface, 560)); setStart(null); }, [surface]);

  // A runner per slice. Seven of them in one call was a third of a second of frozen tab every
  // time a surface or a starting point changed; now the paths arrive one after another.
  const [run] = useRun("descent", info && active.length ? {
    surface, steps: 200, start,
    requested: active.map((key) => ({ key, lr: BY_KEY[key].lr ?? 0 })),
  } : null);

  const runs = run?.runs && Object.keys(run.runs).length ? run.runs : null;
  const busy = !run || run.training;
  const total = runs ? Math.max(...Object.values(runs).map((r) => r.path.length)) : 1;
  const { step, setStep, playing, setPlaying } = usePlayback(total, { fps: 60 });

  return (
    <Plate
      title="Optimizers are not interchangeable"
      note={NOTES[surface]}
    >
      <div className="rig">
        <div ref={stageRef} className="rig-stage">
          <div className="rig-column" style={{ maxWidth: size || undefined }}>
          <Terrain info={info} runs={runs} active={active} step={step} size={size} onPick={setStart} />
          <Scrubber step={step} setStep={setStep} total={total} playing={playing} setPlaying={setPlaying} />
          <MetricBand items={[
            { label: "step", value: step },
            { label: "running", value: active.length },
            { label: "started at", value: start ? start.map((v) => v.toFixed(2)).join(", ") : "default" },
            // What the rig is actually doing. It read "held" while the paths were moving.
            { label: "state", value: busy ? "computing" : playing ? "running" : "held", lit: busy },
          ]} />
          </div>
        </div>

        <aside className="rig-controls">
          <div className="control">
            <span className="overline">surface</span>
            <div className="chips">
              {Object.keys(NOTES).map((s) => (
                <button key={s} className={"chip" + (s === surface ? " is-on" : "")} onClick={() => setSurface(s)}>
                  {s.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>

          {[1, 2].map((order) => (
            <div className="control" key={order}>
              <span className="overline">{order === 1 ? "first order" : "bfgs · newton"}</span>
              <ul className="runners">
                {RUNNERS.filter((r) => r.order === order).map((r) => (
                  <li key={r.key}>
                    <label className={"runner" + (active.includes(r.key) ? " is-on" : "")}>
                      <input
                        type="checkbox" checked={active.includes(r.key)}
                        onChange={() => setActive((a) => a.includes(r.key) ? a.filter((k) => k !== r.key) : [...a, r.key])}
                      />
                      <Swatch runner={r} />
                      <span>{r.name}</span>
                      <span className="runner-note overline">
                        {r.order === 1 ? `lr ${r.lr}` : "line search"}
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <p className="prose control-note">
            Click the terrain to move where they all begin. Every run restarts from the same point,
            which is the only honest way to compare them.
          </p>
        </aside>
      </div>
    </Plate>
  );
}

function Swatch({ runner }) {
  const marks = {
    circle: <circle cx="30" cy="7" r="4" />, square: <rect x="26" y="3" width="8" height="8" />,
    diamond: <polygon points="30,2 35,7 30,12 25,7" />, triangle: <polygon points="30,2 35,11 25,11" />,
    cross: <path d="M26 3l8 8M34 3l-8 8" strokeWidth="2" fill="none" stroke={runner.ink} />,
    bar: <rect x="24" y="4.5" width="12" height="5" />,
    ring: <><circle cx="30" cy="7" r="4.5" fill="none" strokeWidth="2" stroke={runner.ink} /></>,
  };

  return (
    <svg width="38" height="14" aria-hidden="true" className="runner-swatch">
      <line x1="1" y1="7" x2="23" y2="7" stroke={runner.ink} strokeWidth={runner.width}
            strokeDasharray={runner.dash.join(" ") || undefined} />
      <g fill={runner.ink}>{marks[runner.mark]}</g>
    </svg>
  );
}

/**
 * The descent plate, painted. Exported because the home page advertises this plate and has to
 * advertise it with itself: the still there used to be a classification field, which is a
 * picture from a different demo entirely.
 *
 * `labelled` is off for the still, where the drawing has to read at a third of the size and
 * seven names on leaders would close it up.
 */
export function paintDescent(ctx, { info, runs, active, step, size, labelled = true }) {
  paintTerrain(ctx, info.z, info.n, size);

  const [x0, x1] = info.x_range;
  const [y0, y1] = info.y_range;
  const toPixel = (x, y) => [((x - x0) / (x1 - x0)) * size, size - ((y - y0) / (y1 - y0)) * size];

  for (const [mx, my] of info.minima) {
    const [px, py] = toPixel(mx, my);
    ctx.strokeStyle = INK.ground;
    ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(px, py, 8, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px - 4, py); ctx.lineTo(px + 4, py);
    ctx.moveTo(px, py - 4); ctx.lineTo(px, py + 4);
    ctx.stroke();
  }

  // Where every head is right now, so a label can tell whether it is about to land on another
  const named = [];
  const heads = active
    .filter((key) => runs?.[key])
    .map((key) => toPixel(...runs[key].path[Math.min(step, runs[key].path.length - 1)]));

  for (const key of active) {
    const run = runs?.[key];
    if (!run) continue;
    const upTo = Math.min(step, run.path.length - 1);
    const { ink, dash } = BY_KEY[key];

    // An opaque casing wide enough to read, because a trail crossing either black well
    // disappears into terrain drawn in the same values as the line itself.
    const runner = BY_KEY[key];
    for (const [w, colour, pattern] of [[runner.width + 3.4, INK.ground, []], [runner.width, ink, dash]]) {
      ctx.beginPath();
      for (let i = 0; i <= upTo; i++) {
        const [px, py] = toPixel(run.path[i][0], run.path[i][1]);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.setLineDash(pattern);
      ctx.strokeStyle = colour;
      ctx.lineWidth = w;
      ctx.lineJoin = "round";
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // The head: a shape that names the method, ringed in the strike because it is the present.
    // The ring goes *around* the mark rather than over it. Restroking the mark itself in gold
    // spent the shape channel twice and left BFGS's cross with no ink of its own at all.
    const [px, py] = toPixel(run.path[upTo][0], run.path[upTo][1]);

    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    drawMark(ctx, runner.mark, px, py, 8);
    ctx.strokeStyle = INK.ground; ctx.lineWidth = 5; ctx.stroke();

    drawMark(ctx, runner.mark, px, py, 5.5);
    if (runner.mark === "cross") { ctx.strokeStyle = ink; ctx.lineWidth = 2.4; ctx.stroke(); }
    else { ctx.fillStyle = ink; ctx.fill(); }

    ctx.beginPath();
    ctx.arc(px, py, 10, 0, 2 * Math.PI);
    ctx.strokeStyle = INK.ground; ctx.lineWidth = 3.6; ctx.stroke();
    ctx.strokeStyle = INK.accent; ctx.lineWidth = 1.8; ctx.stroke();
    ctx.lineCap = "butt";

    // Divergence is a fact about this runner, not about the present step, so it is written in
    // the runner's own ink. Gold here would claim the surface's one accent for a second
    // meaning, and it is unreadable as type besides.
    if (run.diverged && upTo === run.path.length - 1) {
      ctx.font = "600 12px 'Martian Mono', ui-monospace, monospace";
      ctx.textAlign = "left";
      const [dx, dy] = [Math.min(size - 42, px + 11), Math.max(15, py - 11)];
      ctx.strokeStyle = INK.ground; ctx.lineWidth = 3.5;
      ctx.strokeText("→ ∞", dx, dy);
      ctx.fillStyle = ink;
      ctx.fillText("→ ∞", dx, dy);
    }

    if (labelled) named.push({ key, px, py, ink, runner });
  }

  // Names last, in a pass of their own. Drawn inside the loop, one runner's label was painted
  // over by the next runner's trail — which is how the dashed path of Momentum came to run
  // straight through the gap between the S and the G of "SGD".
  for (const { key, px, py, ink, runner } of named) {
    // Fanned by index so converging paths do not stack, and reached further out the more
    // company a head has: three optimizers sharing one minimum used to pile three labels and
    // three leaders into the same twenty pixels.
    const angle = (RUNNERS.findIndex((r) => r.key === key) / RUNNERS.length) * 2 * Math.PI - Math.PI / 2;
    const crowd = heads.filter(([hx, hy]) => Math.hypot(hx - px, hy - py) < 30).length;
    const reach = Math.min(34 + 24 * (crowd - 1), size * 0.28);
    const [lx, ly] = [px + Math.cos(angle) * reach, py + Math.sin(angle) * reach];

    ctx.beginPath();
    ctx.moveTo(px + Math.cos(angle) * 13, py + Math.sin(angle) * 13);
    ctx.lineTo(lx, ly);
    ctx.strokeStyle = INK.ground; ctx.lineWidth = 3.4; ctx.stroke();
    ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.stroke();

    ctx.font = "600 11px 'Martian Mono', ui-monospace, monospace";
    ctx.textAlign = Math.cos(angle) < -0.3 ? "right" : Math.cos(angle) > 0.3 ? "left" : "center";
    ctx.textBaseline = "middle";
    const nudge = ctx.textAlign === "right" ? -4 : ctx.textAlign === "left" ? 4 : 0;
    ctx.lineJoin = "round";
    ctx.strokeStyle = INK.ground; ctx.lineWidth = 3.5;
    ctx.strokeText(runner.name, lx + nudge, ly);
    ctx.fillStyle = ink;
    ctx.fillText(runner.name, lx + nudge, ly);
  }

  frame(ctx, size);
}

function Terrain({ info, runs, active, step, size, onPick }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !size || !info) return;
    paintDescent(fitCanvas(ref.current, size, size), { info, runs, active, step, size });
  }, [info, runs, active, step, size]);

  const pick = (event) => {
    const box = ref.current.getBoundingClientRect();
    const [x0, x1] = info.x_range;
    const [y0, y1] = info.y_range;
    onPick([
      x0 + ((event.clientX - box.left) / box.width) * (x1 - x0),
      y0 + (1 - (event.clientY - box.top) / box.height) * (y1 - y0),
    ]);
  };

  return (
    <div className="section-plate" style={{ width: size }}>
      <canvas ref={ref} onClick={pick} style={{ cursor: "crosshair" }}
              aria-label="A loss surface with each optimizer's path drawn on it." />
      <p className="section-hint overline">click to move the common starting point</p>
    </div>
  );
}
