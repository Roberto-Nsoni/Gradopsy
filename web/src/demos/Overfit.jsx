import { useEffect, useMemo, useRef, useState } from "react";

import { presetCurve } from "../engine";
import { MetricBand, Plate } from "../components/Chrome";
import Scrubber from "../components/Scrubber";
import { Curves, Legend, useMeasure, usePlayback } from "../components/Section";
import useRun from "../components/Run";
import { INK, fitCanvas, frame } from "../world/section";

// Tuned, not guessed: at this data size and noise the slider reads as underfit, fit, then
// memorise. Wider steps than this and the middle of the scale says nothing.
const CAPACITIES = [
  { label: "2", hidden: [2] },
  { label: "4", hidden: [4] },
  { label: "8", hidden: [8] },
  { label: "16 · 16", hidden: [16, 16] },
  { label: "48 · 48", hidden: [48, 48] },
  { label: "128 · 128", hidden: [128, 128] },
];

const toPixel = (w, h) => (x, y) => [((x + 1) / 2) * w, h - ((y + 1.15) / 2.3) * h];

export default function Overfit() {
  const [data, setData] = useState(null);
  const [capacity, setCapacity] = useState(1);
  const [stageRef, width] = useMeasure(980);
  const [asideRef, asideWidth] = useMeasure(560);

  useEffect(() => { setData(presetCurve()); }, []);

  const hidden = CAPACITIES[capacity].hidden;

  // The widest network on the slider cost three and a half seconds of frozen tab. Now the
  // curve bends towards its answer while it is being fitted.
  const [run, progress] = useRun("overfit",
    data && { points: data.points, hidden, steps: 2000, lr: 0.015, seed: 0 });
  const busy = !run || run.training;

  const total = run?.frames ?? 1;
  const { step, setStep, playing, setPlaying } = usePlayback(total);
  const at = run ? run.keys[Math.min(step, total - 1)] : 0;

  const height = Math.round((width || 400) * 0.56);
  const addPoint = ([x, y]) => setData((d) => ({ points: [...d.points, [x, y]] }));

  const verdict = useMemo(() => {
    if (!run) return "—";
    const test = run.test_loss.slice(0, at + 1);
    const rise = test[test.length - 1] / Math.min(...test);

    if (run.train_loss[at] > 0.012) return "underfitting";
    return rise > 1.25 ? "memorising" : "generalising";
  }, [run, at]);

  return (
    <Plate
      title="Where a model stops learning and starts memorising"
      note="Half of these points train the network; the other half it never sees. Widen it and the
            curve stops describing the trend and starts threading every dot, including the noise."
    >
      <div className="rig rig-wide">
        <div ref={stageRef} className="rig-stage">
          <CurvePlate run={run} frameIndex={step} data={data} width={width} height={height} onPick={addPoint} />

          <Scrubber step={step} setStep={setStep} total={total} playing={playing} setPlaying={setPlaying}
                    shown={at} shownTotal={run ? run.keys[run.frames - 1] : 0} ready={progress} />

          <MetricBand items={[
            { label: "parameters", value: run?.params ?? "—" },
            { label: "training loss", value: run ? run.train_loss[at].toFixed(4) : "—" },
            { label: "held-out loss", value: run ? run.test_loss[at].toFixed(4) : "—", lit: true },
            { label: "verdict", value: busy ? "…" : verdict },
          ]} />
        </div>

        <aside className="rig-controls" ref={asideRef}>
          <div className="control">
            <span className="overline">hidden units · {CAPACITIES[capacity].label}</span>
            <input type="range" min="0" max={CAPACITIES.length - 1} value={capacity}
                   onChange={(e) => setCapacity(+e.target.value)} aria-label="Network capacity" />
            <div className="scale-ends overline"><span>too small</span><span>too large</span></div>
          </div>

          <div className="control">
            <span className="overline">the two losses</span>
            <Curves
              width={asideWidth || 300} height={Math.max(180, Math.round((asideWidth || 300) * 0.62))} logY
              step={at}
              series={[
                { values: run?.train_loss ?? [], colour: INK.poche },
                { values: run?.test_loss ?? [], colour: INK.sunbeamDeep, dash: [4, 3] },
              ]}
            />
            <Legend items={[
              { label: "on data it fitted", colour: INK.poche },
              { label: "on data it never saw", colour: INK.sunbeamDeep, dash: true },
            ]} />
            <p className="prose control-note">
              When the dashed line turns back up while the solid one keeps falling, the network has
              stopped learning the shape and started learning the noise.
            </p>
          </div>
        </aside>
      </div>
    </Plate>
  );
}

function CurvePlate({ run, frameIndex, data, width, height, onPick }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !width) return;
    const ctx = fitCanvas(ref.current, width, height);
    const at = toPixel(width, height);

    ctx.fillStyle = INK.limewash;
    ctx.fillRect(0, 0, width, height);

    // A ruled ground, the way a drawing carries its own graticule
    ctx.strokeStyle = INK.limewashDeep;
    ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
      const y = (i / 8) * height;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    if (run) {
      const i = Math.min(frameIndex, run.frames - 1);
      const n = run.curve_n;
      ctx.beginPath();
      for (let k = 0; k < n; k++) {
        const [px, py] = at(-1 + (2 * k) / (n - 1), run.curve[i * n + k]);
        k === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.strokeStyle = INK.poche;
      ctx.lineWidth = 2.2;
      ctx.lineJoin = "round";
      ctx.stroke();
    }

    const train = new Set(run?.train_idx ?? []);
    (data?.points ?? []).forEach(([x, y], k) => {
      const [px, py] = at(x, y);
      const fitted = !run || train.has(k);

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fillStyle = INK.limewash;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(px, py, fitted ? 3.6 : 4.6, 0, 2 * Math.PI);
      if (fitted) {
        ctx.fillStyle = INK.poche;
        ctx.fill();
      } else {
        ctx.strokeStyle = INK.sunbeamDeep;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    frame(ctx, width, height);
  }, [run, frameIndex, data, width, height]);

  const pick = (event) => {
    const box = ref.current.getBoundingClientRect();
    onPick([
      +(((event.clientX - box.left) / box.width) * 2 - 1).toFixed(4),
      +((1 - (event.clientY - box.top) / box.height) * 2.3 - 1.15).toFixed(4),
    ]);
  };

  return (
    <div className="section-plate section-plate-wide">
      <canvas ref={ref} onClick={pick} style={{ cursor: "crosshair" }}
              aria-label="A curve fitted through points, with held-out points ringed." />
      <p className="section-hint overline">filled points are fitted · ringed points are held out</p>
    </div>
  );
}
