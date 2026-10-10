import { useEffect, useRef, useState } from "react";

import { preset } from "../engine";
import { MetricBand, Plate } from "../components/Chrome";
import Scrubber from "../components/Scrubber";
import { FieldCanvas, useMeasure, usePlayback } from "../components/Section";
import useRun from "../components/Run";
import { INK, fitCanvas, frame, paintTerrain } from "../world/section";

const SETS = ["spirals", "moons", "circles", "pinwheel"];

export default function Lenses() {
  const [set, setSet] = useState("spirals");
  const [data, setData] = useState(null);
  const [stageRef, width] = useMeasure(1180);

  useEffect(() => { setData(preset(set)); }, [set]);

  // Two phases, both sliced: the training, and then the sweep of the loss plane, which cannot
  // choose its own axes until the whole trajectory exists.
  const [run, progress] = useRun("lenses", data && {
    points: data.points, labels: data.labels, hidden: [10, 10],
    lr: 0.05, steps: 240, gridN: 120, surfaceN: 46, seed: 0,
  });
  const busy = !run || run.training;

  const total = run?.frames ?? 1;
  const { step, setStep, playing, setPlaying } = usePlayback(total);
  const at = run ? run.keys[Math.min(step, total - 1)] : 0;

  // Two plates side by side, both fed by the same run and the same step. Once the pair no
  // longer fits abreast it stacks, and a stacked plate takes the whole track: holding the old
  // 220px floor left both drawings at three fifths of a phone's width for no reason.
  const track = width || 600;
  const plate = track >= 2 * 260 + 26 ? Math.floor((track - 26) / 2) : track;

  return (
    <Plate
      title="One run, seen twice at once"
      note="The same training, reported through both lenses on the same step. On the left, what the
            network believes about the plane. On the right, the loss around the path its parameters
            actually took — a plane found by principal components of the run itself, because two
            random directions in this many dimensions would collapse the whole journey to a dot."
    >
      <div ref={stageRef} className="lenses">
        <div className="lenses-pair">
          <figure className="lens">
            <figcaption className="overline">input space · what it believes</figcaption>
            <FieldCanvas run={run} frameIndex={step} points={data?.points} labels={data?.labels} size={plate}
                         hint="material is what it predicts · the gold cut is now" />
          </figure>

          <figure className="lens">
            <figcaption className="overline">parameter space · where it went</figcaption>
            <Landscape run={run} frameIndex={step} size={plate}
                       hint={run?.surface ? "dark is low loss · the lit head is now"
                                          : "measuring the plane the run travelled"} />
          </figure>
        </div>

        <Scrubber step={step} setStep={setStep} total={total} playing={playing} setPlaying={setPlaying}
                    shown={at} shownTotal={run ? run.keys[run.frames - 1] : 0} ready={progress} />

        <MetricBand items={[
          { label: "step", value: at },
          { label: "loss", value: run ? run.loss[at].toFixed(4) : "—" },
          { label: "accuracy", value: run ? (run.accuracy[at] * 100).toFixed(1) : "—", unit: "%", lit: true },
          { label: "parameters", value: run?.params ?? "—" },
        ]} />

        <div className="chips lenses-sets">
          {SETS.map((s) => (
            <button key={s} className={"chip" + (s === set ? " is-on" : "")} onClick={() => setSet(s)}>
              {s}
            </button>
          ))}
        </div>
      </div>
    </Plate>
  );
}

/**
 * The loss over the plane the run actually travelled, with the path drawn on it. Behind the head
 * the path is dim; the head itself is the present moment and carries the strike.
 */
export function paintLandscape(ctx, { run, frameIndex, size }) {
  ctx.fillStyle = INK.ground;
  ctx.fillRect(0, 0, size, size);

  // The plane cannot choose its own axes until the whole trajectory exists, so for the length
  // of the training this plate has nothing yet. It says so, the way an unfinished plate does.
  if (!run?.surface) {
    ctx.strokeStyle = INK.graticule;
    ctx.setLineDash([4, 5]);
    ctx.strokeRect(8.5, 8.5, size - 17, size - 17);
    ctx.setLineDash([]);
    frame(ctx, size);
    return;
  }

  paintTerrain(ctx, run.surface, run.surface_n, size, { levels: 12 });

  const span = run.span;
  const toPixel = (a, b) => [((a + span) / (2 * span)) * size, size - ((b + span) / (2 * span)) * size];
  const upTo = run.keys[Math.min(frameIndex, run.frames - 1)];

  ctx.beginPath();
  run.path.slice(0, upTo + 1).forEach(([a, b], i) => {
    const [px, py] = toPixel(a, b);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.strokeStyle = INK.fieldB;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  const [hx, hy] = toPixel(...run.path[upTo]);
  ctx.beginPath();
  ctx.arc(hx, hy, 5, 0, 2 * Math.PI);
  ctx.fillStyle = INK.accent;
  ctx.fill();
  ctx.strokeStyle = INK.ink;
  ctx.lineWidth = 1.4;
  ctx.stroke();

  frame(ctx, size);
}

function Landscape({ run, frameIndex, size, hint }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !size) return;
    paintLandscape(fitCanvas(ref.current, size, size), { run, frameIndex, size });
  }, [run, frameIndex, size]);

  return (
    <div className="section-plate" style={{ width: size }}>
      <canvas ref={ref} aria-label="The loss around the path the parameters took." />
      <p className="section-hint overline">{hint}</p>
    </div>
  );
}
