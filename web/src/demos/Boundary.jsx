import { useEffect, useMemo, useState } from "react";

import { neuronField, preset } from "../engine";
import { MetricBand, Plate } from "../components/Chrome";
import Scrubber from "../components/Scrubber";
import { FieldCanvas, useMeasure, usePlayback } from "../components/Section";
import useRun from "../components/Run";
import { INK, fitCanvas, paintScalar, frame, material } from "../world/section";

const PRESETS = ["spirals", "moons", "circles", "blobs", "pinwheel"];
const WIDTHS = [4, 8, 12, 16, 24];

export default function Boundary() {
  const [set, setSet] = useState("spirals");
  const [data, setData] = useState(null);
  const [depth, setDepth] = useState(2);
  const [width, setWidth] = useState(12);
  const [activation, setActivation] = useState("tanh");
  const [activeClass, setActiveClass] = useState(0);
  const [neuron, setNeuron] = useState(null);

  const [plateRef, size] = useMeasure(900);

  useEffect(() => { setData(preset(set)); }, [set]);

  const options = useMemo(() => data && ({
    points: data.points, labels: data.labels,
    hidden: Array.from({ length: depth }, () => width),
    activation, optimizer: "adam", lr: 0.05, steps: 240, gridN: 140, seed: 0,
  }), [data, depth, width, activation]);

  // Trained in slices inside animation frames: a chip used to cost a second and a half of
  // frozen tab, and now the boundary forms while it is being computed.
  const [run, progress] = useRun("boundary", options);
  const busy = !run || run.training;

  useEffect(() => { setNeuron(null); }, [options]);

  const total = run?.frames ?? 1;
  const { step, setStep, playing, setPlaying } = usePlayback(total);
  const at = run ? run.keys[Math.min(step, total - 1)] : 0;

  const addPoint = ([x, y]) => {
    setData((d) => ({ points: [...d.points, [+x.toFixed(4), +y.toFixed(4)]], labels: [...d.labels, activeClass] }));
  };

  const classes = run?.classes ?? 3;

  return (
    <Plate
      title="A network drawing a line"
      note="Click the plane to add a point of the active class. The field is what the network
            believes about every location, and it is redrawn from scratch on every change."
    >
      <div className="rig">
        <div ref={plateRef} className="rig-stage">
          <div className="rig-column" style={{ maxWidth: size || undefined }}>
          <FieldCanvas
            run={run} frameIndex={step} points={data?.points} labels={data?.labels}
            size={size} onPick={addPoint}
            hint={busy ? "computing the field" : "material is what it predicts · the gold cut is now"}
          />
          <Scrubber step={step} setStep={setStep} total={total} playing={playing} setPlaying={setPlaying}
                    shown={at} shownTotal={run ? run.keys[run.frames - 1] : 0} ready={progress} />
          <MetricBand items={[
            { label: "step", value: at },
            { label: "loss", value: run ? run.loss[at].toFixed(4) : "—" },
            { label: "accuracy", value: run ? (run.accuracy[at] * 100).toFixed(1) : "—", unit: "%", lit: true },
            { label: "points", value: data?.points.length ?? 0 },
            { label: "parameters", value: run?.params ?? "—" },
          ]} />
          </div>
        </div>

        <aside className="rig-controls">
          <Field label="starting set">
            <div className="chips">
              {PRESETS.map((p) => (
                <button key={p} className={"chip" + (p === set ? " is-on" : "")} onClick={() => setSet(p)}>
                  {p}
                </button>
              ))}
            </div>
          </Field>

          <Field label="active class">
            <div className="chips">
              {Array.from({ length: classes }, (_, i) => (
                <button key={i} className={"chip chip-class" + (i === activeClass ? " is-on" : "")}
                        onClick={() => setActiveClass(i)} aria-label={`Class ${i}`}>
                  <i style={{ background: material(i) }} />{i}
                </button>
              ))}
            </div>
          </Field>

          <Field label={`hidden layers · ${depth}`}>
            <input type="range" min="1" max="3" value={depth} onChange={(e) => setDepth(+e.target.value)} />
          </Field>

          <Field label={`units per layer · ${width}`}>
            <div className="chips">
              {WIDTHS.map((w) => (
                <button key={w} className={"chip" + (w === width ? " is-on" : "")} onClick={() => setWidth(w)}>
                  {w}
                </button>
              ))}
            </div>
          </Field>

          <Field label="activation">
            <div className="chips">
              {["tanh", "relu", "sigmoid"].map((a) => (
                <button key={a} className={"chip" + (a === activation ? " is-on" : "")} onClick={() => setActivation(a)}>
                  {a}
                </button>
              ))}
            </div>
          </Field>

          {options && (
            <Field label="one unit on its own">
              <p className="prose control-note">
                The finished boundary is a blend of simple cuts. Pick one to see the cut it makes.
              </p>
              <div className="chips">
                {Array.from({ length: Math.min(width, 12) }, (_, i) => (
                  <button key={i} className={"chip chip-tight" + (i === neuron ? " is-on" : "")}
                          onClick={() => setNeuron(i === neuron ? null : i)}>
                    {i}
                  </button>
                ))}
              </div>
              {neuron !== null && <NeuronPlate options={options} neuron={neuron} />}
            </Field>
          )}
        </aside>
      </div>
    </Plate>
  );
}

function Field({ label, children }) {
  return (
    <div className="control">
      <span className="overline">{label}</span>
      {children}
    </div>
  );
}

function NeuronPlate({ options, neuron }) {
  const [ref, size] = useMeasure(240);
  const [unit, setUnit] = useState(null);

  useEffect(() => {
    const id = setTimeout(() => setUnit(neuronField({ ...options, gridN: 60 }, neuron)), 20);
    return () => clearTimeout(id);
  }, [options, neuron]);

  return (
    <div className="neuron" ref={ref}>
      {unit && size > 0 && <NeuronCanvas unit={unit} size={size} />}
      <p className="overline neuron-cap">unit {neuron} · dark is off, light is on</p>
    </div>
  );
}

function NeuronCanvas({ unit, size }) {
  const [node, setNode] = useState(null);

  useEffect(() => {
    if (!node) return;
    const ctx = fitCanvas(node, size, size);
    paintScalar(ctx, unit.field, unit.n, size, unit.lo, unit.hi);
    frame(ctx, size, INK.rule);
  }, [node, unit, size]);

  return <canvas ref={setNode} />;
}
