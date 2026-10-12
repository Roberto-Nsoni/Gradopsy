import { useEffect, useState } from "react";

import { LR_MAX, LR_MIN, OPTIMIZERS, LINE_SEARCH, SETS, fmtRate } from "./config";
import { CLASSES, MAX_CLASSES } from "./paint";
import Icon, { ClassMark } from "./icons";

/*
 * The controls, as panel furniture: label above, control beneath, no boxes inside boxes. Every
 * choice among a handful of values is a row of flat chips — one pressed, the rest edged — so the
 * whole state of the panel can be read at a glance without opening anything.
 */

export function Slider({ label, value, min, max, step, onChange, onCommit, format = (v) => v, disabled }) {
  return (
    <label className="pg-field pg-slider">
      <span className="pg-label">{label}<b className="pg-value">{format(value)}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
             onChange={(e) => onChange(Number(e.target.value))}
             onPointerUp={(e) => onCommit?.(Number(e.target.value))}
             onKeyUp={(e) => onCommit?.(Number(e.target.value))} />
    </label>
  );
}

const TOOLS = [
  ["brush", "brush", "B", "Hold to spray points of the active class inside the circle"],
  ["point", "point", "P", "One point per click"],
  ["erase", "erase", "E", "Hold to erase inside the circle (or hold Shift with any tool)"],
  ["move", "move", "V", "Drag a point somewhere else"],
];

export function DataPanel({
  config, pristine, counts, active, tool, radius, canUndo, canRedo,
  onSet, onNoise, onSamples, onReseed, onClasses, onActive, onTool, onRadius, onSplit,
  onUndo, onRedo, onClear, onShake, role, onRole,
}) {
  const total = counts.reduce((a, b) => a + b, 0);
  return (
    <section className="pg-block" aria-labelledby="pg-data">
      <h2 className="pg-block-title" id="pg-data">
        Data
        <span className="pg-row">
          <span className="pg-measure">{total} pts</span>
          <button type="button" className="pg-tiny pg-tiny-word" onClick={onShake} disabled={!total}
                  title="Jitter every point by the noise amount">shake</button>
          <button type="button" className="pg-tiny" onClick={onClear} disabled={!total}
                  aria-label="Clear every point" title="Clear every point (undo brings them back)"><Icon name="clear" size={11} /></button>
        </span>
      </h2>

      <fieldset className="pg-field">
        <legend className="pg-label">start from</legend>
        <div className="pg-chips">
          {SETS.map((s) => (
            <button type="button" key={s.id}
                    className={"pg-chip" + (pristine && s.id === config.set ? " is-on" : "")}
                    aria-pressed={pristine && s.id === config.set} onClick={() => onSet(s.id)}>
              {s.name}
            </button>
          ))}
          <button type="button" className="pg-chip pg-chip-icon" onClick={onReseed}
                  aria-label="Draw the same set again with a new random seed" title="New sample of this set">
            <Icon name="dice" />
          </button>
        </div>
      </fieldset>

      <div className="pg-pair" title={pristine ? undefined : "Noise and samples apply when you pick a set. To roughen a drawing, use shake."}>
        <Slider label="noise" value={config.noise} min={0} max={1} step={0.05}
                format={(v) => `${Math.round(v * 100)}%`} onChange={onNoise} />
        <Slider label="samples" value={config.n} min={40} max={600} step={20} onChange={onSamples} />
      </div>

      <fieldset className="pg-field">
        <legend className="pg-legend-row">
          <span>class <span className="pg-note">· keys 1–{config.classes}</span></span>
          <span className="pg-row">
            <button type="button" className="pg-tiny" disabled={config.classes <= 2}
                    onClick={() => onClasses(-1)} aria-label={`Remove class ${CLASSES[config.classes - 1].name} and its points`}
                    title="Remove the last class and its points">
              <Icon name="minus" size={10} />
            </button>
            <button type="button" className="pg-tiny" disabled={config.classes >= MAX_CLASSES}
                    onClick={() => onClasses(1)} aria-label="Add a class" title="Add a class">
              <Icon name="plus" size={10} />
            </button>
          </span>
        </legend>
        <div className="pg-chips">
          {CLASSES.slice(0, config.classes).map((c, i) => (
            <button type="button" key={c.name} className={"pg-chip pg-class" + (i === active ? " is-on" : "")}
                    aria-pressed={i === active} onClick={() => onActive(i)}
                    aria-label={`Class ${c.name}, ${counts[i] ?? 0} points`}>
              <ClassMark shape={c.shape} colour={c.mark} />
              <span>{c.name}</span>
              <span className="pg-measure">{counts[i] ?? 0}</span>
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="pg-field">
        <legend className="pg-legend-row">
          <span>pen</span>
          <span className="pg-seg" role="group" aria-label="Where drawn points go">
            {[["split", "Drawn points follow the test split"], ["train", "Drawn points are always training points"],
              ["test", "Drawn points are always held out for testing"]].map(([id, help]) => (
              <button type="button" key={id} className={"pg-tiny pg-tiny-word" + (role === id ? " is-on" : "")}
                      aria-pressed={role === id} onClick={() => onRole(id)} title={`${help} (T)`}>{id}</button>
            ))}
          </span>
          <span className="pg-row">
            <button type="button" className="pg-tiny" onClick={onUndo} disabled={!canUndo}
                    aria-label="Undo" title="Undo (Ctrl+Z)"><Icon name="undo" size={11} /></button>
            <button type="button" className="pg-tiny" onClick={onRedo} disabled={!canRedo}
                    aria-label="Redo" title="Redo (Ctrl+Shift+Z)"><Icon name="redo" size={11} /></button>
          </span>
        </legend>
        <div className="pg-tools">
          {TOOLS.map(([id, icon, keyName, help]) => (
            <button type="button" key={id} className={"pg-chip pg-tool" + (tool === id ? " is-on" : "")}
                    aria-pressed={tool === id} onClick={() => onTool(id)} title={`${help} (${keyName})`}
                    aria-label={`${id}: ${help}`}>
              <Icon name={icon} /><span>{id}</span>
            </button>
          ))}
        </div>
      </fieldset>

      <div className="pg-pair">
        <Slider label="pen · [ ]" value={radius} min={0.03} max={0.4} step={0.01}
                format={(v) => v.toFixed(2)} onChange={onRadius} disabled={tool === "point" || tool === "move"} />
        <Slider label="test" value={config.split} min={0} max={1} step={0.05}
                format={(v) => `${Math.round(v * 100)}%`} onChange={onSplit} />
      </div>

    </section>
  );
}

/** A number field that commits on Enter or when it loses focus, so typing never thrashes the run. */
function NumberField({ value, onCommit, label, placeholder, disabled, min, max, step, format = String, parse = Number }) {
  const [draft, setDraft] = useState(format(value));
  useEffect(() => setDraft(format(value)), [value, format]);
  const commit = () => {
    const v = parse(draft);
    if (v == null || Number.isNaN(v)) { setDraft(format(value)); return; }
    onCommit(v);
  };
  return (
    <input className="pg-number" type="text" inputMode="decimal" value={draft} placeholder={placeholder}
           aria-label={label} disabled={disabled} min={min} max={max} step={step}
           onChange={(e) => setDraft(e.target.value)} onBlur={commit}
           onKeyDown={(e) => { if (e.key === "Enter") { e.currentTarget.blur(); } if (e.key === "Escape") { setDraft(format(value)); e.currentTarget.blur(); } }} />
  );
}

const OPT_NAMES = {
  sgd: "SGD", momentum: "Momentum", rmsprop: "RMSProp", adam: "Adam",
  bfgs: "BFGS · quasi-Newton",
};

const LOG_MIN = Math.log10(LR_MIN), LOG_MAX = Math.log10(LR_MAX);
const roundRate = (v) => +Math.min(LR_MAX, Math.max(LR_MIN, v)).toPrecision(2);

export function LearnPanel({ config, trainCount, onChange, onReseed }) {
  const searched = LINE_SEARCH.has(config.optimizer);
  const most = Math.max(1, trainCount);
  const batch = config.batch > 0 && config.batch < most ? config.batch : most;   // full is the top of the scale

  return (
    <section className="pg-block pg-learn" aria-labelledby="pg-learn">
      <h2 className="pg-block-title" id="pg-learn">
        Learning
        {searched && <span className="pg-note pg-learn-note">quasi-Newton · full batch, line search</span>}
      </h2>

      <label className="pg-field">
        <span className="pg-label">optimizer</span>
        <select className="pg-select" value={config.optimizer} onChange={(e) => onChange({ optimizer: e.target.value })}>
          {OPTIMIZERS.map((o) => <option key={o} value={o}>{OPT_NAMES[o]}</option>)}
        </select>
      </label>

      <div className="pg-field">
        <span className="pg-label" title={searched ? "The line search picks the step" : undefined}>rate</span>
        <div className="pg-tune">
          <input type="range" className="pg-range" min={LOG_MIN} max={LOG_MAX} step={0.01} disabled={searched}
                 value={Math.log10(config.lr)} aria-label="Learning rate, logarithmic"
                 onChange={(e) => onChange({ lr: roundRate(10 ** Number(e.target.value)) })} />
          <NumberField value={config.lr} label="Learning rate" disabled={searched} format={fmtRate}
                       parse={(t) => { const v = Number(t); return v > 0 ? roundRate(v) : NaN; }}
                       onCommit={(v) => onChange({ lr: v })} />
        </div>
      </div>

      <div className="pg-field">
        <span className="pg-label" title={searched ? "Full batch: the line search needs it" : undefined}>batch</span>
        <div className="pg-tune">
          <input type="range" className="pg-range" min={1} max={most} step={1} disabled={searched}
                 value={searched ? most : batch} aria-label="Batch size, slider"
                 onChange={(e) => { const v = Number(e.target.value); onChange({ batch: v >= most ? 0 : v }); }}
                 aria-valuetext={searched || batch >= most ? "full batch" : `${batch} points`} />
          <NumberField value={searched ? 0 : config.batch >= most ? 0 : config.batch} label="Batch size" disabled={searched}
                       placeholder="full" format={(v) => (v > 0 ? String(v) : "full")}
                       parse={(t) => { const tt = t.trim().toLowerCase(); if (tt === "" || tt === "full") return 0;
                                       const v = Math.round(Number(tt)); return v >= 1 ? (v >= most ? 0 : v) : NaN; }}
                       onCommit={(v) => onChange({ batch: v })} />
        </div>
      </div>

      <div className="pg-field">
        <span className="pg-label" title="Seed for the initial weights">seed</span>
        <div className="pg-tune">
          <NumberField value={config.seed} label="Seed for the initial weights"
                       parse={(t) => { const v = Math.round(Number(t)); return v >= 0 && v < 2 ** 31 ? v : NaN; }}
                       onCommit={(v) => onChange({ seed: v })} />
          <button type="button" className="pg-chip pg-chip-icon" onClick={onReseed}
                  aria-label="Random seed: new initial weights" title="Random seed: new initial weights">
            <Icon name="dice" />
          </button>
        </div>
      </div>
    </section>
  );
}
