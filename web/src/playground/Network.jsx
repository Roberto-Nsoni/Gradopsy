import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { ACTIVATIONS, MAX_UNITS } from "./config";
import { CLASSES, planeCut, planeImage } from "./paint";
import Icon from "./icons";

/*
 * The network, opened up. Every unit is a small screen of its own showing what that unit
 * computes over the whole plane — the same plane the big screen shows — so reading left to right
 * is reading the forward pass: two coordinates, then cuts, then cuts of cuts, then the classes.
 *
 * A wire is a weight. Its thickness is the weight's size on an absolute scale, so a wire that
 * grows during training has really grown; a negative weight is dashed rather than recoloured,
 * because hue here already means class. With a probe on the screen, wires carry how much of that
 * one input's signal actually flows along them, and everything else goes quiet.
 */

const HEAD = 62;          // the strip of controls above each column
const FOOT = 10;

function inputPlanes(n) {
  const x = new Float32Array(n * n), y = new Float32Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    x[r * n + c] = (c / (n - 1)) * 2 - 1;
    y[r * n + c] = (r / (n - 1)) * 2 - 1;
  }
  return [x, y];
}

/** Does a picture from the engine describe the network on the panel right now? */
export const fitsShape = (sizes, layers, classes) =>
  !!sizes && sizes.length === layers.length + 1 &&
  layers.every((l, i) => sizes[i] === l.units) && sizes[layers.length] === classes;

export default function Network({
  view: incoming, layers, classes, probe: incomingProbe, lens, pinned = null, wave, onLens, onPin, onUnits, onAct, onStatus,
}) {
  // For one frame after the architecture changes, the panel already shows the new shape while the
  // engine's picture is still of the old network. Reading one with the other's geometry indexed
  // weights that did not exist and took the whole page down; so a picture that does not fit is
  // simply not drawn until the new one arrives.
  const view = incoming && fitsShape(incoming.sizes, layers, classes) ? incoming : null;
  const probe = incomingProbe && (!incomingProbe.layers || fitsShape(
    [...incomingProbe.layers.map((h) => h.length), incomingProbe.out?.length], layers, classes))
    ? incomingProbe : incomingProbe && { x: incomingProbe.x, y: incomingProbe.y, pinned: incomingProbe.pinned };
  const box = useRef(null);
  const tiles = useRef([]);
  const [size, setSize] = useState({ w: 600, h: 480 });
  const [focus, setFocus] = useState(null);     // the unit being pointed at: its wires stay lit
  // the unit being read: the one under the pointer, or else the one clicked and pinned, which
  // keeps its formula and its wires until it is released

  useLayoutEffect(() => {
    const node = box.current;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  const cols = useMemo(() => [2, ...layers.map((l) => l.units), classes], [layers, classes]);
  const held = focus ?? (pinned && pinned.layer + 1 < cols.length && pinned.unit < cols[pinned.layer + 1]
    ? { col: pinned.layer + 1, unit: pinned.unit } : null);
  const n = view?.n ?? 40;
  const inputs = useMemo(() => inputPlanes(n), [n]);

  // geometry: every tile's box, in the panel's own pixels
  const geo = useMemo(() => {
    const { w, h } = size;
    const most = Math.max(...cols);
    const pad = 14;
    const avail = Math.max(120, h - HEAD - FOOT);
    const T = Math.round(Math.max(22, Math.min(88, (avail / most) * 0.8, ((w - pad * 2) / cols.length) * 0.7)));
    const gap = Math.max(5, Math.min(14, (avail - most * T) / Math.max(most - 1, 1)));
    const span = w - pad * 2 - T;
    const xs = cols.map((_, i) => pad + (cols.length > 1 ? (span * i) / (cols.length - 1) : span / 2));
    const boxes = cols.map((k, i) => {
      const height = k * T + (k - 1) * gap;
      const y0 = HEAD + (avail - height) / 2;
      return Array.from({ length: k }, (_, j) => ({ x: xs[i], y: y0 + j * (T + gap) }));
    });
    return { T, xs, boxes };
  }, [size, cols]);

  // the plane each tile shows, and how to read it
  const planeOf = (col, unit) => {
    if (col === 0) return { plane: inputs[unit], act: "linear", cls: null };
    if (!view) return null;
    const nn = view.n * view.n;
    let offset = 0;
    for (let i = 1; i < col; i++) offset += cols[i] * nn;
    offset += unit * nn;
    const plane = view.acts.subarray(offset, offset + nn);
    const out = col === cols.length - 1;
    return { plane, act: out ? null : layers[col - 1].act, cls: out ? unit : null };
  };

  useEffect(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const T = geo.T;
    cols.forEach((k, col) => {
      for (let unit = 0; unit < k; unit++) {
        const node = tiles.current[`${col}:${unit}`];
        if (!node) continue;
        const px = Math.round(T * dpr);
        if (node.width !== px) { node.width = node.height = px; }
        const ctx = node.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = "#060d09";
        ctx.fillRect(0, 0, px, px);
        const p = planeOf(col, unit);
        if (!p) continue;
        const nn = Math.round(Math.sqrt(p.plane.length));
        ctx.globalAlpha = wave != null && col > wave ? 0.12 : 1;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(planeImage(p.plane, nn, p.act, p.cls), 0, 0, px, px);

        // the unit's own cut: where it changes sign (or crosses one half, for a probability)
        const segs = p.cls != null
          ? planeCut(p.plane.map((v) => v - 0.5), nn, "linear", px)
          : planeCut(p.plane, nn, p.act, px);
        ctx.strokeStyle = "rgba(200, 255, 226, 0.9)";
        ctx.lineWidth = Math.max(1, dpr);
        ctx.beginPath();
        for (const [x0, y0, x1, y1] of segs) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
        ctx.stroke();

        ctx.globalAlpha = 1;
      }
    });
    // the probe is drawn over the tiles in the DOM, so pointing at the screen never repaints them
  }, [view, geo, cols, wave, layers, inputs]);

  // the wires, and how much of the probe's signal each carries
  const wires = [];
  if (view) {
    for (let l = 0; l < cols.length - 1; l++) {
      const W = view.weights[l];
      const from = l === 0 ? (probe ? [probe.x, probe.y] : null) : probe?.layers?.[l - 1];
      let most = 1e-9;
      if (from) for (let a = 0; a < cols[l]; a++) for (let b = 0; b < cols[l + 1]; b++) most = Math.max(most, Math.abs(W[a][b] * from[a]));
      for (let a = 0; a < cols[l]; a++) {
        for (let b = 0; b < cols[l + 1]; b++) {
          const w = W[a][b];
          const A = geo.boxes[l][a], B = geo.boxes[l + 1][b];
          const mag = Math.min(Math.abs(w), 3) / 3;
          let alpha = 0.18 + mag * 0.55;
          if (from) alpha = 0.06 + (Math.abs(w * from[a]) / most) * 0.9;
          if (wave != null) alpha = l + 1 <= wave ? alpha : 0.03;
          // pointing at a unit: only the wires that feed it, which are the terms of its formula
          const into = held && held.col > 0 && l + 1 === held.col && b === held.unit;
          if (held && held.col > 0) alpha = into ? Math.max(alpha, 0.4 + mag * 0.55) : 0.035;
          wires.push({
            key: `${l}:${a}:${b}`, l, a, b, w,
            x1: A.x + geo.T, y1: A.y + geo.T / 2, x2: B.x, y2: B.y + geo.T / 2,
            width: (into ? 1 : 0.5) + mag * (into ? 4 : 3.4), alpha,
          });
        }
      }
    }
  }

  const name = (col, unit) => (col === 0 ? `x${unit === 0 ? "₁" : "₂"}`
    : col === cols.length - 1 ? `class ${CLASSES[unit].name}` : `h${col},${unit + 1}`);

  /** The same name as it is set: h with its layer and unit as a subscript, never "h1·3". */
  const label = (col, unit) => (col === 0 ? <>x<sub>{unit + 1}</sub></>
    : col === cols.length - 1 ? `class ${CLASSES[unit].name}` : <>h<sub>{col},{unit + 1}</sub></>);

  const describeUnit = (col, unit) => {
    if (col === 0) return `${name(col, unit)} — the ${unit === 0 ? "horizontal" : "vertical"} coordinate, as it enters`;
    const b = view?.biases?.[col - 1]?.[unit];
    const act = col === cols.length - 1 ? "softmax" : layers[col - 1].act;
    const value = col === cols.length - 1 ? probe?.out?.[unit] : probe?.layers?.[col - 1]?.[unit];
    return `${name(col, unit)} · ${act} · bias ${b != null ? b.toFixed(3) : "—"}` +
      (value != null ? ` · at the probe ${value.toFixed(3)}` : "");
  };

  /**
   * The unit written out with its real weights, biggest first, bias last — and, with a probe on
   * the screen, the value it takes there. The same numbers as the lit wires and the tile.
   */
  const formula = (col, unit) => {
    if (col >= cols.length || unit >= cols[col]) return null;       // a unit the reshaped network no longer has
    if (col === 0 || !view) return <span className="pg-formula"><b>{label(col, unit)}</b></span>;
    const W = view.weights[col - 1];
    const terms = Array.from({ length: cols[col - 1] }, (_, a) => ({ a, w: W[a][unit] }))
      .sort((p, q) => Math.abs(q.w) - Math.abs(p.w));
    const b = view.biases[col - 1][unit];
    const act = col === cols.length - 1 ? "softmax" : layers[col - 1].act;
    const value = col === cols.length - 1 ? probe?.out?.[unit] : probe?.layers?.[col - 1]?.[unit];
    const sign = (v) => (v <= -0.005 ? "−" : "+") + Math.abs(v).toFixed(2);   // no "−0.00"
    return (
      <span className="pg-formula">
        <b>{label(col, unit)}</b> = {act === "linear" ? "" : act}(
        {terms.map((t) => <span key={t.a}> {sign(t.w)}·{label(col - 1, t.a)}</span>)}
        <span> {sign(b)}</span> )
        {value != null && <span className="pg-formula-value"> = {value <= -0.0005 ? "−" : ""}{Math.abs(value).toFixed(3)}</span>}
      </span>
    );
  };

  // while it trains the weights move, and pointing at the screen moves the probe: the written
  // formula follows both, for the unit under the pointer or the pinned one
  const shown = useRef(false);
  useEffect(() => {
    if (held) { onStatus(formula(held.col, held.unit)); shown.current = true; }
    else if (shown.current) { onStatus(null); shown.current = false; }
  }, [view, probe, held?.col, held?.unit]); // eslint-disable-line react-hooks/exhaustive-deps

  const point = (col, unit) => {
    setFocus({ col, unit });
    onStatus(formula(col, unit));
    if (col > 0) onLens({ layer: col - 1, unit });
  };
  const unpoint = (col) => {
    setFocus(null);
    if (col > 0) onLens(null);
  };

  const lensKey = lens ? `${lens.layer + 1}:${lens.unit}` : null;

  return (
    <div className="pg-net" ref={box}>
      <svg className="pg-net-wires" width={size.w} height={size.h} aria-hidden="true">
        {wires.map((e) => (
          <g key={e.key}>
            <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                  stroke={`rgba(148, 255, 201, ${e.alpha})`} strokeWidth={e.width}
                  strokeDasharray={e.w < 0 ? "3 3" : undefined} />
            <line className="pg-net-hit" x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
                  onPointerEnter={() => onStatus(<span className="pg-formula">{label(e.l, e.a)} → {label(e.l + 1, e.b)} · w = {e.w <= -0.0005 ? "−" : ""}{Math.abs(e.w).toFixed(3)}</span>)}
                  onPointerLeave={() => onStatus(null)} />
          </g>
        ))}
      </svg>

      {cols.map((k, col) => {
        const hidden = col > 0 && col < cols.length - 1;
        const x = geo.xs[col] + geo.T / 2;
        return (
          <div className="pg-net-head" key={`h${col}`} style={{ left: x }}>
            {col === 0 && <span className="pg-net-title">input</span>}
            {col === cols.length - 1 && <span className="pg-net-title">output</span>}
            {hidden && (
              <>
                <span className="pg-net-units">
                  <button type="button" className="pg-tiny" aria-label={`Fewer units in layer ${col}`}
                          disabled={k <= 1} onClick={() => onUnits(col - 1, -1)}><Icon name="minus" size={10} /></button>
                  <b>{k}</b>
                  <button type="button" className="pg-tiny" aria-label={`More units in layer ${col}`}
                          disabled={k >= MAX_UNITS} onClick={() => onUnits(col - 1, 1)}><Icon name="plus" size={10} /></button>
                </span>
                <select className="pg-net-act" value={layers[col - 1].act} aria-label={`Activation of layer ${col}`}
                        onChange={(e) => onAct(col - 1, e.target.value)}>
                  {ACTIVATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
                </select>
              </>
            )}
          </div>
        );
      })}

      {cols.map((k, col) => Array.from({ length: k }, (_, unit) => {
        const b = geo.boxes[col][unit];
        const id = `${col}:${unit}`;
        const lensable = col > 0;
        const value = probe && col > 0
          ? (col === cols.length - 1 ? probe.out?.[unit] : probe.layers?.[col - 1]?.[unit])
          : probe && col === 0 ? (unit === 0 ? probe.x : probe.y) : null;
        const lit = wave == null || col <= wave;
        return (
          <button
            type="button" key={id}
            className={"pg-tile" + (lensKey === id ? " is-lens" : "") + (lens?.pinned && lensKey === id ? " is-pinned" : "")}
            style={{ left: b.x, top: b.y, width: geo.T, height: geo.T,
                     borderColor: col === cols.length - 1 ? CLASSES[unit].field : undefined }}
            aria-label={`${describeUnit(col, unit)}. Show it on the screen.`}
            onPointerEnter={() => point(col, unit)}
            onPointerLeave={() => unpoint(col)}
            onFocus={() => point(col, unit)}
            onBlur={() => unpoint(col)}
            onClick={() => lensable && onPin({ layer: col - 1, unit })}
          >
            <canvas ref={(node) => { tiles.current[id] = node; }} style={{ width: geo.T, height: geo.T }} />
            {probe && (
              <i className={"pg-tile-probe" + (probe.pinned ? " is-pinned" : "")}
                 style={{ left: `${((probe.x + 1) / 2) * 100}%`, top: `${(1 - (probe.y + 1) / 2) * 100}%` }} />
            )}
            {value != null && lit && geo.T >= 30 && <span className="pg-tile-value">{value.toFixed(2)}</span>}
          </button>
        );
      }))}

      {layers.length === 0 && (
        <p className="pg-net-empty" style={{ top: HEAD + 6 }}>
          No hidden layer: a single linear cut per class. Add one to bend it.
        </p>
      )}
    </div>
  );
}
