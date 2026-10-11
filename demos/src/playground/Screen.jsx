import { useEffect, useRef, useState } from "react";

import { paintRaster } from "../looks/scope";
import {
  CLASSES, bracket, fieldCuts, fieldImage, markInk, paintPoints, planeCut, planeImage, shapePath, strokeCut,
} from "./paint";

/*
 * The screen: the plane the data lives on, the network's verdict over it, and the pen.
 *
 * It is painted the way the cover is — a raster at the engine's own resolution upscaled with
 * smoothing, scanlines carved back out, a ruled graticule, and a second layer for the cut and
 * the marks that is faded rather than cleared, so the network is seen chasing rather than
 * arriving. Everything it draws comes out of the store the page keeps; it owns only the pointer.
 */

const SIZE = 900;            // the backing store, fixed and stretched by CSS (the cover's rule)
const FADE = 0.2;
const MIN_GAP = 0.024;       // how close a sprayed point may land to any other: the brush fills, it does not pile

const toWorld = (event, node) => {
  const b = node.getBoundingClientRect();
  return [((event.clientX - b.left) / b.width) * 2 - 1, 1 - ((event.clientY - b.top) / b.height) * 2];
};
const toPx = (v) => ((v + 1) / 2) * SIZE;

export default function Screen({ store, tool, radius, active, role, onEdit, onProbe, onStrokeEnd, label, notice, starter, split }) {
  const [only, setOnly] = useState({ hover: null, pinned: null });   // train or test, picked out on the screen
  const showOnly = only.hover ?? only.pinned;
  const onlyRef = useRef(null);
  onlyRef.current = showOnly;
  const canvas = useRef(null);
  const glow = useRef(null);
  const pen = useRef({ down: false, at: null, grabbed: -1, erase: false, hover: null, last: 0 });
  const props = useRef({});
  props.current = { tool, radius, active, role, onEdit, onProbe, onStrokeEnd };

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    if (!glow.current) glow.current = document.createElement("canvas");
    let cached = { key: null, image: null, segs: null };

    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const node = canvas.current;
      if (!node) return;

      spray(dt);

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const px = Math.round(SIZE * dpr);
      if (node.width !== px) { node.width = node.height = px; glow.current.width = glow.current.height = px; }
      const ctx = node.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // the ground: the output field, or the neuron under the lens, or a bare tube
      const s = store.current;
      const lens = s.lens && s.view ? lensPlane(s) : null;
      // an empty plane shows the bare tube: a verdict over no data would be a picture of nothing
      const key = lens ? `lens:${s.viewStamp}:${s.lens.layer}:${s.lens.unit}` : `field:${s.fieldStamp}:${s.points.length > 0}`;
      if (cached.key !== key) {
        cached = { key, image: null, segs: null };
        if (lens) {
          cached.image = planeImage(lens.plane, lens.n, lens.act, lens.cls);
          cached.segs = planeCut(lens.cls != null ? lens.plane.map((v) => v - 0.5) : lens.plane, lens.n, lens.cls != null ? "linear" : lens.act, SIZE);
        } else if (s.field && !s.diverged && s.points.length) {
          cached.image = fieldImage(s.field.field, s.field.confidence, s.field.n, s.classes);
          cached.segs = fieldCuts(s.field.field, s.field.confidence, s.field.n, s.classes, SIZE);
        }
      }

      ctx.fillStyle = "#060d09";
      ctx.fillRect(0, 0, SIZE, SIZE);
      if (cached.image) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(cached.image, 0, 0, SIZE, SIZE);
      }
      paintRaster(ctx, SIZE);
      axes(ctx);

      // the phosphor: faded, never cleared
      const g = glow.current.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.globalCompositeOperation = "destination-out";
      g.fillStyle = `rgba(0, 0, 0, ${FADE})`;
      g.fillRect(0, 0, SIZE, SIZE);
      g.globalCompositeOperation = "source-over";
      if (cached.segs) strokeCut(g, cached.segs);
      paintPoints(g, s.points, SIZE, SIZE > 620 ? 5.2 : 4.2, pen.current.grabbed, onlyRef.current);

      ctx.globalCompositeOperation = "lighter";
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(glow.current, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // what is not signal: the probe and the pen, drawn fresh so they never smear
      if (s.probe) {
        const x = toPx(s.probe.x), y = SIZE - toPx(s.probe.y);
        bracket(ctx, x, y, 13, s.probe.pinned ? "rgba(201, 255, 74, 0.95)" : "rgba(200, 255, 226, 0.8)");
      }
      reticle(ctx);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [store]);

  /** The neuron the visitor is pointing at in the schematic, as a plane the screen can draw. */
  function lensPlane(s) {
    const { view, lens } = s;
    const n = view.n, size = n * n;
    const sizes = view.sizes;
    // the lens may name a unit of a network that has just been reshaped; then there is nothing to show
    if (!sizes || lens.layer >= sizes.length || lens.unit >= sizes[lens.layer]) return null;
    if (lens.layer < sizes.length - 1 && !s.layers[lens.layer]) return null;
    let offset = 0;
    for (let i = 0; i < lens.layer; i++) offset += sizes[i] * size;
    offset += lens.unit * size;
    const plane = view.acts.subarray(offset, offset + size);
    const out = lens.layer === sizes.length - 1;
    return { plane, n, act: out ? null : s.layers[lens.layer].act, cls: out ? lens.unit : null };
  }

  function axes(ctx) {
    ctx.fillStyle = "rgba(148, 255, 201, 0.55)";
    for (let i = 0; i <= 10; i++) {
      const v = (i / 10) * SIZE, len = i % 5 === 0 ? 12 : 6;
      ctx.fillRect(v - 0.5, SIZE - len, 1, len);
      ctx.fillRect(0, v - 0.5, len, 1);
    }
  }

  function reticle(ctx) {
    const p = pen.current;
    const { tool, radius, active } = props.current;
    if (!p.at || tool === "move") return;
    const x = toPx(p.at[0]), y = SIZE - toPx(p.at[1]);
    const erasing = tool === "erase" || p.erase;
    if (tool === "point" && !erasing) {
      ctx.strokeStyle = markInk(active, 0.9); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x - 14, y); ctx.lineTo(x - 5, y); ctx.moveTo(x + 5, y); ctx.lineTo(x + 14, y);
      ctx.moveTo(x, y - 14); ctx.lineTo(x, y - 5); ctx.moveTo(x, y + 5); ctx.lineTo(x, y + 14); ctx.stroke();
      return;
    }
    const r = (radius / 2) * SIZE;
    ctx.lineWidth = 1.2;
    ctx.setLineDash(erasing ? [5, 4] : []);
    ctx.strokeStyle = erasing ? "rgba(200, 255, 226, 0.8)" : markInk(active, p.down ? 0.95 : 0.7);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    if (!erasing) {
      // the pen's mark says where its points go: filled for training, hollow for held out
      shapePath(ctx, CLASSES[active].shape, x, y, props.current.role === "test" ? 4 : 3.2);
      if (props.current.role === "test") { ctx.strokeStyle = markInk(active, 0.95); ctx.lineWidth = 1.4; ctx.stroke(); }
      else { ctx.fillStyle = markInk(active, 0.95); ctx.fill(); }
    }
  }

  /** Held down, the brush keeps filling its disc until there is no room left in it. */
  function spray(dt) {
    const p = pen.current;
    const { tool, radius, active, onEdit } = props.current;
    if (!p.down || !p.at) return;
    const s = store.current;

    if (tool === "erase" || p.erase) {
      const r2 = radius * radius;
      const kept = s.points.filter((q) => (q.x - p.at[0]) ** 2 + (q.y - p.at[1]) ** 2 > r2);
      if (kept.length !== s.points.length) onEdit(kept);
      return;
    }
    if (tool !== "brush") return;

    p.carry = (p.carry ?? 0) + dt * (30 + 2600 * radius * radius);
    const added = [];
    for (; p.carry >= 1; p.carry -= 1) {
      for (let attempt = 0; attempt < 6; attempt++) {
        const a = Math.random() * Math.PI * 2, d = Math.sqrt(Math.random()) * radius;
        const x = p.at[0] + Math.cos(a) * d, y = p.at[1] + Math.sin(a) * d;
        if (Math.abs(x) > 0.98 || Math.abs(y) > 0.98) continue;
        if (crowded(x, y, s.points, added)) continue;
        added.push(fresh(x, y, active));
        break;
      }
    }
    if (added.length) onEdit([...s.points, ...added]);
  }

  const crowded = (x, y, ...sets) =>
    sets.some((set) => set.some((q) => (q.x - x) ** 2 + (q.y - y) ** 2 < MIN_GAP * MIN_GAP));
  const fresh = (x, y, c) => {
    const role = props.current.role;
    return { x: Math.round(x * 1e4) / 1e4, y: Math.round(y * 1e4) / 1e4, c, u: Math.random(),
             ...(role === "train" || role === "test" ? { role } : {}) };
  };

  const down = (event) => {
    if (event.button > 0) return;
    event.preventDefault();
    const node = canvas.current;
    node.setPointerCapture(event.pointerId);
    const at = toWorld(event, node);
    const p = pen.current;
    const { tool, active, onEdit, onProbe } = props.current;
    p.down = true; p.at = at; p.erase = event.shiftKey; p.carry = 1;
    onProbe(null);

    const s = store.current;
    if (tool === "move" && !p.erase) {
      let best = -1, bestD = 0.07 * 0.07;
      s.points.forEach((q, i) => { const d = (q.x - at[0]) ** 2 + (q.y - at[1]) ** 2; if (d < bestD) { bestD = d; best = i; } });
      p.grabbed = best;
    } else if (tool === "point" && !p.erase) {
      if (!crowded(at[0], at[1], s.points)) onEdit([...s.points, fresh(at[0], at[1], active)]);
    }
  };

  const move = (event) => {
    const node = canvas.current;
    const at = toWorld(event, node);
    const p = pen.current;
    p.at = at;
    const { onEdit, onProbe } = props.current;
    if (p.down && p.grabbed >= 0) {
      const s = store.current;
      const pts = s.points.slice();
      pts[p.grabbed] = { ...pts[p.grabbed], x: Math.max(-0.98, Math.min(0.98, at[0])), y: Math.max(-0.98, Math.min(0.98, at[1])) };
      onEdit(pts);
    }
    if (!p.down) onProbe(at);
  };

  const up = () => {
    const p = pen.current;
    if (!p.down) return;
    p.down = false; p.grabbed = -1; p.erase = false;
    props.current.onStrokeEnd();
  };

  const leave = () => { pen.current.at = null; if (!pen.current.down) props.current.onProbe(null); };

  return (
    <div className="pg-screen">
      <canvas
        ref={canvas} className="pg-screen-canvas"
        style={{ cursor: tool === "move" ? (pen.current.grabbed >= 0 ? "grabbing" : "grab") : "none", touchAction: "none" }}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onPointerLeave={leave}
        onContextMenu={(e) => e.preventDefault()}
        aria-label="The data plane. Draw points with the brush, erase them, or drag them; the field behind them is what the network predicts."
      />
      {label && <p className="pg-screen-label">{label}</p>}
      {starter}
      {notice && <p className="pg-screen-notice" role="status">{notice}</p>}
      {/* each name at the far end of its own ruled edge, pointing the way the value grows */}
      {split && split.train + split.test > 0 && (
        <div className="pg-split" role="group" aria-label="Pick out the training or the test points">
          {[["train", split.train, "the network learns from these"], ["test", split.test, "held out: the network never sees these"]].map(([id, n, help]) => (
            <button type="button" key={id} className={"pg-split-key" + (only.pinned === id ? " is-on" : "")}
                    aria-pressed={only.pinned === id} title={help}
                    onPointerEnter={() => setOnly((o) => ({ ...o, hover: id }))}
                    onPointerLeave={() => setOnly((o) => ({ ...o, hover: null }))}
                    onClick={() => setOnly((o) => ({ ...o, pinned: o.pinned === id ? null : id }))}>
              <i className={"pg-split-mark is-" + id} />
              <span>{id}</span>
              <b>{n}</b>
            </button>
          ))}
        </div>
      )}
      <div className="pg-axes" aria-hidden="true">
        <span className="pg-axis pg-axis-x">
          <span>x<sub>1</sub></span>
          <svg width="14" height="8" viewBox="0 0 14 8"><path d="M0 4h12M9 1l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.3" /></svg>
        </span>
        <span className="pg-axis pg-axis-y">
          <svg width="8" height="14" viewBox="0 0 8 14"><path d="M4 14V2M1 5l3-3 3 3" fill="none" stroke="currentColor" strokeWidth="1.3" /></svg>
          <span>x<sub>2</sub></span>
        </span>
      </div>
    </div>
  );
}
