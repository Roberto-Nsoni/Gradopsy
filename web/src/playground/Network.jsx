import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { pgEmbed } from "../engine";
import { ACTIVATIONS, MAX_UNITS } from "./config";
import { CLASSES, bracket, fieldCuts, fieldImage, markInk, paintPoints, planeCut, planeImage, shapePath } from "./paint";
import Icon from "./icons";
import "./views/views.css";

/*
 * The network, opened up. Every unit is a small screen of its own showing what that unit
 * computes over the whole plane — the same plane the big screen shows — so reading left to right
 * is reading the forward pass: two coordinates, then cuts, then cuts of cuts, then the classes.
 *
 * A wire is a weight. Its thickness is the weight's size on an absolute scale, so a wire that
 * grows during training has really grown; a negative weight is dashed rather than recoloured,
 * because hue here already means class. With a probe on the screen, wires carry how much of that
 * one input's signal actually flows along them, and everything else goes quiet.
 *
 * Under each column sits the other face of the same thing: the data points placed where that
 * layer puts them. Point at or pin a unit and the two faces are tied together — every point, on
 * every screen, lights up by its value in that unit, so the side of the tile's cut a point falls
 * on is the side it is lit on everywhere; the unit's own axis is drawn in its layer's space; and in
 * the last hidden layer the output's cut is drawn where it crosses the picture, straight, because
 * there it is straight.
 *
 * A layer of one or two units is drawn whole. A wider one is drawn through its two principal
 * directions, and the label says how much of the layer that shows; there the unit's axis and the
 * output's cut are what the picture's plane meets, not the whole of them.
 */

const HEAD = 62;          // the strip of controls above each column
const FOOT = 10;
const LABEL = 16;         // the line above each space screen
const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const R = (k) => "ℝ" + String(k).split("").map((d) => SUP[d]).join("");

/** How a unit's value reads, as a 0..1 brightness and the level where it switches. */
function reading(act, values) {
  if (act === "tanh") return { lo: -1, hi: 1, at: 0 };
  if (act === "sigmoid" || act === "prob") return { lo: 0, hi: 1, at: 0.5 };
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (act === "relu") { const top = Math.max(hi, 1e-6); return { lo: 0, hi: top, at: top * 1e-3 }; }
  const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6);
  return { lo: -m, hi: m, at: 0 };
}

/** The top two principal directions of rows (n×d), by power iteration with deflation. */
function pca2(rows) {
  const d = rows[0].length, n = rows.length;
  const mean = Array(d).fill(0);
  for (const r of rows) for (let k = 0; k < d; k++) mean[k] += r[k] / n;
  const C = Array.from({ length: d }, () => Array(d).fill(0));
  for (const r of rows) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
  const trace = C.reduce((t, row, i) => t + row[i], 0) || 1e-12;
  const vecs = [], vals = [];
  for (let c = 0; c < 2; c++) {
    let v = Array.from({ length: d }, (_, k) => 1 + k * 0.37);
    for (let it = 0; it < 80; it++) {
      const w = C.map((row) => row.reduce((a, cij, j) => a + cij * v[j], 0));
      const nrm = Math.hypot(...w) || 1;
      v = w.map((x) => x / nrm);
    }
    const lam = v.reduce((a, vi, i) => a + vi * C[i].reduce((b, cij, j) => b + cij * v[j], 0), 0);
    // two classes have a single direction of spread: the second is left empty, not filled with noise
    if (lam < trace * 1e-9) v = v.map(() => 0);
    vecs.push(v); vals.push(lam);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] -= lam * v[i] * v[j];
  }
  return { mean, vecs, share: Math.max(0, (vals[0] + vals[1]) / trace) };
}

const fmtEnd = (v) => (Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(1)).replace("-", "−");

const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };

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
  points = [], at = null, engineKey = "playground",
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
    // the space screens take the foot of the panel, one per column, as wide as a column allows
    const S = Math.round(Math.max(56, Math.min(132, ((w - pad * 2) / cols.length) * 0.92, h * 0.27)));
    const avail = Math.max(120, h - HEAD - FOOT - S - LABEL - 12);
    const T = Math.round(Math.max(20, Math.min(80, (avail / most) * 0.8, ((w - pad * 2) / cols.length) * 0.62)));
    const gap = Math.max(5, Math.min(14, (avail - most * T) / Math.max(most - 1, 1)));
    const span = w - pad * 2 - T;
    const xs = cols.map((_, i) => pad + (cols.length > 1 ? (span * i) / (cols.length - 1) : span / 2));
    const boxes = cols.map((k, i) => {
      const height = k * T + (k - 1) * gap;
      const y0 = HEAD + (avail - height) / 2;
      return Array.from({ length: k }, (_, j) => ({ x: xs[i], y: y0 + j * (T + gap) }));
    });
    return { T, xs, boxes, S, spaceTop: h - FOOT - S };
  }, [size, cols]);

  // ---------- the spaces ----------

  const [embed, setEmbed] = useState(null);
  const spaces = useRef({});
  useEffect(() => {
    if (!view) return;
    try { setEmbed(pgEmbed(engineKey, at ?? -1)); } catch { setEmbed(null); }
  }, [view, at, engineKey]);
  // one picture per hidden layer, then the output's own (one number per class, before the softmax)
  const embedOk = embed && embed.layers.length === layers.length + 1 &&
    embed.layers.every((l, i) => l.units === (i < layers.length ? layers[i].units : classes) && l.xy.length === points.length);

  /**
   * A hidden layer of one or two units is drawn whole on a scale it always has: tanh on −1..1,
   * sigmoid on 0..1, and relu or linear on the range its points actually reach, written on the
   * screen. Only a wider layer falls back to its shadow, which has no scale of its own.
   */
  const scaleOf = (col, e) => {
    if (!e || col === 0 || col === cols.length - 1 || !e.exact) return null;
    const act = layers[col - 1].act;
    let lo, hi;
    if (act === "tanh") [lo, hi] = [-1, 1];
    else if (act === "sigmoid") [lo, hi] = [0, 1];
    else {
      let m = 1e-6, mn = 0;
      for (const h of e.acts) for (const v of h) { m = Math.max(m, Math.abs(v)); mn = Math.min(mn, v); }
      [lo, hi] = act === "relu" || mn >= 0 ? [0, +m.toPrecision(2)] : [-m, m].map((v) => +v.toPrecision(2));
    }
    const to = (v) => ((v - lo) / (hi - lo || 1)) * 2 - 1;
    return { lo, hi, place: (h) => [to(h[0]), h.length > 1 ? to(h[1]) : 0] };
  };

  /**
   * How the output space is drawn. Two classes: the square of pA and pB, exact. Three: the
   * triangle the three probabilities lie on — it is a flat plane already, so drawing it is exact
   * and PCA would only rotate it. Four: the probabilities fill a tetrahedron, which does not fit,
   * so it is drawn through its PCA like any other wide layer, with the class corners carried
   * through the same projection.
   */
  const pcaSigns = useRef(null);
  const outFrame = useMemo(() => {
    if (!embedOk) return { place: () => [0, 0], corners: [], share: null, invert: null };
    const probs = embed.layers[layers.length].acts.map(softmax);
    if (probs.length < 3) return { place: () => [0, 0], corners: [], share: null, invert: null };
    const { mean, vecs, share } = pca2(probs);
    if (pcaSigns.current && pcaSigns.current[0].length === classes) {
      vecs.forEach((v, j) => { if (v.reduce((a2, x, k) => a2 + x * pcaSigns.current[j][k], 0) < 0) vecs[j] = v.map((x) => -x); });
    }
    pcaSigns.current = vecs;
    const proj = (p) => vecs.map((v) => p.reduce((a2, pk, k) => a2 + (pk - mean[k]) * v[k], 0));
    const corners = Array.from({ length: classes }, (_, k) => proj(Array.from({ length: classes }, (_, j) => (j === k ? 1 : 0))));
    const span = Math.max(1e-6, ...corners.flat().map(Math.abs));
    // when the picture holds all of the spread (two or three classes), every spot of it is one
    // exact mix of probabilities, so the verdict can be painted under the points
    const invert = share > 0.999 ? (x, y, raw = false) => {
      const p = mean.map((m, k) => m + span * (x * vecs[0][k] + y * vecs[1][k]));
      return raw || !p.some((v) => v < -0.01) ? p : null;
    } : null;
    return { place: (p) => proj(p).map((x) => x / span), corners: corners.map((c) => c.map((x) => x / span)), share, invert };
  }, [classes, embedOk, embed, layers.length]);

  /** Every point's value in the unit being read, and how to read it; null when nothing is read. */
  const litBy = useMemo(() => {
    if (!held || !embedOk || !view) return null;
    const { col, unit } = held;
    let values;
    let act;
    if (col === 0) { values = points.map((q) => (unit === 0 ? q.x : q.y)); act = "linear"; }
    else if (col < cols.length - 1) { values = embed.layers[col - 1].acts.map((h) => h[unit]); act = layers[col - 1].act; }
    else {
      values = embed.layers[layers.length].acts.map((z) => softmax(z)[unit]);
      act = "prob";
    }
    return { values, ...reading(act, values) };
  }, [held, embedOk, embed, view, points, layers, cols]);

  useEffect(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const S = geo.S, px = Math.round(S * dpr);
    for (let col = 0; col < cols.length; col++) {
      const node = spaces.current[col];
      if (!node) continue;
      if (node.width !== px) node.width = node.height = px;
      const ctx = node.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#060d09"; ctx.fillRect(0, 0, px, px);

      const e = col > 0 && embedOk ? embed.layers[col - 1] : null;
      if (col > 0 && !e) continue;
      const out = col === cols.length - 1;
      const fixed = scaleOf(col, e);
      const at2 = (x, y) => [((x * 0.9 + 1) / 2) * px, px - ((y * 0.9 + 1) / 2) * px];
      const font = (size) => `600 ${Math.round(size * dpr)}px "Martian Mono", ui-monospace, monospace`;

      // the output, in probabilities: every point is a mix of the class corners, so a sure point
      // sits on its own corner — [1, 0] or [0, 1] with two classes — and the verdict is which
      // probability is larger, a straight line through the middle
      if (out && outFrame.invert) {
        const g = 48;
        const field = new Uint8Array(g * g), conf = new Uint8Array(g * g);
        for (let r = 0; r < g; r++) for (let c = 0; c < g; c++) {
          const x = ((c / (g - 1)) * 2 - 1) / 0.9, y = ((r / (g - 1)) * 2 - 1) / 0.9;
          let p = outFrame.invert(x, y);
          if (!p) {
            // off the simplex no point can be: left unlit, but given the class of the nearest
            // valid mix so its edge is not mistaken for a cut
            const raw = outFrame.invert(x, y, true);
            const clipped = raw.map((v) => Math.max(0, v));
            const total = clipped.reduce((t, v) => t + v, 0) || 1;
            p = clipped.map((v) => v / total);
            let best = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
            field[r * g + c] = best; conf[r * g + c] = 0;
            continue;
          }
          let best = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k;
          field[r * g + c] = best; conf[r * g + c] = Math.round(Math.min(1, Math.max(...p)) * 255);
        }
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(fieldImage(field, conf, g, classes), 0, 0, px, px);
        ctx.lineCap = "round";
        const path = new Path2D();
        for (const [x0, y0, x1, y1] of fieldCuts(field, conf, g, classes, px)) { path.moveTo(x0, y0); path.lineTo(x1, y1); }
        ctx.strokeStyle = "rgba(200, 255, 226, 0.85)"; ctx.lineWidth = 1.4 * dpr; ctx.stroke(path);
      }

      // the graticule, the screen's own ruling
      ctx.strokeStyle = "rgba(148, 255, 201, 0.13)"; ctx.lineWidth = 1;
      for (let k = 1; k < 4; k++) {
        const v = Math.round((k / 4) * px) + 0.5;
        ctx.beginPath(); ctx.moveTo(v, 0); ctx.lineTo(v, px); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(px, v); ctx.stroke();
      }

      const xy = col === 0 ? points.map((q) => [q.x, q.y]) : out ? e.acts.map((z) => outFrame.place(softmax(z))) : fixed ? e.acts.map(fixed.place) : e.xy;
      const r0 = Math.max(1.8, px / 75);

      if (!litBy) {
        paintPoints(ctx, points.map((q, i) => ({ ...q, x: xy[i][0] * 0.9, y: xy[i][1] * 0.9 })), px, r0);
      } else {
        // lit by the unit being read: full ink on the side of its cut where it fires, faint on the other
        points.forEach((q, i) => {
          const v = litBy.values[i];
          const on = v > litBy.at;
          const t = Math.max(0, Math.min(1, (v - litBy.lo) / (litBy.hi - litBy.lo || 1)));
          const [x, y] = at2(xy[i][0], xy[i][1]);
          shapePath(ctx, CLASSES[q.c].shape, x, y, r0 * (on ? 1.15 : 0.9));
          if (on) { ctx.fillStyle = markInk(q.c, 0.35 + 0.65 * t); ctx.fill(); }
          else { ctx.strokeStyle = markInk(q.c, 0.22); ctx.lineWidth = 1; ctx.stroke(); }
        });
      }

      // the scale, written on the screen: the corners of the output, the ends of a drawn axis
      ctx.font = font(8.5); ctx.textBaseline = "middle";
      if (out) {
        // each corner is a sure verdict, named by the class it belongs to
        const names = CLASSES.slice(0, classes).map((c) => c.name);
        const centre = outFrame.place(Array(classes).fill(1 / classes));
        const placed = [];
        outFrame.corners.forEach(([x, y], k) => {
          const [a, b] = at2(x, y);
          ctx.fillStyle = markInk(k, 0.95);
          // the exact corner, marked; its label set off it, away from the middle, inside the screen,
          // and nudged clear of any label already placed (corners can fall close in a projection)
          const [ca, cb] = at2(centre[0], centre[1]);
          ctx.strokeStyle = markInk(k, 0.95); ctx.lineWidth = 1.2 * dpr;
          ctx.beginPath(); ctx.arc(a, b, 4 * dpr, 0, Math.PI * 2); ctx.stroke();
          const w = ctx.measureText(names[k]).width, lh = 11 * dpr;
          const away = Math.hypot(a - ca, b - cb) > 4 * dpr;
          const dx = away ? (a - ca) / Math.hypot(a - ca, b - cb) : 0, dy = away ? (b - cb) / Math.hypot(a - ca, b - cb) : -1;
          let tx = Math.max(3 * dpr + w / 2, Math.min(px - 3 * dpr - w / 2, a + dx * (w / 2 + 8 * dpr)));
          let ty = Math.max(lh / 2 + 2, Math.min(px - lh / 2 - 2, b + dy * 12 * dpr));
          // pushed back onto its own corner by the screen's edge: set it on the inside instead
          if (Math.hypot(tx - a, ty - b) < 11 * dpr) {
            tx = Math.max(3 * dpr + w / 2, Math.min(px - 3 * dpr - w / 2, a - dx * (w / 2 + 10 * dpr)));
            ty = Math.max(lh / 2 + 2, Math.min(px - lh / 2 - 2, b - dy * 14 * dpr));
          }
          for (let tries = 0; tries < 6 && placed.some((r) => Math.abs(r.x - tx) < (r.w + w) / 2 + 2 && Math.abs(r.y - ty) < lh); tries++) {
            ty = ty + (ty < px / 2 ? lh : -lh);
          }
          placed.push({ x: tx, y: ty, w });
          ctx.textAlign = "center";
          ctx.fillText(names[k], tx, ty);
        });
      } else if (col === 0 || fixed) {
        const { lo, hi } = col === 0 ? { lo: -1, hi: 1 } : fixed;
        ctx.fillStyle = "rgba(169, 201, 187, 0.9)";
        ctx.textAlign = "left"; ctx.fillText(fmtEnd(lo), 4 * dpr, px - 8 * dpr);
        ctx.textAlign = "right"; ctx.fillText(fmtEnd(hi), px - 4 * dpr, px - 8 * dpr);
        ctx.textAlign = "left"; ctx.fillText(fmtEnd(hi), 4 * dpr, 9 * dpr);
      }

      // the unit's own axis, in its own layer's space
      if (held && held.col === col && (col === 0 || e)) {
        // the unit's own basis vector e_k, at its true length: from the layer's origin (every unit at
        // 0) to the point where this unit is 1 and the rest are 0, placed exactly the way the data
        // points are. Under a PCA it comes out shorter the less of it the picture's plane holds.
        let from, to;
        const unitVec = (d, k) => Array.from({ length: d }, (_, j) => (j === k ? 1 : 0));
        if (col === 0) { from = at2(0, 0); to = at2(...(held.unit === 0 ? [1, 0] : [0, 1])); }
        else if (out) {
          // probabilities: the origin is the uniform guess, which the projection puts in the middle
          from = at2(...outFrame.place(Array(classes).fill(1 / classes)));
          to = at2(...outFrame.place(unitVec(classes, held.unit)));
        } else if (fixed) {
          from = at2(...fixed.place(Array(e.units).fill(0)));
          to = at2(...fixed.place(unitVec(e.units, held.unit)));
        } else {
          const proj = (h) => [0, 1].map((j) => h.reduce((s2, hv, k) => s2 + (hv - e.mean[k]) * e.axes[j][k], 0) / e.span);
          from = at2(...proj(Array(e.units).fill(0)));
          to = at2(...proj(unitVec(e.units, held.unit)));
        }
        if (from && to) {
          const len = Math.hypot(to[0] - from[0], to[1] - from[1]);
          const ink = "#c9ff4a";
          // the axis itself, across the whole screen: a line has no length, so it is never scaled
          // and always visible, however short the basis vector on it comes out
          if (len > 0.5) {
            const ux = (to[0] - from[0]) / len, uy = (to[1] - from[1]) / len, far = px * 2;
            ctx.strokeStyle = "rgba(201, 255, 74, 0.45)"; ctx.lineWidth = 1 * dpr; ctx.setLineDash([4 * dpr, 4 * dpr]);
            ctx.beginPath(); ctx.moveTo(from[0] - ux * far, from[1] - uy * far); ctx.lineTo(from[0] + ux * far, from[1] + uy * far); ctx.stroke();
            ctx.setLineDash([]);
          }
          // the origin of the layer, where every unit is 0
          ctx.strokeStyle = ink; ctx.lineWidth = 1.2 * dpr;
          ctx.beginPath(); ctx.arc(from[0], from[1], 2.5 * dpr, 0, Math.PI * 2); ctx.stroke();
          // the basis vector at its true length, and its tip marked "1": where this unit equals 1
          if (len > 0.5) {
            ctx.strokeStyle = ink; ctx.lineWidth = 2 * dpr;
            ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
            const ang = Math.atan2(to[1] - from[1], to[0] - from[0]), hd = Math.min(7 * dpr, len * 0.6);
            ctx.beginPath(); ctx.moveTo(to[0], to[1]);
            ctx.lineTo(to[0] - hd * Math.cos(ang - 0.45), to[1] - hd * Math.sin(ang - 0.45));
            ctx.lineTo(to[0] - hd * Math.cos(ang + 0.45), to[1] - hd * Math.sin(ang + 0.45));
            ctx.closePath(); ctx.fillStyle = ink; ctx.fill();
            ctx.font = font(8); ctx.fillStyle = ink; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            const nx = -Math.sin(ang), ny = Math.cos(ang);
            ctx.fillText("1", to[0] + nx * 9 * dpr, to[1] + ny * 9 * dpr);
          }
        }
      }

      // the probe, carried into this space
      if (probe && probe.layers) {
        let p2 = null;
        if (col === 0) p2 = [probe.x, probe.y];
        else if (out && probe.out?.length === classes) p2 = outFrame.place(probe.out);
        else if (e && probe.layers[col - 1]?.length === e.units) {
          const h = probe.layers[col - 1];
          p2 = fixed ? fixed.place(h) : [0, 1].map((j) => h.reduce((s2, hv, k) => s2 + (hv - e.mean[k]) * e.axes[j][k], 0) / e.span);
        }
        if (p2) { const [x, y] = at2(p2[0], p2[1]); bracket(ctx, x, y, 7 * dpr, "rgba(201, 255, 74, 0.95)", 4 * dpr, 1.4 * dpr); }
      }
    }
  }, [geo, cols, embed, embedOk, points, view, litBy, held, probe, classes, outFrame]);

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

      {cols.map((k, col) => {
        const e = col > 0 && embedOk ? embed.layers[col - 1] : null;
        // centred under its column, but never past the panel's edges
        const left = Math.max(6, Math.min(size.w - geo.S - 6, geo.xs[col] + geo.T / 2 - geo.S / 2));
        return (
          <figure className="pgl-space" key={`s${col}`} style={{ left, top: geo.spaceTop - LABEL, width: geo.S }}>
            {col === cols.length - 1 ? (
              <figcaption title={`The probabilities the network gives each point, through their PCA like any other layer: the two directions where the points spread the most, holding ${outFrame.share != null ? Math.round(outFrame.share * 100) : "—"}% of the spread. Each class's corner — the network certain of it — is carried through the same projection.`}>
                <b>p</b>
                {outFrame.share != null && <span>PCA · {Math.round(outFrame.share * 100)}%</span>}
              </figcaption>
            ) : (
              <figcaption title={e && !e.exact ? `${R(k)} cannot be drawn, so this is its PCA: the points on the two directions where they spread the most. Those two directions hold ${Math.round(e.share * 100)}% of the spread; the rest is hidden. A projection has no scale of its own.` : undefined}>
                <b>{R(k)}</b>
                {e && !e.exact && <span>PCA · {Math.round(e.share * 100)}%</span>}
              </figcaption>
            )}
            <canvas ref={(node) => { spaces.current[col] = node; }} style={{ width: geo.S, height: geo.S }} />
          </figure>
        );
      })}

      {layers.length === 0 && (
        <p className="pg-net-empty" style={{ top: HEAD + 6 }}>
          No hidden layer: a single linear cut per class. Add one to bend it.
        </p>
      )}
    </div>
  );
}
