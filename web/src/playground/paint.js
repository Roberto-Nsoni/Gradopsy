/*
 * How the playground draws, on the phosphor screen's own terms.
 *
 * Up to four classes now, where the cover has two. Each class keeps the cover's arrangement — a
 * field ink mixed into the ground by how sure the network is, and a brighter mark ink for the
 * observations — and the two new ones are the other phosphors a tube was ever coated with: P3
 * amber and the red of P22. Hue is never the only thing telling classes apart: every class has
 * its own mark shape as well, so the screen still reads with the colour taken away.
 *
 * A neuron is not a class, so it never borrows a class ink. A neuron's plane is drawn in the
 * phosphor alone, at a brightness that is its activation, with its cut traced where it changes
 * sign. Output units are classes, and those do wear their class's ink.
 */

import { SCOPE } from "../looks/scope";

const hex = (v) => [parseInt(v.slice(1, 3), 16), parseInt(v.slice(3, 5), 16), parseInt(v.slice(5, 7), 16)];
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export const CLASSES = [
  { name: "A", field: "#39ff9e", mark: "#c9ff4a", shape: "disc" },
  { name: "B", field: "#2896ff", mark: "#5ac8ff", shape: "square" },
  { name: "C", field: "#ffae3d", mark: "#ffcf5c", shape: "triangle" },
  { name: "D", field: "#ff4f7b", mark: "#ff86a8", shape: "diamond" },
];
export const MAX_CLASSES = CLASSES.length;

const GROUND = hex(SCOPE.ground);
const PHOSPHOR = hex("#94ffc9");
const FIELD = CLASSES.map((c) => hex(c.field));
const MARK = CLASSES.map((c) => hex(c.mark).join(", "));

/** n x n bytes in grid order (row 0 is y = -1) onto an n x n image, the right way up. */
function rasterise(n, colour) {
  const buf = document.createElement("canvas");
  buf.width = buf.height = n;
  const bctx = buf.getContext("2d");
  const image = bctx.createImageData(n, n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const c = colour((n - 1 - row) * n + col);
      const p = (row * n + col) * 4;
      image.data[p] = c[0]; image.data[p + 1] = c[1]; image.data[p + 2] = c[2]; image.data[p + 3] = 255;
    }
  }
  bctx.putImageData(image, 0, 0);
  return buf;
}

/** Certainty is brightness: a confident region is lit, an unsure one is nearly the bare tube. */
export function fieldImage(field, confidence, n, classes) {
  const floor = 1 / classes;
  return rasterise(n, (i) => {
    const sure = Math.max(0, (confidence[i] / 255 - floor) / (1 - floor));
    return mix(GROUND, FIELD[field[i]] ?? GROUND, 0.04 + sure * 0.26);
  });
}

/**
 * The range a neuron's plane is read against. Bounded activations use their own bounds, so a
 * dim tanh unit really is a unit near zero; unbounded ones are read against their own extent.
 */
export function planeRange(plane, act) {
  if (act === "tanh") return [-1, 1, 0];
  if (act === "sigmoid") return [0, 1, 0.5];
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < plane.length; i++) { const v = plane[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (act === "relu") { const top = Math.max(hi, 1e-6); return [0, top, top * 1e-3]; }
  const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6);
  return [-m, m, 0];
}

/** A neuron's activation over the plane: the phosphor, as bright as the unit is driven. */
export function planeImage(plane, n, act, cls = null) {
  if (cls != null) {
    const ink = FIELD[cls];
    return rasterise(n, (i) => mix(GROUND, ink, 0.04 + Math.max(0, Math.min(1, plane[i])) * 0.62));
  }
  const [lo, hi] = planeRange(plane, act);
  const span = hi - lo || 1;
  return rasterise(n, (i) => mix(GROUND, PHOSPHOR, 0.03 + Math.max(0, Math.min(1, (plane[i] - lo) / span)) * 0.6));
}

/**
 * Marching squares over any scalar on the grid, already in pixel space and the right way up.
 * `value(i)` is signed: the cut is where it crosses zero.
 */
export function contour(value, n, size) {
  const out = [];
  const cross = (a, b) => a / (a - b);
  const px = (v) => (v / (n - 1)) * size;
  const py = (v) => size - (v / (n - 1)) * size;

  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const tl = value(r * n + c), tr = value(r * n + c + 1);
      const br = value((r + 1) * n + c + 1), bl = value((r + 1) * n + c);
      const mask = (tl > 0 ? 8 : 0) | (tr > 0 ? 4 : 0) | (br > 0 ? 2 : 0) | (bl > 0 ? 1 : 0);
      if (mask === 0 || mask === 15) continue;

      const top = [c + cross(tl, tr), r], right = [c + 1, r + cross(tr, br)];
      const bottom = [c + cross(bl, br), r + 1], left = [c, r + cross(tl, bl)];
      const pairs = {
        1: [left, bottom], 2: [bottom, right], 3: [left, right], 4: [top, right],
        5: [left, top, bottom, right], 6: [top, bottom], 7: [left, top], 8: [left, top],
        9: [top, bottom], 10: [left, bottom, top, right], 11: [top, right], 12: [left, right],
        13: [bottom, right], 14: [left, bottom],
      }[mask];

      for (let k = 0; k < pairs.length; k += 2) {
        out.push([px(pairs[k][0]), py(pairs[k][1]), px(pairs[k + 1][0]), py(pairs[k + 1][1])]);
      }
    }
  }
  return out;
}

/** Every cut between classes: class k against the rest, for all but the last class. */
export function fieldCuts(field, confidence, n, classes, size) {
  const segs = [];
  for (let k = 0; k < Math.max(1, classes - 1); k++) {
    segs.push(...contour((i) => (field[i] === k ? 1 : -1) * Math.max(confidence[i] / 255, 0.02), n, size));
  }
  return segs;
}

export const planeCut = (plane, n, act, size) => {
  const [, , at] = planeRange(plane, act);
  return contour((i) => plane[i] - at, n, size);
};

/** The cut in three passes, the way the cover draws it: a wide faint bloom, a body, a core. */
export function strokeCut(ctx, segs, scale = 1) {
  const path = new Path2D();
  for (const [x0, y0, x1, y1] of segs) { path.moveTo(x0, y0); path.lineTo(x1, y1); }
  ctx.lineCap = "round";
  for (const [colour, width] of [["rgba(125, 255, 190, 0.055)", 15],
                                 ["rgba(125, 255, 190, 0.13)", 7],
                                 ["rgba(200, 255, 226, 0.85)", 1.7]]) {
    ctx.strokeStyle = colour; ctx.lineWidth = width * scale; ctx.stroke(path);
  }
}

/** One mark in its class's shape. */
export function shapePath(ctx, shape, x, y, r) {
  ctx.beginPath();
  if (shape === "disc") ctx.arc(x, y, r, 0, Math.PI * 2);
  else if (shape === "square") { const s = r * 0.9; ctx.rect(x - s, y - s, s * 2, s * 2); }
  else if (shape === "triangle") {
    const s = r * 1.25;
    ctx.moveTo(x, y - s); ctx.lineTo(x + s * 0.94, y + s * 0.62); ctx.lineTo(x - s * 0.94, y + s * 0.62);
    ctx.closePath();
  } else { const s = r * 1.25; ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); }
}

/**
 * The observations. Training points are filled; held-out points are only an outline, because
 * the network never sees them and the drawing should not pretend it does.
 */
export function paintPoints(ctx, points, size, r, held = -1, only = null) {
  points.forEach((p, i) => {
    // with one set picked out (train or test), the other is left as a faint trace
    ctx.globalAlpha = only && (only === "test") !== !!p.test ? 0.12 : 1;
    const x = ((p.x + 1) / 2) * size, y = size - ((p.y + 1) / 2) * size;
    const ink = MARK[p.c];
    const shape = CLASSES[p.c].shape;
    const grabbed = i === held;
    const rr = r * (grabbed ? 1.3 : 1);

    ctx.beginPath(); ctx.arc(x, y, rr * (grabbed ? 4.2 : 2.6), 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${ink}, ${grabbed ? 0.2 : 0.11})`; ctx.fill();

    shapePath(ctx, shape, x, y, rr);
    if (p.test) {
      ctx.fillStyle = "rgba(6, 13, 9, 0.9)"; ctx.fill();
      ctx.strokeStyle = `rgba(${ink}, 0.95)`; ctx.lineWidth = 1.6; ctx.stroke();
    } else {
      ctx.fillStyle = `rgba(${ink}, 0.95)`; ctx.fill();
    }
  });
  ctx.globalAlpha = 1;
}

/** Brackets at the corners, the way an instrument marks the channel it is tracking. */
export function bracket(ctx, x, y, b, ink, arm = 6, width = 1.6) {
  ctx.strokeStyle = ink; ctx.lineWidth = width;
  for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    ctx.beginPath();
    ctx.moveTo(x + sx * b, y + sy * b - sy * arm);
    ctx.lineTo(x + sx * b, y + sy * b);
    ctx.lineTo(x + sx * b - sx * arm, y + sy * b);
    ctx.stroke();
  }
}

export const markInk = (c, alpha = 1) => `rgba(${MARK[c]}, ${alpha})`;
