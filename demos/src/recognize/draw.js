/*
 * How recognize paints numbers. Two readings, the same ones the playground uses:
 *
 *   lit     — an activation, as bright as it fires: the bare tube at zero, the phosphor at the top
 *   signed  — a weight, a product or a pre-activation: phosphor above zero, a cold low blue below,
 *             the bare tube at zero, on a scale symmetric around it
 *
 * A stage is drawn as a grid of maps (channels), each map a raster of its values, so the shape of
 * the tensor is the shape of the picture.
 */

const GROUND = [6, 13, 9];
const INK = [148, 255, 201];
const COLD = [70, 135, 230];

function rgb(value, mode, top) {
  if (mode === "lit") {
    const a = Math.max(0, Math.min(1, value / (top || 1)));
    return GROUND.map((g, i) => g + (INK[i] - g) * a * 0.9);
  }
  const t = Math.max(-1, Math.min(1, value / (top || 1)));
  const c = t >= 0 ? INK : COLD;
  return GROUND.map((g, i) => g + (c[i] - g) * Math.abs(t) * 0.85);
}

/**
 * The largest |value|, so a whole stage shares one scale and its maps can be compared. `floor` keeps
 * the scale from shrinking below a real size: with nothing drawn, a stage holds only its biases,
 * and a bias of 0.001 must not light up as if the unit were firing.
 */
export function scaleOf(values, mode, floor = 1e-9) {
  let top = floor;
  for (let i = 0; i < values.length; i++) top = Math.max(top, mode === "lit" ? values[i] : Math.abs(values[i]));
  return top;
}

/** One h x w raster of `values` (row-major), drawn into a canvas at x, y, w*cell by h*cell. */
export function paintRaster(ctx, values, h, w, x, y, cell, mode, top) {
  const buf = document.createElement("canvas");
  buf.width = w; buf.height = h;
  const bctx = buf.getContext("2d");
  const img = bctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    const c = rgb(values[i], mode, top);
    img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
  }
  bctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(buf, x, y, w * cell, h * cell);
}

/**
 * Where each map of a (C, H, W) stage goes in a grid of `cols` columns, with `gap` between maps
 * and half a gap around the outside, where the ruling goes. Returned so drawing and pointing use
 * the same arithmetic.
 *
 * Everything is snapped to whole device pixels first (`dev`), and the CSS sizes are those divided
 * by `dpr` exactly. A canvas whose CSS size is not a whole number of device pixels is resampled by
 * the browser, and with nearest-neighbour sampling that drops whole rows and columns — a ruling
 * line here, a line there, different at every window size and on every screen scale.
 */
export function gridLayout(C, H, W, cols, cell, gap, dpr = 1) {
  const rows = Math.ceil(C / cols);
  const cd = Math.max(1, Math.floor(cell * dpr));        // device pixels per value
  const gd = Math.max(3, Math.round(gap * dpr));         // device pixels between maps
  const half = Math.floor(gd / 2);
  const px = W * cd + gd, py = H * cd + gd;
  const devMaps = Array.from({ length: C }, (_, c) => ({ x: half + (c % cols) * px, y: half + Math.floor(c / cols) * py }));
  const dev = { maps: devMaps, cell: cd, gap: gd, half, px, py, width: cols * px, height: rows * py };
  return {
    dev, dpr, cols, rows, H, W,
    maps: devMaps.map((m) => ({ x: m.x / dpr, y: m.y / dpr })),
    cell: cd / dpr, width: dev.width / dpr, height: dev.height / dpr,
  };
}

/**
 * Draw a whole (C, H, W) stage into `node` with the layout above, its maps ruled apart, all in
 * device pixels. Returns the context set back to CSS pixels, for the marks drawn over it.
 */
export function paintStage(node, values, C, H, W, layout, mode, floor) {
  if (!node) return;
  const { dev, dpr } = layout;
  if (node.width !== dev.width || node.height !== dev.height) { node.width = dev.width; node.height = dev.height; }
  const ctx = node.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, dev.width, dev.height);
  const top = scaleOf(values, mode, floor);
  for (let c = 0; c < C; c++) {
    const { x, y } = dev.maps[c];
    paintRaster(ctx, values.subarray(c * H * W, (c + 1) * H * W), H, W, x, y, dev.cell, mode, top);
  }
  // the ruling: one line down the middle of every gap, and round the outside, a whole number of
  // device pixels wide. On a fractional screen scale the browser can still land the canvas half a
  // pixel off and resample it, dropping at most one row or column; a line two pixels wide always
  // keeps one of them
  const lw = Number.isInteger(dpr) ? dpr : Math.max(2, Math.round(dpr));
  const place = (centre, size) => Math.min(size - lw, Math.max(0, Math.round(centre - lw / 2)));
  ctx.fillStyle = "rgba(148, 255, 201, 0.28)";
  for (let k = 0; k <= layout.cols; k++) ctx.fillRect(place(k * dev.px + dev.half - dev.gap / 2, dev.width), 0, lw, dev.height);
  for (let k = 0; k <= layout.rows; k++) ctx.fillRect(0, place(k * dev.py + dev.half - dev.gap / 2, dev.height), dev.width, lw);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** Brackets a box [r0, r1, c0, c1] (inclusive, in values) of map `c`. */
export function markBox(ctx, layout, c, box, ink = "#c9ff4a") {
  const { x, y } = layout.maps[c];
  const cell = layout.cell;
  const bx = x + box[2] * cell, by = y + box[0] * cell;
  const bw = (box[3] - box[2] + 1) * cell, bh = (box[1] - box[0] + 1) * cell;
  ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
  ctx.strokeRect(bx - 1, by - 1, bw + 2, bh + 2);
}

/** Which value of a stage the pointer is on: { c, i, j, index } or null. */
export function pickCell(layout, C, mx, my) {
  for (let c = 0; c < C; c++) {
    const { x, y } = layout.maps[c];
    const j = Math.floor((mx - x) / layout.cell), i = Math.floor((my - y) / layout.cell);
    if (i >= 0 && i < layout.H && j >= 0 && j < layout.W) return { c, i, j, index: c * layout.H * layout.W + i * layout.W + j };
  }
  return null;
}

/** A tensor's shape as a picture: vectors are folded into the nearest readable rectangle. */
export function asMaps(shape) {
  if (shape.length === 3) return { C: shape[0], H: shape[1], W: shape[2] };
  const n = shape[0];
  if (n === 784) return { C: 1, H: 28, W: 28 };
  if (n === 256) return { C: 1, H: 16, W: 16 };
  if (n === 128) return { C: 1, H: 8, W: 16 };
  return { C: 1, H: n, W: 1 };
}

/** A number to `d` places, with a real minus sign, and no sign at all on a value that rounds to 0. */
export const fmt = (v, d = 2) => {
  const s = Math.abs(v).toFixed(d);
  return (v < 0 && Number(s) !== 0 ? "−" : "") + s;
};
