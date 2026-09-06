// The drawing vocabulary. Everything the demos paint goes through here, so a plot is built
// from the page's own materials rather than dropped in as a foreign canvas.

export const INK = {
  limewash: "#f2efea",
  limewashDeep: "#e7e2d9",
  poche: "#1c1f22",
  pocheLift: "#2a2e33",
  sunbeam: "#d6a43a",
  sunbeamDeep: "#b8862a",
  sunbeamText: "#8a6210",     // 4.77:1 on limewash; the fill golds are not legible as type
  shadowBlue: "#5b6b7c",
  stone: "#a8a39a",
  stoneDeep: "#6f6a61",
  rule: "#c8c2b6",
  ruleStrong: "#8b8578",      // 3.20:1: what a control edge needs, where a separator does not
};

// Classes are the section's own materials. The ramp alternates dark and light so any two
// neighbouring classes differ by value and not only by hue, and the drawing survives being read
// without colour. The sunbeam is never one of them: it belongs to the present moment.
export const MATERIALS = [INK.poche, "#b9c3cb", "#5b6b7c", "#e2ddd2", "#8b8578"];

export const material = (i) => MATERIALS[i % MATERIALS.length];

/**
 * Canvas sized for the device, drawn in CSS pixels. Returns the 2D context.
 *
 * Assigning `width` reallocates the backing store, so it is only done when the size actually
 * changed: the live section calls this sixty times a second and would otherwise spend a good
 * part of its frame budget in the allocator.
 */
export function fitCanvas(canvas, width, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(width * dpr);
  const h = Math.round(height * dpr);

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  return ctx;
}

/** One scratch canvas for every raster that gets scaled up to display size. */
const scratch = { canvas: null, n: 0 };

function scratchAt(n) {
  if (!scratch.canvas) scratch.canvas = document.createElement("canvas");
  if (scratch.n !== n) {
    scratch.canvas.width = scratch.canvas.height = n;
    scratch.n = n;
  }

  return scratch.canvas;
}

const hex = (value) => [
  parseInt(value.slice(1, 3), 16),
  parseInt(value.slice(3, 5), 16),
  parseInt(value.slice(5, 7), 16),
];

/*
 * A note on which way up a grid is, because getting it wrong is silent.
 *
 * `make_grid` builds its points with `meshgrid(xs, ys)`, so row 0 of every field this file is
 * handed is y = -1 — the BOTTOM of the plane. A canvas draws row 0 at the top. Every raster and
 * every contour therefore has to be turned over on the way out, exactly as `paintTerrain` has
 * always done, or the drawing is mirrored about the horizontal axis while the observations on
 * top of it are not.
 *
 * This was live for a while and it hid well: the observations are drawn from their own
 * coordinates and were always right, the cut lines are derived from the same rows as the field
 * and so were wrong in step with it, and the accuracy readout is computed in Python and kept
 * saying 100%. Only `circles` — near enough symmetric about that axis — looked correct, which
 * is exactly the dataset a quick glance lands on.
 */

/** Draw a grid raster the right way up: row 0 of the field is the bottom of the plane. */
function drawUpright(ctx, buffer, size) {
  ctx.imageSmoothingEnabled = true;
  ctx.save();
  ctx.scale(1, -1);
  ctx.drawImage(buffer, 0, -size, size, size);
  ctx.restore();
}

/**
 * The classified plane as a cut section: real material where the model is sure, washing back to
 * the limewash ground where it is not.
 */
export function paintField(ctx, { field, confidence, n, offset }, size, classes) {
  const image = ctx.createImageData(n, n);
  const ground = hex(INK.limewash);
  const inks = Array.from({ length: classes }, (_, i) => hex(material(i)));

  for (let i = 0; i < n * n; i++) {
    const ink = inks[field[offset + i]] ?? ground;

    // 1/classes is chance level, 1 is certain. Below chance-plus-a-bit the field is bare paper;
    // where the model is sure the material arrives at full weight, because a section without
    // real mass is only a wash, and mass against limewash is this world's whole identity.
    const sure = Math.max(0, (confidence[offset + i] / 255 - 1 / classes) / (1 - 1 / classes));
    const weight = 0.1 + 0.9 * sure ** 0.75;

    const p = i * 4;
    for (let c = 0; c < 3; c++) image.data[p + c] = ground[c] + (ink[c] - ground[c]) * weight;
    image.data[p + 3] = 255;
  }

  const buffer = scratchAt(n);
  buffer.getContext("2d").putImageData(image, 0, 0);
  drawUpright(ctx, buffer, size);
}

/**
 * Marching squares over any signed field, so a level line comes out as a real curve at display
 * resolution instead of a staircase of grid cells. Segments come back in cell coordinates.
 *
 * Both level lines in this world go through here: the class boundary, and the contours of a
 * loss surface. Drawing one of them properly and the other by lightening single raster pixels
 * was the reason the parameter-space plate read as a compression artefact next to a plate whose
 * cut line is vector.
 */
function isolines(at, n) {
  const cross = (a, b) => a / (a - b);
  const segments = [];

  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const tl = at(r, c), tr = at(r, c + 1), br = at(r + 1, c + 1), bl = at(r + 1, c);
      const mask = (tl > 0 ? 8 : 0) | (tr > 0 ? 4 : 0) | (br > 0 ? 2 : 0) | (bl > 0 ? 1 : 0);
      if (mask === 0 || mask === 15) continue;

      // Where the indicator crosses zero along each edge of the cell
      const top = [c + cross(tl, tr), r];
      const right = [c + 1, r + cross(tr, br)];
      const bottom = [c + cross(bl, br), r + 1];
      const left = [c, r + cross(tl, bl)];

      const pairs = {
        1: [left, bottom], 2: [bottom, right], 3: [left, right], 4: [top, right],
        5: [left, top, bottom, right], 6: [top, bottom], 7: [left, top],
        8: [left, top], 9: [top, bottom], 10: [left, bottom, top, right],
        11: [top, right], 12: [left, right], 13: [bottom, right], 14: [left, bottom],
      }[mask];

      for (let k = 0; k < pairs.length; k += 2) {
        segments.push([pairs[k][0], pairs[k][1], pairs[k + 1][0], pairs[k + 1][1]]);
      }
    }
  }

  return segments;
}

/** One class's own confidence-signed indicator, in unit coordinates. */
function contour(field, confidence, n, offset, cls) {
  const at = (r, c) => {
    const i = offset + r * n + c;
    return (field[i] === cls ? 1 : -1) * (confidence[i] / 255);
  };

  return isolines(at, n).map((seg) => seg.map((v) => v / (n - 1)));
}

/**
 * One moment's cut, as a reusable path. The live section holds on to these: the four cuts
 * behind the present one only change every fifth frame, and running marching squares over all
 * five of them sixty times a second spends the frame budget on work already done.
 */
export function cutPath(field, confidence, n, offset, classes, size) {
  const path = new Path2D();

  for (let cls = 0; cls < classes; cls++) {
    for (const [x0, y0, x1, y1] of contour(field, confidence, n, offset, cls)) {
      path.moveTo(x0 * size, size - y0 * size);
      path.lineTo(x1 * size, size - y1 * size);
    }
  }

  return path;
}

/** How a cut is inked: mass under a thread of light, or shadow if it has already passed. */
export function strokeCut(ctx, path, { colour, width, alpha = 1 }) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = colour;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke(path);
  ctx.restore();
}

export const CUT_NOW = [
  { colour: INK.poche, width: 2.6, alpha: 0.92 },
  { colour: INK.sunbeam, width: 1.2, alpha: 1 },
];

export const cutPast = (back, trail) =>
  ({ colour: INK.shadowBlue, width: 1, alpha: 0.1 + 0.055 * (trail - back) });

/**
 * The cut lines, and the band of light.
 *
 * The boundary of the step you are looking at is drawn in sunbeam: it is the present moment, and
 * watching it sweep across the plane as training advances is the thesis of this world made
 * visible. A few earlier boundaries stay behind it in shadow blue, so the drawing carries where
 * it came from as well as where it is now.
 */
export function drawCuts(ctx, run, frameIndex, size, { trail = 4, spacing = 2 } = {}) {
  const cells = run.n * run.n;
  const now = Math.min(frameIndex, run.frames - 1);
  const at = (i) => cutPath(run.field, run.confidence, run.n, i * cells, run.classes, size);

  for (let back = trail; back >= 1; back--) {          // what has already happened, receding
    const i = now - back * spacing;
    if (i >= 0) strokeCut(ctx, at(i), cutPast(back, trail));
  }

  const here = at(now);
  for (const ink of CUT_NOW) strokeCut(ctx, here, ink);
}

/** A field of one continuous value, for a single neuron's output. */
export function paintScalar(ctx, values, n, size, lo, hi) {
  const image = ctx.createImageData(n, n);
  const ground = hex(INK.limewash);
  const dark = hex(INK.poche);
  const warm = hex(INK.sunbeam);

  for (let i = 0; i < n * n; i++) {
    const t = (values[i] - lo) / (hi - lo || 1);
    const [from, to, k] = t < 0.5 ? [dark, ground, t * 2] : [ground, warm, (t - 0.5) * 2];

    const p = i * 4;
    for (let c = 0; c < 3; c++) image.data[p + c] = from[c] + (to[c] - from[c]) * k;
    image.data[p + 3] = 255;
  }

  const buffer = document.createElement("canvas");
  buffer.width = buffer.height = n;
  buffer.getContext("2d").putImageData(image, 0, 0);
  drawUpright(ctx, buffer, size);
}

const luminance = (colour) => {
  const [r, g, b] = hex(colour);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

/**
 * The observations, each on the limewash halo that keeps it off the material underneath.
 *
 * The outline takes the opposite value to the observation's own material, because a class's
 * points sit on that class's own ground: a poche dot ringed in poche, inside the poche region
 * the model is confident about, is an invisible dot. Value, not hue, is what separates a mark
 * from what it is lying on.
 */
export function drawPoints(ctx, points, labels, toPixel, { radius = 4.5 } = {}) {
  points.forEach(([x, y], i) => {
    const [px, py] = toPixel(x, y);
    const ink = material(labels[i]);

    ctx.beginPath();
    ctx.arc(px, py, radius + 1.8, 0, 2 * Math.PI);
    ctx.fillStyle = INK.limewash;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, radius, 0, 2 * Math.PI);
    ctx.fillStyle = ink;
    ctx.fill();
    ctx.strokeStyle = luminance(ink) < 120 ? INK.limewash : INK.poche;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  });
}

/** A hairline frame, the way a drawing is bounded. */
export function frame(ctx, width, height = width, colour = INK.ruleStrong) {
  ctx.strokeStyle = colour;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
}

/**
 * A loss surface as ground: deep valleys in poche, high land washing out to limewash, with the
 * quantised bands edge-detected into contour lines.
 *
 * Mapped by percentile rather than by value, because these surfaces run from a span of 11
 * (double well) to 10^6 (Goldstein-Price) and a linear ramp turns all but one of them into a
 * flat field with a dot in it.
 *
 * Both lenses that draw terrain — the parameter-space plate and the descent plate — come
 * through here, so a valley is the same depth of ink wherever it is drawn.
 */
/**
 * A loss surface as ground: deep valleys in poche, high land washing out to limewash, with the
 * contours of the quantised bands drawn over it as real curves.
 *
 * Mapped by percentile rather than by value, because these surfaces run from a span of 11
 * (double well) to 10^6 (Goldstein-Price) and a linear ramp turns all but one of them into a
 * flat field with a dot in it.
 *
 * The result is cached against the surface it was built from: the descent plate repaints on
 * every one of sixty frames a second, and rebuilding a 560x560 field each time is work nobody
 * asked for. What changes between frames is the paths, not the ground.
 */
const terrains = new WeakMap();

export function paintTerrain(ctx, z, n, size, { levels = 14 } = {}) {
  let byShape = terrains.get(z);
  if (!byShape) terrains.set(z, (byShape = new Map()));

  const key = `${size}:${levels}`;
  let ground = byShape.get(key);

  if (!ground) {
    ground = buildTerrain(z, n, size, levels);
    byShape.clear();                       // one size per surface is all any plate ever needs
    byShape.set(key, ground);
  }

  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(ground, 0, 0, size, size);
}

function buildTerrain(z, n, size, levels) {
  const sorted = Float32Array.from(z).sort();
  const last = sorted.length - 1;
  const rank = (v) => {
    let lo = 0, hi = last;
    while (lo < hi) { const m = (lo + hi) >> 1; sorted[m] < v ? (lo = m + 1) : (hi = m); }
    return lo / last;
  };

  const t = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) t[i] = rank(z[i]);

  const image = new ImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const k = t[i] ** 0.85;
    const p = i * 4;
    image.data[p] = 28 + k * 214;
    image.data[p + 1] = 31 + k * 208;
    image.data[p + 2] = 34 + k * 200;
    image.data[p + 3] = 255;
  }

  const buffer = document.createElement("canvas");
  buffer.width = buffer.height = n;
  buffer.getContext("2d").putImageData(image, 0, 0);

  const out = document.createElement("canvas");
  out.width = out.height = Math.max(1, Math.round(size));
  const ctx = out.getContext("2d");

  // The grid runs bottom-up in mathematics and top-down on a canvas
  ctx.imageSmoothingEnabled = true;
  ctx.save();
  ctx.scale(1, -1);
  ctx.drawImage(buffer, 0, -size, size, size);
  ctx.restore();

  // Contours, added rather than painted: `lighter` reproduces exactly the +26/24/18 the raster
  // used to add to an edge pixel, so a line stays visible on the dark floor of a valley and
  // stays discreet on the pale high ground, at any resolution.
  const px = (c) => (c / (n - 1)) * size;
  const py = (r) => size - (r / (n - 1)) * size;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = "rgb(26, 24, 18)";
  ctx.lineWidth = 1;
  ctx.beginPath();

  for (let level = 1; level < levels; level++) {
    const threshold = level / levels;
    for (const [x0, y0, x1, y1] of isolines((r, c) => t[r * n + c] - threshold, n)) {
      ctx.moveTo(px(x0), py(y0));
      ctx.lineTo(px(x1), py(y1));
    }
  }
  ctx.stroke();
  ctx.restore();

  return out;
}
