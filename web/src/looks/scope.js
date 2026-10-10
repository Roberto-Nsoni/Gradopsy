/*
 * The phosphor screen.
 *
 * The world is settled, so this file is only about how a signal looks on a cathode ray tube:
 * the raster and its scanlines, a ruled graticule, and a trace that glows and then does not
 * stop glowing for a while.
 *
 * The persistence is the point. Painting the ground fresh every frame and stamping the cut on
 * top made the network look instantaneous — the space around a dragged observation just
 * repainted. A phosphor that decays over a few frames shows the cut that was there a moment ago
 * while the new one arrives, which is the network visibly chasing rather than simply arriving.
 */

import { boundarySegments, sampler } from "../world/lobes";

const hex = (v) => [parseInt(v.slice(1, 3), 16), parseInt(v.slice(3, 5), 16), parseInt(v.slice(5, 7), 16)];
const mix = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);

/*
 * A phosphor whitens as it is driven harder, which is why the palette runs from the graticule at
 * 13% up through the lettering to the trace. `ink` is the lettering's drive, a step above the
 * screen's own light and a long way below the cut's.
 */
export const SCOPE = {
  ground: "#060d09",
  ink: "#94ffc9",
  quiet: "#2f6a4d",
  accent: "#c9ff4a",
  classInk: ["201, 255, 74", "90, 200, 255"],
};

/** The lit surface: the classified plane as a raster, its scanlines, and the graticule. */
export function paintGround(ctx, run, size) {
  const bg = hex(SCOPE.ground), green = hex("#39ff9e"), blue = hex("#2896ff");
  const sample = sampler(run.field, run.confidence, run.n);
  const image = ctx.createImageData(run.n, run.n);

  for (let row = 0; row < run.n; row++) {
    for (let col = 0; col < run.n; col++) {
      const { cls, sure } = sample(col, row);
      // How hard the field is driven. Pulled back from 0.40 and 0.31: the classified ground is
      // what the lettering has to be read against, and a sixth off it is worth more than any
      // brightening of the type — 1.5 points of contrast against 0.2.
      const c = sure < 0.04 ? bg
        : mix(bg, cls === 0 ? green : blue, cls === 0 ? 0.05 + sure * 0.29 : 0.04 + sure * 0.23);
      const p = (row * run.n + col) * 4;
      image.data[p] = c[0]; image.data[p + 1] = c[1]; image.data[p + 2] = c[2]; image.data[p + 3] = 255;
    }
  }

  const buf = document.createElement("canvas");
  buf.width = buf.height = run.n;
  buf.getContext("2d").putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(buf, 0, 0, size, size);

  paintRaster(ctx, size);
}

/**
 * The tube itself: the scanlines and the ruling, which belong to the glass and not to the signal.
 *
 * Kept separate because they are drawn twice — once under every frame of a live run, and once on
 * their own while the engine is still loading, where they are the whole picture.
 */
export function paintRaster(ctx, size) {
  // full-width bars of ground carved back out. This used to be drawn as a quarter of a million
  // little rectangles a frame, because the loop walked x — and a scanline does not.
  ctx.fillStyle = SCOPE.ground;
  for (let y = 0; y < size; y += 3) ctx.fillRect(0, y + 1.6, size, 1.4);

  ctx.strokeStyle = "rgba(125, 255, 190, 0.13)"; ctx.lineWidth = 1;
  for (let i = 1; i < 10; i++) {
    const v = (i / 10) * size;
    ctx.beginPath(); ctx.moveTo(v, 0); ctx.lineTo(v, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(size, v); ctx.stroke();
  }
}

/** An unlit tube: ruled and scanned, with nothing on it yet. */
export function paintTube(ctx, size) {
  ctx.fillStyle = SCOPE.ground;
  ctx.fillRect(0, 0, size, size);
  paintRaster(ctx, size);
}

/** The custom properties the whole world is drawn through, so nothing redeclares them. */
export const VARS = {
  "--ground": SCOPE.ground, "--ink": SCOPE.ink,
  "--quiet": SCOPE.quiet, "--accent": SCOPE.accent,
  "--prose": "#a9c9bb",
  // A separator and the edge of a control are not the same power. The hairline only has to be
  // seen; an edge has to be seen *against the field*, and at 28% it measures 1.86:1 there.
  "--hairline": "rgba(148, 255, 201, 0.28)",
  "--edge": "rgba(148, 255, 201, 0.6)",
};

/** The trace and the observations, drawn onto a layer that is faded rather than cleared. */
export function paintTrace(ctx, run, size, pts, labels, held) {
  const segs = boundarySegments(run.field, run.confidence, run.n, size);
  const trace = new Path2D();
  for (const [x0, y0, x1, y1] of segs) { trace.moveTo(x0, y0); trace.lineTo(x1, y1); }

  ctx.lineCap = "round";
  for (const [colour, width] of [["rgba(125, 255, 190, 0.055)", 15],
                                 ["rgba(125, 255, 190, 0.13)", 7],
                                 ["rgba(200, 255, 226, 0.85)", 1.7]]) {
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.stroke(trace);
  }

  pts.forEach(([x, y], i) => {
    const px = ((x + 1) / 2) * size, py = size - ((y + 1) / 2) * size;
    const ink = SCOPE.classInk[labels[i]];
    const grabbed = held === i;
    const r = (size > 620 ? 7 : 5.5) * (grabbed ? 1.3 : 1);

    ctx.beginPath(); ctx.arc(px, py, r * (grabbed ? 4.2 : 2.6), 0, 7);
    ctx.fillStyle = `rgba(${ink}, ${grabbed ? 0.2 : 0.13})`; ctx.fill();
    ctx.beginPath(); ctx.arc(px, py, r, 0, 7);
    ctx.fillStyle = `rgba(${ink}, 0.95)`; ctx.fill();

    // a held reading is bracketed, the way an instrument marks the channel it is tracking
    if (grabbed) {
      const b = r + 11;
      ctx.strokeStyle = `rgba(${ink}, 0.85)`; ctx.lineWidth = 1.6;
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        ctx.beginPath();
        ctx.moveTo(px + sx * b, py + sy * b - sy * 6);
        ctx.lineTo(px + sx * b, py + sy * b);
        ctx.lineTo(px + sx * b - sx * 6, py + sy * b);
        ctx.stroke();
      }
    }
  });
}
