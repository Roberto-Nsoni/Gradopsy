import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

import { paintRaster } from "../looks/scope";

/*
 * The pad: where a digit is drawn, and the 28x28 pixels the network reads, which are the same
 * thing. The strokes are drawn on a surface ten times MNIST's frame and averaged down to 28x28, and
 * nothing else: no crop, no rescale, no centring, so a pixel never moves once it is drawn.
 *
 * That only works because the networks were trained on digits moved and resized at random
 * (scripts/train_mnist.py, augment), not only on MNIST's own, which were all fitted into a 20x20
 * box at the centre. The dashed box shows that size: a digit about that big is the one they know
 * best, but it can sit anywhere.
 */

const INK = 280;            // the drawing surface, ten times MNIST's frame
const SIZE = 560;           // the pad's backing store, stretched by CSS
const PEN = 21;             // stroke width on the 280 surface: about two MNIST pixels

/** Ink on the 280 surface -> the 784 values the network reads (0..1), or null when empty. */
function prepare(surface) {
  // two steps down, so each pixel is the average of the ink under it rather than one sample of it
  const mid = document.createElement("canvas");
  mid.width = mid.height = 112;
  const mctx = mid.getContext("2d");
  mctx.imageSmoothingEnabled = true; mctx.imageSmoothingQuality = "high";
  mctx.drawImage(surface, 0, 0, 112, 112);
  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.imageSmoothingEnabled = true; sctx.imageSmoothingQuality = "high";
  sctx.drawImage(mid, 0, 0, 28, 28);
  const px = sctx.getImageData(0, 0, 28, 28).data;

  const out = new Float32Array(784);
  let any = false;
  for (let i = 0; i < 784; i++) {
    out[i] = px[i * 4] / 255;
    if (out[i] > 0) any = true;
  }
  return any ? out : null;
}

const Pad = forwardRef(function Pad({ field, grad, onInput, onSettle }, ref) {
  const canvas = useRef(null);
  const surface = useRef(null);
  const strokes = useRef([]);         // every stroke as its points, for undo
  const pen = useRef({ down: false, last: null });
  const drawn = useRef(null);         // the 784 values, as last drawn

  if (!surface.current && typeof document !== "undefined") {
    surface.current = document.createElement("canvas");
    surface.current.width = surface.current.height = INK;
  }

  const redrawSurface = () => {
    const ctx = surface.current.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, INK, INK);
    ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff"; ctx.lineWidth = PEN; ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const s of strokes.current) {
      ctx.beginPath();
      s.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (s.length === 1) { ctx.arc(s[0][0], s[0][1], PEN / 2, 0, Math.PI * 2); ctx.fill(); }
      else ctx.stroke();
    }
  };

  const commit = () => {
    redrawSurface();
    drawn.current = prepare(surface.current);
    onInput(drawn.current);
  };

  useImperativeHandle(ref, () => ({
    clear() { strokes.current = []; commit(); },
    undo() { strokes.current.pop(); commit(); },
    redraw() { commit(); },
  }));

  // the screen
  useEffect(() => {
    const node = canvas.current;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const px = Math.round(SIZE * dpr);
    if (node.width !== px) node.width = node.height = px;
    const ctx = node.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#060d09"; ctx.fillRect(0, 0, SIZE, SIZE);

    // the drawing at 28x28, one cell per pixel, as bright as the pixel is
    const cell = SIZE / 28;
    if (drawn.current) for (let i = 0; i < 784; i++) {
      const v = drawn.current[i];
      if (v <= 0.01) continue;
      ctx.fillStyle = `rgba(148, 255, 201, ${0.12 + v * 0.85})`;
      ctx.fillRect((i % 28) * cell, Math.floor(i / 28) * cell, cell - 1, cell - 1);
    }

    // the 20x20 box MNIST's own digits were fitted into: the size the networks know best
    ctx.strokeStyle = "rgba(148, 255, 201, 0.22)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.strokeRect(4 * cell, 4 * cell, 20 * cell, 20 * cell); ctx.setLineDash([]);

    // saliency: which pixels push the pinned value up (phosphor) or down (cold blue)
    if (grad) {
      let m = 1e-9;
      for (let i = 0; i < 784; i++) m = Math.max(m, Math.abs(grad[i]));
      // only the pixels that matter: under a fifth of the strongest is left unpainted, or the
      // small, nowhere-zero gradient of an MLP paints the whole screen as noise
      for (let i = 0; i < 784; i++) {
        const t = grad[i] / m;
        if (Math.abs(t) < 0.2) continue;
        const a = (Math.abs(t) - 0.2) / 0.8;
        ctx.fillStyle = t > 0 ? `rgba(201, 255, 74, ${0.25 + a * 0.6})` : `rgba(90, 160, 255, ${0.25 + a * 0.6})`;
        ctx.fillRect((i % 28) * cell + 2, Math.floor(i / 28) * cell + 2, cell - 4, cell - 4);
      }
    }

    // the pixels the value being read can see at all
    if (field) {
      const [r0, r1, c0, c1] = field;
      ctx.fillStyle = "rgba(201, 255, 74, 0.08)";
      ctx.fillRect(c0 * cell, r0 * cell, (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell);
      ctx.strokeStyle = "#c9ff4a"; ctx.lineWidth = 2;
      ctx.strokeRect(c0 * cell + 1, r0 * cell + 1, (c1 - c0 + 1) * cell - 2, (r1 - r0 + 1) * cell - 2);
    }

    paintRaster(ctx, SIZE);
  });

  const at = (e) => {
    const b = canvas.current.getBoundingClientRect();
    return [((e.clientX - b.left) / b.width) * INK, ((e.clientY - b.top) / b.height) * INK];
  };

  const down = (e) => {
    if (e.button > 0) return;
    e.preventDefault();
    canvas.current.setPointerCapture(e.pointerId);
    pen.current.down = true;
    strokes.current.push([at(e)]);
    commit();
  };
  const move = (e) => {
    if (!pen.current.down) return;
    strokes.current[strokes.current.length - 1].push(at(e));
    commit();
  };
  const up = () => { if (pen.current.down) onSettle?.(drawn.current); pen.current.down = false; };

  return (
    <canvas ref={canvas} className="rc-pad-canvas" style={{ touchAction: "none" }}
            onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
            aria-label="Draw a digit here with the pointer." />
  );
});

export default Pad;
