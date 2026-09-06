import { useCallback, useEffect, useRef, useState } from "react";
import { INK, fitCanvas, paintField, drawCuts, drawPoints, frame } from "../world/section";

/**
 * Width of whatever the returned ref is attached to. It must be a callback ref: the wait screen
 * renders first, so an effect with an empty dependency list would run while the node does not
 * exist and would silently never attach.
 */
export function useMeasure(max = Infinity) {
  const [width, setWidth] = useState(0);
  const observer = useRef(null);

  const ref = useCallback((node) => {
    observer.current?.disconnect();
    if (!node) return;

    observer.current = new ResizeObserver(([entry]) => {
      setWidth(Math.min(max, Math.round(entry.contentRect.width)));
    });
    observer.current.observe(node);
  }, [max]);

  return [ref, width];
}

/** The viewport's own height, for drawings that have to fit above the fold. */
export function useViewport() {
  const [height, setHeight] = useState(() => (typeof window === "undefined" ? 800 : window.innerHeight));

  useEffect(() => {
    const onResize = () => setHeight(window.innerHeight);
    window.addEventListener("resize", onResize);

    return () => window.removeEventListener("resize", onResize);
  }, []);

  return height;
}

/** Playback over recorded frames. The run is computed whole; this only walks it. */
export function usePlayback(total, { fps = 30 } = {}) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing || total < 2) return;

    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      if (now - last >= 1000 / fps) {
        last = now;
        setStep((s) => (s + 1 >= total ? (setPlaying(false), total - 1) : s + 1));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(raf);
  }, [playing, total, fps]);

  const restart = useCallback(() => { setStep(0); setPlaying(true); }, []);

  return { step, setStep, playing, setPlaying, restart };
}

const toPixel = (size) => (x, y) => [((x + 1) / 2) * size, size - ((y + 1) / 2) * size];

/**
 * The classified plane, drawn as a cut section: solid material where the model is sure, bare
 * limewash where it is not, with the observations laid on top.
 */
export function FieldCanvas({ run, frameIndex, points, labels, size, onPick, hint }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !size) return;
    const ctx = fitCanvas(ref.current, size, size);

    ctx.fillStyle = INK.limewash;
    ctx.fillRect(0, 0, size, size);

    if (run) {
      const cells = run.n * run.n;
      const i = Math.min(frameIndex, run.frames - 1);
      paintField(ctx, { field: run.field, confidence: run.confidence, n: run.n, offset: i * cells },
                 size, run.classes);
      drawCuts(ctx, run, i, size);
    }

    if (points?.length) drawPoints(ctx, points, labels, toPixel(size));
    frame(ctx, size);
  }, [run, frameIndex, points, labels, size]);

  const pick = (event) => {
    if (!onPick) return;
    const box = ref.current.getBoundingClientRect();
    onPick([
      ((event.clientX - box.left) / box.width) * 2 - 1,
      1 - ((event.clientY - box.top) / box.height) * 2,
    ]);
  };

  return (
    <div className="section-plate" style={{ width: size }}>
      <canvas ref={ref} onClick={pick} style={{ cursor: onPick ? "crosshair" : "default" }}
              aria-label="The plane, coloured by what the network predicts there." />
      {hint && <p className="section-hint overline">{hint}</p>}
    </div>
  );
}

/**
 * Two series against training step, drawn as the section's own line work. The head of each
 * line is the present moment, so it carries the sunbeam.
 */
export function Curves({ series, step, width, height = 170, logY = false }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current || !width) return;
    const ctx = fitCanvas(ref.current, width, height);

    const pad = { l: 52, r: 12, t: 12, b: 24 };
    const w = width - pad.l - pad.r;
    const h = height - pad.t - pad.b;
    if (w <= 0) return;

    const all = series.flatMap((s) => s.values.filter(Number.isFinite));
    if (!all.length) return;

    const scale = logY ? (v) => Math.log10(Math.max(v, 1e-6)) : (v) => v;
    const lo = scale(Math.min(...all));
    const hi = scale(Math.max(...all));
    const toY = (v) => pad.t + h - ((scale(v) - lo) / (hi - lo || 1)) * h;
    const toX = (i, n) => pad.l + (i / Math.max(n - 1, 1)) * w;

    ctx.strokeStyle = INK.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + h); ctx.lineTo(pad.l + w, pad.t + h);
    ctx.stroke();

    // Flush against the axis they belong to, not floating forty pixels off it
    ctx.fillStyle = INK.stoneDeep;
    ctx.font = "500 10px 'Martian Mono', ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(Math.max(...all).toPrecision(2), pad.l - 8, pad.t);
    ctx.fillText(Math.min(...all).toPrecision(2), pad.l - 8, pad.t + h);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    for (const { values, colour, dash } of series) {
      const upTo = Math.min(step, values.length - 1);
      ctx.beginPath();
      for (let i = 0; i <= upTo; i++) {
        const [px, py] = [toX(i, values.length), toY(values[i])];
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.setLineDash(dash ?? []);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.8;
      ctx.stroke();
      ctx.setLineDash([]);

      const [hx, hy] = [toX(upTo, values.length), toY(values[upTo])];
      ctx.beginPath();
      ctx.arc(hx, hy, 3.2, 0, 2 * Math.PI);
      ctx.fillStyle = INK.sunbeam;
      ctx.fill();
      ctx.strokeStyle = INK.poche;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [series, step, width, height, logY]);

  return <canvas ref={ref} className="curves" aria-label="Loss against training step." />;
}

export function Legend({ items }) {
  return (
    <ul className="legend">
      {items.map(({ label, colour, dash }) => (
        <li key={label}>
          <span className="legend-key" style={{ background: dash ? "transparent" : colour,
                borderTop: dash ? `2px dashed ${colour}` : "none" }} />
          <span className="overline">{label}</span>
        </li>
      ))}
    </ul>
  );
}
