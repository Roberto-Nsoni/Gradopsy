import { useEffect, useRef, useState } from "react";

import { runOpen, runAdvance, runClose } from "../engine";

/*
 * A run, trained in slices inside animation frames.
 *
 * The batch calls this replaces froze the tab for one and a half to three and a half seconds
 * every time a chip was clicked or a point was added. Nothing about the mathematics wanted
 * that: the trainer was steppable from the first commit, and only the loop sat on the wrong
 * side of the bridge.
 *
 * What comes back is the same shape the plates already read — frames, keys, field, loss — it
 * simply grows. So the scrubber fills as the run is computed and the boundary forms in front
 * of you, which is the thing the page is about rather than a spinner over the top of it.
 */

const BUDGET_MS = 12;              // what one slice may spend, leaving the frame its own time
const DRAW_EVERY_MS = 50;          // how often the growing run is handed to React to redraw

let counter = 0;

/** Append into a growing typed array, doubling the backing store rather than copying per chunk. */
function grow(box, chunk, Kind) {
  if (!chunk?.length) return box;
  if (!box) return { data: chunk.slice(), length: chunk.length };

  let { data, length } = box;
  if (length + chunk.length > data.length) {
    const bigger = new Kind(Math.max(data.length * 2, length + chunk.length));
    bigger.set(data.subarray(0, length));
    data = bigger;
  }
  data.set(chunk, length);

  return { data, length: length + chunk.length };
}

/**
 * `spec` is the run to compute, or null for none. It is compared by value, so a component may
 * rebuild it every render; changing any field starts a new run and abandons the old one.
 */
export default function useRun(kind, spec) {
  const [run, setRun] = useState(null);
  const [progress, setProgress] = useState(0);
  const signature = JSON.stringify(spec);
  const boxes = useRef(null);

  useEffect(() => {
    if (!spec) return;

    const key = `run${counter++}`;
    let raf = 0;
    let live = true;
    let meta = null;

    try {
      meta = runOpen(key, kind, spec);
    } catch (e) {
      console.error("could not open a run", e);
      return;
    }

    const acc = { field: null, confidence: null, curve: null, keys: [], loss: [],
                  accuracy: [], train_loss: [], test_loss: [], runs: {} };
    boxes.current = acc;

    let painted = 0;

    const pump = () => {
      if (!live) return;

      let chunk;
      try {
        chunk = runAdvance(key, BUDGET_MS);
      } catch (e) {
        console.error("run stopped", e);
        return;
      }

      acc.field = grow(acc.field, chunk.field, Uint8Array);
      acc.confidence = grow(acc.confidence, chunk.confidence, Uint8Array);
      acc.curve = grow(acc.curve, chunk.curve, Float32Array);
      for (const list of ["keys", "loss", "accuracy", "train_loss", "test_loss"]) {
        if (chunk[list]?.length) acc[list].push(...chunk[list]);
      }
      if (chunk.runs) Object.assign(acc.runs, chunk.runs);   // descent arrives a runner at a time

      setProgress(chunk.done ? 1 : Math.min(0.99, chunk.k / meta.total));

      const nothingYet = !acc.keys.length && !Object.keys(acc.runs).length;
      // Slicing costs wall time — only part of each frame is work — so the frame is not also
      // spent re-rendering. Twenty repaints a second is more than anyone can read a boundary at.
      const now = performance.now();
      if (!chunk.done && (nothingYet || now - painted < DRAW_EVERY_MS)) {
        raf = requestAnimationFrame(pump);
        return;
      }
      painted = now;

      setRun({
        ...meta,
        frames: acc.keys.length,
        keys: acc.keys,
        runs: acc.runs,
        field: acc.field?.data,
        confidence: acc.confidence?.data,
        curve: acc.curve?.data,
        loss: acc.loss,
        accuracy: acc.accuracy,
        train_loss: acc.train_loss,
        test_loss: acc.test_loss,
        // the loss surface can only pick its plane once the whole trajectory exists
        surface: chunk.surface ?? undefined,
        surface_n: chunk.surface_n,
        span: chunk.span,
        path: chunk.path,
        training: !chunk.done,
      });

      if (!chunk.done) raf = requestAnimationFrame(pump);
      else runClose(key);
    };

    raf = requestAnimationFrame(pump);

    return () => {
      live = false;
      cancelAnimationFrame(raf);
      setProgress(0);
      try { runClose(key); } catch { /* the session outlived the engine; nothing to release */ }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, signature]);

  return [run, progress];
}
