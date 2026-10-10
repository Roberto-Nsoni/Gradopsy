import { useEffect, useRef } from "react";

import { VARS, paintTube } from "./scope";
import "./looks.css";

/*
 * The wait, which is the first thing every visitor sees.
 *
 * Fifteen megabytes of Python have to arrive before a single number can be computed, and the
 * sceptical visitor — the one this whole site is for — spends those seconds with nothing to go
 * on. So the wait is not a spinner over a blank page: it is the tube, ruled and scanned, with
 * no signal on it yet. The instrument is already there; only the mathematics is missing.
 *
 * Nothing on this screen animates. The world has no CSS animation in it, and a loading dance
 * would be the one moving thing here that is not something really happening — the progress line
 * changes because the engine actually reached the next stage, and that is the only motion.
 *
 * The composition is the cover's — same column, same wordmark, same sizes — but centred rather
 * than hung off the right edge. On the cover the type stands aside so the drawing can have the
 * screen; here there is no drawing yet to stand aside from, and a column pushed to one edge of
 * an empty tube reads as something missing rather than something waiting.
 */
export default function Waiting({ progress, failed }) {
  const tube = useRef(null);

  useEffect(() => {
    const paint = () => {
      const node = tube.current;
      if (!node) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = Math.max(window.innerWidth, window.innerHeight);
      node.width = node.height = Math.round(size * dpr);
      node.style.width = node.style.height = `${size}px`;
      node.style.left = `${(window.innerWidth - size) / 2}px`;
      node.style.top = `${(window.innerHeight - size) / 2}px`;
      const ctx = node.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      paintTube(ctx, size);
    };
    paint();
    window.addEventListener("resize", paint);

    return () => window.removeEventListener("resize", paint);
  }, []);

  return (
    <div className="look" style={VARS}>
      <canvas ref={tube} className="look-canvas" aria-hidden="true" />

      <div className="look-column look-column-mid">
        <h1 className="look-column-mark">visualML</h1>
        <p className="look-boot" role="status">
          <i>{failed ? "failed" : "starting"}</i>
          {progress}
        </p>
        <p className="look-column-say">
          Python and NumPy are loading into this page. About 15&nbsp;MB, once.
        </p>
      </div>
    </div>
  );
}
