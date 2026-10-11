import { useEffect, useLayoutEffect, useRef, useState } from "react";

/*
 * The training run, as it was logged: the loss on the minibatches (solid, log scale) and the
 * accuracy on a fixed slice of the test set (dashed), against steps on a log axis, because almost
 * everything happens in the first few hundred. The ticks are the snapshots the page holds; the
 * head can stop on any of them, and the whole page is then read by the network as it was there.
 */

export default function RunLog({ log, snaps, at, onScrub, onLive }) {
  const canvas = useRef(null);
  const box = useRef(null);
  const [w, setW] = useState(600);
  const H = 64;

  useLayoutEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  const points = log?.log ?? [];
  const last = points.length ? points[points.length - 1].step : 1;
  // log axis on steps from step 1, which sits just inside the left edge; step 0 is the edge itself
  const xOf = (step) => (step <= 0 ? 2 : 6 + (Math.log10(step) / Math.log10(last)) * (w - 8));
  // the snapshot steps, as scripts/train_mnist.py chooses them, until the real list is loaded
  const ticks = snaps ?? [...new Set([0, ...Array.from({ length: 11 }, (_, i) => Math.floor(last ** (i / 10)))])];

  useEffect(() => {
    const node = canvas.current;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    node.width = Math.round(w * dpr); node.height = Math.round(H * dpr);
    const ctx = node.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    ctx.strokeStyle = "rgba(148, 255, 201, 0.13)"; ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) { const y = Math.round((i / 4) * H) + 0.5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    if (!points.length) return;

    const losses = points.map((p) => p.loss).filter((v) => v > 0);
    const lo = Math.log10(Math.min(...losses)), hi = Math.log10(Math.max(...losses));
    const yLoss = (v) => 4 + (1 - (Math.log10(v) - lo) / Math.max(hi - lo, 0.3)) * (H - 8);
    const yAcc = (v) => 4 + (1 - v) * (H - 8);

    ctx.strokeStyle = "rgba(148, 255, 201, 0.75)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath(); points.forEach((p, i) => (i ? ctx.lineTo(xOf(p.step), yAcc(p.acc)) : ctx.moveTo(xOf(p.step), yAcc(p.acc)))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(200, 255, 226, 0.95)";
    ctx.beginPath();
    let started = false;
    for (const p of points) { if (!(p.loss > 0)) continue; const x = xOf(p.step), y = yLoss(p.loss); started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true; }
    ctx.stroke();

    // the snapshots, and where the head is
    for (let i = 0; i < ticks.length; i++) {
      const x = Math.round(xOf(ticks[i])) + 0.5;
      ctx.strokeStyle = i === at ? "#c9ff4a" : "rgba(148, 255, 201, 0.35)"; ctx.lineWidth = i === at ? 2 : 1;
      ctx.beginPath(); ctx.moveTo(x, i === at ? 0 : H - 8); ctx.lineTo(x, H); ctx.stroke();
    }
    if (at < 0) {
      const x = Math.round(xOf(last)) - 0.5;
      ctx.strokeStyle = "rgba(200, 255, 226, 0.9)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
  });

  const pick = (e) => {
    const b = canvas.current.getBoundingClientRect();
    const x = e.clientX - b.left;
    let best = 0, d = Infinity;
    ticks.forEach((s, i) => { const dx = Math.abs(xOf(s) - x); if (dx < d) { d = dx; best = i; } });
    best === ticks.length - 1 ? onLive() : onScrub(best);
  };
  const key = (e) => {
    const n = ticks.length;
    const i = at < 0 ? n - 1 : at;
    if (e.key === "ArrowLeft" && i > 0) { e.preventDefault(); onScrub(i - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); i + 1 >= n - 1 ? onLive() : onScrub(i + 1); }
  };

  const step = at < 0 ? last : ticks[at];

  return (
    <div className="pg-time" ref={box}>
      <canvas ref={canvas} className="pg-time-canvas" style={{ width: "100%", height: H }}
              role="slider" tabIndex={0} aria-label="Training run: pick a snapshot to read the digit with the network as it was"
              aria-valuemin={0} aria-valuemax={last} aria-valuenow={step} aria-valuetext={`step ${step}${at < 0 ? ", trained" : ""}`}
              onPointerDown={pick} onKeyDown={key} />
      <div className="pg-time-under">
        <div className="pg-time-legend" aria-hidden="true">
          <span><svg width="18" height="6"><path d="M0 3h18" stroke="#c8ffe2" strokeWidth="1.5" /></svg>train loss</span>
          <span><svg width="18" height="6"><path d="M0 3h18" stroke="#94ffc9" strokeWidth="1.5" strokeDasharray="4 3" /></svg>accuracy</span>
        </div>
        <p className="pg-stopped">{at < 0 ? `trained · ${last} steps` : `step ${step} of ${last}`}</p>
      </div>
    </div>
  );
}
