import { useEffect, useLayoutEffect, useRef, useState } from "react";

/*
 * The run, as something you can walk back through. The two losses over every recorded epoch on
 * a log scale — training solid, held-out dashed, because the screen already uses filled and
 * hollow for the same distinction — and a head that is wherever the page is looking.
 *
 * Dragging the head is not a replay of pictures: every snapshot is a parameter vector, and the
 * engine re-renders the screen and every unit from it.
 */

export default function Timeline({ history, at, onScrub, onLive, children }) {
  const canvas = useRef(null);
  const box = useRef(null);
  const [w, setW] = useState(600);
  const dragging = useRef(false);
  const H = 64;

  useLayoutEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  const last = history.length ? history[history.length - 1].epoch : 0;
  const xOf = (epoch) => (last > 0 ? (epoch / last) * (w - 2) + 1 : 1);

  useEffect(() => {
    const node = canvas.current;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    node.width = Math.round(w * dpr); node.height = Math.round(H * dpr);
    const ctx = node.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);

    ctx.strokeStyle = "rgba(148, 255, 201, 0.13)"; ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) { const y = Math.round((i / 4) * H) + 0.5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    const values = history.flatMap((m) => [m.train_loss, m.test_loss]).filter((v) => v != null && v > 0);
    if (!values.length) return;
    const lo = Math.log10(Math.max(Math.min(...values), 1e-5)), hi = Math.log10(Math.max(...values));
    const span = Math.max(hi - lo, 0.5);
    const yOf = (v) => 4 + (1 - (Math.log10(Math.max(v, 1e-5)) - lo) / span) * (H - 8);

    for (const [key, style, dash] of [["test_loss", "rgba(148, 255, 201, 0.75)", [4, 3]],
                                      ["train_loss", "rgba(200, 255, 226, 0.95)", []]]) {
      ctx.strokeStyle = style; ctx.lineWidth = 1.5; ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (const m of history) {
        if (m[key] == null) { started = false; continue; }
        const x = xOf(m.epoch), y = yOf(m[key]);
        if (started) ctx.lineTo(x, y); else { ctx.moveTo(x, y); started = true; }
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const head = at == null ? last : history[at]?.epoch ?? last;
    const hx = Math.round(xOf(head)) + 0.5;
    ctx.strokeStyle = at == null ? "rgba(200, 255, 226, 0.9)" : "#c9ff4a";
    ctx.lineWidth = at == null ? 1 : 2;
    ctx.beginPath(); ctx.moveTo(hx, 0); ctx.lineTo(hx, H); ctx.stroke();
  }, [history, at, w, last]);

  const indexAt = (event) => {
    const b = canvas.current.getBoundingClientRect();
    const epoch = ((event.clientX - b.left) / b.width) * last;
    let best = 0, d = Infinity;
    history.forEach((m, i) => { const e = Math.abs(m.epoch - epoch); if (e < d) { d = e; best = i; } });
    return best;
  };

  const down = (event) => {
    if (history.length < 2) return;
    dragging.current = true;
    canvas.current.setPointerCapture(event.pointerId);
    const i = indexAt(event);
    i === history.length - 1 ? onLive() : onScrub(i);
  };
  const move = (event) => {
    if (!dragging.current) return;
    const i = indexAt(event);
    i === history.length - 1 ? onLive() : onScrub(i);
  };
  const up = () => { dragging.current = false; };

  const key = (event) => {
    if (history.length < 2) return;
    const i = at ?? history.length - 1;
    const step = event.shiftKey ? 10 : 1;
    if (event.key === "ArrowLeft") { event.preventDefault(); onScrub(Math.max(0, i - step)); }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      const j = i + step;
      j >= history.length - 1 ? onLive() : onScrub(j);
    }
    if (event.key === "End") { event.preventDefault(); onLive(); }
    if (event.key === "Home") { event.preventDefault(); onScrub(0); }
  };

  const shown = at == null ? last : history[at]?.epoch ?? 0;

  return (
    <div className="pg-time" ref={box}>
      <canvas
        ref={canvas} style={{ width: "100%", height: H }} className="pg-time-canvas"
        role="slider" tabIndex={0} aria-label="Training history: drag to rewind to any recorded epoch"
        aria-valuemin={0} aria-valuemax={last} aria-valuenow={shown}
        aria-valuetext={`epoch ${shown}${at == null ? ", live" : ""}`}
        onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onKeyDown={key}
      />
      <div className="pg-time-under">
        <div className="pg-time-legend" aria-hidden="true">
          <span><svg width="18" height="6"><path d="M0 3h18" stroke="#c8ffe2" strokeWidth="1.5" /></svg>train loss</span>
          <span><svg width="18" height="6"><path d="M0 3h18" stroke="#94ffc9" strokeWidth="1.5" strokeDasharray="4 3" /></svg>test loss</span>
        </div>
        {children}
      </div>
    </div>
  );
}
