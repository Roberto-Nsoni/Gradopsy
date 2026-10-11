import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { asMaps, fmt, gridLayout, markBox, paintStage, pickCell, scaleOf } from "./draw";
import { Sym } from "./Inspector";

/*
 * The forward pass, laid out left to right across the page: the digit, then every tensor it
 * becomes, then the ten numbers it ends as.
 *
 * Every stage is drawn at the same size per value, so the picture itself shows what the network
 * does to its input — the image shrinks and deepens, 28x28 with one channel to 7x7 with sixteen —
 * and pointing at any value hands it to the inspector below, which shows how it was computed.
 */

const GAP = 4;           // between maps inside a stage, where the ruling runs
const FLOOR = 1;         // the smallest scale a stage is drawn at (see scaleOf)
const NEAR = "#c9ff4a";                   // the stage the value is computed from
const FAR = "rgba(201, 255, 74, 0.45)";   // every stage before that
const HEAD = 34;         // the operation's name above each stage
const FOOT = 18;         // the tensor's shape below

/** One stage's part of where the value being read comes from: boxes in some maps, or all of it. */
function paintMark(ctx, layout, w, h, mark, ink) {
  if (mark === "all") {
    ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
    ctx.strokeRect(0.75, 0.75, w - 1.5, h - 1.5);
    return;
  }
  const cs = mark.cs === "all" ? layout.maps.map((_, c) => c) : mark.cs;
  for (const c of cs) markBox(ctx, layout, c, mark.box, ink);
}

/** Grid columns for a stage of C maps, so tall stacks stay inside the strip's height. */
const colsFor = (C) => (C <= 1 ? 1 : C <= 8 ? 2 : 4);

export default function Strip({ trace, displays, pad, focus, marks, onPoint, onPin, probs, logits, chosen, onChoose, fool }) {
  const box = useRef(null);
  const cv = useRef({});
  const [size, setSize] = useState({ w: 1200, h: 380 });
  // the screen's own pixel ratio, uncapped: the stages are laid out in its whole pixels (see
  // gridLayout), and it changes when the window moves to another screen or the page is zoomed
  const [dpr, setDpr] = useState(() => window.devicePixelRatio || 1);
  useEffect(() => {
    const onResize = () => setDpr(window.devicePixelRatio || 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useLayoutEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);

  // one size per value for every map stage, the largest at which the tallest stage still fits.
  // On a narrow screen the strip runs down the page and its height follows its content, so there
  // the stages are sized from the width instead — sizing from the height would feed on itself.
  const narrow = typeof window !== "undefined" && window.matchMedia("(max-width: 1100px)").matches;
  const geo = useMemo(() => {
    const inner = narrow ? Math.min(size.w - 24, 420) : Math.max(160, size.h - HEAD - FOOT);
    const maps = displays.filter((d) => d.kind === "maps");
    let cell = 40;
    for (const d of maps) {
      const { C, H, W } = asMaps(d.shape);
      const cols = colsFor(C), rows = Math.ceil(C / cols);
      cell = Math.min(cell, (inner - rows * GAP) / (rows * H));
      if (narrow) cell = Math.min(cell, (size.w - 24 - cols * GAP) / (cols * W));
    }
    const layouts = displays.map((d) => {
      if (d.kind !== "maps") return null;
      const { C, H, W } = asMaps(d.shape);
      return gridLayout(C, H, W, colsFor(C), cell, GAP, dpr);
    });
    return { inner, cell, layouts };
  }, [displays, size, narrow, dpr]);

  // The pad takes what is left: drawn at the full height first, and if the strip then runs past
  // the page, smaller by exactly the overflow. Measured again whenever the strip changes size.
  const [padFit, setPadFit] = useState(Infinity);
  useLayoutEffect(() => { setPadFit(Infinity); }, [size.w, size.h, displays.length]);
  useLayoutEffect(() => {
    if (narrow || padFit !== Infinity) return;
    const over = box.current.scrollWidth - box.current.clientWidth;
    if (over > 0) setPadFit(Math.max(140, geo.inner - over - 2));
  });
  const padSide = Math.min(geo.inner, padFit);

  // paint every map stage, and the marks for the value being read
  useEffect(() => {
    if (!trace) return;
    displays.forEach((d, k) => {
      if (d.kind !== "maps" && d.kind !== "column") return;
      const node = cv.current[k];
      if (!node) return;
      const values = trace.slices[d.view];
      if (d.kind === "column") {
        const h = geo.inner, w = 10;
        if (node.width !== Math.round(w * dpr)) { node.width = Math.round(w * dpr); node.height = Math.round(h * dpr); }
        const ctx = node.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        const top = scaleOf(values, "lit", FLOOR);
        const img = ctx.createImageData(1, values.length);
        for (let i = 0; i < values.length; i++) {
          const a = Math.max(0, Math.min(1, values[i] / top));
          img.data[i * 4] = 6 + 142 * a * 0.9; img.data[i * 4 + 1] = 13 + 242 * a * 0.9; img.data[i * 4 + 2] = 9 + 192 * a * 0.9; img.data[i * 4 + 3] = 255;
        }
        const buf = document.createElement("canvas"); buf.width = 1; buf.height = values.length;
        buf.getContext("2d").putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(buf, 0, 0, node.width, node.height);
        if (focus?.d === k) {
          const y = (focus.index / values.length) * node.height;
          ctx.fillStyle = "#c9ff4a"; ctx.fillRect(0, y - dpr, node.width, 2 * dpr);
        }
        if (marks?.[k] === "all") {
          ctx.strokeStyle = focus.d - k === 1 ? NEAR : FAR; ctx.lineWidth = 1.5 * dpr;
          ctx.strokeRect(0.75 * dpr, 0.75 * dpr, node.width - 1.5 * dpr, node.height - 1.5 * dpr);
        }
        return;
      }
      const { C, H, W } = asMaps(d.shape);
      const layout = geo.layouts[k];
      const ctx = paintStage(node, values, C, H, W, layout, d.mode, FLOOR);
      if (focus?.d === k) markBox(ctx, layout, focus.c, [focus.i, focus.i, focus.j, focus.j]);
      if (marks?.[k]) paintMark(ctx, layout, layout.width, layout.height, marks[k], focus.d - k === 1 ? NEAR : FAR);
    });
  });

  const point = (k, e, pin) => {
    const d = displays[k];
    const node = cv.current[k];
    const b = node.getBoundingClientRect();
    if (d.kind === "column") {
      const index = Math.max(0, Math.min(d.size - 1, Math.floor(((e.clientY - b.top) / b.height) * d.size)));
      const f = { d: k, index };
      pin ? onPin(f) : onPoint(f);
      return;
    }
    const { C } = asMaps(d.shape);
    const cell = pickCell(geo.layouts[k], C, e.clientX - b.left, e.clientY - b.top);
    const f = cell ? { d: k, ...cell } : null;
    pin ? f && onPin(f) : onPoint(f);
  };

  const winner = probs ? probs.indexOf(Math.max(...probs)) : -1;
  const zTop = logits ? Math.max(FLOOR, ...logits.map((v) => Math.abs(v))) : 1;

  return (
    <div className="rc-strip" ref={box}>
      {displays.map((d, k) => (
        <figure key={k} className={"rc-stage is-" + d.kind + (focus?.d === k ? " is-focus" : "")}>
          <figcaption className="rc-stage-op">{d.sym && <><span className="rc-stage-sym"><Sym n={d.sym} /></span> · </>}{d.op}</figcaption>

          {d.kind === "input" && <div className="rc-stage-pad" style={{ width: padSide, height: padSide }}>{pad}</div>}

          {(d.kind === "maps" || d.kind === "column") && (
            <canvas ref={(n) => { cv.current[k] = n; }} className="rc-stage-canvas"
                    style={d.kind === "column" ? { width: 10, height: geo.inner }
                      : { width: geo.layouts[k]?.width, height: geo.layouts[k]?.height }}
                    onPointerMove={(e) => point(k, e, false)} onPointerLeave={() => onPoint(null)}
                    onClick={(e) => point(k, e, true)}
                    aria-label={`${d.op}: ${d.shape.join(" × ")} values. Point at one to see how it was computed.`} />
          )}

          {d.kind === "logits" && (
            <ol className={"rc-logits" + (marks?.[k] === "all" ? " is-marked" : "")} style={{ height: geo.inner }} onPointerLeave={() => onPoint(null)}>
              {Array.from({ length: 10 }, (_, c) => {
                const z = logits?.[c] ?? 0;
                return (
                  <li key={c} className={focus?.d === k && focus.index === c ? "is-focus" : ""}
                      onPointerEnter={() => onPoint({ d: k, index: c })} onClick={() => onPin({ d: k, index: c })}>
                    <b>{c}</b>
                    <span className="rc-zbar"><i className={z < 0 ? "is-neg" : ""}
                      style={{ left: `${50 + Math.min(0, z / zTop) * 50}%`, width: `${(Math.abs(z) / zTop) * 50}%` }} /></span>
                    <span className="rc-num">{logits ? fmt(z, 1) : "—"}</span>
                  </li>
                );
              })}
            </ol>
          )}

          {d.kind === "softmax" && (
            <div className="rc-soft" style={{ height: geo.inner }}>
              <ol className="rc-probs" onPointerLeave={() => onPoint(null)}>
                {Array.from({ length: 10 }, (_, c) => {
                  const p = probs?.[c] ?? 0;
                  return (
                    <li key={c} className={(c === winner ? "is-top" : "") + (c === chosen ? " is-chosen" : "")}>
                      <button type="button" onPointerEnter={() => onPoint({ d: k, index: c })}
                              onClick={() => onChoose(c)} aria-pressed={c === chosen}
                              title={`p(${c}) — press to see which pixels push it up or down`}>
                        <b>{c}</b>
                        <span className="rc-bar"><i style={{ width: `${p * 100}%` }} /></span>
                        <span className="rc-num">{probs ? `${(p * 100).toFixed(1)}%` : "—"}</span>
                      </button>
                    </li>
                  );
                })}
              </ol>
              {fool}
            </div>
          )}

          <span className="rc-stage-shape">{d.shape.join(" × ")}</span>
        </figure>
      ))}
    </div>
  );
}
