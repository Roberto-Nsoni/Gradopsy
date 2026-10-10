import { useEffect, useMemo, useRef, useState } from "react";

import { pgEmbed } from "../../engine";
import { paintPoints } from "../paint";
import { LayerControls, useBox } from "./shared";
import "./views.css";

/*
 * PARKED, NOT ROUTED. The author liked this study and wants it for another visualisation, but as
 * the network panel it was harder to read than the schematic. To try it again, render it in place
 * of <Network> in Playground.jsx and pass `points`, `at` and `engineKey` ("playground").
 * It needs `pg_embed` in engine.py.
 *
 * Study E · warp. Not what each unit draws, but where each layer puts the data.
 *
 * A layer of k units turns every point into k numbers: a new place for it, in a space of its
 * own. This study shows the same points in each of those spaces, one screen per layer, left to
 * right as the data flows. The first screen is the plane as drawn. Training is visible as the
 * classes being pulled apart from screen to screen, until in the last one a single straight cut
 * would separate them — which is all the output layer can do.
 *
 * A layer wider than two units cannot be drawn whole, so it is shown through its two main
 * directions (principal components), and its screen says how much of the layer that shows.
 */

export default function NetWarp({ view, layers, points, at, engineKey, onUnits, onAct }) {
  const box = useRef(null);
  const size = useBox(box);
  const cv = useRef({});
  const [embed, setEmbed] = useState(null);

  // the view refreshes a few times a second while training; the embedding follows it
  useEffect(() => {
    if (!view) return;
    try { setEmbed(pgEmbed(engineKey, at ?? -1)); } catch { setEmbed(null); }
  }, [view, at, engineKey]);

  const screens = useMemo(() => {
    const list = [{ name: "input", note: "the plane as drawn", xy: points.map((q) => [q.x, q.y]), exact: true }];
    const ok = embed && embed.layers.length === layers.length && embed.layers.every((l) => l.xy.length === points.length);
    layers.forEach((l, i) => {
      const e = ok ? embed.layers[i] : null;
      list.push({
        name: `layer ${i + 1}`, layer: i,
        note: !e ? "…" : e.units === 1 ? "1 unit: a line" : e.exact ? "2 units: shown whole" : `${e.units} units · ${Math.round(e.share * 100)}% shown`,
        xy: e ? e.xy : null,
      });
    });
    return list;
  }, [embed, points, layers]);

  const per = screens.length <= 2 ? 2 : screens.length <= 4 ? 2 : 3;
  const rows = Math.ceil(screens.length / per);
  const S = Math.round(Math.max(80, Math.min((size.w - 16 - (per - 1) * 14) / per, (size.h - 44) / rows - 58)));

  useEffect(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const px = Math.round(S * dpr);
    screens.forEach((s, i) => {
      const node = cv.current[i];
      if (!node) return;
      if (node.width !== px) node.width = node.height = px;
      const ctx = node.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#060d09"; ctx.fillRect(0, 0, px, px);
      ctx.strokeStyle = "rgba(148, 255, 201, 0.13)"; ctx.lineWidth = 1;
      for (let k = 1; k < 4; k++) {
        const v = Math.round((k / 4) * px) + 0.5;
        ctx.beginPath(); ctx.moveTo(v, 0); ctx.lineTo(v, px); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(px, v); ctx.stroke();
      }
      if (!s.xy) return;
      const pts = points.map((q, j) => ({ ...q, x: s.xy[j][0] * 0.9, y: s.xy[j][1] * 0.9 }));
      paintPoints(ctx, pts, px, Math.max(2.2, px / 110));
    });
  }, [screens, S, points]);

  return (
    <div className="pg-net pgv-warp" ref={box}>
      <p className="pgv-warp-say">Each screen: the same points, placed where that layer puts them. Training pulls the colours apart; by the last screen one straight cut should be enough.</p>
      <ol className="pgv-warp-grid" style={{ gridTemplateColumns: `repeat(${per}, ${S}px)` }}>
        {screens.map((s, i) => (
          <li key={i}>
            <header>
              <b>{i > 0 && <span className="pgv-warp-step">→ </span>}{s.name}</b>
              <span>{s.note}</span>
            </header>
            <canvas ref={(node) => { cv.current[i] = node; }} style={{ width: S, height: S }} />
            {s.layer != null && layers[s.layer] && (
              <LayerControls index={s.layer} layer={layers[s.layer]} onUnits={onUnits} onAct={onAct} compact />
            )}
          </li>
        ))}
      </ol>
      {!points.length && <p className="pgv-note">Draw some points or pick a set to see them travel.</p>}
    </div>
  );
}
