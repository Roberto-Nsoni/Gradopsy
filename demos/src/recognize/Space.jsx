import { useEffect, useRef, useState } from "react";

import { bracket } from "../playground/paint";

/*
 * Where it lands: two thousand test digits in the network's last hidden layer, through its PCA,
 * each drawn as the numeral it is — ten classes is more than the world has inks for, and a
 * numeral needs no key. The digit on the pad is bracketed among them, and moves as it is drawn.
 */

const font = (px, dpr) => `600 ${Math.round(px * dpr)}px "Martian Mono", ui-monospace, monospace`;

export function Space({ embed, here, onHover }) {
  const cv = useRef(null);
  const box = useRef(null);
  const [only, setOnly] = useState(null);

  useEffect(() => {
    const node = cv.current;
    if (!node || !embed) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    // as large as the panel allows, in both directions
    const room = box.current.parentElement?.clientHeight ? box.current.parentElement.clientHeight - 110 : Infinity;
    const S = Math.max(140, Math.min(box.current.clientWidth, room));
    const px = Math.round(S * dpr);
    if (node.width !== px) node.width = node.height = px;
    node.style.width = node.style.height = `${S}px`;
    const ctx = node.getContext("2d");
    ctx.fillStyle = "#060d09"; ctx.fillRect(0, 0, px, px);
    ctx.strokeStyle = "rgba(148, 255, 201, 0.13)"; ctx.lineWidth = 1;
    for (let k = 1; k < 4; k++) {
      const v = Math.round((k / 4) * px) + 0.5;
      ctx.beginPath(); ctx.moveTo(v, 0); ctx.lineTo(v, px); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(px, v); ctx.stroke();
    }
    const at = (x, y) => [((x * 0.92 + 1) / 2) * px, px - ((y * 0.92 + 1) / 2) * px];
    ctx.font = font(9, dpr); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < embed.labels.length; i++) {
      const d = embed.labels[i];
      const [x, y] = at(embed.xy[2 * i], embed.xy[2 * i + 1]);
      ctx.fillStyle = only == null || only === d ? `rgba(148, 255, 201, ${only == null ? 0.3 : 0.9})` : "rgba(148, 255, 201, 0.05)";
      ctx.fillText(String(d), x, y);
    }
    if (here) {
      const [x, y] = at(here[0], here[1]);
      bracket(ctx, x, y, 9 * dpr, "#c9ff4a", 5 * dpr, 1.8 * dpr);
    }
  });

  return (
    <div className="rc-space" ref={box}>
      <header className="rc-space-head">
        <span className="pg-label">last hidden layer · {embed ? `PCA · ${Math.round(embed.share * 100)}%` : "…"}</span>
        <span className="rc-digits" role="group" aria-label="Pick out one digit">
          {Array.from({ length: 10 }, (_, d) => (
            <button type="button" key={d} className={"pg-tiny" + (only === d ? " is-on" : "")} aria-pressed={only === d}
                    onPointerEnter={() => onHover?.(d)} onPointerLeave={() => onHover?.(null)}
                    onClick={() => setOnly((o) => (o === d ? null : d))}>{d}</button>
          ))}
        </span>
      </header>
      <canvas ref={cv} className="rc-space-canvas"
              aria-label="Two thousand test digits placed by the network's last hidden layer; the drawn digit is bracketed." />
    </div>
  );
}
