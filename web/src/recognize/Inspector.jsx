import { useEffect, useRef } from "react";

import { asMaps, fmt, gridLayout, paintStage } from "./draw";

/*
 * The inspector: one number of the forward pass, and exactly how it was computed, with the
 * network's own numbers. A convolution is its patch times its kernel, summed, plus a bias, through
 * a ReLU; a linear unit is every input times its weight, summed; a max-pool is the largest of a
 * block; a probability is a softmax of ten logits. The pictures are the operands, captioned with
 * the words of the formula above them; the line under them is the arithmetic, in the same words.
 *
 * Every value is written the way the strip labels its stage, and the way the playground writes a
 * unit: the stage's symbol, its layer as a superscript, its position as subscripts counted from 1
 * (c¹₃,₆,₂₄ is filter 3 of the first convolution, row 6, column 24). Logits and probabilities are
 * named by their digit.
 */

/** A value's name: { s: "c", l: 1, i: [3, 6, 24] } -> c¹₃,₆,₂₄. */
export const Sym = ({ n }) => (
  <span className="rc-name">
    {n.s}
    {n.l != null && n.i ? (
      // both: stacked in one column after the letter, layer over position, the way LaTeX sets m^2_{6,2,5}
      <span className="rc-ss"><span>{n.l}</span><span>{n.i.join(",")}</span></span>
    ) : (
      <>{n.l != null && <sup>{n.l}</sup>}{n.i && <sub>{n.i.join(",")}</sub>}</>
    )}
  </span>
);

function Raster({ values, shape, mode, cell, cols, label }) {
  const node = useRef(null);
  const { C, H, W } = asMaps(shape);
  const layout = gridLayout(C, H, W, cols ?? (C <= 1 ? 1 : 4), cell, 2, window.devicePixelRatio || 1);
  useEffect(() => { paintStage(node.current, values, C, H, W, layout, mode); });
  return (
    <figure className="rc-op">
      <canvas ref={node} style={{ width: layout.width, height: layout.height }} />
      {label && <figcaption>{label}</figcaption>}
    </figure>
  );
}

/** A small grid of the actual numbers, each cell shaded the way the picture would be. */
function NumGrid({ values, k, mode, label }) {
  let top = 1e-9;
  for (const v of values) top = Math.max(top, mode === "lit" ? v : Math.abs(v));
  return (
    <figure className="rc-op">
      <div className="rc-nums" style={{ gridTemplateColumns: `repeat(${k}, 1fr)` }}>
        {Array.from(values).map((v, i) => {
          const a = Math.min(1, Math.abs(v) / top);
          const bg = mode === "lit" || v >= 0 ? `rgba(148, 255, 201, ${a * 0.32})` : `rgba(70, 135, 230, ${a * 0.45})`;
          return <span key={i} style={{ background: bg }}>{fmt(v)}</span>;
        })}
      </div>
      {label && <figcaption>{label}</figcaption>}
    </figure>
  );
}

const Eq = ({ children }) => <p className="rc-eq">{children}</p>;
const Op = ({ children }) => <span className="rc-sym" aria-hidden="true">{children}</span>;

/**
 * The one line of arithmetic a convolution and a linear unit share, in the words of their formula:
 * Σ (the products added up), + b, = z, and through the ReLU when there is one. The value being read
 * is the last number on the line, in the accent. z and b are rounded and Σ is shown as their
 * difference, so the line adds up exactly as printed (it is within 0.0005 of the true Σ).
 */
function Sum({ sum, bias, relu }) {
  const z = sum + bias;
  const zr = Number(z.toFixed(3)), br = Number(bias.toFixed(3));
  return (
    <Eq>
      Σ <b>{fmt(zr - br, 3)}</b> &nbsp;+ b <b>{fmt(br, 3)}</b> &nbsp;= z{" "}
      <b className={relu == null ? "rc-result" : ""}>{fmt(z, 3)}</b>
      {relu != null && <> &nbsp;→ relu <b className="rc-result">{fmt(relu, 3)}</b></>}
      {relu != null && z < 0 && <span className="rc-cut"> cut to 0</span>}
    </Eq>
  );
}

function Conv({ e, name, relu }) {
  const k = e.k, C = e.channels;
  const cell = C === 1 ? 14 : 6;
  // the sum split by input map, so a deep convolution reads as C small ones added up
  const perMap = Array.from({ length: C }, (_, c) => e.products.subarray(c * k * k, (c + 1) * k * k).reduce((s, v) => s + v, 0));
  return (
    <>
      <Eq><b><Sym n={name} /></b> = relu( Σ patch ⊙ kernel + b )</Eq>
      {C === 1 ? (
        <div className="rc-ops">
          <NumGrid values={e.patch} k={k} mode="lit" label="patch" />
          <Op>⊙</Op>
          <NumGrid values={e.kernel} k={k} mode="signed" label="kernel" />
          <Op>=</Op>
          <NumGrid values={e.products} k={k} mode="signed" label="products" />
        </div>
      ) : (
        <div className="rc-ops is-deep">
          <Raster values={e.patch} shape={[C, k, k]} mode="lit" cell={cell} cols={4} label={`patch · ${C} maps`} />
          <Op>⊙</Op>
          <Raster values={e.kernel} shape={[C, k, k]} mode="signed" cell={cell} cols={4} label={`kernel · ${C} maps`} />
          <Op>=</Op>
          <Raster values={e.products} shape={[C, k, k]} mode="signed" cell={cell} cols={4} label={`products · ${C} maps`} />
        </div>
      )}
      {C > 1 && (
        <p className="rc-terms">
          Σ = {perMap.map((v, c) => <span key={c}>{c ? (fmt(v).startsWith("−") ? " − " : " + ") : ""}{c ? fmt(Math.abs(v)) : fmt(v)}</span>)}
        </p>
      )}
      <Sum sum={e.sum} bias={e.bias} relu={relu} />
    </>
  );
}

function Pool({ e, name }) {
  const top = Math.max(...e.block);
  return (
    <>
      <Eq><b><Sym n={name} /></b> = max of {e.k}×{e.k}</Eq>
      <div className="rc-block" style={{ gridTemplateColumns: `repeat(${e.k}, 1fr)` }}>
        {e.block.map((v, i) => <span key={i} className={v === top ? "is-max" : ""}>{fmt(v, 3)}</span>)}
      </div>
      <Eq>max( {e.block.map((v) => fmt(v, 3)).join(", ")} ) = <b className="rc-result">{fmt(e.value, 3)}</b></Eq>
      <p className="rc-note">Only this one value travels on; the other three are dropped. That is all a max-pool learns: nothing — it has no weights.</p>
    </>
  );
}

function Lin({ e, name, inShape, relu, inNames }) {
  const { C, H } = asMaps(inShape);
  // the same picture size whatever the input: 16 maps of 7x7, a 28x28 image, a 16x16 layer
  const cell = C > 1 ? 4 : Math.max(4, Math.round(112 / H));
  return (
    <>
      <Eq><b><Sym n={name} /></b> = {relu != null ? "relu( " : ""}Σ input ⊙ weights + b{relu != null ? " )" : ""}</Eq>
      <div className="rc-ops">
        <Raster values={e.inputs} shape={inShape} mode="lit" cell={cell} label="input" />
        <Op>⊙</Op>
        <Raster values={e.weights} shape={inShape} mode="signed" cell={cell} label="weights" />
        <Op>=</Op>
        <Raster values={e.products} shape={inShape} mode="signed" cell={cell} label="products" />
      </div>
      <p className="rc-terms">
        Σ = {e.top.map(([i, a, w], k) => (
          <span key={i}>{k ? " + " : ""}<Sym n={inNames(i)} /> {fmt(a)}×{w < 0 ? `(${fmt(w)})` : fmt(w)}</span>
        ))}
        <span className="rc-more"> + {e.n_in - e.top.length} more</span>
      </p>
      <Sum sum={e.sum} bias={e.bias} relu={relu} />
    </>
  );
}

function Softmax({ logits, probs, k }) {
  const exps = logits.map((z) => Math.exp(z - Math.max(...logits)));
  const total = exps.reduce((s, v) => s + v, 0);
  const small = (v) => (v < 0.001 ? v.toExponential(0).replace("-", "−") : v.toFixed(3));
  return (
    <>
      <Eq>
        <b><Sym n={{ s: "p", i: [k] }} /></b> = e<sup><Sym n={{ s: "z", i: [k] }} /></sup> / Σ<sub>j</sub> e<sup><Sym n={{ s: "z", i: ["j"] }} /></sup>
      </Eq>
      <table className="rc-softmax">
        <thead><tr><th>digit</th>{logits.map((_, c) => <th key={c} className={c === k ? "is-k" : ""}>{c}</th>)}</tr></thead>
        <tbody>
          <tr><th>z</th>{logits.map((z, c) => <td key={c} className={c === k ? "is-k" : ""}>{fmt(z, 2)}</td>)}</tr>
          <tr><th>e<sup>z − max</sup></th>{exps.map((v, c) => <td key={c} className={c === k ? "is-k" : ""}>{small(v)}</td>)}</tr>
          <tr><th>p</th>{probs.map((p, c) => <td key={c} className={c === k ? "is-k" : ""}>{(p * 100).toFixed(2)}%</td>)}</tr>
        </tbody>
      </table>
      <Eq>
        <Sym n={{ s: "p", i: [k] }} /> = {small(exps[k])} / {total.toFixed(3)} = <b className="rc-result">{(probs[k] * 100).toFixed(2)}%</b>
      </Eq>
      <p className="rc-note">Subtracting the largest z first changes nothing in the ratio and keeps e<sup>z</sup> from overflowing — the engine does the same.</p>
    </>
  );
}

export default function Inspector({ display, explain, trace, focus }) {
  if (!trace) return null;

  let body = null;
  if (!display || display.kind === "softmax" || display.kind === "input") {
    const k = display?.kind === "softmax" && focus ? focus.index : trace.probs.indexOf(Math.max(...trace.probs));
    body = <Softmax logits={trace.logits} probs={trace.probs} k={k} />;
  } else if (explain && display.kind === "logits") {
    body = <Lin e={explain} name={display.unitName(focus)} inShape={display.inShape} relu={null} inNames={display.inNames} />;
  } else if (explain?.kind === "conv") {
    body = <Conv e={explain} name={display.unitName(focus)} relu={trace.slices[display.view][focus.index]} />;
  } else if (explain?.kind === "linear") {
    body = <Lin e={explain} name={display.unitName(focus)} inShape={display.inShape} relu={trace.slices[display.view][focus.index]} inNames={display.inNames} />;
  } else if (explain?.kind === "pool") {
    body = <Pool e={explain} name={display.unitName(focus)} />;
  } else if (explain?.kind === "flatten") {
    body = (
      <>
        <Eq><b><Sym n={display.unitName(focus)} /></b> = <Sym n={display.sourceName(focus.index)} /> = <b className="rc-result">{fmt(explain.value, 3)}</b></Eq>
        <p className="rc-note">The sixteen 7×7 maps laid end to end, so a linear layer can weigh every one of their {display.shape[0]} values.</p>
      </>
    );
  }

  return (
    <div className="rc-inspect">
      {body}
    </div>
  );
}
