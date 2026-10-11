import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  recAt, recEmbed, recEmbedAxes, recEmbedPart, recExplain, recHistory, recLoad, recSaliency, recTestset, recTrace,
} from "../engine";
import { VARS, paintTube } from "../looks/scope";
import Icon from "../playground/icons";
import Pad from "./Pad";
import Strip from "./Strip";
import Inspector, { Sym } from "./Inspector";
import { Space } from "./Space";
import RunLog from "./RunLog";
import "../looks/looks.css";
import "../playground/playground.css";
import "./recognize.css";

/*
 * recognize: the forward pass, read.
 *
 * The playground is a bench for building and training a network. Nothing is built or trained here;
 * a trained network is read. A digit drawn on the pad becomes every tensor the network turns it
 * into, laid across the page left to right, and any number in any of them can be pointed at to see
 * exactly how it was computed — the patch and the kernel of a convolution, the inputs and weights
 * of a linear unit, the block of a max-pool, the softmax of the logits — with the network's own
 * numbers, all computed by the engine in this tab.
 *
 * Both networks were trained offline by scripts/train_mnist.py with the same engine.
 */

const BASE = import.meta.env.BASE_URL + "recognize/";
const REPO = "https://github.com/Roberto-Nsoni/visualML";
const MODELS = { mlp: "MLP", cnn: "CNN" };
const EMBED_DIGITS = 2000;          // test digits in the "where it lands" picture
const PCA_STEPS = 50;               // subspace iterations for its two axes (see rec_embed_axes)

async function fetchOk(path) {
  const res = await fetch(BASE + path);
  // a missing file comes back as the site's own page from a static host, not as a 404
  if (!res.ok || (res.headers.get("content-type") || "").includes("text/html")) throw new Error(`${path} is not there`);
  return res;
}
const fetchBytes = async (path) => new Uint8Array(await (await fetchOk(path)).arrayBuffer());
const fetchJson = async (path) => (await fetchOk(path)).json();

/**
 * The engine's modules as the strip draws them. A convolution or linear layer followed by a ReLU
 * is one stage (its maps are drawn after the ReLU; the inspector shows both steps), a max-pool
 * and a flatten are stages of their own, the last linear is the logits, and a softmax closes it.
 *
 * Every stage carries the symbol its values are written with — c¹, m¹, c², m², f, h¹, h², z, p —
 * and `unitName` names one value the way the playground does: the layer as a superscript, the
 * position as subscripts counted from 1. A logit and a probability are named by their digit.
 */
function buildDisplays(stages) {
  const out = [{ kind: "input", op: "input", sym: { s: "x" }, shape: [28, 28] }];
  const at3 = (shape, n) => {
    const [, H, W] = shape;
    return [Math.floor(n / (H * W)) + 1, Math.floor(n / W) % H + 1, (n % W) + 1];
  };
  let convs = 0, pools = 0, linears = 0;
  let last = { s: "x" };               // the symbol of the stage before, which a linear unit reads
  for (let m = 0; m < stages.length; m++) {
    const st = stages[m], next = stages[m + 1];
    const prev = stages[m - 1];
    const inShape = !prev ? [784] : prev.kind === "flatten" ? stages[m - 2].shape : prev.shape;
    const from = last;
    // a linear unit's inputs, named as the values they are: pixels, pooled maps, or hidden units
    const inNames = (n) => {
      if (inShape.length === 3) return { ...from, i: at3(inShape, n) };
      if (!prev) return { s: "x", i: [Math.floor(n / 28) + 1, (n % 28) + 1] };
      return { ...from, i: [n + 1] };
    };
    if (st.kind === "conv" && next?.kind === "relu") {
      const sym = { s: "c", l: ++convs };
      out.push({ kind: "maps", op: `conv ${st.k}×${st.k} · relu`, sym, shape: st.shape, view: m + 1, explain: m, mode: "lit",
                 unitName: (f) => ({ ...sym, i: [f.c + 1, f.i + 1, f.j + 1] }) });
      last = sym;
      m += 1;
    } else if (st.kind === "linear" && next?.kind === "relu") {
      const sym = { s: "h", l: ++linears };
      out.push({ kind: "maps", op: "linear · relu", sym, shape: st.shape, view: m + 1, explain: m, mode: "lit", inShape, inNames,
                 unitName: (f) => ({ ...sym, i: [f.index + 1] }) });
      last = sym;
      m += 1;
    } else if (st.kind === "pool") {
      const sym = { s: "m", l: ++pools };
      out.push({ kind: "maps", op: `max ${st.k}×${st.k}`, sym, shape: st.shape, view: m, explain: m, mode: "lit",
                 unitName: (f) => ({ ...sym, i: [f.c + 1, f.i + 1, f.j + 1] }) });
      last = sym;
    } else if (st.kind === "flatten") {
      const [C, H, W] = prev.shape;
      const source = last;
      out.push({ kind: "column", op: "flatten", sym: { s: "f" }, shape: st.shape, view: m, explain: m, size: st.shape[0], from: { C, H, W },
                 unitName: (f) => ({ s: "f", i: [f.index + 1] }), sourceName: (n) => ({ ...source, i: at3(prev.shape, n) }) });
    } else if (st.kind === "linear") {
      out.push({ kind: "logits", op: "linear", sym: { s: "z" }, shape: st.shape, view: m, explain: m, inShape, inNames,
                 unitName: (f) => ({ s: "z", i: [f.index] }) });
    }
  }
  out.push({ kind: "softmax", op: "softmax", sym: { s: "p" }, shape: [10] });
  return out;
}

export default function Recognize() {
  const [model, setModel] = useState("cnn");
  const [status, setStatus] = useState("loading the networks");
  const [ready, setReady] = useState({});
  const [logs, setLogs] = useState({});
  const [results, setResults] = useState(null);
  const [pixels, setPixels] = useState(null);
  const [traced, setTrace] = useState(null);
  const [hover, setHover] = useState(null);
  const [pinned, setPinned] = useState(null);
  const [grad, setGrad] = useState(null);
  const [snaps, setSnaps] = useState({});
  const [at, setAt] = useState(-1);
  const [embed, setEmbed] = useState(null);
  const testLoaded = useRef(false);
  const pad = useRef(null);

  // ---------- loading ----------

  useEffect(() => {
    let alive = true;
    (async () => {
      const soft = (p) => p.catch((e) => { console.warn("recognize: not available", e); return null; });
      const [res, mlpLog, cnnLog] = await Promise.all([soft(fetchJson("results.json")), soft(fetchJson("mlp-log.json")), soft(fetchJson("cnn-log.json"))]);
      if (!alive) return;
      setResults(res); setLogs({ mlp: mlpLog, cnn: cnnLog });
      let any = false;
      for (const m of ["cnn", "mlp"]) {
        try {
          setStatus(`loading the ${MODELS[m]}`);
          recLoad(m, await fetchBytes(`${m}.npz`));
          if (!alive) return;
          setReady((r) => ({ ...r, [m]: true }));
          any = true;
        } catch (e) { console.warn(`recognize: the ${m} could not be loaded`, e); }
      }
      setStatus(any ? null : "the networks could not be loaded");
    })();
    return () => { alive = false; };
  }, []);

  // ---------- the trace: every stage of the digit ----------

  // the strip's shape comes from the network, not from a digit: laid out from a blank one as soon as
  // the network is loaded, so nothing moves under the pen when the first stroke arrives
  const skeleton = useMemo(() => (ready[model] ? recTrace(model, new Float32Array(784)).stages : null), [model, ready]);
  const displays = useMemo(() => (skeleton ? buildDisplays(skeleton) : [{ kind: "input", op: "input", sym: { s: "x" }, shape: [28, 28] }]), [skeleton]);

  // the pinned value is the one whose saliency is painted on the pad, recomputed with every stroke:
  // the gradient of its z (before any ReLU, so a unit that is off still shows what would turn it on)
  const target = pinned ? displays[pinned.d] : null;
  const salStage = target?.explain ?? null;
  const salIndex = salStage != null ? pinned.index : null;

  // with nothing drawn the network still runs, on a digit of zeros: every stage shows what its
  // biases alone make of nothing, and clearing the pad brings exactly that back.
  // The saliency is a backward pass on top of the forward one, so while the pen moves it is
  // redone at most every 120 ms, and always when the pen lifts (`force`).
  const lastSal = useRef(0);
  const read = useCallback((drawn, force = false) => {
    if (!ready[model]) { setTrace(null); setGrad(null); return; }
    const px = drawn ?? new Float32Array(784);
    const t = recTrace(model, px);
    t.model = model;
    // one view per module, as a typed array into the one buffer
    let offset = 0;
    t.slices = t.stages.map((st) => {
      const n = st.shape.reduce((a, b) => a * b, 1);
      const sl = t.data.subarray(offset, offset + n);
      offset += n;
      return sl;
    });
    setTrace(t);
    if (salStage == null) setGrad(null);
    else if (force || performance.now() - lastSal.current > 120) {
      setGrad(recSaliency(model, px, salStage, salIndex).grad);
      lastSal.current = performance.now();
    }
  }, [model, ready, salStage, salIndex]);

  useEffect(() => { read(pixels); }, [pixels]);                     // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { read(pixels, true); }, [read, at]);              // eslint-disable-line react-hooks/exhaustive-deps


  // a trace belongs to the network that made it: for the one render between switching networks and
  // reading the digit again, the strip has the new network's stages and no values for them
  const trace = traced?.model === model ? traced : null;

  const focus = hover ?? pinned;
  const display = focus ? displays[focus.d] : null;
  const explain = useMemo(() => {
    if (!trace || !display || display.explain == null) return null;
    try { return recExplain(model, display.explain, focus.index); } catch { return null; }
  }, [trace, display, focus, model]);

  // what to mark: the receptive field on the pad, and the source region on the stage before
  // Where the value being read comes from, in every stage before it: the engine walks the region
  // back module by module (explain.fields), and a stage it depends on entirely is marked "all".
  // A probability depends on everything.
  const { field, marks } = useMemo(() => {
    if (!focus || !display) return { field: null, marks: null };
    if (display.kind === "softmax") return { field: [0, 27, 0, 27], marks: displays.map((d, k) => (k > 0 && k < focus.d ? "all" : null)) };
    if (!explain) return { field: null, marks: null };
    return {
      field: explain.field,
      marks: displays.map((d, k) => {
        if (k === 0 || k >= focus.d) return null;
        return explain.fields?.[d.view] ?? null;
      }),
    };
  }, [explain, focus, display, displays]);

  // ---------- inputs and saliency ----------

  const onInput = useCallback((px) => { setPixels(px); }, []);

  // pressing a digit in the softmax pins its logit, z of that digit
  const logitsAt = displays.findIndex((d) => d.kind === "logits");
  const choose = (digit) => setPinned((p) => (p && p.d === logitsAt && p.index === digit ? null : { d: logitsAt, index: digit }));
  const chosen = pinned && pinned.d === logitsAt ? pinned.index : null;

  const clear = () => { pad.current?.clear(); setPinned(null); };

  // ---------- time ----------

  const scrub = async (idx) => {
    if (!snaps[model]) {
      const h = recHistory(model, await fetchBytes(`${model}-history.npz`));
      setSnaps((s) => ({ ...s, [model]: h.steps }));
    }
    recAt(model, idx);
    setAt(idx);
    setEmbed(null);
  };

  useEffect(() => {
    if (!ready[model]) return;
    recAt(model, -1); setAt(-1); setEmbed(null); setPinned(null); setHover(null);
  }, [model, ready]);

  // ---------- where it lands ----------

  // Two thousand digits through the CNN take seconds, so they go a few at a time with the browser
  // let back in between: the page answers the pointer the whole time and the picture lands when
  // it is done. Each chunk is sized to take about 30 ms, whatever the machine.
  useEffect(() => {
    if (embed || !ready[model]) return;
    let alive = true;
    (async () => {
      try {
        if (!testLoaded.current) {
          const [gz, labels] = await Promise.all([fetchBytes("test-images.dat"), fetchBytes("test-labels.u8")]);
          if (!alive) return;
          recTestset(gz, labels);
          testLoaded.current = true;
        }
        let start = 0, count = 10;
        while (start < EMBED_DIGITS) {
          await new Promise((r) => setTimeout(r, 0));
          if (!alive) return;
          count = Math.min(count, EMBED_DIGITS - start);
          const t = performance.now();
          recEmbedPart(model, start, count);
          start += count;
          const took = performance.now() - t;
          if (took < 15) count *= 2;
          else if (took > 45 && count > 1) count = Math.ceil(count / 2);
        }
        for (let i = 0; i < PCA_STEPS; i += 2) {
          await new Promise((r) => setTimeout(r, 0));
          if (!alive) return;
          recEmbedAxes(model, 2);
        }
        setEmbed(recEmbed(model));
      } catch (e) { console.warn("recognize: no test digits for the space picture", e); }
    })();
    return () => { alive = false; };
  }, [embed, model, ready, at]);

  const here = useMemo(() => {
    if (!embed || !trace?.last || !pixels) return null;
    return [0, 1].map((j) => trace.last.reduce((s, v, k) => s + (v - embed.mean[k]) * embed.axes[j][k], 0) / embed.span);
  }, [embed, trace, pixels]);

  // ---------- keys ----------

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); pad.current?.undo(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (k === "c" || k === "backspace" || k === "delete") clear();
      else if (k === "escape") setPinned(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ---------- the tube ----------

  const tube = useRef(null);
  useEffect(() => {
    const paint = () => {
      const node = tube.current;
      if (!node) return;
      const size = Math.max(window.innerWidth, window.innerHeight);
      node.width = node.height = Math.min(1400, size);
      node.style.width = node.style.height = `${size}px`;
      paintTube(node.getContext("2d"), node.width);
    };
    paint();
    window.addEventListener("resize", paint);
    return () => window.removeEventListener("resize", paint);
  }, []);

  const res = results?.[model];

  const padNode = (
    <div className="pg-screen rc-pad">
      <Pad ref={pad} field={field} grad={grad} onInput={onInput} onSettle={(px) => read(px, true)} />
      <span className="rc-pad-tools">
        <button type="button" className="pg-tiny" onClick={() => pad.current?.undo()} aria-label="Undo the last stroke" title="Undo (Ctrl+Z)"><Icon name="undo" size={11} /></button>
        <button type="button" className="pg-tiny" onClick={clear} aria-label="Clear the pad" title="Clear (C)"><Icon name="clear" size={11} /></button>
      </span>
      {!pixels && !status && <p className="rc-empty">draw a digit</p>}
      {status && <p className="pg-screen-notice" role="status">{status}</p>}
      {target && grad && <p className="rc-pad-label">saliency · <Sym n={target.unitName(pinned)} /></p>}
    </div>
  );

  return (
    <div className="rc" style={VARS}>
      <canvas ref={tube} className="pg-tube" aria-hidden="true" />

      <header className="pg-head rc-head">
        <a className="look-mark" href="/">visualML</a>
        <h1 className="pg-visually-hidden">visualML recognize</h1>
        <nav className="pg-nav" aria-label="Demos">
          <a href="/playground">playground</a>
          <a href="/recognize" aria-current="page">recognize</a>
        </nav>
        <div className="rc-models" role="group" aria-label="Which network reads the digit">
          {Object.entries(MODELS).map(([id, name]) => (
            <button type="button" key={id} className={"pg-chip" + (model === id ? " is-on" : "")} aria-pressed={model === id}
                    disabled={!ready[id]} onClick={() => setModel(id)}>
              <span>{name}</span>
              <b className="pg-measure">{results?.[id] ? `${(results[id].mean * 100).toFixed(2)}%` : ""}</b>
            </button>
          ))}
        </div>
        <nav className="look-source pg-source"><a href={REPO} target="_blank" rel="noreferrer">source</a></nav>
      </header>

      <section className="rc-pass" aria-label="The forward pass">
        <Strip trace={trace} displays={displays} pad={padNode} focus={focus} marks={marks}
               onPoint={setHover} onPin={(f) => setPinned((p) => (p && p.d === f.d && p.index === f.index ? null : f))}
               probs={trace?.probs} logits={trace?.logits} chosen={chosen} onChoose={choose} />
        <div className="rc-run">
          <RunLog log={logs[model]} snaps={snaps[model]} at={at} onScrub={scrub} onLive={() => scrub(-1)} />
        </div>
      </section>

      <section className="rc-below">
        <div className="rc-explain" aria-live="polite">
          <Inspector display={display} explain={explain} trace={trace} focus={focus} />
        </div>
        <aside className="rc-where" aria-label="Where it lands">
          <Space embed={embed} here={here} />
          {res && (
            <p className="rc-facts">
              {MODELS[model]} · {res.parameters.toLocaleString("en")} weights · test accuracy {(res.mean * 100).toFixed(2)}% ± {(res.std * 100).toFixed(2)} over {res.runs.length} seeds
            </p>
          )}
        </aside>
      </section>
    </div>
  );
}
