import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  pgAdvance, pgClose, pgData, pgDataset, pgOpen, pgProbe, pgRewind, pgSet, pgView, pgWatch,
} from "../engine";
import { VARS, paintTube } from "../looks/scope";
import {
  DEFAULTS, MAX_LAYERS, MAX_POINTS, SECOND_ORDER, SETS, FRAME_BUDGET, RATE_MAX, RATE_MIN, archKey, countParams, pythonFor,
} from "./config";
import { CLASSES } from "./paint";
import Screen from "./Screen";
import Network from "./Network";
import Timeline from "./Timeline";
import { DataPanel, LearnPanel } from "./Panel";
import Icon from "./icons";
import "../looks/looks.css";
import "./playground.css";

/*
 * The playground: draw the data, build the network, and watch every unit in it learn.
 *
 * The page owns the data and the loop; the engine — gradopsy, in this tab, through Pyodide —
 * owns the model, the optimizer and the history. Training runs a few milliseconds per animation
 * frame, so nothing ever freezes; the schematic is re-rendered from the engine a few times a
 * second; and any recorded moment of the run can be brought back, exactly, by dragging the head
 * of the timeline.
 */

const KEY = "playground";
const REPO = "https://github.com/Roberto-Nsoni/visualML";
const VIEW_N = 40;
const REFRESH_MS = 110;
const PATIENCE = 500;       // epochs without a 1% gain in training loss before the run stops itself

// said quietly, beside the run it describes, not over the drawing
const STOP_WHY = {
  perfect: "100% train accuracy",
  plateau: `no gain in ${PATIENCE} epochs`,
};

/** A seeded uniform, so a set drawn with the same seed holds out the same points. */
function seeded(seed) {
  let a = (seed * 2654435761 + 0x9e3779b9) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A named set from the engine, as the page's own points. */
function setPoints0(c) {
  const d = pgDataset(c.set, c.n, c.noise, c.classes, c.dataSeed);
  const rand = seeded(c.dataSeed + 17);
  return d.points.map(([x, y], i) => ({ x, y, c: d.labels[i], u: rand() }));
}

/**
 * Which points the network never sees. A point drawn with the pen set to train or test keeps that
 * role whatever the split; every other point is held out by its own fixed draw against the split,
 * so moving the slider moves the fewest points it can.
 */
const isTest = (p, split) => (p.role ? p.role === "test" : p.u < split);
const annotate = (points, split) => points.map((p) => (p.test === isTest(p, split) ? p : { ...p, test: isTest(p, split) }));

/** 1, 2 … 9, 10, 12 … 95, 100: a speed reads as a round number. */
const roundSpeed = (v) => Math.min(RATE_MAX, Math.max(RATE_MIN, v < 10 ? Math.round(v) : +v.toPrecision(2)));

const fmt = (v) => (v == null || !Number.isFinite(v) ? "—" : v !== 0 && v < 0.001 ? v.toExponential(1) : v.toFixed(4));
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}`);

export default function Playground() {
  // /playground/linked tries the schematic with each layer's space under it; /playground is untouched
  // It opens empty and paused: the first thing to do is give the network something to learn,
  // and the screen says so. Nothing trains until the visitor asks it to.
  const [config, setConfig] = useState(DEFAULTS);
  const [points, setPointsState] = useState([]);
  const [pristine, setPristine] = useState(false);
  const [tool, setTool] = useState("brush");
  const [radius, setRadius] = useState(0.1);
  const [active, setActive] = useState(0);
  const [role, setRole] = useState("split");     // where newly drawn points go: by the split, or pinned
  const [playing, setPlaying] = useState(false);
  const [at, setAt] = useState(null);
  const [view, setView] = useState(null);
  const [history, setHistory] = useState([]);
  const [now, setNow] = useState(null);
  const [hoverLens, setHoverLens] = useState(null);
  const [pinnedLens, setPinnedLens] = useState(null);
  const [probe, setProbe] = useState(null);
  const [status, setStatus] = useState(null);
  const [wave, setWave] = useState(null);
  const [tab, setTab] = useState("network");
  const [copied, setCopied] = useState(false);
  const [, setUndoTick] = useState(0);
  const [autoStop, setAutoStop] = useState(true);
  const [stopped, setStopped] = useState(null);

  const store = useRef({
    points, field: null, fieldStamp: 0, view: null, viewStamp: 0,
    classes: config.classes, layers: config.layers, lens: null, probe: null, diverged: false,
  });
  const live = useRef({ open: false, history: [], last: null, refreshAt: 0, probeAsk: null, dataDirty: false });
  const undo = useRef({ past: [], future: [], stroke: false });
  const state = useRef({});
  state.current = { playing, at, config, probe, pinnedLens, autoStop, stopped };
  // the 100% rule fires once: after a visitor says "go on", it waits until accuracy dips again
  const armed = useRef({ perfect: true });

  const lens = hoverLens ?? pinnedLens;
  store.current.lens = lens;
  store.current.classes = config.classes;
  store.current.layers = config.layers;
  store.current.probe = probe;

  // ---------- data ----------

  const setPoints = useCallback((next, { record = false, keepPristine = false } = {}) => {
    const s = store.current;
    if (record) {
      undo.current.past.push(s.points);
      if (undo.current.past.length > 120) undo.current.past.shift();
      undo.current.future = [];
      setUndoTick((n) => n + 1);
    }
    const annotated = annotate(next.slice(0, MAX_POINTS), state.current.config.split);
    s.points = annotated;
    live.current.dataDirty = true;
    setPointsState(annotated);
    setStopped(null);
    if (!keepPristine) setPristine(false);
  }, []);

  /** Edits from the pen: the first one in a stroke is what undo goes back to. */
  const onEdit = useCallback((next) => {
    if (!undo.current.stroke) {
      undo.current.stroke = true;
      setPoints(next, { record: true });
    } else {
      setPoints(next);
    }
  }, [setPoints]);
  const onStrokeEnd = useCallback(() => { undo.current.stroke = false; }, []);

  const loadSet = useCallback((c, record = true) => {
    setPoints(setPoints0(c), { record, keepPristine: true });
    setPristine(true);
  }, [setPoints]);

  const updateConfig = useCallback((patch) => setConfig((c) => ({ ...c, ...patch })), []);

  const chooseSet = (id) => {
    const set = SETS.find((s) => s.id === id);
    const classes = set.classes.includes(config.classes) ? config.classes : set.classes[0];
    // every pick is a fresh sample of that set, not the same one again
    const next = { ...config, set: id, classes, dataSeed: Math.floor(Math.random() * 1e6) };
    setConfig(next);
    if (active >= classes) setActive(0);
    loadSet(next);
  };

  /** Noise, samples and the dice redraw the set on screen; over a drawing they only take note. */
  const redraw = (patch) => {
    const next = { ...config, ...patch };
    setConfig(next);
    if (pristine) loadSet(next);
  };

  const changeClasses = (delta) => {
    const k = config.classes + delta;
    if (k < 2 || k > CLASSES.length) return;
    const set = SETS.find((s) => s.id === config.set);
    if (pristine && set.classes.includes(k)) {
      const next = { ...config, classes: k };
      setConfig(next);
      loadSet(next);
    } else {
      setConfig((c) => ({ ...c, classes: k }));
      if (delta < 0) setPoints(store.current.points.filter((p) => p.c < k), { record: true });
      else setPristine(false);
    }
    setActive(delta > 0 ? k - 1 : Math.min(active, k - 1));
  };

  const shake = () => {
    const sd = 0.02 + config.noise * 0.08;
    const gauss = () => Math.sqrt(-2 * Math.log(Math.random() || 1e-9)) * Math.cos(2 * Math.PI * Math.random());
    setPoints(store.current.points.map((p) => ({
      ...p, x: Math.max(-0.98, Math.min(0.98, p.x + gauss() * sd)), y: Math.max(-0.98, Math.min(0.98, p.y + gauss() * sd)),
    })), { record: true });
  };

  const doUndo = useCallback(() => {
    const u = undo.current;
    if (!u.past.length) return;
    u.future.push(store.current.points);
    setPoints(u.past.pop());
    setUndoTick((n) => n + 1);
  }, [setPoints]);
  const doRedo = useCallback(() => {
    const u = undo.current;
    if (!u.future.length) return;
    u.past.push(store.current.points);
    setPoints(u.future.pop());
    setUndoTick((n) => n + 1);
  }, [setPoints]);

  // re-flag held-out points when the split moves
  useEffect(() => {
    const s = store.current;
    s.points = annotate(s.points, config.split);
    setPointsState(s.points);
    live.current.dataDirty = true;
  }, [config.split]);

  const counts = useMemo(() => {
    const c = Array(config.classes).fill(0);
    for (const p of points) if (p.c < c.length) c[p.c] += 1;
    return c;
  }, [points, config.classes]);

  // ---------- the session ----------

  const pushData = () => {
    const pts = store.current.points;
    pgData(KEY, pts.map((p) => [p.x, p.y]), pts.map((p) => p.c), pts.map((p) => !!p.test));
    live.current.dataDirty = false;
  };

  const showView = useCallback((idx) => {
    const v = pgView(KEY, idx, VIEW_N, true);
    const s = store.current;
    s.view = { n: v.n, sizes: v.sizes, acts: v.acts, weights: v.weights, biases: v.biases };
    s.viewStamp += 1;
    s.field = { field: v.field, confidence: v.confidence, n: 72 };
    s.fieldStamp += 1;
    setView(s.view);
    return v;
  }, []);

  const arch = archKey(config);
  useEffect(() => {
    const c = state.current.config;
    const pts = store.current.points;
    try {
      pgClose(KEY);
      const meta = pgOpen(KEY, {
        classes: c.classes, layers: c.layers, optimizer: c.optimizer, lr: c.lr, batch: c.batch,
        seed: c.seed, grid_n: 72,
        points: pts.map((p) => [p.x, p.y]), labels: pts.map((p) => p.c), test: pts.map((p) => !!p.test),
      });
      live.current = { ...live.current, open: true, history: meta.history, dataDirty: false, last: null };
      store.current.diverged = false;
      setHistory(meta.history.slice());
      setNow({ epoch: 0, ...meta.history[0] });
      setAt(null);
      setWave(null);
      setStopped(null);
      armed.current.perfect = true;
      // a pinned unit that the new shape no longer has is let go, and so is the hovered one
      const sizes = [...c.layers.map((x) => x.units), c.classes];
      setPinnedLens((l) => (l && l.layer < sizes.length && l.unit < sizes[l.layer] ? l : null));
      setHoverLens(null);
      if (state.current.probe) {
        const p = state.current.probe;
        live.current.probeAsk = { x: p.x, y: p.y, pinned: p.pinned };
      }
      showView(-1);
    } catch (e) {
      console.error("the playground could not open its network", e);
      live.current.open = false;
    }
    return undefined;
  }, [arch, showView]);

  useEffect(() => () => { try { pgClose(KEY); } catch { /* gone */ } }, []);

  // how it learns can change without forgetting what it has learnt
  const firstSet = useRef(true);
  useEffect(() => {
    if (firstSet.current) { firstSet.current = false; return; }
    if (!live.current.open) return;
    pgSet(KEY, { optimizer: config.optimizer, lr: config.lr, batch: config.batch });
  }, [config.optimizer, config.lr, config.batch]);

  // ---------- the loop ----------

  const absorb = useCallback((out) => {
    const s = store.current;
    const L = live.current;
    s.field = { field: out.field, confidence: out.confidence, n: 72 };
    s.fieldStamp += 1;
    if (out.history) L.history = out.history;
    else if (out.recorded?.length) L.history.push(...out.recorded);
    L.last = out;
    if (out.now?.train_acc != null && out.now.train_acc < 1) armed.current.perfect = true;
    if (out.stopped) {
      setPlaying(false);
      setStopped({ reason: out.stopped, epoch: out.epoch });
    }
    if (out.diverged && !s.diverged) {
      s.diverged = true;
      setPlaying(false);
    }
  }, []);

  const refresh = useCallback((force = false) => {
    const L = live.current;
    const t = performance.now();
    if (!force && t - L.refreshAt < REFRESH_MS) return;
    L.refreshAt = t;
    if (L.last) setNow({ epoch: L.last.epoch, ...L.last.now, learnable: L.last.learnable, diverged: L.last.diverged });
    setHistory(L.history.slice());
    if (state.current.at == null) {
      const v = pgView(KEY, -1, VIEW_N, false);
      const s = store.current;
      s.view = { n: v.n, sizes: v.sizes, acts: v.acts, weights: v.weights, biases: v.biases };
      s.viewStamp += 1;
      setView(s.view);
    }
    const p = state.current.probe;
    if (p) L.probeAsk = { x: p.x, y: p.y, pinned: p.pinned };
  }, []);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let owed = 0;             // epochs the chosen speed has earned and the engine has not yet run
    const tick = (now) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      const L = live.current;
      if (!L.open) return;
      const { playing: go, at: here, config: c } = state.current;
      try {
        const due = go && here == null && !store.current.diverged;
        if (due) owed = Math.min(owed + c.rate * dt, Math.max(1, c.rate * 0.25));
        else owed = 0;
        const epochs = Math.floor(owed);

        if (L.dataDirty) {
          pushData();
          if (here == null && (!go || epochs < 1)) {
            // paused: the verdict over new data is still the network's, so show it
            absorb(pgAdvance(KEY, 0, -1));
            refresh(true);
          }
        }
        if (due && epochs >= 1) {
          const rules = state.current.autoStop ? { perfect: armed.current.perfect, patience: PATIENCE } : null;
          const before = L.last?.epoch ?? 0;
          const out = pgAdvance(KEY, FRAME_BUDGET, 0, rules, epochs);
          // pay off what actually ran; whatever the budget cut short stays owed (within the cap)
          owed -= Math.max(1, out.epoch - before);
          absorb(out);
          // a run that has just ended gets one last full refresh, or the readings stay a beat behind
          refresh(!!out.stopped || !!out.diverged);
        }
        if (L.probeAsk) {
          const ask = L.probeAsk;
          L.probeAsk = null;
          const r = pgProbe(KEY, here ?? -1, ask.x, ask.y);
          setProbe({ x: ask.x, y: ask.y, pinned: ask.pinned, layers: r.layers, out: r.out });
        }
      } catch (e) {
        console.error("the playground stopped", e);
        L.open = false;
      }
      if (typeof window !== "undefined") window.__pg = { store: store.current, live: L };
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [absorb, refresh]);

  // ---------- time ----------

  const scrub = useCallback((idx) => {
    setPlaying(false);
    setAt(idx);
    showView(idx);
    const p = state.current.probe;
    if (p) live.current.probeAsk = { x: p.x, y: p.y, pinned: p.pinned };
  }, [showView]);

  const goLive = useCallback(() => {
    setAt(null);
    showView(-1);
    const p = state.current.probe;
    if (p) live.current.probeAsk = { x: p.x, y: p.y, pinned: p.pinned };
  }, [showView]);

  /** Resume from wherever the head is. From the past, that discards the future. */
  const resumeHere = useCallback(() => {
    const here = state.current.at;
    if (here != null) {
      pgRewind(KEY, here);
      live.current.history = live.current.history.slice(0, here + 1);
      store.current.diverged = false;
      setAt(null);
    }
  }, []);

  const togglePlay = useCallback(() => {
    if (state.current.playing) { setPlaying(false); return; }
    if (store.current.diverged && state.current.at == null) return;
    resumeHere();
    // going on after a stop is the visitor overruling it: not again for the same reason
    if (state.current.stopped?.reason === "perfect") armed.current.perfect = false;
    pgWatch(KEY);
    setStopped(null);
    setPlaying(true);
  }, [resumeHere]);

  /**
   * One epoch forward. Rewound, that is the next recorded epoch, and nothing is thrown away;
   * live, it is one epoch of real training.
   */
  const stepOnce = useCallback(() => {
    if (!live.current.open) return;
    setPlaying(false);
    setStopped(null);
    const here = state.current.at;
    if (here != null) {
      here + 1 >= live.current.history.length - 1 ? goLive() : scrub(here + 1);
      return;
    }
    const c = state.current.config;
    const train = store.current.points.filter((p) => !p.test).length;
    const steps = c.batch > 0 && !SECOND_ORDER.has(c.optimizer) ? Math.ceil(train / c.batch) : 1;
    absorb(pgAdvance(KEY, 0, Math.max(1, steps)));
    refresh(true);
  }, [absorb, refresh, scrub, goLive]);

  /** One recorded epoch back: the head moves, the network is not touched until you resume. */
  const stepBack = useCallback(() => {
    const L = live.current;
    if (!L.open || L.history.length < 2) return;
    setPlaying(false);
    const here = state.current.at ?? L.history.length - 1;
    if (here > 0) scrub(here - 1);
  }, [scrub]);

  /**
   * Back to the first epoch, keeping whatever it was doing: paused stays paused, training carries
   * on from the start. The weights are the same initial ones; new ones come from the seed.
   */
  const toStart = useCallback(() => {
    const L = live.current;
    if (!L.open) return;
    pgRewind(KEY, 0);
    setStopped(null);
    armed.current.perfect = true;
    L.history = L.history.slice(0, 1);
    store.current.diverged = false;
    setAt(null);
    setHistory(L.history.slice());
    showView(-1);
    absorb(pgAdvance(KEY, 0, -1));
    refresh(true);
  }, [absorb, refresh, showView]);

  // ---------- the probe and the forward pass ----------

  const onProbe = useCallback((p) => {
    if (wave != null) return;
    if (!p) { if (!state.current.probe?.pinned) { setProbe(null); live.current.probeAsk = null; } return; }
    live.current.probeAsk = { x: p[0], y: p[1], pinned: false };
  }, [wave]);

  const propagate = () => {
    const train = store.current.points.filter((p) => !p.test);
    const from = probe ?? (train.length ? train[Math.floor(Math.random() * train.length)] : { x: 0.3, y: 0.4 });
    live.current.probeAsk = { x: from.x, y: from.y, pinned: true };
    setWave(0);
  };

  useEffect(() => {
    if (wave == null) return undefined;
    const last = config.layers.length + 1;
    const t = setTimeout(() => setWave(wave >= last ? null : wave + 1), wave >= last ? 1600 : 340);
    return () => clearTimeout(t);
  }, [wave, config.layers.length]);

  // ---------- the architecture, edited on the schematic ----------

  const setLayers = (layers) => updateConfig({ layers });
  const onUnits = (i, d) => setLayers(config.layers.map((l, j) => (j === i ? { ...l, units: Math.max(1, Math.min(8, l.units + d)) } : l)));
  const onAct = (i, act) => setLayers(config.layers.map((l, j) => (j === i ? { ...l, act } : l)));
  const addLayer = () => config.layers.length < MAX_LAYERS &&
    setLayers([...config.layers, { units: config.layers.at(-1)?.units ?? 4, act: config.layers.at(-1)?.act ?? "tanh" }]);
  const removeLayer = () => config.layers.length > 0 && setLayers(config.layers.slice(0, -1));

  const pin = (l) => setPinnedLens((p) => (p && p.layer === l.layer && p.unit === l.unit ? null : { ...l, pinned: true }));

  // ---------- keys ----------

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.target.getAttribute?.("role") === "slider") return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); return; }
      if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); doRedo(); return; }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (k === " " && tag !== "BUTTON") { e.preventDefault(); togglePlay(); }
      else if (k === "arrowright") { e.preventDefault(); stepOnce(); }
      else if (k === "arrowleft") { e.preventDefault(); stepBack(); }
      else if (k === "r" || k === "home") toStart();
      else if (k === "b") setTool("brush");
      else if (k === "p") setTool("point");
      else if (k === "e") setTool("erase");
      else if (k === "v") setTool("move");
      else if (k === "t") setRole((r) => ({ split: "train", train: "test", test: "split" })[r]);
      else if (k === "[") setRadius((r) => Math.max(0.03, +(r - 0.02).toFixed(2)));
      else if (k === "]") setRadius((r) => Math.min(0.4, +(r + 0.02).toFixed(2)));
      else if (k === "escape") { setPinnedLens(null); setProbe(null); }
      else if (/^[1-4]$/.test(k) && Number(k) <= state.current.config.classes) setActive(Number(k) - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [doRedo, doUndo, togglePlay, stepOnce, stepBack, toStart]);

  // ---------- the tube behind everything ----------

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

  // ---------- what the page says ----------

  const counted = counts.filter((n) => n > 0).length;
  const trainCount = points.filter((p) => !p.test).length;
  const learnable = counted >= 2 && trainCount > 0 && !store.current.diverged;
  let notice = null;
  if (store.current.diverged) notice = `Diverged at epoch ${now?.epoch ?? "?"}: the loss left the numbers. Lower the learning rate, then go back to the first epoch.`;
  else if (points.length && counted < 2) notice = `Only class ${CLASSES[counts.findIndex((n) => n > 0)]?.name ?? "A"} so far. Switch class (keys 1–${config.classes}) and paint a second one.`;
  else if (points.length && trainCount === 0) notice = "Every point is held out for testing. Lower the test split so the network has something to train on.";

  // the first screen: nothing on it yet, and two ways to begin
  const starter = !points.length && (
    <div className="pg-start">
      <h2 className="pg-start-title">Give it something to learn</h2>
      <ol className="pg-start-ways">
        <li>
          <b>Draw</b>
          <span>
            Paint on this screen. Hold to fill the circle;{" "}
            switch class with <kbd>1</kbd>–<kbd>{config.classes}</kbd>. You need two classes.
          </span>
        </li>
        <li>
          <b>Or start from a set</b>
          <span className="pg-chips">
            {SETS.map((set) => (
              <button type="button" key={set.id} className="pg-chip" onClick={() => chooseSet(set.id)}>{set.name}</button>
            ))}
          </span>
        </li>
      </ol>
    </div>
  );

  const histAt = at != null ? history[at] : null;
  const shown = histAt ?? now ?? {};
  // a pinned unit can outlive its layer for one render, between the change and the reopen
  const lensValid = lens && (lens.layer === config.layers.length ? lens.unit < config.classes
    : lens.layer < config.layers.length && lens.unit < config.layers[lens.layer].units);
  const lensName = lensValid
    ? (lens.layer === config.layers.length ? `class ${CLASSES[lens.unit].name} probability` : <>h<sub>{lens.layer + 1},{lens.unit + 1}</sub> · {config.layers[lens.layer].act}</>)
    : null;
  const label = lensName
    ? <>{lensName}{lens.pinned && !hoverLens ? " · pinned, esc to release" : ""}</>
    : null;     // the output needs no caption: the epoch is in the band below

  const params = countParams(config);
  const code = useMemo(() => pythonFor(config), [config]);

  const copy = async () => {
    try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1400); }
    catch { setCopied(false); }
  };

  return (
    <div className="pg" style={VARS}>
      <canvas ref={tube} className="pg-tube" aria-hidden="true" />

      <header className="pg-head">
        <a className="look-mark" href="/">Gradopsy</a>
        <h1 className="pg-visually-hidden">Gradopsy playground</h1>
        <nav className="pg-nav" aria-label="Demos">
          <a href="/playground" aria-current="page">playground</a>
          <a href="/recognize">recognize</a>
        </nav>
        <nav className="look-source pg-source"><a href={REPO} target="_blank" rel="noreferrer">source</a></nav>
      </header>

      <main className="pg-main">
        <aside className="pg-side">
          <DataPanel
            config={config} pristine={pristine} counts={counts} active={active} tool={tool} radius={radius}
            canUndo={undo.current.past.length > 0} canRedo={undo.current.future.length > 0}
            onSet={chooseSet}
            onNoise={(v) => redraw({ noise: v })}
            onSamples={(v) => redraw({ n: v })}
            onReseed={() => redraw({ dataSeed: config.dataSeed + 1 })}
            onClasses={changeClasses} onActive={setActive} onTool={setTool} onRadius={setRadius}
            onSplit={(v) => updateConfig({ split: v })}
            role={role} onRole={setRole}
            onUndo={doUndo} onRedo={doRedo}
            onClear={() => setPoints([], { record: true })}
            onShake={shake}
          />
          <LearnPanel config={config} trainCount={trainCount} onChange={updateConfig}
                      onReseed={() => updateConfig({ seed: Math.floor(Math.random() * 10000) })} />
        </aside>

        <section className="pg-stage" aria-label="The plane">
          <Screen store={store} tool={tool} radius={radius} active={active} role={role}
                  onEdit={onEdit} onProbe={onProbe} onStrokeEnd={onStrokeEnd}
                  label={label} notice={notice} starter={starter}
                  split={{ train: trainCount, test: points.length - trainCount }} />
        </section>

        <section className="pg-right" aria-label="The network">
          <div className="pg-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={tab === "network"} className={"pg-tab" + (tab === "network" ? " is-on" : "")}
                    onClick={() => setTab("network")}>network</button>
            <button type="button" role="tab" aria-selected={tab === "code"} className={"pg-tab" + (tab === "code" ? " is-on" : "")}
                    onClick={() => setTab("code")}>code</button>
            <span className="pg-tabs-end">
              {tab === "network" ? (
                <>
                  <span className="pg-label">hidden layers</span>
                  <button type="button" className="pg-tiny" onClick={removeLayer} disabled={!config.layers.length}
                          aria-label="Remove the last hidden layer"><Icon name="minus" size={10} /></button>
                  <b className="pg-measure">{config.layers.length}</b>
                  <button type="button" className="pg-tiny" onClick={addLayer} disabled={config.layers.length >= MAX_LAYERS}
                          aria-label="Add a hidden layer"><Icon name="plus" size={10} /></button>
                  <button type="button" className="pg-chip pg-propagate" onClick={propagate} disabled={wave != null || !view}
                          title="Carry one point through the network, layer by layer">
                    <Icon name="pulse" /><span>forward pass</span>
                  </button>
                </>
              ) : (
                <button type="button" className="pg-chip" onClick={copy}>
                  <Icon name="copy" /><span>{copied ? "copied" : "copy"}</span>
                </button>
              )}
            </span>
          </div>

          {tab === "network" ? (
            <>
              <Network
                view={view} layers={config.layers} classes={config.classes} probe={probe} lens={lens} wave={wave}
                pinned={pinnedLens} points={points} at={at} engineKey={KEY}
                onLens={setHoverLens} onPin={pin} onUnits={onUnits} onAct={onAct} onStatus={setStatus}
              />
              <p className="pg-status" aria-live="polite">
                {status ?? (wave != null
                  ? `forward pass · ${wave === 0 ? "the input" : wave > config.layers.length ? "the output" : `layer ${wave}`}`
                  : probe
                    ? "Each unit is showing its value at the probe; wires carry how much of it flows on."
                    : "Point at a unit to put it on the screen; click to pin it. Point at the screen to probe.")}
              </p>
            </>
          ) : (
            <div className="pg-code">
              <p className="pg-hint">
                What this page is running, written out. The engine builds exactly these objects from the
                gradopsy package; X_train and y_train are the filled points on the screen.
              </p>
              <pre><code>{code}</code></pre>
            </div>
          )}
        </section>
      </main>

      <footer className="pg-foot">
        <div className="pg-transport">
          <button type="button" className="pg-act" onClick={togglePlay} disabled={!playing && !learnable && at == null}
                  aria-label={playing ? "Pause" : at != null ? "Resume from this epoch" : learnable ? "Train" : "Draw points first"}>
            <Icon name={playing ? "pause" : "play"} />
            <span>{playing ? "pause" : at != null ? "resume here" : learnable ? "train" : "draw first"}</span>
          </button>
          <button type="button" className="pg-chip pg-chip-icon" onClick={stepOnce} disabled={!learnable && at == null}
                  aria-label="One epoch forward" title="One epoch forward (→)">
            <Icon name="step" />
          </button>
          <button type="button" className="pg-chip pg-chip-icon" onClick={stepBack} disabled={history.length < 2 || at === 0}
                  aria-label="One epoch back" title="One epoch back (←)">
            <Icon name="back" />
          </button>
          <button type="button" className="pg-chip pg-chip-icon" onClick={toStart} disabled={history.length < 2}
                  aria-label="Back to the first epoch" title="Back to the first epoch, keeping play or pause (R)">
            <Icon name="reset" />
          </button>
          <p className="pg-keys">
            <b>space</b> train · <b>← →</b> epoch · <b>r</b> first · <b>b p e v</b> pen · <b>t</b> train/test · <b>1–4</b> class
          </p>
          <label className="pg-speed" title="Epochs per second. On big networks or small batches the machine may top out lower.">
            <span className="pg-label">speed</span>
            <input type="range" className="pg-range" min={Math.log10(RATE_MIN)} max={Math.log10(RATE_MAX)} step={0.01}
                   value={Math.log10(config.rate)} aria-label="Training speed, epochs per second"
                   aria-valuetext={`${config.rate} epochs per second`}
                   onChange={(e) => updateConfig({ rate: roundSpeed(10 ** Number(e.target.value)) })} />
            <b className="pg-measure">{config.rate}<small> ep/s</small></b>
          </label>
          <button type="button" className={"pg-chip pg-auto" + (autoStop ? " is-on" : "")} aria-pressed={autoStop}
                  onClick={() => setAutoStop((v) => !v)}
                  title={`Stop on its own at 100% training accuracy, or after ${PATIENCE} epochs without a 1% gain in loss`}>
            auto stop
          </button>
        </div>

        <div className="pg-timeline">
          <Timeline history={history} at={at} onScrub={scrub} onLive={goLive}>
            {stopped && !playing && at == null && (
              <p className="pg-stopped" role="status">
                auto stop · epoch {stopped.epoch} · {STOP_WHY[stopped.reason]}
              </p>
            )}
            {at != null && (
              <p className="pg-rewound">
                Rewound to epoch {histAt?.epoch}. Resume here and the {history.at(-1).epoch - (histAt?.epoch ?? 0)} epochs after it are discarded.
                <button type="button" className="pg-link" onClick={goLive}>back to live</button>
              </p>
            )}
          </Timeline>
        </div>

        <dl className="pg-band">
          <div><dt>epoch</dt><dd>{shown.epoch ?? 0}</dd></div>
          <div><dt>train loss</dt><dd>{fmt(shown.train_loss)}</dd></div>
          <div><dt>test loss</dt><dd>{fmt(shown.test_loss)}</dd></div>
          <div><dt>train acc</dt><dd>{pct(shown.train_acc)}<small>%</small></dd></div>
          <div><dt>test acc</dt><dd>{pct(shown.test_acc)}<small>%</small></dd></div>
          <div><dt>weights</dt><dd>{params}</dd></div>
        </dl>
      </footer>
    </div>
  );
}

