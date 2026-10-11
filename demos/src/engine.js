// The bridge to Pyodide. There is no mathematics in this file: the real visualml package does
// all of it, inside the browser. This only carries data across.
import enginePy from "./engine.py?raw";

// Pyodide arrives through a <script> tag rather than through the bundler: the npm package
// carries Node branches a bundler has to shim and that then break in the browser.
//
// The tag is written here rather than in index.html because a classic script in the head blocks
// parsing of the body, and a deferred one still runs in document order ahead of the app module.
// Either way the CDN decided when the page could first paint, which meant a white screen where
// the wait screen — the whole point of which is to hold someone through this download — should
// have been. Now the shell paints at once and the runtime arrives underneath it.
const INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.6/full/";

function loadRuntime() {
  if (window.loadPyodide) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = `${INDEX_URL}pyodide.js`;
    tag.onload = () => resolve();
    tag.onerror = () => reject(new Error("could not reach the Pyodide CDN"));
    document.head.append(tag);
  });
}

let pyodide = null;
let booting = null;

export function boot(onProgress) {
  if (booting) return booting;

  booting = (async () => {
    onProgress?.("downloading Python and NumPy");
    await loadRuntime();
    pyodide = await window.loadPyodide({ indexURL: INDEX_URL });
    await pyodide.loadPackage(["numpy", "micropip"]);

    onProgress?.("installing visualml");
    const micropip = pyodide.pyimport("micropip");

    // The full wheel filename matters: micropip rejects anything that is not PEP 427
    // from the site's root, not the page's path: relative to `/look/wave` it would miss
    const root = new URL(import.meta.env.BASE_URL, window.location.origin);
    const wheel = (await (await fetch(new URL("wheel.txt", root))).text()).trim();
    await micropip.install(new URL(wheel, root).href);

    pyodide.runPython(enginePy);
    onProgress?.("ready");

    return pyodide;
  })();

  return booting;
}

export const ready = () => pyodide !== null;

function call(fn, ...args) {
  const py = pyodide.globals.get(fn);
  // `null` crosses as a JsNull, which is not None and fails every `is not None` guard on the
  // other side; `undefined` is the one that arrives as None.
  const raw = py(...args.map((a) => {
    if (a === null || a === undefined) return undefined;
    return typeof a === "object" ? pyodide.toPy(a) : a;
  }));
  py.destroy();

  return raw;
}

/** Python dict -> plain object, with byte fields already turned into typed arrays. */
function unwrap(raw, buffers = {}) {
  const out = raw.toJs({ dict_converter: Object.fromEntries });
  raw.destroy();

  for (const [key, kind] of Object.entries(buffers)) {
    const bytes = out[key];
    if (!bytes) continue;                       // a streamed chunk only carries its own kind
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    out[key] = kind === "u8" ? new Uint8Array(buffer) : new Float32Array(buffer);
  }

  return out;
}

export const preset = (name) => unwrap(call("preset", name));
export const presetCurve = () => unwrap(call("preset_curve"));

export const boundaryRun = (options) =>
  unwrap(call("boundary_run", options.points, options.labels, options.hidden,
    options.activation, options.optimizer, options.lr, options.steps, options.gridN, options.seed),
  { field: "u8", confidence: "u8" });

export const neuronField = (options, neuron) =>
  unwrap(call("neuron_field", options.points, options.labels, options.hidden, options.activation,
    options.optimizer, options.lr, options.steps, options.gridN, options.seed, neuron),
  { field: "f32" });

export const overfitRun = (points, hidden, steps, lr) =>
  unwrap(call("overfit_run", points, hidden, steps, lr), { curve: "f32" });

export const lensesRun = (options) =>
  unwrap(call("lenses_run", options.points, options.labels, options.hidden, options.lr,
    options.steps, options.gridN, options.surfaceN, options.seed),
  { field: "u8", confidence: "u8", surface: "f32" });

export const surfaceInfo = (name, n) => {
  const info = unwrap(call("surface_info", name, n));
  info.z = new Float32Array(unwrapBytes(call("surface_z", name, n)));

  return info;
};

export const descentRun = (surface, runners, steps, start) =>
  unwrap(call("descent_run", surface, runners, steps, start ?? null));

/**
 * A training session the page keeps open. `liveStep` advances a few iterations and hands back
 * the field as it stands, so the loop belongs to an animation frame instead of to one blocking
 * call — which is what lets the page stay alive while it trains, and what lets a drawing whose
 * points are moving have a model chasing them.
 */
export const liveOpen = (key, options) =>
  unwrap(call("live_open", key, options.points, options.labels, options.hidden,
    options.activation ?? "tanh", options.optimizer ?? "adam", options.lr ?? 0.05,
    options.gridN ?? 72, options.seed ?? 0));

export const liveStep = (key, steps, points, labels, lr) =>
  unwrap(call("live_step", key, steps, points ?? null, labels ?? null, lr ?? null),
    { field: "u8", confidence: "u8" });

export const liveClose = (key) => call("live_close", key);

/**
 * A run the page trains in slices. `runAdvance` spends about `budgetMs` and hands back the
 * keyframes it managed, so the tab never stops answering and the drawing forms as it is
 * computed instead of appearing after a freeze.
 */
export const runOpen = (key, kind, o) => {
  // The page speaks camelCase and the engine speaks snake_case; keys with nothing behind them
  // are dropped so Python's own defaults apply instead of arriving as None.
  const spec = { points: o.points, labels: o.labels, hidden: o.hidden, activation: o.activation,
                 optimizer: o.optimizer, lr: o.lr, steps: o.steps, grid_n: o.gridN,
                 surface_n: o.surfaceN, curve_n: o.curveN, seed: o.seed,
                 surface: o.surface, requested: o.requested, start: o.start };
  for (const k of Object.keys(spec)) if (spec[k] === undefined) delete spec[k];

  return unwrap(call("run_open", key, kind, spec));
};

export const runAdvance = (key, budgetMs) =>
  unwrap(call("run_advance", key, budgetMs), { field: "u8", confidence: "u8", curve: "f32", surface: "f32" });

export const runClose = (key) => call("run_close", key);

export function surfaceNames() {
  const proxy = pyodide.runPython("sorted(SURFACES)");
  const names = proxy.toJs();
  proxy.destroy();

  return names;
}

function unwrapBytes(raw) {
  const bytes = raw.toJs();
  raw.destroy();

  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

/**
 * The playground: one network the visitor builds. The page owns the data and the loop; the
 * engine owns the model, its optimizer and a history of parameter vectors that any moment of the
 * run can be re-rendered from.
 */
export const pgDataset = (kind, n, noise, classes, seed) =>
  unwrap(call("pg_dataset", kind, n, noise, classes, seed));

export const pgOpen = (key, spec) => unwrap(call("pg_open", key, spec));

export const pgData = (key, points, labels, test) => call("pg_data", key, points, labels, test);

export const pgSet = (key, spec) => call("pg_set", key, spec);

export const pgAdvance = (key, budgetMs, maxSteps = 0, rules = null, maxEpochs = 0) =>
  unwrap(call("pg_advance", key, budgetMs, maxSteps, rules, maxEpochs), { field: "u8", confidence: "u8" });

export const pgWatch = (key) => call("pg_watch", key);

export const pgView = (key, idx, n, withField = false) =>
  unwrap(call("pg_view", key, idx, n, withField), { acts: "f32", field: "u8", confidence: "u8" });

export const pgEmbed = (key, idx) => unwrap(call("pg_embed", key, idx));

export const pgProbe = (key, idx, x, y) => unwrap(call("pg_probe", key, idx, x, y));

export const pgRewind = (key, idx) => unwrap(call("pg_rewind", key, idx));

export const pgClose = (key) => call("pg_close", key);

/**
 * recognize: MNIST networks trained offline by scripts/train_mnist.py with the same engine. Byte
 * buffers go in as Uint8Arrays and come back as typed arrays; everything is computed in Python.
 */
const bytesIn = (buf) => (buf instanceof Uint8Array ? buf : new Uint8Array(buf));

export const recLoad = (model, weights) => unwrap(call("rec_load", model, bytesIn(weights)));
export const recHistory = (model, history) => unwrap(call("rec_history", model, bytesIn(history)));
export const recAt = (model, idx) => call("rec_at", model, idx);
export const recSaliency = (model, pixels, stage, index) =>
  unwrap(call("rec_saliency", model, Array.from(pixels), stage, index), { grad: "f32" });
export const recTestset = (imagesGz, labels) => unwrap(call("rec_testset", bytesIn(imagesGz), bytesIn(labels)));
export const recEmbedPart = (model, start, count) => call("rec_embed_part", model, start, count);
export const recEmbedAxes = (model, iters) => call("rec_embed_axes", model, iters);
export const recEmbed = (model) => unwrap(call("rec_embed", model), { xy: "f32" });
export const recTrace = (model, pixels) => unwrap(call("rec_trace", model, Array.from(pixels)), { data: "f32" });
export const recExplain = (model, stage, index) =>
  unwrap(call("rec_explain", model, stage, index), { patch: "f32", kernel: "f32", products: "f32", weights: "f32", inputs: "f32" });
