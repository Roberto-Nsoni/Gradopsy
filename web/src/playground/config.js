/*
 * Everything the visitor can set, in one object, and the one way it leaves the page: as the
 * Python that is actually running.
 */

export const SETS = [
  { id: "circles", name: "circles", classes: [2, 3, 4] },
  { id: "moons", name: "moons", classes: [2] },
  { id: "spiral", name: "spiral", classes: [2, 3, 4] },
  { id: "xor", name: "xor", classes: [2, 4] },
  { id: "gaussians", name: "gaussians", classes: [2, 3, 4] },
];

export const ACTIVATIONS = ["tanh", "relu", "sigmoid", "linear"];
export const OPTIMIZERS = ["sgd", "momentum", "rmsprop", "adam", "lbfgs", "bfgs"];
export const SECOND_ORDER = new Set(["lbfgs", "bfgs"]);
export const LR_MIN = 0.0001;
export const LR_MAX = 3;
// Training speed, in epochs per second, 1 to 100. Ten by default: slow enough to watch every cut move; the top of
// the scale is a request, not a promise: a frame never trains for more than FRAME_BUDGET ms, so a
// big network or a small batch reaches what the machine can do and the page stays responsive.
export const RATE_MIN = 1;
export const RATE_MAX = 100;
export const FRAME_BUDGET = 14;

export const MAX_LAYERS = 6;
export const MAX_UNITS = 8;
export const MAX_POINTS = 800;

export const DEFAULTS = {
  set: "circles", n: 240, noise: 0.25, dataSeed: 42, classes: 2, split: 0.3,
  layers: [{ units: 4, act: "tanh" }, { units: 4, act: "tanh" }],
  optimizer: "adam", lr: 0.03, batch: 0, seed: 42, rate: 10,
};

export const archKey = (c) => JSON.stringify([c.classes, c.layers, c.seed]);

export const countParams = (c) => {
  let fan = 2, total = 0;
  for (const l of [...c.layers, { units: c.classes }]) { total += fan * l.units + l.units; fan = l.units; }
  return total;
};

const PY_ACT = { tanh: "Tanh", relu: "ReLU", sigmoid: "Sigmoid" };

/** A learning rate as a person would type it: 0.03, 0.001, 1e-4. */
export const fmtRate = (v) => (v >= 0.001 ? String(+v.toPrecision(2)) : v.toExponential(0));

/**
 * The Python this page is running, written out. It is the same classes the engine builds in
 * `engine.py`; if this snippet did not run as it stands, the page would be lying about itself.
 */
export function pythonFor(c) {
  const acts = [...new Set(c.layers.map((l) => l.act).filter((a) => a !== "linear"))].map((a) => PY_ACT[a]);
  const second = SECOND_ORDER.has(c.optimizer);
  const opt = { sgd: "SGD", momentum: "Momentum", rmsprop: "RMSProp", adam: "Adam", lbfgs: "LBFGS", bfgs: "BFGS" }[c.optimizer];
  const mini = !second && c.batch > 0;

  const chain = [];
  let fan = 2;
  for (const l of c.layers) {
    chain.push(`    Linear(${fan}, ${l.units}),` + (l.act === "linear" ? "" : ` ${PY_ACT[l.act]}(),`));
    fan = l.units;
  }
  chain.push(`    Linear(${fan}, ${c.classes}),`);

  const lines = [
    "import numpy as np",
    "",
    `from ml.nn import Sequential, Linear${acts.length ? ", " + acts.join(", ") : ""}, CrossEntropyFromLogits`,
    `from ml.optim import ${opt}`,
    "from ml.training import Trainer",
    ...(mini ? ["from ml.data import iterate_minibatches"] : []),
    "",
    `np.random.seed(${c.seed})`,
    "model = Sequential(",
    ...chain,
    ")",
    "",
  ];

  lines.push("loss_f = CrossEntropyFromLogits()");

  lines.push(
    `trainer = Trainer(model, loss_f, ${opt}(model.parameters()${second ? "" : `, lr=${c.lr}`}))`,
    "",
    "for epoch in range(epochs):",
    ...(mini
      ? [`    for X_b, y_b in iterate_minibatches(X_train, y_train, batch_size=${c.batch}):`,
         "        trainer.step(X_b, y_b)"]
      : [`    trainer.step(X_train, y_train)${second ? "          # full batch: the line search needs it" : ""}`]),
  );

  return lines.join("\n");
}
