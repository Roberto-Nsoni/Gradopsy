"""
The bridge the page talks to. Everything here runs inside Pyodide, on the real gradopsy
package: nothing is reimplemented in JavaScript.

Every run is computed whole and then replayed by the page. That is what makes scrubbing
instant, and it is why each function returns frames rather than a live handle. Grids travel
as raw bytes; as JSON a single boundary run would be several megabytes.
"""
import numpy as np

from gradopsy.tensor import Tensor
from gradopsy.metrics import accuracy
from gradopsy.nn import MLP, ReLU, Tanh, Sigmoid, MSE, CrossEntropyFromLogits
from gradopsy.data import SURFACES, make_spirals, make_moons, make_circles, make_blobs
from gradopsy.optim import SGD, Momentum, RMSProp, Adam, BFGS, Newton, flat_params
from gradopsy.training import Trainer, Recorder, SurfaceRunner
from gradopsy.viz import make_grid, predict_grid, trajectory_directions, loss_surface, project_trajectory

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam,
              "bfgs": BFGS, "newton": Newton}
ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}
FIRST_ORDER = {"sgd", "momentum", "rmsprop", "adam"}

FRAMES = 60             # how many moments of a run the page can scrub through


def _keyframes(steps):
    """Which steps get recorded. A run is scrubbed, not watched frame by frame."""
    return sorted(set(np.linspace(0, steps - 1, min(FRAMES, steps)).astype(int).tolist()))


def _optimizer(model, key, lr):
    kwargs = {"lr": lr} if key in FIRST_ORDER else {}
    return OPTIMIZERS[key](model.parameters(), **kwargs)


# ---------- presets, for visitors who would rather not draw ----------

PRESETS = {
    "spirals": lambda: make_spirals(60, 3, noise=0.14, seed=0),
    "moons": lambda: make_moons(200, noise=0.18, seed=0),
    "circles": lambda: make_circles(200, noise=0.09, seed=0),
    "blobs": lambda: make_blobs(210, n_classes=3, cluster_std=0.55, radius=1.6, seed=0),
    "pinwheel": lambda: make_spirals(50, 4, noise=0.10, seed=3),
}


def preset_curve(n=18, noise=0.17, seed=4):
    """
    A noisy 1D sample. These numbers are tuned, not guessed: at this count and noise the
    capacity slider reads as underfit, fit, then memorise, with the held-out loss visibly
    turning back up. Cleaner data makes a wide net look harmless and the demo say nothing.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(-0.92, 0.92, n)
    ys = 0.6 * np.sin(3.4 * xs) + rng.normal(scale=noise, size=n)

    return {"points": np.c_[xs, ys].round(4).tolist()}


def preset(name):
    """Points and labels for one of the named starting sets, scaled into the unit box."""
    X, y = PRESETS[name]()
    X = X / np.abs(X).max() * 0.86

    return {"points": X.round(4).tolist(), "labels": y.astype(int).tolist()}


# ---------- boundary: a network separating drawn points ----------

def boundary_run(points, labels, hidden, activation="tanh", optimizer="adam",
                 lr=0.05, steps=240, grid_n=140, seed=0):
    """
    Trains an MLP on the given 2D points and records the decision field as it forms.

    Returns the grid as two flat byte buffers over all keyframes: the argmax class per cell,
    and the confidence (max softmax probability) that fades the field where the net is unsure.
    """
    np.random.seed(seed)
    X = np.asarray(points, dtype=float)
    y = np.asarray(labels, dtype=int)
    classes = int(y.max()) + 1

    model = MLP([2] + list(hidden) + [classes], ACTIVATIONS[activation])
    trainer = Trainer(model, CrossEntropyFromLogits(), _optimizer(model, optimizer, lr))

    _, _, grid = make_grid((-1.0, 1.0), (-1.0, 1.0), n=grid_n)
    keys = set(_keyframes(steps))

    fields, confidences, loss, acc = [], [], [], []
    for k in range(steps):
        snapshot = trainer.step(X, y)
        loss.append(snapshot["loss"])
        acc.append(accuracy(model(Tensor(X)), y))

        if k in keys:
            field, confidence = predict_grid(model, grid)
            fields.append(field.astype(np.uint8))
            confidences.append((confidence * 255).astype(np.uint8))

    return {
        "n": grid_n,
        "classes": classes,
        "frames": len(fields),
        "keys": sorted(keys),
        "field": np.concatenate(fields).tobytes(),
        "confidence": np.concatenate(confidences).tobytes(),
        "loss": [round(v, 6) for v in loss],
        "accuracy": [round(v, 4) for v in acc],
        "params": int(sum(p.size for _, p in model.parameters())),
    }


def neuron_field(points, labels, hidden, activation, optimizer, lr, steps, grid_n, seed, neuron):
    """
    One hidden unit's own output over the plane, after the same training run. Seeing the
    final boundary as a blend of simple cuts is the point of this demo.
    """
    np.random.seed(seed)
    X = np.asarray(points, dtype=float)
    y = np.asarray(labels, dtype=int)

    model = MLP([2] + list(hidden) + [int(y.max()) + 1], ACTIVATIONS[activation])
    trainer = Trainer(model, CrossEntropyFromLogits(), _optimizer(model, optimizer, lr))
    for _ in range(steps):
        trainer.step(X, y)

    _, _, grid = make_grid((-1.0, 1.0), (-1.0, 1.0), n=grid_n)
    first = list(model._modules.values())[0]
    activation_layer = ACTIVATIONS[activation]()

    unit = activation_layer(first(Tensor(grid))).data[:, neuron]

    return {"n": grid_n, "field": unit.astype(np.float32).tobytes(),
            "lo": float(unit.min()), "hi": float(unit.max())}


# ---------- overfit: capacity against honesty in one dimension ----------

def overfit_run(points, hidden, steps=2000, lr=0.015, seed=0, curve_n=220):
    """
    Fits a 1D curve through drawn points and records it as it bends. `hidden` is the whole
    argument: a small net finds the trend, a large one memorises the noise.
    """
    np.random.seed(seed)
    data = np.asarray(points, dtype=float)
    X, y = data[:, :1], data[:, 1:]

    # Half the points are held out. Overfitting is only visible against data not fitted.
    order = np.random.default_rng(seed).permutation(len(X))
    train, test = order[: max(len(X) // 2, 1)], order[max(len(X) // 2, 1):]
    if len(test) == 0:
        test = train

    model = MLP([1] + list(hidden) + [1], Tanh)
    trainer = Trainer(model, MSE(), Adam(model.parameters(), lr=lr))

    xs = np.linspace(-1.0, 1.0, curve_n).reshape(-1, 1)
    keys = set(_keyframes(steps))

    curves, train_loss, test_loss = [], [], []
    for k in range(steps):
        snapshot = trainer.step(X[train], y[train])
        train_loss.append(snapshot["loss"])
        test_loss.append(float(MSE()(model(Tensor(X[test])), Tensor(y[test])).data.item()))

        if k in keys:
            curves.append(model(Tensor(xs)).data.reshape(-1).astype(np.float32))

    return {
        "frames": len(curves),
        "keys": sorted(keys),
        "curve_n": curve_n,
        "curve": np.concatenate(curves).tobytes(),
        "train_loss": [round(v, 6) for v in train_loss],
        "test_loss": [round(v, 6) for v in test_loss],
        "train_idx": train.tolist(),
        "test_idx": test.tolist(),
        "params": int(sum(p.size for _, p in model.parameters())),
    }


# ---------- two lenses: the same run, seen twice ----------

def lenses_run(points, labels, hidden, lr=0.05, steps=240, grid_n=120, surface_n=44, seed=0):
    """
    One training run reported through both lenses at once: the decision field in the input
    space, and the loss surface around the path the parameters actually took.

    The plane comes from a PCA of the trajectory, not from random directions. Two random
    directions in a few hundred dimensions are almost orthogonal to any real path, so the run
    projected onto them collapses to a dot.
    """
    np.random.seed(seed)
    X = np.asarray(points, dtype=float)
    y = np.asarray(labels, dtype=int)
    classes = int(y.max()) + 1

    model = MLP([2] + list(hidden) + [classes], Tanh)
    loss_f = CrossEntropyFromLogits()
    trainer = Trainer(model, loss_f, Adam(model.parameters(), lr=lr))

    _, _, grid = make_grid((-1.0, 1.0), (-1.0, 1.0), n=grid_n)
    keys = set(_keyframes(steps))

    traj, fields, confidences, loss, acc = [], [], [], [], []
    for k in range(steps):
        snapshot = trainer.step(X, y)
        traj.append(snapshot["params"])
        loss.append(snapshot["loss"])
        acc.append(accuracy(model(Tensor(X)), y))

        if k in keys:
            field, confidence = predict_grid(model, grid)
            fields.append(field.astype(np.uint8))
            confidences.append((confidence * 255).astype(np.uint8))

    params = [p for _, p in model.parameters()]
    centre = flat_params(params)
    d1, d2 = trajectory_directions(traj)

    a, b = project_trajectory(traj, centre, d1, d2)
    span = float(max(np.abs(a).max(), np.abs(b).max())) * 1.35 or 1.0

    coords, Z = loss_surface(model, loss_f, X, y, centre, d1, d2, span=span, n=surface_n)

    return {
        "n": grid_n,
        "classes": classes,
        "frames": len(fields),
        "keys": sorted(keys),
        "field": np.concatenate(fields).tobytes(),
        "confidence": np.concatenate(confidences).tobytes(),
        "loss": [round(v, 6) for v in loss],
        "accuracy": [round(v, 4) for v in acc],
        "surface_n": surface_n,
        "surface": Z.astype(np.float32).tobytes(),
        "span": span,
        "path": np.c_[a, b].round(5).tolist(),
        "params": int(sum(p.size for p in params)),
    }


# ---------- descent: optimizers over a fixed landscape ----------

def surface_info(name, n=560):
    surface = SURFACES[name]
    _, _, Z = surface.grid(n)

    return {"name": surface.name, "x_range": list(surface.x_range), "y_range": list(surface.y_range),
            "start": list(surface.start), "minima": [list(m) for m in surface.minima],
            "bounded": surface.bounded, "n": n,
            "z_min": float(Z.min()), "z_max": float(Z.max())}


def surface_z(name, n=560):
    return SURFACES[name].grid(n)[2].astype(np.float32).tobytes()


def descent_run(surface_name, requested, steps=200, start=None):
    surface = SURFACES[surface_name]
    start = tuple(start) if start else None
    out = {}

    for entry in requested:
        key = entry["key"]
        kwargs = {"lr": entry.get("lr", 0.01)} if key in FIRST_ORDER else {}
        runner = SurfaceRunner(surface, OPTIMIZERS[key], start=start, **kwargs)
        rec = Recorder()

        diverged = False
        for _ in range(steps):
            rec.add(runner.step())
            if not np.isfinite(runner.position).all() or abs(runner.position[0]) > 1e12:
                diverged = True
                break

        columns = rec.columns()
        run = {"path": rec.column("params").tolist(),
               "loss": np.where(np.isfinite(rec.column("loss")), rec.column("loss"), np.nan).tolist(),
               "grad_norm": rec.column("grad_norm").tolist(),
               "diverged": diverged, "steps": len(rec)}

        for extra in ("state_velocity", "state_scale", "state_lam", "state_alpha", "state_H"):
            if extra in columns:
                run[extra[6:]] = np.nan_to_num(columns[extra]).tolist()

        out[key] = run

    return out


# ---------- live: a session the page keeps open, instead of a run computed whole ----------
#
# Every function above computes a whole run and hands back frames to replay. That is right for
# a plate you scrub, and wrong for two things: a page that freezes for a second every time a
# parameter changes, and a drawing where the data itself is moving while the model chases it.
#
# A live session keeps the model, the optimizer and the grid alive between calls, so the page
# can advance training a few steps at a time inside an animation frame. The mathematics is the
# same Trainer stepping the same MLP; only the ownership of the loop moves to the page.

LIVE = {}


def live_open(key, points, labels, hidden, activation="tanh", optimizer="adam",
              lr=0.05, grid_n=72, seed=0):
    """Open a session and return what the page needs to size its buffers."""
    np.random.seed(seed)
    X = np.asarray(points, dtype=float)
    y = np.asarray(labels, dtype=int)
    classes = int(y.max()) + 1

    model = MLP([2] + list(hidden) + [classes], ACTIVATIONS[activation])
    trainer = Trainer(model, CrossEntropyFromLogits(), _optimizer(model, optimizer, lr))
    _, _, grid = make_grid((-1.0, 1.0), (-1.0, 1.0), n=grid_n)

    LIVE[key] = {"model": model, "trainer": trainer, "grid": grid, "n": grid_n,
                 "classes": classes, "X": X, "y": y}

    return {"n": grid_n, "classes": classes,
            "params": int(sum(p.size for _, p in model.parameters()))}


def live_step(key, steps=1, points=None, labels=None, lr=None):
    """
    Advance an open session and report where it has got to.

    `points` is passed when the data has moved under the model — a dragged observation, or a
    set that drifts on its own. The model is not reset: it keeps its parameters and chases.

    `lr` is settable per call because the page wants two speeds out of one run: fast enough to
    find the first boundary from a random start in a couple of seconds, then far slower, so that
    a dragged observation is something the network is visibly working at rather than arriving at.
    """
    session = LIVE[key]

    if lr is not None:
        session["trainer"].opt.lr = float(lr)

    if points is not None:
        session["X"] = np.asarray(points, dtype=float)
        if labels is not None:
            session["y"] = np.asarray(labels, dtype=int)

    X, y = session["X"], session["y"]
    trainer = session["trainer"]

    loss = 0.0
    for _ in range(max(1, steps)):
        loss = trainer.step(X, y)["loss"]

    field, confidence = predict_grid(session["model"], session["grid"])

    return {
        "k": trainer.k,
        "loss": round(float(loss), 6),
        "accuracy": round(float(accuracy(session["model"](Tensor(X)), y)), 4),
        "field": field.astype(np.uint8).tobytes(),
        "confidence": (confidence * 255).astype(np.uint8).tobytes(),
    }


def live_close(key):
    LIVE.pop(key, None)
    return True


# ---------- streamed runs: the same mathematics, but the page keeps the loop ----------
#
# `boundary_run`, `overfit_run` and `lenses_run` above each compute a whole run inside one call.
# That is 1.5 to 3.6 seconds of a frozen tab every time a chip is clicked or a point is added,
# which is the worst thing this page does. Nothing about the mathematics needs it: the Trainer
# was steppable from the start, and only the loop was in the wrong place.
#
# A streamed run hands the loop to the browser. `run_advance` trains for a few milliseconds and
# returns the keyframes it managed, so the page stays alive and the drawing forms as it is
# computed — which is a better answer than covering the wait, because the wait *is* the subject.

import time

from gradopsy.optim import set_flat_params

RUNS = {}


def _pack(frames):
    return np.concatenate(frames).tobytes() if frames else b""


def run_open(key, kind, options):
    """Open a streamed run. Same arguments its batch twin takes, same numbers out."""
    opts = dict(options)
    seed = int(opts.get("seed", 0))
    np.random.seed(seed)

    steps = int(opts["steps"])
    keys = set() if kind == "descent" else set(_keyframes(steps))
    session = {"kind": kind, "k": 0, "steps": steps, "keys": keys, "phase": "train", "opts": opts}

    if kind == "descent":
        session.update(surface=opts["surface"], requested=list(opts["requested"]),
                       start=tuple(opts["start"]) if opts.get("start") else None, i=0)
        RUNS[key] = session

        return {"total": steps, "runners": len(session["requested"]), "params": 0}

    if kind == "overfit":
        data = np.asarray(opts["points"], dtype=float)
        X, y = data[:, :1], data[:, 1:]
        order = np.random.default_rng(seed).permutation(len(X))
        train, test = order[: max(len(X) // 2, 1)], order[max(len(X) // 2, 1):]
        if len(test) == 0:
            test = train

        model = MLP([1] + list(opts["hidden"]) + [1], Tanh)
        session.update(
            model=model, trainer=Trainer(model, MSE(), Adam(model.parameters(), lr=opts["lr"])),
            X=X, y=y, train=train, test=test, curve_n=int(opts.get("curve_n", 220)),
            xs=np.linspace(-1.0, 1.0, int(opts.get("curve_n", 220))).reshape(-1, 1))

        meta = {"curve_n": session["curve_n"], "train_idx": train.tolist(), "test_idx": test.tolist()}
    else:
        X = np.asarray(opts["points"], dtype=float)
        y = np.asarray(opts["labels"], dtype=int)
        classes = int(y.max()) + 1
        grid_n = int(opts["grid_n"])

        activation = ACTIVATIONS[opts.get("activation", "tanh")]
        model = MLP([2] + list(opts["hidden"]) + [classes], activation)
        loss_f = CrossEntropyFromLogits()
        optimizer = _optimizer(model, opts.get("optimizer", "adam"), opts["lr"])

        _, _, grid = make_grid((-1.0, 1.0), (-1.0, 1.0), n=grid_n)
        session.update(model=model, loss_f=loss_f, trainer=Trainer(model, loss_f, optimizer),
                       X=X, y=y, grid=grid, n=grid_n, classes=classes, traj=[],
                       surface_n=int(opts.get("surface_n", 46)))

        meta = {"n": grid_n, "classes": classes}

    RUNS[key] = session
    meta.update({"total": steps, "frames_total": len(session["keys"]),
                 "params": int(sum(p.size for _, p in session["model"].parameters()))})

    return meta


def run_advance(key, budget_ms=10):
    """
    Train for about `budget_ms` and report what came of it. Always advances at least one step,
    so a slow machine makes progress rather than a stall.
    """
    session = RUNS[key]
    deadline = time.monotonic() + budget_ms / 1000.0
    kind = session["kind"]

    if kind == "descent":
        return _advance_descent(session, deadline)

    if session["phase"] == "surface":
        return _advance_surface(session, deadline)

    fields, confidences, curves, keys = [], [], [], []
    loss, acc, test_loss = [], [], []

    while session["k"] < session["steps"]:
        k = session["k"]

        if kind == "overfit":
            X, y, train, test = session["X"], session["y"], session["train"], session["test"]
            loss.append(session["trainer"].step(X[train], y[train])["loss"])
            test_loss.append(float(MSE()(session["model"](Tensor(X[test])), Tensor(y[test])).data.item()))
            if k in session["keys"]:
                curves.append(session["model"](Tensor(session["xs"])).data.reshape(-1).astype(np.float32))
                keys.append(k)
        else:
            snapshot = session["trainer"].step(session["X"], session["y"])
            loss.append(snapshot["loss"])
            acc.append(accuracy(session["model"](Tensor(session["X"])), session["y"]))
            if kind == "lenses":
                session["traj"].append(snapshot["params"])
            if k in session["keys"]:
                field, confidence = predict_grid(session["model"], session["grid"])
                fields.append(field.astype(np.uint8))
                confidences.append((confidence * 255).astype(np.uint8))
                keys.append(k)

        session["k"] += 1
        if time.monotonic() >= deadline:
            break

    trained = session["k"] >= session["steps"]
    out = {"k": session["k"], "keys": keys, "done": trained,
           "loss": [round(v, 6) for v in loss]}

    if kind == "overfit":
        out.update({"curve": _pack(curves), "train_loss": out.pop("loss"),
                    "test_loss": [round(v, 6) for v in test_loss]})
    else:
        out.update({"field": _pack(fields), "confidence": _pack(confidences),
                    "accuracy": [round(v, 4) for v in acc]})

    # The loss surface needs the whole trajectory before it can pick its plane, so it is a
    # second phase rather than something interleaved with training.
    if trained and kind == "lenses":
        out["done"] = False
        _open_surface(session)

    return out


def _advance_descent(session, deadline):
    """
    One optimizer at a time. Seven of them over two hundred steps is a third of a second in one
    call, which is a third of a second the page is not answering; a runner apiece is a frame.
    """
    surface = SURFACES[session["surface"]]
    out = {}

    while session["i"] < len(session["requested"]):
        entry = dict(session["requested"][session["i"]])
        key = entry["key"]
        kwargs = {"lr": entry.get("lr", 0.01)} if key in FIRST_ORDER else {}
        runner = SurfaceRunner(surface, OPTIMIZERS[key], start=session["start"], **kwargs)
        rec = Recorder()

        diverged = False
        for _ in range(session["steps"]):
            rec.add(runner.step())
            if not np.isfinite(runner.position).all() or abs(runner.position[0]) > 1e12:
                diverged = True
                break

        out[key] = {"path": rec.column("params").tolist(),
                    "loss": np.where(np.isfinite(rec.column("loss")), rec.column("loss"), np.nan).tolist(),
                    "grad_norm": rec.column("grad_norm").tolist(),
                    "diverged": diverged, "steps": len(rec)}

        session["i"] += 1
        if time.monotonic() >= deadline:
            break

    return {"k": session["i"], "keys": [], "loss": [], "accuracy": [],
            "runs": out, "done": session["i"] >= len(session["requested"])}


def _open_surface(session):
    params = [p for _, p in session["model"].parameters()]
    centre = flat_params(params)
    d1, d2 = trajectory_directions(session["traj"])
    a, b = project_trajectory(session["traj"], centre, d1, d2)
    span = float(max(np.abs(a).max(), np.abs(b).max())) * 1.35 or 1.0

    n = session["surface_n"]
    session.update(phase="surface", params=params, centre=centre, d1=d1, d2=d2, span=span,
                   saved=flat_params(params), coords=np.linspace(-span, span, n),
                   Z=np.zeros((n, n)), row=0,
                   path=np.c_[a, b].round(5).tolist())

    return session


def _advance_surface(session, deadline):
    """
    The same sweep `loss_surface` does, a few rows at a time. Row by row is how the library
    walks it too, so the numbers are the ones the batch call would have produced.
    """
    n = session["surface_n"]
    X_t, y_t = Tensor(session["X"]), Tensor(session["y"])

    while session["row"] < n:
        b = session["coords"][session["row"]]
        for j, a in enumerate(session["coords"]):
            set_flat_params(session["params"], session["centre"] + a * session["d1"] + b * session["d2"])
            session["Z"][session["row"], j] = session["loss_f"](session["model"](X_t), y_t).data.item()

        session["row"] += 1
        if time.monotonic() >= deadline:
            break

    done = session["row"] >= n
    out = {"k": session["steps"], "keys": [], "done": done, "loss": [], "accuracy": [],
           "field": b"", "confidence": b"", "surface_row": session["row"]}

    if done:
        set_flat_params(session["params"], session["saved"])   # leave the model where we found it
        out.update({"surface_n": n, "surface": session["Z"].astype(np.float32).tobytes(),
                    "span": session["span"], "path": session["path"]})

    return out


def run_close(key):
    RUNS.pop(key, None)
    return True


# ---------- playground: a network the visitor builds, opened up neuron by neuron ----------
#
# One session per page. The page owns the data — it is drawn, sprayed and erased in the browser —
# and hands it over whenever it changes; the model keeps its parameters and chases, the way the
# cover does. Everything else is the real package: `Sequential` of `Linear` and activation
# modules, `CrossEntropyFromLogits`, the optimizers, the `Trainer`.
#
# What the page can scrub is a history of flat parameter vectors, not of pictures. A network of a
# few hundred parameters costs a few kilobytes a snapshot, and any moment of the run can be
# re-rendered exactly — every neuron, every weight — by loading it into a mirror of the model.

from gradopsy.nn import Sequential, Linear

PG = {}
PG_ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "linear": None}
PG_LINE_SEARCH = {"bfgs"}
PG_HISTORY = 2000         # snapshots kept, one per epoch until then; past it the history is thinned and the cadence doubles


def _fit_box(X, room=0.88, recentre=False):
    """
    Scale a set into the unit box. Sets built around the origin (rings, spirals, blobs on a
    circle) stay on it: re-centring them on their bounding box shifts the spiral's own centre off
    (0, 0) by however lopsided the noise happened to be. Only moons, which is not built around
    the origin, is re-centred.
    """
    X = np.asarray(X, dtype=float)
    if recentre:
        X = X - (X.max(axis=0) + X.min(axis=0)) / 2
    return X / (np.abs(X).max() or 1.0) * room


def pg_dataset(kind, n=200, noise=0.2, classes=2, seed=0):
    """
    A starting set in the unit box. `noise` runs 0..1 for every kind and is mapped onto each
    generator's own scale, so the slider means the same amount of mess whatever is on screen.
    """
    rng = np.random.default_rng(seed)
    n, classes = int(n), int(classes)

    if kind == "circles":
        # concentric rings, one per class, innermost last
        per = n // classes
        X, y = [], []
        for c in range(classes):
            r = 1.0 - c * (0.8 / max(classes - 1, 1)) if classes > 1 else 0.6
            t = rng.uniform(0, 2 * np.pi, per)
            X.append(np.c_[np.cos(t), np.sin(t)] * r)
            y.append(np.full(per, c))
        X, y = np.vstack(X), np.concatenate(y)
        X += rng.normal(scale=0.02 + noise * 0.16, size=X.shape)
        X = _fit_box(X)
    elif kind == "moons":
        X, y = make_moons(n, noise=0.02 + noise * 0.3, seed=seed)
        X = _fit_box(X, recentre=True)
    elif kind == "spiral":
        X, y = make_spirals(n // classes, classes, noise=0.02 + noise * 0.55, seed=seed)
        X = _fit_box(X)
    elif kind == "xor":
        X = rng.uniform(-0.88, 0.88, size=(n * 2, 2))
        X = X[np.abs(X).min(axis=1) > 0.04 + 0.06 * (1 - noise)][:n]   # a margin that noise erodes
        quadrant = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
        y = (X[:, 0] * X[:, 1] > 0).astype(int) if classes == 2 else np.array([0, 1, 3, 2])[quadrant] % classes
        X = np.clip(X + rng.normal(scale=noise * 0.18, size=X.shape), -0.94, 0.94)
    elif kind == "gaussians":
        X, y = make_blobs(n, n_classes=classes, cluster_std=0.18 + noise * 0.75, radius=1.2, seed=seed)
        X = _fit_box(X)
    else:
        raise ValueError(f"unknown set {kind!r}")

    return {"points": np.asarray(X).round(4).tolist(), "labels": np.asarray(y).astype(int).tolist()}


def _pg_build(layers, classes):
    """`layers` is [{"units": 4, "act": "tanh"}, ...]: one Linear per layer and its activation."""
    chain, fan = [], 2
    for layer in layers:
        units = int(layer["units"])
        chain.append(Linear(fan, units))
        act = PG_ACTIVATIONS[layer["act"]]
        if act is not None:
            chain.append(act())
        fan = units
    chain.append(Linear(fan, int(classes)))

    return Sequential(*chain)


def _pg_optimizer(s):
    model, key = s["model"], s["optimizer"]
    if key in PG_LINE_SEARCH:
        return OPTIMIZERS[key](model.parameters())
    return OPTIMIZERS[key](model.parameters(), lr=s["lr"])


def _pg_split(s):
    X, y, test = s["X_all"], s["y_all"], s["test_mask"]
    s["X"], s["y"] = X[~test], y[~test]
    s["Xt"], s["yt"] = X[test], y[test]
    s["order"], s["cursor"] = None, 0


def pg_open(key, spec):
    """Build the network the page describes and its first snapshot. `spec` mirrors the panel."""
    spec = dict(spec)
    seed = int(spec.get("seed", 0))
    np.random.seed(seed)

    classes = int(spec["classes"])
    layers = [dict(l) for l in spec["layers"]]
    model = _pg_build(layers, classes)
    mirror = _pg_build(layers, classes)

    s = {"model": model, "mirror": mirror, "layers": layers, "classes": classes,
         "optimizer": spec.get("optimizer", "adam"), "lr": float(spec.get("lr", 0.01)),
         "batch": int(spec.get("batch", 0)), "rng": np.random.default_rng(seed),
         "epoch": 0, "steps": 0, "every": 1, "history": [], "diverged": False,
         "best_loss": np.inf, "best_epoch": 0}
    s["loss_f"] = CrossEntropyFromLogits()
    s["trainer"] = Trainer(model, s["loss_f"], _pg_optimizer(s))
    s["grid"] = make_grid((-1.0, 1.0), (-1.0, 1.0), n=int(spec.get("grid_n", 72)))[2]

    PG[key] = s
    pg_data(key, spec.get("points", []), spec.get("labels", []), spec.get("test", []))
    _pg_record(s)

    return {"params": int(sum(p.size for _, p in model.parameters())),
            "history": [s["history"][-1]["m"]]}


def pg_data(key, points, labels, test):
    """New data under the same model. It is not reset; it chases."""
    s = PG[key]
    s["X_all"] = np.asarray(points, dtype=float).reshape(-1, 2)
    s["y_all"] = np.asarray(labels, dtype=int).reshape(-1)
    mask = np.asarray(test, dtype=bool).reshape(-1)
    s["test_mask"] = mask if len(mask) == len(s["X_all"]) else np.zeros(len(s["X_all"]), dtype=bool)
    _pg_split(s)
    pg_watch(key)
    return True


def pg_set(key, spec):
    """Change how it learns without touching what it has learnt."""
    s = PG[key]
    spec = dict(spec)
    if "batch" in spec:
        s["batch"] = int(spec["batch"])
        s["order"], s["cursor"] = None, 0
    rebuild = "optimizer" in spec and spec["optimizer"] != s["optimizer"]
    if "optimizer" in spec:
        s["optimizer"] = spec["optimizer"]
    if "lr" in spec:
        s["lr"] = float(spec["lr"])
        if not rebuild and s["optimizer"] not in PG_LINE_SEARCH:
            s["trainer"].opt.lr = s["lr"]
    if rebuild:
        s["trainer"] = Trainer(s["model"], s["loss_f"], _pg_optimizer(s))
    return True


def _pg_metrics(s, model):
    out = {}
    for name, X, y in (("train", s["X"], s["y"]), ("test", s["Xt"], s["yt"])):
        if len(X) == 0:
            out[name + "_loss"], out[name + "_acc"] = None, None
            continue
        logits = model(Tensor(X))
        loss = float(s["loss_f"](logits, Tensor(y)).data.item())
        out[name + "_loss"] = round(loss, 6) if np.isfinite(loss) else None
        out[name + "_acc"] = round(float(accuracy(logits, y)), 4)
    return out


def _pg_record(s):
    params = [p for _, p in s["model"].parameters()]
    entry = {"x": flat_params(params), "m": {"epoch": s["epoch"], **_pg_metrics(s, s["model"])}}
    s["history"].append(entry)
    if len(s["history"]) > PG_HISTORY:
        # keep the first and every other one after it, and record half as often from now on
        s["history"] = s["history"][::2]
        s["every"] *= 2
    return entry


def _pg_batch(s):
    """The next minibatch of the current epoch, or the whole set. Returns (X, y, epoch_finished)."""
    n = len(s["X"])
    b = s["batch"]
    if b <= 0 or b >= n or s["optimizer"] in PG_LINE_SEARCH:
        return s["X"], s["y"], True
    if s["order"] is None or s["cursor"] >= n:
        s["order"], s["cursor"] = s["rng"].permutation(n), 0
    idx = s["order"][s["cursor"]: s["cursor"] + b]
    s["cursor"] += b
    return s["X"][idx], s["y"][idx], s["cursor"] >= n


# How much better the best training loss must get, and within how many epochs, before the run
# counts as still learning. Relative, so it means the same at a loss of 0.5 and of 0.0005.
PG_PLATEAU_GAIN = 0.01


def pg_watch(key):
    """Start the plateau count again from here: new data, a rewind, or a visitor who said go on."""
    s = PG[key]
    s["best_loss"], s["best_epoch"] = np.inf, s["epoch"]
    return True


def _pg_should_stop(s, rules):
    """Checked once per epoch, on the whole training set: has it finished, or stopped getting better?"""
    logits = s["model"](Tensor(s["X"]))
    if rules.get("perfect") and accuracy(logits, s["y"]) >= 1.0:
        return "perfect"
    patience = int(rules.get("patience") or 0)
    if patience > 0:
        loss = float(s["loss_f"](logits, Tensor(s["y"])).data.item())
        if loss < s["best_loss"] * (1 - PG_PLATEAU_GAIN):
            s["best_loss"], s["best_epoch"] = loss, s["epoch"]
        elif s["epoch"] - s["best_epoch"] >= patience:
            return "plateau"
    return None


def pg_advance(key, budget_ms=8, max_steps=0, rules=None, max_epochs=0):
    """
    Train for about `budget_ms` (or exactly `max_steps`, when given) and report. Always takes
    at least one step, so a slow machine is slow rather than stuck. A negative `max_steps` takes
    none: the page wants the verdict over data that has just changed under a paused network.

    `rules`, when given, can end the run on its own at the close of an epoch: `perfect` stops it
    at 100% on the training points, `patience` after that many epochs without a 1% gain.

    `max_epochs` ends the call once that many epochs have closed, still inside the budget: the
    page asks for the epochs a frame owes at the speed the visitor chose, and no more.
    """
    s = PG[key]
    rules = dict(rules) if rules else None
    stopped = None
    recorded = []
    learnable = len(s["X"]) > 0 and len(np.unique(s["y"])) > 1

    if learnable and not s["diverged"] and max_steps >= 0:
        deadline = time.monotonic() + budget_ms / 1000.0
        taken = 0
        closed = 0
        while True:
            X, y, finished = _pg_batch(s)
            loss = s["trainer"].step(X, y)["loss"]
            s["steps"] += 1
            taken += 1
            if not np.isfinite(loss):
                s["diverged"] = True
                break
            if finished:
                s["epoch"] += 1
                closed += 1
                if s["epoch"] % s["every"] == 0:
                    before = len(s["history"])
                    entry = _pg_record(s)
                    recorded.append(entry["m"])
                    if len(s["history"]) < before:      # thinned: the page must rebuild its copy
                        recorded = None
                if rules:
                    stopped = _pg_should_stop(s, rules)
                    if stopped:
                        break
            if max_epochs and closed >= max_epochs:
                break
            if max_steps and taken >= max_steps:
                break
            if not max_steps and time.monotonic() >= deadline:
                break
            if recorded is None:
                break

    field, confidence = predict_grid(s["model"], s["grid"])
    out = {"epoch": s["epoch"], "steps": s["steps"], "diverged": s["diverged"], "stopped": stopped,
           "learnable": bool(learnable), "now": _pg_metrics(s, s["model"]),
           "field": field.astype(np.uint8).tobytes(),
           "confidence": np.nan_to_num(confidence * 255).astype(np.uint8).tobytes()}
    if recorded is None:
        out["history"] = [e["m"] for e in s["history"]]
    else:
        out["recorded"] = recorded
    return out


def _pg_at(s, idx):
    """The model as it stood at snapshot `idx` (or live, for -1)."""
    if idx is None or idx < 0 or idx >= len(s["history"]):
        return s["model"]
    set_flat_params([p for _, p in s["mirror"].parameters()], s["history"][idx]["x"])
    return s["mirror"]


def _pg_forward(model, X):
    """Every layer's post-activation output on X, plus the output's softmax. One list per layer."""
    a = Tensor(X)
    layers = []
    mods = list(model._modules.values())
    for i, m in enumerate(mods):
        a = m(a)
        nxt = mods[i + 1] if i + 1 < len(mods) else None
        if isinstance(m, Linear) and isinstance(nxt, Linear):
            layers.append(a.data)                   # a linear layer with no activation after it
        elif not isinstance(m, Linear):
            layers.append(a.data)
    probs = np.exp(a.data - a.data.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return layers, probs


def pg_view(key, idx=-1, n=40, with_field=False):
    """
    Everything the schematic draws at one moment: each neuron's activation over the plane, every
    weight and bias, and (when asked) the output field for the screen.
    """
    s = PG[key]
    model = _pg_at(s, idx)
    _, _, grid = make_grid((-1.0, 1.0), (-1.0, 1.0), n=int(n))
    hidden, probs = _pg_forward(model, grid)

    planes = [h.T.astype(np.float32) for h in hidden] + [probs.T.astype(np.float32)]
    linears = [m for m in model._modules.values() if isinstance(m, Linear)]

    out = {"n": int(n),
           "sizes": [int(h.shape[1]) for h in hidden] + [int(probs.shape[1])],
           "acts": np.concatenate([p.reshape(-1) for p in planes]).tobytes(),
           "weights": [np.nan_to_num(l.W.data).round(5).tolist() for l in linears],
           "biases": [np.nan_to_num(l.b.data).round(5).tolist() for l in linears],
           "epoch": s["history"][idx]["m"]["epoch"] if 0 <= idx < len(s["history"]) else s["epoch"]}
    if with_field:
        field, confidence = predict_grid(model, s["grid"])
        out["field"] = field.astype(np.uint8).tobytes()
        out["confidence"] = np.nan_to_num(confidence * 255).astype(np.uint8).tobytes()
    return out


def pg_embed(key, idx=-1):
    """
    The data as each hidden layer sees it: every point's activations in that layer, brought
    down to two coordinates. A layer of one or two units is shown exactly; a wider one through
    its first two principal components, whose share of the variance is reported so the picture
    can say how much of the layer it is showing.

    Principal axes have no sign of their own, so each is flipped to agree with the one this
    session showed last time, or the picture would mirror itself from one frame to the next.
    """
    s = PG[key]
    X = s["X_all"]
    if len(X) == 0:
        return {"layers": []}
    model = _pg_at(s, idx)
    hidden, _ = _pg_forward(model, X)
    # the output's own space, before the softmax: one number per class. With two classes it is a
    # plane, drawn whole, and the network's verdict there is the straight line where they are equal
    logits = model(Tensor(X)).data
    memory = s.setdefault("embed_axes", {})
    out = []
    for i, h in enumerate(hidden + [logits]):
        units = h.shape[1]
        if units <= 2:
            xy = h if units == 2 else np.c_[h[:, 0], np.zeros(len(h))]
            share, exact = 1.0, True
        else:
            centred = h - h.mean(axis=0)
            _, sv, vt = np.linalg.svd(centred, full_matrices=False)
            axes = vt[:2]
            prev = memory.get((i, units))
            if prev is not None:
                for k in range(2):
                    if axes[k] @ prev[k] < 0:
                        axes[k] = -axes[k]
            memory[(i, units)] = axes.copy()
            xy = centred @ axes.T
            share = float((sv[:2] ** 2).sum() / max((sv ** 2).sum(), 1e-12))
            exact = False
        if exact:
            mean = np.zeros(units)
            axes = np.eye(2, units)                 # one unit: the second axis is empty
        else:
            mean = h.mean(axis=0)
        span = float(np.abs(xy).max()) or 1.0
        # With mean, axes and span the page can map any spot of this picture back into the layer
        # (h = mean + span * (u * axes[0] + v * axes[1])) and ask the next layer about it; `acts`
        # is every point's own value in every unit, which the picture alone cannot give exactly.
        out.append({"xy": np.nan_to_num(xy / span).round(4).tolist(), "units": int(units),
                    "share": round(share, 3), "exact": exact,
                    "mean": np.nan_to_num(mean).round(5).tolist(), "axes": np.nan_to_num(axes).round(5).tolist(),
                    "span": span, "acts": np.nan_to_num(h).round(4).tolist()})
    return {"layers": out}


def pg_probe(key, idx, x, y):
    """One input carried through the network: the activation of every neuron it meets."""
    s = PG[key]
    hidden, probs = _pg_forward(_pg_at(s, idx), np.array([[float(x), float(y)]]))
    return {"layers": [np.nan_to_num(h[0]).round(4).tolist() for h in hidden],
            "out": np.nan_to_num(probs[0]).round(4).tolist()}


def pg_rewind(key, idx):
    """Resume from a past snapshot: the future is discarded and the optimizer starts fresh."""
    s = PG[key]
    entry = s["history"][idx]
    set_flat_params([p for _, p in s["model"].parameters()], entry["x"])
    s["history"] = s["history"][: idx + 1]
    s["epoch"] = entry["m"]["epoch"]
    s["diverged"] = False
    s["best_loss"], s["best_epoch"] = np.inf, s["epoch"]
    s["order"], s["cursor"] = None, 0
    s["trainer"] = Trainer(s["model"], s["loss_f"], _pg_optimizer(s))
    return {"epoch": s["epoch"]}


def pg_close(key):
    PG.pop(key, None)
    return True


# ---------- recognize: MNIST, with weights trained offline by scripts/train_mnist.py ----------
#
# Training sixty thousand digits is minutes of work and does not belong in a tab, so the networks
# arrive trained — by the same engine, with the script in the repository. Everything the page
# shows from then on is computed here, on those weights: every prediction, every value of every
# stage, and every gradient with respect to a pixel.

import gzip
import io

from gradopsy.nn import Conv2D as _Conv2D, MaxPool2D as _MaxPool2D, Flatten as _Flatten

REC = {}
REC_TEST = {}


def _rec_build(model):
    """The two architectures, exactly as scripts/train_mnist.py builds them."""
    if model == "mlp":
        return Sequential(Linear(784, 256), ReLU(), Linear(256, 128), ReLU(), Linear(128, 10))
    return Sequential(_Conv2D(1, 8, 5, padding=2), ReLU(), _MaxPool2D(2),
                      _Conv2D(8, 16, 5, padding=2), ReLU(), _MaxPool2D(2),
                      _Flatten(), Linear(16 * 7 * 7, 10))


def _rec_shape(model):
    return (-1, 784) if model == "mlp" else (-1, 1, 28, 28)


def rec_load(model, weights_bytes):
    """Build the network and load its shipped weights (an .npz of float32 arrays by name)."""
    np.random.seed(0)
    net = _rec_build(model)
    saved = np.load(io.BytesIO(bytes(weights_bytes)))
    for name, p in net.parameters():
        p.data = saved[name].astype(np.float64)
    params = [p for _, p in net.parameters()]
    # trained, so only ever used for predictions: no graph through the weights. The list above is
    # taken first, because a frozen model's parameters() skips its (now non-trainable) tensors
    net.freeze()
    REC[model] = {"net": net, "params": params, "final": flat_params(params), "history": None, "at": -1}
    return {"parameters": int(sum(p.size for p in params))}


def rec_history(model, history_bytes):
    """The training snapshots, for the timeline: float16 parameter vectors at the given steps."""
    saved = np.load(io.BytesIO(bytes(history_bytes)))
    REC[model]["history"] = {"steps": saved["steps"].astype(int), "params": saved["params"]}
    return {"steps": saved["steps"].astype(int).tolist()}


def rec_at(model, idx):
    """Put the network at snapshot `idx` of its history, or back to the shipped weights for -1."""
    s = REC[model]
    if idx == s["at"]:
        return True
    vec = s["final"] if idx < 0 else s["history"]["params"][idx].astype(np.float64)
    set_flat_params(s["params"], vec)
    s["at"] = idx
    return True


def _rec_layers(model, x):
    """Every layer's output for one input batch: the activations the page draws."""
    out = []
    a = x
    for m in REC[model]["net"]._modules.values():
        a = m(a)
        out.append(a)
    return out


def rec_saliency(model, pixels, stage, index):
    """
    How much each pixel pushes one value of the network up or down: the gradient, with respect to
    the input, of the value at flat `index` of module `stage`'s output, by the engine's own
    backward pass. The value is picked out by multiplying with a one-hot mask and summing, so no
    indexing op is needed. For the last module and a digit it is that digit's logit.
    """
    x = Tensor(np.asarray(pixels, dtype=float).reshape(_rec_shape(model)), requires_grad=True)
    a = x
    for s, m in enumerate(REC[model]["net"]._modules.values()):
        a = m(a)
        if s == stage:
            break
    pick = np.zeros(a.shape)
    pick.reshape(-1)[int(index)] = 1.0
    (a * Tensor(pick)).sum().backward()
    return {"grad": x.grad.reshape(-1).astype(np.float32).tobytes()}


def rec_testset(images_gz, labels):
    """The 10 000 MNIST test digits, as the page fetched them: gzipped uint8 and uint8 labels."""
    X = np.frombuffer(gzip.decompress(bytes(images_gz)), dtype=np.uint8).reshape(-1, 28, 28)
    y = np.frombuffer(bytes(labels), dtype=np.uint8).astype(int)
    REC_TEST.update(X=X, y=y)
    return {"n": int(len(y))}


def rec_embed_part(model, start, count):
    """
    Run test digits start..start+count through the network and keep their last hidden layer.
    The page asks for the picture a few digits at a time, between frames, because two thousand
    digits through the CNN in one go holds the tab for several seconds.
    """
    if start == 0:
        REC[model]["embed_h"] = []
        REC[model]["embed_A"] = None
    batch = REC_TEST["X"][start:start + count].astype(np.float64).reshape(_rec_shape(model)) / 255.0
    last = 3 if model == "mlp" else 6
    REC[model]["embed_h"].append(_rec_layers(model, Tensor(batch))[last].data)
    return True


def rec_embed_axes(model, iters):
    """
    `iters` more steps towards the two directions of largest variance of the digits run so far.

    Only those two are needed, so instead of a full SVD (which takes over ten seconds in the
    browser's NumPy) this is subspace iteration: multiply two vectors by the covariance hc.T @ hc
    and orthonormalise them, again and again, until they settle on its top two eigenvectors. Like
    the digits, the steps are taken a few at a time between frames.
    """
    s = REC[model]
    if s.get("embed_A") is None:
        h = np.concatenate(s["embed_h"])
        s["embed_mean"] = h.mean(axis=0)
        s["embed_hc"] = h - s["embed_mean"]
        s["embed_A"] = np.random.RandomState(0).randn(h.shape[1], 2)
    hc, A = s["embed_hc"], s["embed_A"]
    for _ in range(iters):
        A, _ = np.linalg.qr(hc.T @ (hc @ A))
    s["embed_A"] = A
    return True


def rec_embed(model):
    """
    The test digits run so far, in the network's last hidden layer, through its PCA: where the
    network puts each kind of digit. Also the projection itself, so a drawn digit can be placed.
    """
    s = REC[model]
    hc, A, mean = s["embed_hc"], s["embed_A"], s["embed_mean"]
    y = REC_TEST["y"][:len(hc)]
    # the iteration found the right plane; a 2x2 eigenproblem turns the two vectors to the exact
    # axes inside it
    lam, R = np.linalg.eigh((hc @ A).T @ (hc @ A))
    axes = (A @ R[:, ::-1]).T
    memory = REC[model].setdefault("embed_axes", None)
    if memory is not None:
        for k in range(2):
            if axes[k] @ memory[k] < 0:
                axes[k] = -axes[k]
    REC[model]["embed_axes"] = axes.copy()
    xy = hc @ axes.T
    span = float(np.abs(xy).max()) or 1.0
    return {"xy": (xy / span).astype(np.float32).tobytes(), "labels": y.tolist(),
            "mean": mean.round(5).tolist(), "axes": axes.round(6).tolist(), "span": span,
            "share": round(float(lam.sum() / max((hc ** 2).sum(), 1e-12)), 3)}


# ---------- recognize: the forward pass, read ----------
#
# `rec_trace` runs one digit through the network and keeps every stage's output, so the page can
# draw them all. `rec_explain` then answers "how was this one number computed?" with the numbers
# themselves: for a convolution, the patch it looked at, the kernel, their products and the sum;
# for a linear unit, every input times its weight; for a max-pool, the block it chose from. It also
# says which pixels of the input that number can see at all (its receptive field).

def _rec_stages(model):
    """What each module does, in order: the page labels and draws the stages from this."""
    out = []
    for m in REC[model]["net"]._modules.values():
        if isinstance(m, _Conv2D):
            out.append({"kind": "conv", "k": int(m.W.shape[2]), "padding": int(m.padding)})
        elif isinstance(m, _MaxPool2D):
            out.append({"kind": "pool", "k": int(m.k)})
        elif isinstance(m, _Flatten):
            out.append({"kind": "flatten"})
        elif isinstance(m, Linear):
            out.append({"kind": "linear"})
        else:
            out.append({"kind": type(m).__name__.lower()})
    return out


def rec_trace(model, pixels):
    """Every stage of one digit's forward pass. `data` holds them one after another as float32."""
    x = np.asarray(pixels, dtype=float).reshape(_rec_shape(model))
    acts = [a.data[0] for a in _rec_layers(model, Tensor(x))]
    REC[model]["trace"] = {"input": x[0], "acts": acts}
    logits = acts[-1]
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    stages = _rec_stages(model)
    for st, a in zip(stages, acts):
        st["shape"] = list(a.shape)
    return {"stages": stages, "data": np.concatenate([a.reshape(-1) for a in acts]).astype(np.float32).tobytes(),
            "logits": logits.round(5).tolist(), "probs": probs.round(6).tolist(),
            "last": acts[-2].reshape(-1).round(5).tolist()}


def _rec_fields(model, stage, index):
    """
    Which values of every earlier stage the number at flat `index` of `stage`'s output depends on,
    walking back one module at a time: a k x k convolution with padding p sees a box k - 1 wider in
    every input map, a k x k pool sees its block in the same map, a ReLU sees the same value, a
    flatten turns a position back into (map, row, column), and a linear unit sees everything.

    Returns (fields, field): fields[s] is the region of module s's output, for every s < stage, as
    {"cs": maps or "all", "box": [r0, r1, c0, c1]} (rows and columns, inclusive) or "all"; field is
    the box of input pixels on the 28x28 digit.
    """
    stages = _rec_stages(model)
    acts = REC[model]["trace"]["acts"]
    shape = acts[stage].shape
    if len(shape) == 3:
        C, H, W = shape
        i, j = (index // W) % H, index % W
        reg = {"cs": [int(index // (H * W))], "box": [int(i), int(i), int(j), int(j)]}
    else:
        reg = {"n": int(index)}

    fields = [None] * stage
    for s in range(stage, -1, -1):
        kind = stages[s]["kind"]
        if reg == "all" or kind == "linear":
            reg = "all"
        elif kind == "conv":
            k, p = stages[s]["k"], stages[s]["padding"]
            r0, r1, c0, c1 = reg["box"]
            reg = {"cs": "all", "box": [r0 - p, r1 - p + k - 1, c0 - p, c1 - p + k - 1]}
        elif kind == "pool":
            k = stages[s]["k"]
            r0, r1, c0, c1 = reg["box"]
            reg = {"cs": reg["cs"], "box": [r0 * k, r1 * k + k - 1, c0 * k, c1 * k + k - 1]}
        elif kind == "flatten":
            C, H, W = acts[s - 1].shape
            n = reg["n"]
            reg = {"cs": [n // (H * W)], "box": [(n // W) % H, (n // W) % H, n % W, n % W]}
        # a relu changes nothing about where a value comes from
        if s > 0:
            fields[s - 1] = _rec_clip(reg, acts[s - 1].shape)

    if reg == "all" or "box" not in reg:
        field = [0, 27, 0, 27]
    else:
        field = _rec_clip(reg, (1, 28, 28))["box"]
    return fields, field


def _rec_clip(reg, shape):
    """A region cut to the tensor it lives in (padding reaches past the edges)."""
    if reg == "all" or "box" not in reg:
        return reg
    H, W = shape[1], shape[2]
    r0, r1, c0, c1 = reg["box"]
    return {"cs": reg["cs"], "box": [max(0, r0), min(H - 1, r1), max(0, c0), min(W - 1, c1)]}


def rec_explain(model, stage, index):
    """How the number at flat `index` of `stage`'s output (for the last traced digit) came about."""
    tr = REC[model]["trace"]
    mods = list(REC[model]["net"]._modules.values())
    m = mods[stage]
    out = tr["acts"][stage]
    prev = tr["input"] if stage == 0 else tr["acts"][stage - 1]
    value = float(out.reshape(-1)[index])
    fields, field = _rec_fields(model, stage, index)
    res = {"value": round(value, 5), "fields": fields, "field": field}

    if isinstance(m, _Conv2D):
        O, H, W = out.shape
        o, i, j = index // (H * W), (index // W) % H, index % W
        k, p = m.W.shape[2], m.padding
        src = prev if prev.ndim == 3 else prev.reshape(1, 28, 28)
        padded = np.pad(src, ((0, 0), (p, p), (p, p)))
        patch = padded[:, i:i + k, j:j + k]
        kernel = m.W.data[o]
        products = patch * kernel
        res.update(kind="conv", o=int(o), i=int(i), j=int(j), k=int(k), channels=int(src.shape[0]),
                   patch=patch.astype(np.float32).tobytes(), kernel=kernel.astype(np.float32).tobytes(),
                   products=products.astype(np.float32).tobytes(),
                   sum=round(float(products.sum()), 5), bias=round(float(m.b.data[o]), 5))
    elif isinstance(m, _MaxPool2D):
        C, H, W = out.shape
        c, i, j = index // (H * W), (index // W) % H, index % W
        k = m.k
        block = prev[c, i * k:(i + 1) * k, j * k:(j + 1) * k]
        res.update(kind="pool", c=int(c), i=int(i), j=int(j), k=int(k),
                   block=block.reshape(-1).round(5).tolist())
    elif isinstance(m, Linear):
        a = prev.reshape(-1)
        w = m.W.data[:, index]
        products = a * w
        order = np.argsort(-np.abs(products))[:8]
        res.update(kind="linear", unit=int(index), n_in=int(len(a)),
                   products=products.astype(np.float32).tobytes(), weights=w.astype(np.float32).tobytes(),
                   inputs=a.astype(np.float32).tobytes(),
                   top=[[int(t), round(float(a[t]), 4), round(float(w[t]), 4)] for t in order],
                   sum=round(float(products.sum()), 5), bias=round(float(m.b.data[index]), 5))
    else:
        # relu or flatten: the same number as before, or zero
        res.update(kind=type(m).__name__.lower(), before=round(float(prev.reshape(-1)[index]), 5))
    return res
