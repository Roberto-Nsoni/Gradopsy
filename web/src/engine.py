"""
The bridge the page talks to. Everything here runs inside Pyodide, on the real visualml
package: nothing is reimplemented in JavaScript.

Every run is computed whole and then replayed by the page. That is what makes scrubbing
instant, and it is why each function returns frames rather than a live handle. Grids travel
as raw bytes; as JSON a single boundary run would be several megabytes.
"""
import numpy as np

from ml.tensor import Tensor
from ml.metrics import accuracy
from ml.nn import MLP, ReLU, Tanh, Sigmoid, MSE, CrossEntropyFromLogits
from ml.data import SURFACES, make_spirals, make_moons, make_circles, make_blobs
from ml.optim import SGD, Momentum, RMSProp, Adam, BFGS, LBFGS, Newton, flat_params
from ml.training import Trainer, Recorder, SurfaceRunner
from ml.viz import make_grid, predict_grid, trajectory_directions, loss_surface, project_trajectory

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam,
              "bfgs": BFGS, "lbfgs": LBFGS, "newton": Newton}
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

from ml.optim import set_flat_params

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
