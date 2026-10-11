"""
Train the recognize models on MNIST with visualml, and write what the web page shows.

Nothing in the page's numbers is typed by hand: every accuracy on /recognize comes from the files
this script writes.

    python scripts/get_mnist.py                      # once
    python scripts/train_mnist.py --model mlp --seed 0
    python scripts/train_mnist.py --report           # every model x 3 seeds -> results.json

For each run it writes, into demos/public/recognize/:
    <model>.npz            final weights, float32 (only for the seed the page ships: 0)
    <model>-history.npz    ~12 snapshots spaced logarithmically in steps, float16
    <model>-log.json       training loss and a held-out accuracy at log-spaced steps
and, with --report, results.json: test accuracy over 3 seeds per model, mean and spread.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from visualml.data import load_mnist, iterate_minibatches                      # noqa: E402
from visualml.nn import (Sequential, Linear, ReLU, Conv2D, MaxPool2D, Flatten,   # noqa: E402
                   CrossEntropyFromLogits)
from visualml.optim import Adam, LinearDecay, flat_params                                  # noqa: E402
from visualml.tensor import Tensor                                            # noqa: E402
from visualml.training import Trainer                                         # noqa: E402

DATA = ROOT / "data" / "raw" / "mnist"
OUT = ROOT / "demos" / "public" / "recognize"

BATCH = 64
LR = 1e-3
LOG_POINTS = 60           # log points, spaced logarithmically in steps like the page's axis
SNAPSHOTS = 12            # the page's timeline
MONITOR = 2000            # test images used for the running accuracy in the log (a fixed slice)
SEEDS = (0, 1, 2)


def build(model):
    """The two architectures. The page rebuilds exactly these from engine.py."""
    if model == "mlp":
        return Sequential(Linear(784, 256), ReLU(), Linear(256, 128), ReLU(), Linear(128, 10))
    if model == "cnn":
        return Sequential(Conv2D(1, 8, 5, padding=2), ReLU(), MaxPool2D(2),
                          Conv2D(8, 16, 5, padding=2), ReLU(), MaxPool2D(2),
                          Flatten(), Linear(16 * 7 * 7, 10))
    raise ValueError(model)


EPOCHS = {"mlp": 15, "cnn": 10}
SHAPE = {"mlp": (-1, 784), "cnn": (-1, 1, 28, 28)}


def augment(X, rng):
    """
    Every digit resized and moved at random, so the networks learn to read a digit however large
    and wherever it is drawn, not only MNIST's centred 20x20 ones. Only the training digits are
    augmented; the test accuracy is measured on MNIST's own test set as it is.

    Each digit is scaled by s in [0.7, 1.3] about the centre and shifted by up to as much as keeps
    it inside the frame (a 20x20 digit scaled by s is 10s from centre to edge). The new image is
    built backwards: for every output pixel, the point of the original it comes from, read by
    bilinear interpolation between the four pixels around it, and 0 outside the frame.
    """
    n = len(X)
    img = X.reshape(n, 28, 28)
    s = rng.uniform(0.7, 1.3, size=(n, 1, 1))
    room = np.maximum(0, 13 - 10 * s)
    dy = rng.uniform(-1, 1, size=(n, 1, 1)) * room
    dx = rng.uniform(-1, 1, size=(n, 1, 1)) * room

    rows, cols = np.mgrid[0:28, 0:28]
    sy = (rows - 13.5 - dy) / s + 13.5
    sx = (cols - 13.5 - dx) / s + 13.5
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    wy, wx = sy - y0, sx - x0
    which = np.arange(n)[:, None, None]

    def pixel(yy, xx):
        inside = (yy >= 0) & (yy < 28) & (xx >= 0) & (xx < 28)
        return img[which, np.clip(yy, 0, 27), np.clip(xx, 0, 27)] * inside

    out = (pixel(y0, x0) * (1 - wy) * (1 - wx) + pixel(y0, x0 + 1) * (1 - wy) * wx
           + pixel(y0 + 1, x0) * wy * (1 - wx) + pixel(y0 + 1, x0 + 1) * wy * wx)
    return out.reshape(X.shape)


def evaluate(net, X, y, shape, chunk=1000):
    """Accuracy over X in chunks, and every prediction."""
    preds = []
    for i in range(0, len(X), chunk):
        preds.append(net(Tensor(X[i:i + chunk].reshape(shape))).data.argmax(axis=1))
    preds = np.concatenate(preds)
    return float((preds == y).mean()), preds


def run(model, seed, save):
    X, y = load_mnist(DATA, "train")
    Xt, yt = load_mnist(DATA, "test")
    X, Xt = X.astype(np.float64) / 255.0, Xt.astype(np.float64) / 255.0

    np.random.seed(seed)
    net = build(model)
    params = [p for _, p in net.parameters()]
    shape = SHAPE[model]
    steps_per_epoch = int(np.ceil(len(X) / BATCH))
    total = steps_per_epoch * EPOCHS[model]

    # Adam, with the rate falling linearly from LR to LR/100 over the whole run: the last epochs
    # settle instead of rattling around the minimum, which is worth a few tenths of a point here
    opt = Adam(net.parameters(), lr=LR)
    trainer = Trainer(net, CrossEntropyFromLogits(), opt, scheduler=LinearDecay(opt, LR, total))
    # snapshot and log steps: 0, then log-spaced to the end, because learning is fastest at the start
    snap_steps = sorted(set([0] + np.unique(np.geomspace(1, total, SNAPSHOTS - 1).astype(int)).tolist()))
    log_steps = set([0, total] + np.unique(np.geomspace(1, total, LOG_POINTS).astype(int)).tolist())
    # the same moved and resized test digits every run, to measure what the augmentation buys
    Xt_moved = augment(Xt, np.random.default_rng(1234))

    snaps, log, running = {}, [], []
    rng = np.random.default_rng(seed)
    k = 0
    started = time.time()

    def checkpoint():
        if k in snap_steps:
            snaps[k] = flat_params(params).astype(np.float16)
        if k in log_steps:
            acc, _ = evaluate(net, Xt[:MONITOR], yt[:MONITOR], shape)
            log.append({"step": k, "epoch": round(k / steps_per_epoch, 3),
                        "loss": round(float(np.mean(running)), 5) if running else None,
                        "acc": round(acc, 4)})
            running.clear()

    checkpoint()
    for epoch in range(EPOCHS[model]):
        for Xb, yb in iterate_minibatches(X, y, BATCH, rng=rng):
            running.append(trainer.step(augment(Xb, rng).reshape(shape), yb)["loss"])
            k += 1
            checkpoint()
        print(f"  {model} seed {seed} epoch {epoch + 1}/{EPOCHS[model]}  loss {log[-1]['loss']}  "
              f"acc@{MONITOR} {log[-1]['acc']}  {time.time() - started:.0f}s", flush=True)

    # the page loads the weights as float32, so the published number is measured on exactly those
    for p in params:
        p.data = p.data.astype(np.float32).astype(np.float64)
    test_acc, preds = evaluate(net, Xt, yt, shape)
    moved_acc, _ = evaluate(net, Xt_moved, yt, shape)
    elapsed = time.time() - started
    n_params = int(sum(p.size for p in params))
    print(f"{model} seed {seed}: test accuracy {test_acc:.4f}, moved {moved_acc:.4f} ({n_params} parameters, {elapsed:.0f}s)")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"{model}.npz", **{name: p.data.astype(np.float32) for name, p in net.parameters()})
        steps = sorted(snaps)
        np.savez(OUT / f"{model}-history.npz", steps=np.array(steps), params=np.stack([snaps[s] for s in steps]))
        (OUT / f"{model}-log.json").write_text(json.dumps({
            "model": model, "seed": seed, "steps_per_epoch": steps_per_epoch, "epochs": EPOCHS[model],
            "batch": BATCH, "lr": LR, "schedule": "linear decay to lr/100 over the run", "monitor": MONITOR, "augment": "scale 0.7-1.3, shift inside the frame, bilinear",
            "log": log, "test_accuracy": test_acc, "test_accuracy_moved": moved_acc,
            "parameters": n_params, "seconds": round(elapsed, 1)}))

    return {"seed": seed, "test_accuracy": test_acc, "test_accuracy_moved": moved_acc,
            "parameters": n_params, "seconds": round(elapsed, 1)}


def report():
    results = {}
    for model in ("mlp", "cnn"):
        runs = [run(model, seed, save=(seed == 0)) for seed in SEEDS]
        accs = np.array([r["test_accuracy"] for r in runs])
        moved = np.array([r["test_accuracy_moved"] for r in runs])
        results[model] = {"runs": runs, "mean": round(float(accs.mean()), 4),
                          "std": round(float(accs.std(ddof=1)), 4),
                          "mean_moved": round(float(moved.mean()), 4), "std_moved": round(float(moved.std(ddof=1)), 4),
                          "parameters": runs[0]["parameters"],
                          "epochs": EPOCHS[model], "shipped_seed": 0}
        print(f"== {model}: {accs.mean():.4f} ± {accs.std(ddof=1):.4f} over seeds {SEEDS}")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))


def export_test_set():
    """
    The 10 000 test digits for the page, raw uint8, gzipped — under a name that does not end in
    .gz, because static servers send .gz files with Content-Encoding: gzip and the browser would
    unpack them before the page ever saw the bytes.
    """
    import gzip
    Xt, yt = load_mnist(DATA, "test")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test-images.dat").write_bytes(gzip.compress(Xt.astype(np.uint8).tobytes(), 9))
    (OUT / "test-labels.u8").write_bytes(yt.astype(np.uint8).tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--export-test", action="store_true")
    a = ap.parse_args()
    if a.export_test:
        export_test_set()
    elif a.report:
        export_test_set()
        report()
    else:
        run(a.model, a.seed, save=a.seed == 0)
