import numpy as np

from gradopsy.data import SURFACES
from gradopsy.optim import Adam, SGD, BFGS
from gradopsy.training import SurfaceRunner, Recorder


def record(optimizer_class=Adam, steps=50, every=1, **kwargs):
    runner = SurfaceRunner(SURFACES["rosenbrock"], optimizer_class, **kwargs)
    rec = Recorder(every=every)

    for _ in range(steps):
        rec.add(runner.step())

    return rec


def test_records_every_step():
    assert len(record(steps=50, lr=0.05)) == 50


def test_every_subsamples():
    """On big models you cannot keep one frame per step, so `every` thins the recording."""
    assert len(record(steps=50, every=10, lr=0.05)) == 5


def test_add_returns_the_snapshot_so_it_chains():
    runner = SurfaceRunner(SURFACES["rosenbrock"], SGD, lr=1e-3)
    rec = Recorder()

    assert rec.add(runner.step())["k"] == 1


def test_indexing_and_column():
    rec = record(steps=30, lr=0.05)

    assert rec[0]["k"] == 1
    assert rec[-1]["k"] == 30
    assert rec.column("loss").shape == (30,)
    assert rec.column("params").shape == (30, 2)
    assert rec.column("loss")[-1] < rec.column("loss")[0]


def test_columns_flattens_the_optimizer_state():
    columns = record(steps=20, lr=0.05).columns()

    assert set(columns) == {"k", "loss", "lr", "grad_norm", "params", "state_velocity", "state_scale"}
    assert columns["state_velocity"].shape == (20, 2)


def test_save_round_trip(tmp_path):
    rec = record(steps=20, lr=0.05)
    path = tmp_path / "run.npz"
    rec.save(path)

    saved = np.load(path)
    assert np.allclose(saved["params"], rec.column("params"))
    assert np.allclose(saved["loss"], rec.column("loss"))


def test_sgd_records_no_state():
    """SGD carries nothing, and that emptiness is the contrast the demo is built on."""
    columns = record(SGD, steps=10, lr=1e-3).columns()

    assert set(columns) == {"k", "loss", "lr", "grad_norm", "params"}


def test_bfgs_records_its_own_state():
    columns = record(BFGS, steps=10).columns()

    assert columns["state_H"].shape == (10, 2, 2)
    assert columns["state_alpha"].shape == (10,)
