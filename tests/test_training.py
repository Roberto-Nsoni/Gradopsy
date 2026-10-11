import numpy as np

from visualml.tensor import Tensor
from visualml.metrics import accuracy
from visualml.nn import MLP, ReLU, Tanh, CrossEntropyFromLogits
from visualml.optim import SGD, Adam, BFGS, LinearDecay, flat_params
from visualml.data import make_spirals, make_moons, iterate_minibatches, from_points
from visualml.training import Trainer, EarlyStopping
from visualml.viz import make_grid


def spirals():
    np.random.seed(0)
    return make_spirals(100, 3, noise=0.15, seed=0)


def test_full_batch_training_learns_spirals():
    X, y = spirals()
    model = MLP([2, 16, 16, 3], ReLU)
    trainer = Trainer(model, CrossEntropyFromLogits(), Adam(model.parameters(), lr=0.05))

    for _ in range(400):
        snapshot = trainer.step(X, y)

    assert snapshot["k"] == 400
    assert accuracy(model(Tensor(X)), y) > 0.9


def test_bfgs_learns_spirals_in_far_fewer_steps():
    """Second order on full batch: the point of the whole Stage 2.5."""
    X, y = spirals()
    model = MLP([2, 16, 16, 3], Tanh)
    trainer = Trainer(model, CrossEntropyFromLogits(), BFGS(model.parameters()))

    for _ in range(60):
        trainer.step(X, y)

    assert accuracy(model(Tensor(X)), y) > 0.9


def test_snapshot_contents():
    X, y = spirals()
    model = MLP([2, 8, 3], ReLU)
    trainer = Trainer(model, CrossEntropyFromLogits(), SGD(model.parameters(), lr=0.1))

    _, _, grid = make_grid((-1, 1), (-1, 1), n=10)
    snapshot = trainer.step(X, y, grid=grid)

    assert snapshot["loss"] > 0
    assert snapshot["lr"] == 0.1
    assert snapshot["grad_norm"] > 0
    assert snapshot["params"].shape == flat_params([p for _, p in model.parameters()]).shape
    assert snapshot["labels"].shape == (100,)
    assert snapshot["confidence"].shape == (100,)


def test_snapshot_has_no_grid_when_not_asked():
    X, y = spirals()
    model = MLP([2, 8, 3], ReLU)
    trainer = Trainer(model, CrossEntropyFromLogits(), SGD(model.parameters(), lr=0.1))

    assert "labels" not in trainer.step(X, y)


def test_epoch_walks_the_whole_dataset():
    X, y = spirals()
    model = MLP([2, 8, 3], ReLU)
    trainer = Trainer(model, CrossEntropyFromLogits(), SGD(model.parameters(), lr=0.1))

    loss = trainer.epoch(X, y, batch_size=32, rng=np.random.default_rng(0))

    assert trainer.k == int(np.ceil(300 / 32))
    assert loss > 0


def test_minibatches_cover_every_sample_once():
    X, y = spirals()
    seen = np.concatenate([y_b for _, y_b in iterate_minibatches(X, y, 32, rng=np.random.default_rng(0))])

    assert len(seen) == len(y)
    assert sorted(seen) == sorted(y)


def test_sgd_with_linear_decay_and_early_stopping():
    """The stochastic gradient method of the MATLAB, assembled from its three pieces."""
    X, y = make_moons(400, noise=0.2, seed=0)

    perm = np.random.default_rng(0).permutation(len(X))     # make_moons comes sorted by class
    X, y = X[perm], y[perm]
    X_tr, y_tr, X_te, y_te = X[:300], y[:300], X[300:], y[300:]

    m, emax, beta = 32, 40, 0.6
    k_max = emax * int(np.ceil(len(X_tr) / m))

    model = MLP([2, 8, 2], Tanh)
    loss_f = CrossEntropyFromLogits()
    opt = SGD(model.parameters(), lr=0.5)
    trainer = Trainer(model, loss_f, opt, scheduler=LinearDecay(opt, alpha_0=0.5, k_ld=int(beta * k_max)))

    stopper = EarlyStopping(patience=5)
    params = [p for _, p in model.parameters()]
    rng = np.random.default_rng(0)

    for _ in range(emax):
        trainer.epoch(X_tr, y_tr, m, rng=rng)
        test_loss = loss_f(model(Tensor(X_te)), Tensor(y_te)).data.item()
        if stopper.update(test_loss, params):
            break

    stopper.restore(params)

    assert stopper.best_epoch >= 1
    assert accuracy(model(Tensor(X_te)), y_te) > 0.85


# ------- Early stopping on its own --------

def test_early_stopping_keeps_the_best_epoch():
    p = Tensor([1.0], requires_grad=True)
    params = [p]
    stopper = EarlyStopping(patience=2)

    assert not stopper.update(1.0, params)      # best so far
    p.data = np.array([2.0])
    assert not stopper.update(2.0, params)      # 1 epoch without improving
    p.data = np.array([3.0])
    assert stopper.update(3.0, params)          # 2 in a row -> stop

    assert stopper.best_epoch == 1
    assert stopper.best_loss == 1.0

    stopper.restore(params)
    assert np.allclose(p.data, [1.0])


def test_early_stopping_resets_the_counter_on_improvement():
    p = Tensor([0.0], requires_grad=True)
    stopper = EarlyStopping(patience=2)

    stopper.update(3.0, [p])
    stopper.update(4.0, [p])       # worse
    stopper.update(1.0, [p])       # better -> counter back to zero

    assert stopper.worse == 0
    assert stopper.best_epoch == 3


def test_from_points():
    X, y = from_points([[0.0, 1.0], [2.0, 3.0]], [0, 1])

    assert X.shape == (2, 2)
    assert y.dtype == int
