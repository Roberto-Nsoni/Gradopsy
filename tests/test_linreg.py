import numpy as np

from gradopsy.tensor import Tensor
from gradopsy.nn import Linear, MSE
from gradopsy.optim import SGD

N = 200   # number of samples
P = 3     # number of features


def calc_analytical_solution(X_np, y_np):
    W = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np
    return W.reshape(-1, 1)


def train_model(X_np, y_np, lr=0.01, steps=10000):
    X_t = Tensor(X_np, requires_grad=False)
    y_t = Tensor(y_np, requires_grad=False)

    model = Linear(X_np.shape[1], 1, bias=False)
    loss_f = MSE()
    opt = SGD(model.parameters(), lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        loss_f(model(X_t), y_t).backward()
        opt.step()

    return model.W.data


def run_experiment(X_np, y_np, atol=1e-4, **kwargs):
    """Gradient descent has to land on the same place as the normal equations."""
    learned_W = train_model(X_np, y_np, **kwargs)
    analytical_W = calc_analytical_solution(X_np, y_np)

    assert np.isfinite(learned_W).all()
    assert np.allclose(learned_W, analytical_W, atol=atol)


def test_linreg_gaussian():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, P))
    W = rng.normal(size=(P, 1))
    y = X @ W + 0.1 * rng.normal(size=(N, 1))

    run_experiment(X, y)


def test_linreg_scaled():
    """
    Features on wildly different scales (x1, x10, x100). The Hessian is 2/n X'X, so its
    largest eigenvalue goes as 100^2 and any lr that does not blow up leaves the small
    directions frozen: plain gradient descent simply cannot do this one. Standardizing
    is not a nicety here, it is the fix.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(N, P)) * np.array([1, 10, 100])
    W = rng.normal(size=(P, 1))
    y = X @ W + 0.2 * rng.normal(size=(N, 1))

    X_std = X / X.std(axis=0)
    run_experiment(X_std, y, atol=1e-3)


def test_linreg_scaled_diverges_without_standardizing():
    """The other half of the lesson above: same data, same lr, and it blows up."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(N, P)) * np.array([1, 10, 100])
    W = rng.normal(size=(P, 1))
    y = X @ W + 0.2 * rng.normal(size=(N, 1))

    assert np.abs(train_model(X, y, steps=100)).max() > 1e10


def test_linreg_colinear():
    """
    Two nearly identical columns: cond(X'X) ~ 1e4, so the valley is a flat ridge and
    only their SUM is pinned down, not each coefficient. So the weights are not what
    converges here, the predictions are, and that is what gets asserted.
    """
    rng = np.random.default_rng(3)
    X_base = rng.normal(size=(N, 1))
    X = np.hstack([X_base, X_base + 0.01 * rng.normal(size=(N, 1)), rng.normal(size=(N, 1))])
    W = rng.normal(size=(P, 1))
    y = X @ W + 0.2 * rng.normal(size=(N, 1))

    learned_W = train_model(X, y, steps=20000)
    analytical_W = calc_analytical_solution(X, y)

    assert np.isfinite(learned_W).all()
    assert np.allclose(X @ learned_W, X @ analytical_W, atol=1e-1)


def test_linreg_outliers():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(N, P))
    W = rng.normal(size=(P, 1))
    y = X @ W
    y[rng.choice(N, 5, replace=False)] += 20 * rng.normal(size=(5, 1))

    run_experiment(X, y)


def test_linreg_nonlinear():
    """The model cannot fit this, but it still has to reach the best linear fit."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(N, P))
    W = rng.normal(size=(P, 1))
    y = X @ W + 0.5 * np.sin(X[:, 0:1]) + 0.2 * (X[:, 1:2] ** 2) + 0.3 * rng.normal(size=(N, 1))

    run_experiment(X, y)
