import numpy as np
from src.ml.tensor import Tensor
from src.ml.ops import *

np.random.seed(42)

# ----- Tensor management ------

def test_tensor_creation():
    t = Tensor([1, 2, 3])

    assert t.shape == (3,)
    assert t.rank == 1
    assert t.size == 3


def test_tensor_operations_chain():
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([0.5, -1.0, 2.0])

    z = (x * y + x).data

    expected = x.data * y.data + x.data
    assert np.allclose(z, expected)

# ------- Operations tests --------

def test_add():
    a = Tensor(np.random.randn(3, 2), requires_grad=True)
    b = Tensor(np.random.randn(2), requires_grad=True)

    out = (a + b)
    out.sum().backward()

    assert out.shape == (3, 2)
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    assert np.allclose(b.grad, np.full_like(b.data, 3.0))


def test_mul():
    a = Tensor(np.random.randn(4, 3), requires_grad=True)
    b = Tensor(np.random.randn(3), requires_grad=True)

    out = (a * b)
    out.sum().backward()

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    # grad b = sum over rows of a
    assert np.allclose(b.grad, a.data.sum(axis=0))


def test_matmul():
    a = Tensor(np.random.randn(5, 3), requires_grad=True)
    b = Tensor(np.random.randn(3, 4), requires_grad=True)

    out = (a @ b)
    out.sum().backward()

    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

    expected_b = a.data.T @ np.ones((5, 4))
    assert np.allclose(b.grad, expected_b)


def test_pow():
    a = Tensor(np.random.randn(4) + 2.0, requires_grad=True)

    out = (a ** 3)
    out.sum().backward()

    expected = 3 * (a.data ** 2)
    assert np.allclose(a.grad, expected)


def test_exp():
    a = Tensor(np.random.randn(5), requires_grad=True)

    out = a.exp()
    out.sum().backward()

    assert np.allclose(a.grad, np.exp(a.data))


def test_log():
    a = Tensor(np.random.rand(5) + 0.5, requires_grad=True)

    out = a.log()
    out.sum().backward()

    assert np.allclose(a.grad, 1 / a.data)


def test_sum():
    a = Tensor(np.random.randn(3, 4), requires_grad=True)

    out = a.sum(axis=1)
    out.sum().backward()

    assert a.grad.shape == a.shape
    assert np.allclose(a.grad, np.ones_like(a.data))


def test_mean():
    a = Tensor(np.random.randn(2, 4), requires_grad=True)

    out = a.mean(axis=1)
    out.sum().backward()

    expected = np.ones_like(a.data) / 4
    assert np.allclose(a.grad, expected)


def test_neg():
    a = Tensor(np.random.randn(5), requires_grad=True)

    out = -a
    out.sum().backward()

    assert np.allclose(a.grad, -np.ones_like(a.data))

def test_combined_ops():
    a = Tensor(np.random.randn(3), requires_grad=True)
    b = Tensor(np.random.randn(3), requires_grad=True)

    out = (a * b).exp().log() + 1.0
    out = out.sum()
    out.backward()

    # log(exp(x)) = x -> grad = b
    assert np.allclose(a.grad, b.data)

# ------ Activation tests -------

def test_relu_backward():
    t = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = t.relu()
    y.sum().backward()

    assert np.allclose(t.grad, [0.0, 0.0, 1.0])


def test_sigmoid_backward():
    t = Tensor([0.0], requires_grad=True)
    y = t.sigmoid()
    y.backward()

    # sigmoid(0) = 0.5 -> grad = 0.25
    assert np.allclose(t.grad, [0.25])


def test_tanh_backward():
    t = Tensor([0.0], requires_grad=True)
    y = t.tanh()
    y.backward()

    assert np.allclose(t.grad, [1.0])


# ------ Grad management ------

def test_no_grad():
    t1 = Tensor([2.0], requires_grad=False)
    t2 = Tensor([3.0], requires_grad=True)

    a = t1 * t2
    a.backward()

    assert t1.grad is None
    assert np.allclose(t2.grad, [2.0])


def test_zero_grad():
    t = Tensor([1.0], requires_grad=True)

    y = t * 2.0
    y.backward()
    assert np.allclose(t.grad, [2.0])

    t.zero_grad()
    assert np.allclose(t.grad, 0.0)


def test_grad_accumulation():
    t = Tensor([1.0], requires_grad=True)

    y1 = t * 2.0
    y1.backward()

    y2 = t * 3.0
    y2.backward()

    # accumulated: 2 + 3
    assert np.allclose(t.grad, [5.0])


# ------ Misc -------

def test_linear_regression_grad():
    x = Tensor([[1., 2.]], requires_grad=False)
    W = Tensor([[1., 3.], [2., 4.]], requires_grad=True)
    b = Tensor([[1., 1.]], requires_grad=True)

    y = x @ W.T + b
    L = y.mean()
    L.backward()

    assert np.allclose(W.grad, [[0.5, 1.0], [0.5, 1.0]])
    assert np.allclose(b.grad, [[0.5, 0.5]])


def test_forward_neuron():
    x = Tensor([2.0, 3.0])
    w = Tensor([0.5, -0.1], requires_grad=True)
    b = Tensor(0.1, requires_grad=True)

    y = (x * w).sum() + b
    y.backward()

    assert np.allclose(y.data, 0.8)
    assert np.allclose(w.grad, [2.0, 3.0])
    assert np.allclose(b.grad, 1.0)

import numpy as np
from src.ml.tensor import Tensor
from src.ml.nn.module import Linear


# -----------------------------
# Analytical solution (closed form)
# -----------------------------
def calc_analytical_solution(X_np, y_np):
    W = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np
    return W.reshape(-1, 1)


# -----------------------------
# Synthetic data generators
# -----------------------------
def make_dataset(kind="noise", N=100, P=3, seed=0):
    np.random.seed(seed)
    X = np.random.randn(N, P)

    if kind == "noise":
        W = np.random.randn(P, 1)
        y = X @ W + 0.1 * np.random.randn(N, 1)

    elif kind == "scaled":
        X = X * np.array([1, 10, 100])
        W = np.random.randn(P, 1)
        y = X @ W + 0.2 * np.random.randn(N, 1)

    elif kind == "collinear":
        X_base = np.random.randn(N, 1)
        X = np.hstack([
            X_base,
            X_base + 0.01 * np.random.randn(N, 1),
            np.random.randn(N, 1)
        ])
        W = np.random.randn(P, 1)
        y = X @ W + 0.2 * np.random.randn(N, 1)

    elif kind == "outliers":
        W = np.random.randn(P, 1)
        y = X @ W
        outliers = np.random.choice(N, 5, replace=False)
        y[outliers] += 20 * np.random.randn(5, 1)

    elif kind == "nonlinear":
        W = np.random.randn(P, 1)
        y = (
            X @ W
            + 0.5 * np.sin(X[:, 0:1])
            + 0.2 * (X[:, 1:2] ** 2)
            + 0.3 * np.random.randn(N, 1)
        )

    else:
        raise ValueError("Unknown dataset type")

    return X, y, W


# -----------------------------
# Training loop
# -----------------------------
def train_model(X_np, y_np, P, lr=0.01, steps=2000, bias=False):
    X_t = Tensor(X_np, requires_grad=False)
    y_t = Tensor(y_np, requires_grad=False)

    model = Linear(P, 1, bias=bias)

    for i in range(steps):
        pred = model(X_t)
        loss = ((pred - y_t) ** 2).mean()

        model.zero_grad()
        loss.backward()

        for _, p in model.parameters():
            p.data -= lr * p.grad

        if i % 200 == 0:
            print(f"[step {i}] loss = {loss.data}")

    learned_W = None
    for _, p in model.parameters():
        learned_W = p.data

    print("Learned W:\n", learned_W)
    return learned_W


# -----------------------------
# Single experiment runner
# -----------------------------
def run_experiment(X_np, true_W, y_np):
    print("True W:\n", true_W)

    W_analytical = calc_analytical_solution(X_np, y_np)
    print("Analytical W:\n", W_analytical)

    learned_W = train_model(X_np, y_np, P=X_np.shape[1])

    print("Difference (analytical - learned):\n", W_analytical - learned_W)


# -----------------------------
# Main execution (sequential pipeline)
# -----------------------------
def main():
    N = 200   # you can scale dataset size here
    P = 3     # number of features

    # ---- Gaussian noise -----
    print("\n" + "=" * 20 + "Gaussian noise" + "=" * 20)
    X = np.random.randn(N, P)
    W = np.random.randn(P, 1)
    y = X @ W + 0.1 * np.random.randn(N, 1)
    run_experiment(X, W, y)

    # ---- Scaled -----
    print("\n" + "=" * 20 + "Scaled" + "=" * 20)
    X = X * np.array([1, 10, 100])
    W = np.random.randn(P, 1)
    y = X @ W + 0.2 * np.random.randn(N, 1)
    run_experiment(X, W, y)

    # ---- Collinear -----
    print("\n" + "=" * 20 + "Collinear" + "=" * 20)
    X_base = np.random.randn(N, 1)
    X = np.hstack([
        X_base,
        X_base + 0.01 * np.random.randn(N, 1),
        np.random.randn(N, 1)
    ])
    W = np.random.randn(P, 1)
    y = X @ W + 0.2 * np.random.randn(N, 1)
    run_experiment(X, W, y)

    # ---- Outliers -----
    print("\n" + "=" * 20 + "Outliers" + "=" * 20)
    W = np.random.randn(P, 1)
    y = X @ W
    outliers = np.random.choice(N, 5, replace=False)
    y[outliers] += 20 * np.random.randn(5, 1)
    run_experiment(X, W, y)

    # ---- Non-linear -----
    print("\n" + "=" * 20 + "Non-linear" + "=" * 20)
    W = np.random.randn(P, 1)
    y = (
        X @ W
        + 0.5 * np.sin(X[:, 0:1])
        + 0.2 * (X[:, 1:2] ** 2)
        + 0.3 * np.random.randn(N, 1)
    )
    run_experiment(X, W, y)


if __name__ == "__main__":
    main()