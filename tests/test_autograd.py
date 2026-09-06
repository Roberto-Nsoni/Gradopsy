import numpy as np

from ml.tensor import Tensor
from ml.nn import MLP, ReLU, Tanh, CrossEntropyFromLogits

rng = np.random.default_rng(0)


def numeric_grad(f, t, eps=1e-6):
    """d/dt sum(f()) by central differences, one entry of t at a time."""
    grad = np.zeros_like(t.data)
    it = np.nditer(t.data, flags=["multi_index"])

    while not it.finished:
        i = it.multi_index
        original = t.data[i]

        t.data[i] = original + eps
        f_plus = f().data.sum()

        t.data[i] = original - eps
        f_minus = f().data.sum()

        t.data[i] = original
        grad[i] = (f_plus - f_minus) / (2 * eps)

        it.iternext()

    return grad


def check_grad(f, tensors, atol=1e-6):
    """The safety net: analytical grad vs numeric grad for every op we add."""
    for t in tensors:
        t.zero_grad()

    f().sum().backward()

    for t in tensors:
        assert np.allclose(t.grad, numeric_grad(f, t), atol=atol), f"bad grad for {t}"


# ------- Op by op --------

def test_add():
    a = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    b = Tensor(rng.normal(size=(2,)), requires_grad=True)
    check_grad(lambda: a + b, [a, b])

def test_mul():
    a = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    b = Tensor(rng.normal(size=(3,)), requires_grad=True)
    check_grad(lambda: a * b, [a, b])

def test_matmul():
    a = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    b = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    check_grad(lambda: a @ b, [a, b])

def test_pow():
    a = Tensor(rng.normal(size=(4,)) + 2.0, requires_grad=True)
    check_grad(lambda: a ** 3, [a])

def test_exp():
    a = Tensor(rng.normal(size=(5,)), requires_grad=True)
    check_grad(lambda: a.exp(), [a])

def test_log():
    a = Tensor(rng.random(size=(5,)) + 0.5, requires_grad=True)
    check_grad(lambda: a.log(), [a])

def test_sum():
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    check_grad(lambda: a.sum(axis=1), [a])

def test_mean():
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    check_grad(lambda: a.mean(axis=0), [a])

def test_neg():
    a = Tensor(rng.normal(size=(5,)), requires_grad=True)
    check_grad(lambda: -a, [a])

def test_transpose():
    a = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    b = Tensor(rng.normal(size=(2, 3)), requires_grad=True)
    check_grad(lambda: a.T * b, [a, b])

def test_relu():
    a = Tensor(rng.normal(size=(6,)), requires_grad=True)
    check_grad(lambda: a.relu(), [a])

def test_sigmoid():
    a = Tensor(rng.normal(size=(6,)), requires_grad=True)
    check_grad(lambda: a.sigmoid(), [a])

def test_tanh():
    a = Tensor(rng.normal(size=(6,)), requires_grad=True)
    check_grad(lambda: a.tanh(), [a])


# ------- Graphs that are not a simple chain --------

def test_diamond():
    """x reaches the output through two different paths, the grads have to add up."""
    x = Tensor(rng.normal(size=(4,)), requires_grad=True)
    check_grad(lambda: (x * x) + x.exp(), [x])

def test_shared_subexpression():
    x = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    y = Tensor(rng.normal(size=(3, 2)), requires_grad=True)
    check_grad(lambda: (x * y).tanh() * (x + y), [x, y])


# ------- The whole thing --------

def test_mlp_with_cross_entropy():
    X = Tensor(rng.normal(size=(8, 2)))
    y = Tensor(rng.integers(0, 3, size=8))

    model = MLP([2, 5, 4, 3], Tanh)
    loss_f = CrossEntropyFromLogits()

    params = [p for _, p in model.parameters()]
    check_grad(lambda: loss_f(model(X), y), params, atol=1e-5)


def test_mlp_with_relu():
    X = Tensor(rng.normal(size=(8, 2)))
    y = Tensor(rng.integers(0, 3, size=8))

    model = MLP([2, 6, 3], ReLU)
    loss_f = CrossEntropyFromLogits()

    params = [p for _, p in model.parameters()]
    check_grad(lambda: loss_f(model(X), y), params, atol=1e-5)
