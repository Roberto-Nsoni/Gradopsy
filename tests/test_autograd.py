import numpy as np

from gradopsy.tensor import Tensor
from gradopsy.nn import MLP, ReLU, Tanh, CrossEntropyFromLogits

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


# ------- Shape and image ops (recognize) --------

def test_reshape():
    a = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True)
    check_grad(lambda: a.reshape(4, 6) * Tensor(np.arange(24.0).reshape(4, 6)), [a])

def test_conv2d_no_padding():
    x = Tensor(rng.normal(size=(2, 3, 6, 5)), requires_grad=True)
    w = Tensor(rng.normal(size=(4, 3, 3, 2)), requires_grad=True)
    b = Tensor(rng.normal(size=(4,)), requires_grad=True)
    # a weighted sum, so every output position counts differently
    k = Tensor(rng.normal(size=(2, 4, 4, 4)))
    check_grad(lambda: x.conv2d(w, b, padding=0) * k, [x, w, b])

def test_conv2d_padding():
    x = Tensor(rng.normal(size=(2, 2, 5, 5)), requires_grad=True)
    w = Tensor(rng.normal(size=(3, 2, 3, 3)), requires_grad=True)
    b = Tensor(rng.normal(size=(3,)), requires_grad=True)
    k = Tensor(rng.normal(size=(2, 3, 5, 5)))
    check_grad(lambda: x.conv2d(w, b, padding=1) * k, [x, w, b])

def test_conv2d_matches_direct_sum():
    """Not only consistent gradients: the forward is the cross-correlation it claims to be."""
    x = rng.normal(size=(1, 2, 5, 5))
    w = rng.normal(size=(3, 2, 3, 3))
    b = rng.normal(size=(3,))
    out = Tensor(x).conv2d(Tensor(w), Tensor(b), padding=1).data
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    ref = np.zeros((1, 3, 5, 5))
    for o in range(3):
        for i in range(5):
            for j in range(5):
                ref[0, o, i, j] = b[o] + (w[o] * xp[0, :, i:i + 3, j:j + 3]).sum()
    assert np.allclose(out, ref)

def test_maxpool2d():
    x = Tensor(rng.normal(size=(2, 3, 4, 6)), requires_grad=True)
    k = Tensor(rng.normal(size=(2, 3, 2, 3)))
    check_grad(lambda: x.maxpool2d(2) * k, [x])

def test_small_cnn_with_cross_entropy():
    """conv -> relu -> pool -> flatten -> linear -> CE, every parameter checked end to end."""
    from gradopsy.nn import Conv2D, MaxPool2D, Flatten, Sequential, Linear
    np.random.seed(1)
    net = Sequential(Conv2D(1, 2, 3, padding=1), ReLU(), MaxPool2D(2), Flatten(), Linear(2 * 2 * 2, 3))
    X = Tensor(rng.normal(size=(3, 1, 4, 4)))
    y = Tensor(np.array([0, 2, 1]))
    loss_f = CrossEntropyFromLogits()
    params = [p for _, p in net.parameters()]
    check_grad(lambda: loss_f(net(X), y), params, atol=1e-5)



def test_freeze():
    """A frozen model builds no graph through its weights, but still differentiates its input."""
    from gradopsy.nn import Sequential, Linear
    np.random.seed(2)
    net = Sequential(Linear(3, 4), Tanh(), Linear(4, 2))
    net.freeze()
    out = net(Tensor(rng.normal(size=(5, 3))))
    assert not out.requires_grad and not out._prev

    x = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    check_grad(lambda: net(x), [x])
