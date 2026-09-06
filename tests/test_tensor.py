import numpy as np
from ml.tensor import Tensor
from ml.ops import *

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