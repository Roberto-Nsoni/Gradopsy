import numpy as np
from src.ml.tensor import Tensor

np.random.seed(42)

def test_backward_add():
    t1 = Tensor([2.0], requires_grad=True)
    t2 = Tensor([3.0], requires_grad=True)

    a = t1 + t2
    a.backward()

    assert np.allclose(t1.grad, [1.0])
    assert np.allclose(t2.grad, [1.0])


def test_backward_mul():
    t1 = Tensor([2.0], requires_grad=True)
    t2 = Tensor([3.0], requires_grad=True)

    a = t1 * t2
    a.backward()

    assert np.allclose(t1.grad, [3.0])
    assert np.allclose(t2.grad, [2.0])


def test_backward_add_mul_chain():
    t1 = Tensor([2.0], requires_grad=True)
    t2 = Tensor([3.0], requires_grad=True)

    a = t1 * t2 + t1  # 2*3 + 2 = 8
    a.backward()

    assert np.allclose(t1.grad, [4.0])  # t2 + 1
    assert np.allclose(t2.grad, [2.0])  # t1


def test_backward_matmul():
    t1 = Tensor(np.random.randn(3, 2), requires_grad=True)
    t2 = Tensor(np.random.randn(2, 4), requires_grad=True)
    
    a = t1 @ t2
    a.sum().backward()
    
    # Gradient formulas for sum(X@W): d(sum)/dX = 1 @ W.T, d(sum)/dW = X.T @ 1
    expected_b = t1.data.T @ np.ones((3, 4))
    expected_a = np.ones((3, 4)) @ t2.data.T
    
    assert np.allclose(t2.grad, expected_b)
    assert np.allclose(t1.grad, expected_a)


def test_backward_mean():
    t1 = Tensor([3.0, 2.0, 1.0, 4.0], requires_grad=True)
    a = t1.mean()
    a.backward()
    
    assert np.allclose(t1.grad, 0.25)


def test_no_grad():
    t1 = Tensor([2.0], requires_grad=False)
    t2 = Tensor([3.0], requires_grad=True)

    a = t1 * t2
    a.backward()

    assert t1.grad is None
    assert np.allclose(t2.grad, [2.0])


def test_broadcasting_add():
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    t2 = Tensor([10.0], requires_grad=True)
    
    a = t1 + t2
    a.sum().backward()
    
    # t2 is broadcasted to all elements, gradient = sum of all incoming gradients
    assert t1.grad.shape == (2, 2)
    assert t2.grad.shape == (1,)
    assert np.allclose(t2.grad, 4.0)


def test_zero_grad():
    t1 = Tensor([1.0], requires_grad=True)
    a = t1 * 2.0
    a.backward()
    assert t1.grad == 2.0
    
    t1.zero_grad()
    assert np.allclose(t1.grad, 0.0)

def test_linear_regression_grad():
    x = Tensor([[1.,2.]], requires_grad=False)
    W = Tensor([[1.,3.],[2.,4.]], requires_grad=True)
    b = Tensor([[1.,1.]], requires_grad=True)

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
