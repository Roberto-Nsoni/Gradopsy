import numpy as np

from ml.tensor import Tensor
from ml.ops import Softmax
from ml.nn import MSE, CrossEntropyFromLogits, BCEWithLogitsLoss

rng = np.random.default_rng(1)


def test_mse_value_and_grad():
    y_pred = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    y_true = Tensor([[1.5], [2.0], [1.0]])

    loss = MSE()(y_pred, y_true)
    loss.backward()

    assert np.allclose(loss.data, ((np.array([1.0, 2.0, 3.0]) - [1.5, 2.0, 1.0]) ** 2).mean())
    assert np.allclose(y_pred.grad, 2 * np.array([[-0.5], [0.0], [2.0]]) / 3)


def test_cross_entropy_value():
    z = np.array([[1.0, 2.0, 3.0]])
    loss = CrossEntropyFromLogits()(Tensor(z), Tensor([2]))

    p = np.exp(z) / np.exp(z).sum()
    assert np.allclose(loss.data, -np.log(p[0, 2]))


def test_cross_entropy_backward_is_softmax_minus_onehot():
    z = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    y = np.array([0, 2, 1, 1])

    CrossEntropyFromLogits()(z, Tensor(y)).backward()

    expected = Softmax.forward(z.data)
    expected[np.arange(4), y] -= 1

    assert np.allclose(z.grad, expected / 4)


def test_cross_entropy_accepts_column_labels():
    z = Tensor(rng.normal(size=(4, 3)))
    y = np.array([0, 2, 1, 1])

    flat = CrossEntropyFromLogits()(z, Tensor(y))
    column = CrossEntropyFromLogits()(z, Tensor(y.reshape(-1, 1)))

    assert np.allclose(flat.data, column.data)


def test_cross_entropy_does_not_overflow():
    """The whole point of fusing it: huge logits must not turn into inf/nan."""
    z = Tensor([[1000.0, 0.0, -1000.0]], requires_grad=True)
    loss = CrossEntropyFromLogits()(z, Tensor([0]))
    loss.backward()

    assert np.isfinite(loss.data).all()
    assert np.isfinite(z.grad).all()
    assert np.allclose(loss.data, 0.0)          # it is already perfectly confident and right


def test_bce_with_logits_does_not_overflow():
    """This is what the old BCE got wrong: log(1 - y_pred) with y_pred -> 1."""
    z = Tensor([[50.0], [-50.0]], requires_grad=True)
    loss = BCEWithLogitsLoss()(z, Tensor([[1.0], [0.0]]))
    loss.backward()

    assert np.isfinite(loss.data).all()
    assert np.isfinite(z.grad).all()
    assert loss.data < 1e-10


def test_bce_with_logits_matches_the_naive_formula():
    """Where the naive version is still safe, both have to agree."""
    z = rng.normal(size=(5, 1))
    y = rng.integers(0, 2, size=(5, 1)).astype(float)

    p = 1 / (1 + np.exp(-z))
    naive = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    assert np.allclose(BCEWithLogitsLoss()(Tensor(z), Tensor(y)).data, naive)
