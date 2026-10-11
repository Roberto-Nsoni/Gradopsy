import numpy as np

from visualml.tensor import Tensor
from visualml.ops import Softmax
from visualml.nn.module import Linear

# Numbers the input space lens eats. Nothing here draws anything, it all comes back as arrays.

def make_grid(x_range, y_range, n=100):
    """Regular n x n grid over the box. `points` is the (n*n, 2) matrix the model is fed."""
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)

    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel()]

    return XX, YY, points


def predict_grid(model, points):
    """
    Runs the model over the grid and returns (labels, confidence): the argmax class,
    which draws the k regions, and the max softmax probability, which is the alpha
    that makes the boundary fade where the net is unsure.
    """
    logits = model(Tensor(points)).data
    probs = Softmax.forward(logits)

    return probs.argmax(axis=1), probs.max(axis=1)


def layer_hyperplanes(linear: Linear):
    """
    The lines w0*x + w1*y + b = 0 each neuron of a Linear(2, h) layer draws on the
    input plane. Returns (w, b) normalized, so every row of w is a unit normal and
    -b is the signed distance from the origin to the line.
    """
    W = linear.W.data.T                                                 # (h, 2), one row per neuron
    b = linear.b.data if linear.b is not None else np.zeros(W.shape[0])

    norms = np.linalg.norm(W, axis=1, keepdims=True)

    return W / norms, b / norms[:, 0]
