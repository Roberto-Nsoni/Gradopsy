import numpy as np

from visualml.tensor import Tensor
from visualml.nn import Linear, MLP, ReLU, CrossEntropyFromLogits
from visualml.optim import Adam
from visualml.training import Trainer
from visualml.optim import flat_params
from visualml.data import make_spirals
from visualml.viz import make_grid, predict_grid, layer_hyperplanes
from visualml.viz import random_directions, trajectory_directions, loss_surface, project_trajectory

np.random.seed(0)


def small_model():
    return MLP([2, 4, 3], ReLU)


# ------- Lens A: input space --------

def test_make_grid():
    XX, YY, points = make_grid((-1, 1), (-2, 2), n=10)

    assert XX.shape == (10, 10)
    assert points.shape == (100, 2)
    assert np.allclose([points[:, 0].min(), points[:, 0].max()], [-1, 1])
    assert np.allclose([points[:, 1].min(), points[:, 1].max()], [-2, 2])


def test_predict_grid():
    _, _, points = make_grid((-1, 1), (-1, 1), n=8)
    labels, confidence = predict_grid(small_model(), points)

    assert labels.shape == (64,)
    assert set(np.unique(labels)) <= {0, 1, 2}
    assert ((confidence >= 1 / 3) & (confidence <= 1)).all()    # it is a max over 3 probabilities


def test_layer_hyperplanes():
    layer = Linear(2, 5)
    w, b = layer_hyperplanes(layer)

    assert w.shape == (5, 2)
    assert b.shape == (5,)
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0)


def test_hyperplane_is_where_the_neuron_is_zero():
    """A point on the line w'x + b = 0 must give exactly 0 before the activation."""
    layer = Linear(2, 3)
    w, b = layer_hyperplanes(layer)

    point = -b[0] * w[0]                        # closest point of neuron 0's line to the origin
    out = layer(Tensor(point.reshape(1, 2))).data

    assert np.isclose(out[0, 0], 0.0)


# ------- Lens B: loss landscape --------

def test_random_directions_are_filter_normalized():
    model = small_model()
    params = [p for _, p in model.parameters()]
    d1, d2 = random_directions(params, seed=0)

    assert d1.shape == flat_params(params).shape

    W = params[0].data                          # first Linear, shape (2, 4)
    dW = d1[:W.size].reshape(W.shape)
    assert np.allclose(np.linalg.norm(dW, axis=0), np.linalg.norm(W, axis=0))

    b = params[1].data                          # its bias, left at zero as in the paper
    assert np.allclose(d1[W.size:W.size + b.size], 0)


def test_random_directions_are_reproducible():
    params = [p for _, p in small_model().parameters()]

    assert np.allclose(random_directions(params, seed=7)[0], random_directions(params, seed=7)[0])


def test_loss_surface_shape_and_restores_the_model():
    X, y = make_spirals(20, 3, seed=0)
    model = small_model()
    params = [p for _, p in model.parameters()]

    center = flat_params(params)
    d1, d2 = random_directions(params, seed=0)

    coords, Z = loss_surface(model, CrossEntropyFromLogits(), X, y, center, d1, d2, span=0.5, n=5)

    assert coords.shape == (5,)
    assert Z.shape == (5, 5)
    assert np.isfinite(Z).all()
    assert np.allclose(flat_params(params), center)     # weights left where we found them


def test_loss_surface_center_matches_the_real_loss():
    X, y = make_spirals(20, 3, seed=0)
    model = small_model()
    params = [p for _, p in model.parameters()]

    center = flat_params(params)
    d1, d2 = random_directions(params, seed=0)
    loss_f = CrossEntropyFromLogits()

    _, Z = loss_surface(model, loss_f, X, y, center, d1, d2, span=1.0, n=5)

    assert np.isclose(Z[2, 2], loss_f(model(Tensor(X)), Tensor(y)).data.item())


def test_project_trajectory():
    params = [p for _, p in small_model().parameters()]
    center = flat_params(params)
    d1, d2 = random_directions(params, seed=0)

    traj = [center, center + 2 * d1, center - 3 * d2, center + d1 + d2]
    a, b = project_trajectory(traj, center, d1, d2)

    assert np.allclose(a, [0, 2, 0, 1])
    assert np.allclose(b, [0, 0, -3, 1])


def test_trajectory_directions_span_the_actual_run():
    """
    Random directions collapse a real run to a dot; the PCA ones do not. That gap is the
    whole reason this function exists, so it gets asserted rather than described.
    """
    np.random.seed(0)
    X, y = make_spirals(30, 3, seed=0)
    model = MLP([2, 8, 3], ReLU)
    trainer = Trainer(model, CrossEntropyFromLogits(), Adam(model.parameters(), lr=0.05))

    traj = [trainer.step(X, y)["params"] for _ in range(60)]
    centre = traj[-1]

    d1, d2 = trajectory_directions(traj)
    a, b = project_trajectory(traj, centre, d1, d2)
    spread_pca = max(a.max() - a.min(), b.max() - b.min())

    r1, r2 = random_directions([p for _, p in model.parameters()], seed=0)
    ra, rb = project_trajectory(traj, centre, r1, r2)
    spread_random = max(ra.max() - ra.min(), rb.max() - rb.min())

    assert spread_pca > 20 * spread_random


def test_grid_rows_run_bottom_up():
    """
    Row 0 of a grid is y = y_range[0] — the BOTTOM of the plane.

    Nothing in Python cares, but every canvas draws row 0 at the top, so anything rendering a
    field has to turn it over. Getting that wrong is silent: the observations keep being drawn
    from their own coordinates and stay right, the boundary is derived from the same rows as the
    field and goes wrong in step with it, and the accuracy number is computed here and keeps
    saying what it always said. The web build shipped mirrored for a while on exactly this.
    """
    n = 6
    _, _, points = make_grid((-1.0, 1.0), (-1.0, 1.0), n=n)

    assert points[0][1] == -1.0                       # first row is the bottom
    assert points[(n - 1) * n][1] == 1.0              # last row is the top
    assert points[0][0] == -1.0                       # and columns run left to right
    assert points[n - 1][0] == 1.0

    # index = row * n + col, with row indexing y ascending
    ys = points[:, 1].reshape(n, n)
    assert np.all(np.diff(ys[:, 0]) > 0)
    assert np.all(ys[3] == ys[3][0])
