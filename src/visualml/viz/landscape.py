import numpy as np

from visualml.tensor import Tensor
from visualml.optim import flat_params, set_flat_params

# Numbers the loss landscape lens eats.

def random_directions(params, seed=None):
    """
    Two random directions in parameter space, filter-normalized (Li et al. 2018):
    each neuron's slice of the direction is rescaled to the norm of that same neuron's
    weights, so the picture is not dominated by whichever layer has the big weights.
    Biases are left at zero, as in the paper.
    """
    rng = np.random.default_rng(seed)
    directions = []

    for _ in range(2):
        d = []

        for p in params:
            if p.rank < 2:                          # bias
                d.append(np.zeros_like(p.data))
                continue

            d_p = rng.normal(size=p.shape)
            for j in range(p.shape[1]):             # one column of W = one output neuron
                d_p[:, j] *= np.linalg.norm(p.data[:, j]) / np.linalg.norm(d_p[:, j])

            d.append(d_p)

        directions.append(np.concatenate([d_p.reshape(-1) for d_p in d]))

    return directions[0], directions[1]


def loss_surface(model, loss_f, X, y, center, d1, d2, span=1.0, n=25):
    """
    Loss over the plane center + a*d1 + b*d2, on an n x n grid of (a, b) in [-span, span].
    Returns (coords, Z). Be honest about what this is: a 2D slice of a surface living in
    as many dimensions as the net has parameters, not the landscape itself.
    """
    params = [p for _, p in model.parameters()]
    saved = flat_params(params)

    X_t, y_t = Tensor(X), Tensor(y)
    coords = np.linspace(-span, span, n)
    Z = np.zeros((n, n))

    for i, b in enumerate(coords):
        for j, a in enumerate(coords):
            set_flat_params(params, center + a * d1 + b * d2)
            Z[i, j] = loss_f(model(X_t), y_t).data.item()

    set_flat_params(params, saved)      # leave the model where we found it

    return coords, Z


def project_trajectory(traj, center, d1, d2):
    """
    Coordinates (a, b) of each visited point on that same plane, by least squares on
    [d1 d2]. `traj` is the list of flat parameter vectors the optimizer went through.
    """
    D = np.c_[d1, d2]
    offsets = (np.asarray(traj) - center).T

    coefs, *_ = np.linalg.lstsq(D, offsets, rcond=None)

    return coefs[0], coefs[1]


def trajectory_directions(traj):
    """
    The plane the run actually moved in, from a PCA of its own trajectory.

    Two random directions are very nearly orthogonal to any path in a space of hundreds of
    dimensions, so a real run projected onto them collapses into a dot. These two do not:
    they are the directions the parameters genuinely travelled along.
    """
    points = np.asarray(traj)
    centred = points - points.mean(axis=0)

    _, _, right = np.linalg.svd(centred, full_matrices=False)

    return right[0], right[1]
