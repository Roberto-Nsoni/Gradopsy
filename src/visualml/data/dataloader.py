import numpy as np

def iterate_minibatches(X, y, batch_size, shuffle=True, rng=None):
    """Yields the (X, y) minibatches of one epoch over a random permutation."""
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    idx = rng.permutation(n) if shuffle else np.arange(n)

    for i in range(0, n, batch_size):
        batch_idx = idx[i:i+batch_size]

        yield X[batch_idx], y[batch_idx]
