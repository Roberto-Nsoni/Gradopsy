import numpy as np

def make_moons(n_samples, noise=0.15, seed=None):
    rng = np.random.default_rng(seed)
    n0, n1 = n_samples // 2, n_samples - n_samples // 2

    t0 = np.linspace(0, np.pi, n0)
    X0 = np.c_[np.cos(t0), np.sin(t0)]

    t1 = np.linspace(0, np.pi, n1)
    X1 = np.c_[1 - np.cos(t1), 1 - np.sin(t1) - 0.5]

    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    X += rng.normal(scale=noise, size=X.shape)

    return X, y

def make_circles(n_samples, noise=0.08, factor=0.5, seed=None):
    rng = np.random.default_rng(seed)
    n0, n1 = n_samples // 2, n_samples - n_samples // 2

    t_out = rng.uniform(0, 2*np.pi, n0)
    X_out = np.c_[np.cos(t_out), np.sin(t_out)]

    t_in = rng.uniform(0, 2*np.pi, n1)
    X_in = factor * np.c_[np.cos(t_in), np.sin(t_in)]

    X = np.vstack([X_out, X_in])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    X += rng.normal(scale=noise, size=X.shape)
    
    return X, y

def make_blobs(n_samples, n_classes=3, cluster_std=0.6, radius=3.0, seed=None):
    rng = np.random.default_rng(seed)
    per_class = n_samples // n_classes

    centers = radius * np.c_[
        np.cos(2*np.pi*np.arange(n_classes)/n_classes),
        np.sin(2*np.pi*np.arange(n_classes)/n_classes)
    ]

    X, y = [], []

    for c in range(n_classes):
        X.append(centers[c] + rng.normal(scale=cluster_std, size=(per_class, 2)))
        y.append(np.full(per_class, c))

    return np.vstack(X), np.concatenate(y)

def make_spirals(n_samples_per_class, n_classes, noise=0.2, seed=None):
    rng = np.random.default_rng(seed)

    X, y = [], []

    for c in range(n_classes):
        r = np.linspace(0.1, 1.0, n_samples_per_class)
        t = (
            np.linspace(c * 4, (c + 1) * 4, n_samples_per_class)
            + rng.normal(scale=noise, size=n_samples_per_class)
        )

        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_samples_per_class, c, dtype=int))

    return np.vstack(X), np.concatenate(y)

def from_points(points, labels):
    """Path for the points the user draws by hand: two lists in, the usual (X, y) out."""
    return np.asarray(points, dtype=float), np.asarray(labels, dtype=int)
