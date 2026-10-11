import gzip

import numpy as np

from ml.data import load_mnist


def test_load_mnist(tmp_path):
    """A tiny fake MNIST in the same format: 16-byte header + images, 8-byte header + labels."""
    X = np.random.default_rng(0).integers(0, 256, size=(5, 28, 28)).astype(np.uint8)
    y = np.array([3, 1, 4, 1, 5], dtype=np.uint8)
    with gzip.open(tmp_path / "t10k-images-idx3-ubyte.gz", "wb") as f:
        f.write(bytes(16) + X.tobytes())
    with gzip.open(tmp_path / "t10k-labels-idx1-ubyte.gz", "wb") as f:
        f.write(bytes(8) + y.tobytes())

    Xr, yr = load_mnist(tmp_path, "test")
    assert Xr.shape == (5, 28, 28) and np.array_equal(Xr, X)
    assert yr.tolist() == [3, 1, 4, 1, 5]
