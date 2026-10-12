import gzip
from pathlib import Path

import numpy as np

# MNIST comes as 4 gzipped files (scripts/get_mnist.py downloads them).
# Images: 16 bytes of header, then 28*28 uint8 per image. Labels: 8 bytes of header, then 1 uint8 each.
FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def load_mnist(folder, split="train"):
    """(X, y): X uint8 of shape (n, 28, 28), y int of shape (n,)."""
    images_file, labels_file = FILES[split]

    with gzip.open(Path(folder) / images_file, "rb") as f:
        X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

    with gzip.open(Path(folder) / labels_file, "rb") as f:
        y = np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(int)

    return X, y
