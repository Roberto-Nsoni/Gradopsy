import numpy as np

from ml.tensor import Tensor

def accuracy(logits: Tensor, y) -> float:
    pred = logits.data.argmax(axis=1)
    y = np.asarray(y).astype(int).reshape(-1)

    return float((pred == y).mean())
