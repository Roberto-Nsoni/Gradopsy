import numpy as np

from src.ml.tensor import Tensor
from src.ml.nn.module import Module
from src.ml.nn.loss import Loss

class Optimizer:
    def __init__(self, f: Module, X: Tensor, y: Tensor, lr: float | str):
        self.f = f
        self.X = X
        self.y = y
        self.d = None # Direction of descent
        self.k = 1 # Iterations
        self.lr = lr

        if type(lr) == float:
            self.al = lr
        else:
            self.al = 1

        self.log: list[dict[str, float | str]] = []

    def zero_grad(self):
        for _, p in self.f.parameters():
            p.zero_grad()

    def compute_grad(self):

        self.zero_grad()

        pred = self.f(self.X)

        loss = self.loss_fn(pred, self.y)

        loss.backward()

        self.g = flatten_grads(self.f)

        return loss.data
