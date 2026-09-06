import numpy as np

from ml.tensor import Tensor
from ml.nn import Module, Loss
from ml.optim import Optimizer, flat_params, flat_grad
from ml.data import iterate_minibatches
from ml.viz import predict_grid


def make_snapshot(k: int, loss: float, params, opt: Optimizer) -> dict:
    """
    What one step looked like. This dict is the contract with the visualization, and both
    the Trainer and the SurfaceRunner build it, so both feed the same viz.
    """
    return {
        "k": k,
        "loss": loss,
        "lr": opt.lr,
        "grad_norm": float(np.linalg.norm(flat_grad(params))),
        "params": flat_params(params),
        "state": opt.state(),
    }


class Trainer:
    def __init__(self, model: Module, loss_f: Loss, optimizer: Optimizer, scheduler=None):
        self.model = model
        self.loss_f = loss_f
        self.opt = optimizer
        self.scheduler = scheduler

        self.k = 0

    def step(self, X, y, grid=None) -> dict:
        X_t, y_t = Tensor(X), Tensor(y)

        def closure():
            self.opt.zero_grad()
            loss = self.loss_f(self.model(X_t), y_t)
            loss.backward()
            return loss.data.item()

        loss = self.opt.step(closure)

        if self.scheduler is not None:
            self.scheduler.step()

        self.k += 1

        return self.snapshot(loss, grid)

    def epoch(self, X, y, batch_size, rng=None) -> float:
        """One full pass over the data in minibatches. Returns the mean training loss."""
        losses = [self.step(X_b, y_b)["loss"] for X_b, y_b in iterate_minibatches(X, y, batch_size, rng=rng)]

        return float(np.mean(losses))

    def snapshot(self, loss: float, grid=None) -> dict:
        params = [p for _, p in self.model.parameters()]
        snap = make_snapshot(self.k, loss, params, self.opt)

        if grid is not None:
            snap["labels"], snap["confidence"] = predict_grid(self.model, grid)

        return snap
