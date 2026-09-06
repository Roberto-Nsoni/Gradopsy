import numpy as np
from typing import Iterator

from ml.tensor import Tensor
from ml.optim.optimizer import Optimizer, Closure


class SGD(Optimizer):
    """
    Plain gradient descent (stochastic if you feed it minibatches):
        x_{k+1} = x_k - alpha * g_k
    """
    def step(self, closure: Closure | None = None) -> float | None:
        loss = closure() if closure is not None else None

        for param in self.params:
            param.data -= self.lr * param.grad

        return loss


class Momentum(Optimizer):
    """
    Gradient descent with momentum:
        v_{k+1} = beta * v_k + g_k
        x_{k+1} = x_k - alpha * v_{k+1}
    """
    def __init__(self, params: Iterator[tuple[str, Tensor]], lr: float, beta: float = 0.9):
        super().__init__(params, lr)
        self.beta = beta
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self, closure: Closure | None = None) -> float | None:
        loss = closure() if closure is not None else None

        for i, param in enumerate(self.params):
            self.v[i] = self.beta * self.v[i] + param.grad
            param.data -= self.lr * self.v[i]

        return loss

    def state(self) -> dict:
        return {"velocity": np.concatenate([v.reshape(-1) for v in self.v])}


class LinearDecay:
    """
    Linear decay until iteration k_{ld} = beta * k_max and then a constant value till the last iteration
    alpha_k =
        if k <= k_{ld}: (1 - k / k_{ld}) * alpha_0 + (k / k_{ld}) * (0.01 * alpha_0) 
        else: 0.01 * alpha_0
    """
    def __init__(self, optimizer: Optimizer, alpha_0: float, k_ld: int):
        self.opt = optimizer
        self.alpha_0 = alpha_0
        self.k_ld = max(k_ld, 1)

        self.k = 1
        self.opt.lr = self.alpha(self.k)

    def alpha(self, k: int) -> float:
        if k > self.k_ld:
            return 0.01 * self.alpha_0
        return (1 - k / self.k_ld) * self.alpha_0 + (k / self.k_ld) * (0.01 * self.alpha_0)

    def step(self) -> None:
        self.k += 1
        self.opt.lr = self.alpha(self.k)
