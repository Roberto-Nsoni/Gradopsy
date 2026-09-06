import numpy as np
from typing import Iterator

from ml.tensor import Tensor
from ml.optim.optimizer import Optimizer, Closure


class RMSProp(Optimizer):
    """
    Scales each coordinate by a running average of its squared gradient:
        r_{k+1} = rho * r_k + (1 - rho) * g_k^2
        x_{k+1} = x_k - alpha * g_k / (sqrt(r_{k+1}) + eps)
    """
    def __init__(self, params: Iterator[tuple[str, Tensor]], lr: float, rho: float = 0.9, eps: float = 1e-8):
        super().__init__(params, lr)
        self.rho = rho
        self.eps = eps
        self.r = [np.zeros_like(p.data) for p in self.params]

    def step(self, closure: Closure | None = None) -> float | None:
        loss = closure() if closure is not None else None

        for i, param in enumerate(self.params):
            self.r[i] = self.rho * self.r[i] + (1 - self.rho) * param.grad ** 2
            param.data -= self.lr * param.grad / (np.sqrt(self.r[i]) + self.eps)

        return loss

    def state(self) -> dict:
        """`scale` is the per coordinate step multiplier: that is the whole trick of RMSProp."""
        scale = [1 / (np.sqrt(r) + self.eps) for r in self.r]

        return {"scale": np.concatenate([s.reshape(-1) for s in scale])}


class Adam(Optimizer):
    """
    Momentum and RMSProp together, both bias corrected:
        m_k = b1 * m_{k-1} + (1 - b1) * g_k
        hat{m} = m_k / (1 - b1^k)
        v_k = b2 * v_{k-1} + (1 - b2) * g_k^2      hat{v} = v_k / (1 - b2^k)
        x_{k+1} = x_k - alpha * hat{m} / (sqrt(hat{v}) + eps)
    """
    def __init__(self, params: Iterator[tuple[str, Tensor]], lr: float = 0.001,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.b1, self.b2 = b1, b2
        self.eps = eps

        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.k = 0

    def step(self, closure: Closure | None = None) -> float | None:
        loss = closure() if closure is not None else None
        self.k += 1

        for i, param in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * param.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * param.grad ** 2

            m_hat = self.m[i] / (1 - self.b1 ** self.k)
            v_hat = self.v[i] / (1 - self.b2 ** self.k)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return loss

    def state(self) -> dict:
        """Adam is momentum and RMSProp at once, so it has both: a direction and a scaling."""
        if self.k == 0:
            return {}

        velocity = [m / (1 - self.b1 ** self.k) for m in self.m]
        scale = [1 / (np.sqrt(v / (1 - self.b2 ** self.k)) + self.eps) for v in self.v]

        return {"velocity": np.concatenate([m.reshape(-1) for m in velocity]),
                "scale": np.concatenate([s.reshape(-1) for s in scale])}
