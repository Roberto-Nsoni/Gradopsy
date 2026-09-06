import numpy as np
from typing import Callable, Iterator

from ml.tensor import Tensor
from ml.ops import Array

type Closure = Callable[[], float]

def flat_params(params: list[Tensor]) -> Array:
    return np.concatenate([p.data.reshape(-1) for p in params])

def set_flat_params(params: list[Tensor], x: Array) -> None:
    i = 0
    for p in params:
        p.data = x[i:i + p.size].reshape(p.shape).copy()
        i += p.size

def flat_grad(params: list[Tensor]) -> Array:
    return np.concatenate([p.grad.reshape(-1) for p in params])

def make_fg(params: list[Tensor], closure: Closure) -> tuple[Callable[[Array], float], Callable[[Array], Array]]:
    """
    Turns the closure into the f(x) / g(x) pair the second order methods need:
    both move the parameters to x and reevaluate the model there.
    """
    def f(x: Array) -> float:
        set_flat_params(params, x)
        return closure()

    def g(x: Array) -> Array:
        set_flat_params(params, x)
        closure()
        return flat_grad(params)

    return f, g


class Optimizer:
    def __init__(self, params: Iterator[tuple[str, Tensor]], lr: float):
        self.params = [p for _, p in params]
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def step(self, closure: Closure | None = None) -> float | None:
        raise NotImplementedError

    def state(self) -> dict:
        return {}
