import numpy as np
from typing import Any

from src.ml.ops import Array

class ReLU:
    @staticmethod
    def forward(a: Array) -> Array:
        return np.maximum(0, a)

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        a = ctx["a"]
        return ret_grad * (a > 0)


class Sigmoid:
    @staticmethod
    def forward(a: Array) -> Array:
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        sig = ctx["sig"] # reuse forward calc for efficiency
        grad = ret_grad * sig * (1 - sig)
        return grad

class Tanh:
    @staticmethod
    def forward(a: Array) -> Array:
        return np.tanh(a)

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        tanh = ctx["tanh"] # reuse forward calc for efficiency
        grad = ret_grad * (1 - tanh * tanh)
        return grad
