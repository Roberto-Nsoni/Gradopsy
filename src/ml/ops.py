import numpy as np
from typing import Any

type Array = np.typing.NDArray[Any]
type ScalarLike = int | float

def reduce_grad(grad: Array, shape: tuple[int, ...]) -> Array:
    # Expand dims
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis = 0)

    # If some dim = 1 because of the "keepdims" of sum() and mean()
    for i in range(len(grad.shape)):
        (new_dim, orig_dim) = shape[i], grad.shape[i]
        if new_dim == 1 and orig_dim != 1:
            grad = grad.sum(axis = i, keepdims=True)

    return grad

class Neg:
    @staticmethod
    def forward(a: Array) -> Array:
        return -a
    
    @staticmethod
    def backward(ret_grad: Array) -> Array:
        return -ret_grad

class Add:
    @staticmethod
    def forward(a: Array, b: Array) -> Array:
        return a + b

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> tuple[Array, Array]:
        a_shape, b_shape = ctx['a_shape'], ctx['b_shape']
        a_grad = reduce_grad(ret_grad, a_shape)
        b_grad = reduce_grad(ret_grad, b_shape)
        return a_grad, b_grad

class Mul:
    @staticmethod
    def forward(a: Array, b: Array) -> Array:
        return a * b

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> tuple[Array, Array]:
        a, b = ctx["a"], ctx["b"]
        a_shape, b_shape = ctx["a_shape"], ctx["b_shape"]

        a_grad = ret_grad * b
        b_grad = ret_grad * a

        a_grad = reduce_grad(a_grad, a_shape)
        b_grad = reduce_grad(b_grad, b_shape)

        return a_grad, b_grad

class MatMul:
    @staticmethod
    def forward(a: Array, b: Array) -> Array:
        return a @ b

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> tuple[Array, Array]:
        a, b = ctx["a"], ctx["b"]

        a_grad = ret_grad @ b.T
        b_grad = a.T @ ret_grad

        return a_grad, b_grad

class Pow:
    @staticmethod
    def forward(a: Array, b: int) -> Array:
        return a ** b

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        a, b = ctx["a"], ctx["b"]
        a_grad = ret_grad * b * (a ** (b - 1)) 
        return a_grad
    
class Exp:
    @staticmethod
    def forward(a: Array, b: ScalarLike | None) -> Array:
        if b is None: b = np.e
        return b ** a

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        a, b = ctx["a"], ctx["b"]
        if b is None: b = np.e
        a_grad = ret_grad * (b ** a) * np.log(b)
        return a_grad

class Log:
    @staticmethod
    def forward(a: Array, b: ScalarLike | None = None) -> Array:
        if b is None:
            return np.log(a)
        return np.log(a) / np.log(b)

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        a, b = ctx["a"], ctx["b"]
        if b is None:
            return ret_grad * 1 / a
        a_grad = ret_grad * 1 / (a * np.log(b))
        return a_grad

class Sum:
    @staticmethod
    def forward(a: Array, axis: Any) -> Array:
        return a.sum(axis=axis, keepdims=True)

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> Array:
        a_shape = ctx["a_shape"]
        grad = np.broadcast_to(ret_grad, a_shape)
        return grad


class Mean:
    @staticmethod
    def forward(a: Array, axis: Any) -> Array:
        return a.mean(axis=axis, keepdims=True)

    @staticmethod
    def backward(ctx: dict[str, Any], ret_grad: Array) -> tuple[Array]:
        a_shape = ctx["a_shape"]
        size = ctx["a_size"]
        out_size = ctx["out_size"]

        N = size / out_size
        grad = np.broadcast_to(ret_grad, a_shape) / N

        return grad

# --- Activation functions ---

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
