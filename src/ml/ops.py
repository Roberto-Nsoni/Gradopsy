import numpy as np
from typing import Any

Array = np.typing.NDArray[Any]

def reduce_grad(grad: Array, shape: tuple[int, ...]) -> Array:
    # Expand dims
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # If some dim = 1 because of the "keepdims" of sum() and mean()
    for i in range(len(grad.shape)):
        (new_dim, orig_dim) = shape[i], grad.shape[i]
        if new_dim == 1 and orig_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Add:
    @staticmethod
    def forward(a: Array, b: Array) -> Array:
        return a + b

    @staticmethod
    def backward(ctx: dict[str, Any], grad_output: Array) -> tuple[Array, Array]:
        a_shape, b_shape = ctx['a_shape'], ctx['b_shape']
        grad_a = reduce_grad(grad_output, a_shape)
        grad_b = reduce_grad(grad_output, b_shape)
        return grad_a, grad_b


class Mul:
    @staticmethod
    def forward(a: Array, b: Array) -> Array:
        return a * b

    @staticmethod
    def backward(ctx: dict[str, Any], grad_output: Array) -> tuple[Array, Array]:
        a, b = ctx["a"], ctx["b"]
        a_shape, b_shape = ctx["a_shape"], ctx["b_shape"]

        grad_a = grad_output * b
        grad_b = grad_output * a

        grad_a = reduce_grad(grad_a, a_shape)
        grad_b = reduce_grad(grad_b, b_shape)

        return grad_a, grad_b


class MatMul:
    @staticmethod
    def forward(a: Array, b: Array) -> Array:
        return a @ b

    @staticmethod
    def backward(ctx: dict[str, Any], grad_output: Array) -> tuple[Array, Array]:
        a, b = ctx["a"], ctx["b"]

        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output

        return grad_a, grad_b


class Sum:
    @staticmethod
    def forward(a: Array, axis: Any) -> Array:
        return a.sum(axis=axis, keepdims=True)

    @staticmethod
    def backward(ctx: dict[str, Any], grad_output: Array) -> tuple[Array]:
        a_shape = ctx["a_shape"]
        grad = np.broadcast_to(grad_output, a_shape)
        return (grad,)


class Mean:
    @staticmethod
    def forward(a: Array, axis: Any) -> Array:
        return a.mean(axis=axis, keepdims=True)

    @staticmethod
    def backward(ctx: dict[str, Any], grad_output: Array) -> tuple[Array]:
        a_shape = ctx["a_shape"]
        size = ctx["a_size"]
        out_size = ctx["out_size"]

        N = size / out_size
        grad = np.broadcast_to(grad_output, a_shape) / N

        return (grad,)
