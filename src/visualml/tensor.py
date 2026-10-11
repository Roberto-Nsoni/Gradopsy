from __future__ import annotations
import numpy as np
from typing import Any

from visualml.ops import (
    Array, ScalarLike, Neg, Add, Mul, Pow, MatMul, Exp, Log, Sum, Mean, ReLU, Sigmoid, Tanh
    )

class Tensor:
    def __init__(self, data: np.typing.ArrayLike, requires_grad: bool = False) -> None:
        self.data: Array = np.asarray(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self._grad: Array | None = None

        self._prev: set[Tensor] = set()
        self._backward = lambda: None
        self._op: str = ""

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"
    
    def __str__(self) -> str:
        return f"Tensor({self.data})"

    def __neg__(self):
        ret = Tensor(Neg.forward(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward():
                self.grad += Neg.backward(ret.grad)
        
            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "neg"

        return ret

    def __add__(self, other: Tensor | ScalarLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        requires_grad = self.requires_grad or other.requires_grad
        ret = Tensor(Add.forward(self.data, other.data), requires_grad=requires_grad)

        if requires_grad:
            ctx = {"a_shape": self.shape, "b_shape": other.shape}

            def _backward():
                g_self, g_other = Add.backward(ctx, ret.grad)
                if self.requires_grad: self.grad += g_self
                if other.requires_grad: other.grad += g_other

            ret._prev = {self, other}
            ret._backward = _backward  
            ret._op = "add"
        
        return ret

    def __mul__(self, other: Tensor | ScalarLike) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
            
        requires_grad = self.requires_grad or other.requires_grad
        ret = Tensor(Mul.forward(self.data, other.data), requires_grad=requires_grad)
        
        if requires_grad:
            ctx: dict[str, Any] = {"a": self.data, "b": other.data,
                                   "a_shape": self.shape, "b_shape": other.shape}

            def _backward():
                g_self, g_other = Mul.backward(ctx, ret.grad)
                if self.requires_grad: self.grad += g_self
                if other.requires_grad: other.grad += g_other

            ret._prev = {self, other}
            ret._backward = _backward  
            ret._op = "mul"

        return ret

    def __pow__(self, other: int) -> Tensor:
        ret = Tensor(Pow.forward(self.data, other), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx: dict[str, Any] = {"a": self.data, "b": other}

            def _backward():
                g_self = Pow.backward(ctx, ret.grad)
                if self.requires_grad: self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "pow"
        
        return ret

    def __matmul__(self, other: Tensor) -> Tensor:
        requires_grad = self.requires_grad or other.requires_grad
        ret = Tensor(MatMul.forward(self.data, other.data), requires_grad=requires_grad)

        if requires_grad:
            ctx = {"a": self.data, "b": other.data}

            def _backward():
                g_self, g_other = MatMul.backward(ctx, ret.grad)
                if self.requires_grad: self.grad += g_self
                if other.requires_grad: other.grad += g_other

            ret._prev = {self, other}
            ret._backward = _backward  
            ret._op = "matmul"
        
        return ret
    
    def __sub__(self, other: Tensor | ScalarLike) -> Tensor: return self + (-other)
    def __truediv__(self, other: Tensor | ScalarLike) -> Tensor: return self * (other ** -1)
    def __radd__(self, other: Tensor | ScalarLike) -> Tensor: return self + other
    def __rsub__(self, other: Tensor | ScalarLike) -> Tensor: return other + (-self)
    def __rmul__(self, other: Tensor | ScalarLike) -> Tensor: return self * other

    def exp(self, base: ScalarLike | None = None) -> Tensor:
        ret = Tensor(Exp.forward(self.data, base), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx: dict[str, Any] = {"a": self.data, "b": base}

            def _backward():
                g_self = Exp.backward(ctx, ret.grad)
                if self.requires_grad: self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "exp"
        
        return ret

    def log(self, base: ScalarLike | None = None) -> Tensor:
        ret = Tensor(Log.forward(self.data, base), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx: dict[str, Any] = {"a": self.data, "b": base}

            def _backward():
                g_self = Log.backward(ctx, ret.grad)
                if self.requires_grad: self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "log"
        
        return ret

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Tensor:
        ret = Tensor(Sum.forward(self.data, axis), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx = {"a_shape": self.shape}

            def _backward():
                g_self = Sum.backward(ctx, ret.grad)
                self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "sum"
        return ret
    
    def mean(self, axis: int | tuple[int, ...] | None = None) -> Tensor:
        ret = Tensor(Mean.forward(self.data, axis), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx: dict[str, Any] = {"a_shape": self.shape, "a_size": self.size, "out_size": ret.size}

            def _backward():
                g_self = Mean.backward(ctx, ret.grad)
                self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "mean"
        return ret

    def relu(self):
        ret = Tensor(ReLU.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            ctx = {"a": self.data}
            def _backward():
                self.grad += ReLU.backward(ctx, ret.grad)
            ret._prev = {self}
            ret._backward = _backward
        return ret
    
    def sigmoid(self):
        ret = Tensor(Sigmoid.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            ctx = {"sig": ret.data}
            def _backward():
                self.grad += Sigmoid.backward(ctx, ret.grad)
            ret._prev = {self}
            ret._backward = _backward
        return ret
    
    def tanh(self):
        ret = Tensor(Tanh.forward(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            ctx = {"tanh": ret.data}
            def _backward():
                self.grad += Tanh.backward(ctx, ret.grad)
            ret._prev = {self}
            ret._backward = _backward
        return ret

    def reshape(self, *shape: int) -> Tensor:
        # Needed to go from conv outputs (N, C, H, W) to the (N, C*H*W) a Linear eats
        ret = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self.grad += ret.grad.reshape(self.shape)
            ret._prev = {self}
            ret._backward = _backward
        return ret

    def conv2d(self, w: Tensor, b: Tensor | None = None, padding: int = 0) -> Tensor:
        """
        2D convolution (really cross-correlation, like every framework) with stride 1.
        x: (N, C, H, W), w: (O, C, kh, kw), b: (O,)
            out[n, o, i, j] = b[o] + sum_{c, u, v} w[o, c, u, v] * x_pad[n, c, i + u, j + v]
        Instead of looping over every output pixel, loop over the kh*kw positions of the kernel:
        each one is a shifted slice of the input times one (O, C) matrix of weights.
        optimize=True lets einsum hand each product to a matrix multiply (about 5x faster).
        """
        x = np.pad(self.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        O, C, kh, kw = w.shape
        ho, wo = x.shape[2] - kh + 1, x.shape[3] - kw + 1

        out = np.zeros((x.shape[0], O, ho, wo))
        for u in range(kh):
            for v in range(kw):
                out += np.einsum("nchw,oc->nohw", x[:, :, u:u+ho, v:v+wo], w.data[:, :, u, v], optimize=True)
        if b is not None:
            out += b.data[None, :, None, None]

        requires_grad = self.requires_grad or w.requires_grad or (b is not None and b.requires_grad)
        ret = Tensor(out, requires_grad=requires_grad)

        if requires_grad:
            def _backward():
                g = ret.grad
                dx = np.zeros_like(x)
                for u in range(kh):
                    for v in range(kw):
                        patch = x[:, :, u:u+ho, v:v+wo]
                        if w.requires_grad:
                            w.grad[:, :, u, v] += np.einsum("nchw,nohw->oc", patch, g, optimize=True)
                        dx[:, :, u:u+ho, v:v+wo] += np.einsum("nohw,oc->nchw", g, w.data[:, :, u, v], optimize=True)
                if self.requires_grad:
                    # cut the padding back off
                    self.grad += dx[:, :, padding:dx.shape[2]-padding, padding:dx.shape[3]-padding]
                if b is not None and b.requires_grad:
                    b.grad += g.sum(axis=(0, 2, 3))

            ret._prev = {t for t in (self, w, b) if t is not None}
            ret._backward = _backward
            ret._op = "conv2d"

        return ret

    def maxpool2d(self, k: int = 2) -> Tensor:
        """
        Max over non-overlapping k x k blocks (H and W must be multiples of k).
        The gradient only goes to the max of each block (to all of them if there is a tie).
        """
        N, C, H, W = self.shape
        blocks = self.data.reshape(N, C, H // k, k, W // k, k)
        out = blocks.max(axis=(3, 5))

        ret = Tensor(out, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                # out and its grad stretched back to the input size
                big_out = out.repeat(k, axis=2).repeat(k, axis=3)
                big_grad = ret.grad.repeat(k, axis=2).repeat(k, axis=3)
                self.grad += (self.data == big_out) * big_grad
            ret._prev = {self}
            ret._backward = _backward
            ret._op = "maxpool2d"
        return ret

    def backward(self) -> None:
        # Find topological sort
        topo_order: list[Tensor] = []
        visited: set[Tensor] = set()
        pile: list[Tensor] = [self]
        while pile:
            curr = pile[-1]
            if curr not in visited:
                visited.add(curr)
                for adj in curr._prev:
                    if adj not in visited:
                        pile.append(adj)
            else:
                topo_order.append(pile.pop())

        # Reset all grads in the graph to zero before starting? 
        # Or just assume user called zero_grad()
        self.grad = np.ones_like(self.data)

        for t in reversed(topo_order):
            t._backward()

    def zero_grad(self):
        self._grad = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
    
    @property
    def rank(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> Tensor:
        ret = Tensor(self.data.T, requires_grad=self.requires_grad) # initialize if = None

        if self.requires_grad:
            def _backward():
                self.grad += ret.grad.T
        
            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "T"
        
        return ret

    @property
    def grad(self) -> Array:
        # Initialize if = None
        if self.requires_grad and self._grad is None:
            self._grad = np.zeros_like(self.data)
        return self._grad # type: ignore (ts is consistent but pylint doesn't accept it D:<)

    @grad.setter
    def grad(self, value: Array):     
        self._grad = value

if __name__ == "__main__":
    print("OK")