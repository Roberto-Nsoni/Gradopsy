from __future__ import annotations
import numpy as np
from typing import Any

from src.ml.ops import (
    Array, ScalarLike, Neg, Add, Mul, Pow, MatMul, Exp, Log, Sum, Mean
    )

from src.ml.nn.activations import ReLU, Sigmoid, Tanh

class Tensor:
    def __init__(self, data: np.typing.ArrayLike, requires_grad: bool = False) -> None:
        self.data: Array = np.array(data)

        self.requires_grad = requires_grad
        self._grad: Array | None = None

        self._prev: set[Tensor] = set()
        self._backward = lambda: None
        self._op: str = ""

    def __repr__(self) -> str:
        return str(self.data)
    
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
