import numpy as np
from typing import Any

from src.ml.ops import *

Array = np.typing.NDArray[Any]

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
    
    def __add__(self, other: "Tensor | int | float") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor([other])

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

    def __mul__(self, other: "Tensor | int | float") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor([other])
            
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

    def __matmul__(self, other: "Tensor") -> "Tensor":
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

    def sum(self, axis: int | tuple[int, ...] | None = None) -> "Tensor":
        ret = Tensor(Sum.forward(self.data, axis), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx = {"a_shape": self.shape}

            def _backward():
                (g_self,) = Sum.backward(ctx, ret.grad)
                self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "sum"
        return ret
    
    def mean(self, axis: int | tuple[int, ...] | None = None) -> "Tensor":
        ret = Tensor(Mean.forward(self.data, axis), requires_grad=self.requires_grad)

        if self.requires_grad:
            ctx: dict[str, Any] = {"a_shape": self.shape, "a_size": self.size, "out_size": ret.size}

            def _backward():
                (g_self,) = Mean.backward(ctx, ret.grad)
                self.grad += g_self

            ret._prev = {self}
            ret._backward = _backward  
            ret._op = "mean"
        return ret

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
    def T(self) -> "Tensor":
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

if __name__ == "__main__":
    x = Tensor([2.0, 3.0])
    w = Tensor([0.5, -0.1], requires_grad=True)
    b = Tensor(0.1, requires_grad=True)

    y = (x * w).sum() + b 
    y.backward()

    print(f"Result: {y.data}") # Should be (2*0.5 + 3*-0.1) + 0.1 = 0.8
    print(f"Gradient w: {w.grad}") # Should be [2.0, 3.0]
    print(f"Gradient b: {b.grad}") # Should be 1.0


