import numpy as np
from typing import Any, Iterator

from src.ml.tensor import Tensor
from src.ml.ops import ScalarLike

np.random.seed(42)

class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, Module] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        if isinstance(value, Tensor):
            self._parameters[name] = value

        elif isinstance(value, Module):
            self._modules[name] = value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def parameters(self) -> Iterator[tuple[str, Tensor]]:
        yield from self._parameters.items()

        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self) -> None:
        for _, par in self.parameters():
            par.zero_grad()
        
    def forward(self, *args: Any, **kwds: Any) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """yt = xt @ Wt + bt"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.W = Tensor(np.random.random((in_features, out_features)), requires_grad=True)
        
        self.b = None
        if bias: self.b = Tensor(np.random.random((1, out_features)), requires_grad=True)

    def forward(self, x: Any) -> Tensor:
        ret = x @ self.W
        if self.b is not None: ret += self.b
        return ret

if __name__ == "__main__":
    lin = Linear(3, 2)
    x = Tensor([[1,2,3]])
    y = lin(x)
    