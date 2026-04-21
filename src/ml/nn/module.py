import numpy as np
from typing import Any, Iterator

from src.ml.tensor import Tensor



class Module:
    def __init__(self) -> None:
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, Module] = {}

    def __setattr__(self, name: str, value: Any) -> None:

        # Case of a overwrite param
        if hasattr(self, name):
            if name in self._parameters:
                del self._parameters[name]
            if name in self._modules:
                del self._modules[name]

        object.__setattr__(self, name, value)

        if isinstance(value, Tensor):
            self._parameters[name] = value

        elif isinstance(value, Module):
            self._modules[name] = value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def parameters(self, include_non_trainable: bool = False) -> Iterator[tuple[str, Tensor]]:
        for name, value in self._parameters.items():
            if not (include_non_trainable or value.requires_grad):
                continue
            yield name, value

        for module_name, module in self._modules.items():
            for name, value in module.parameters(include_non_trainable=include_non_trainable):
                yield f"{module_name}.{name}", value

    def zero_grad(self) -> None:
        for _, par in self.parameters():
            par.zero_grad()
        
    def forward(self, *args: Any, **kwds: Any) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """
    Given row-vectors:
    y = x @ W + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        # See the effects of the init at later stages
        W = np.random.normal(0, np.sqrt(2/in_features), (in_features, out_features))  # ~ N (0, sqrt(2/in_features)) (He normal init)
        self.W = Tensor(W, requires_grad=True)
        
        self.b = None
        if bias: 
            b = np.zeros(out_features)
            self.b = Tensor(b, requires_grad=True)
            
    def forward(self, x: Tensor) -> Tensor:
        ret = x @ self.W
        if self.b is not None:
            ret = ret + self.b
        return ret

if __name__ == "__main__":

    np.random.seed(42)

    x = Tensor([[1., 2., 3.]], requires_grad=False)

    layer1 = Linear(3, 4)
    layer2 = Linear(4, 2)

    y = layer2(layer1(x))
    loss = y.sum()

    loss.backward()

    for name, p in layer1.parameters():
        print(name, p.grad)

    for name, p in layer2.parameters():
        print(name, p.grad)