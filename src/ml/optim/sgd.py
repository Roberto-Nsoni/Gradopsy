from src.ml.tensor import Tensor
from src.ml.nn.module import Module
from ml.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, f: Module, X: Tensor, y: Tensor, lr: float | str):
        super().__init__(f, X, y, lr)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.al * p.grad
    
    
    def blsnw(self) -> None:
        assert type(self.lr) != float

        if self.k != 1:
            self.al = 2 * (self.f(self.X) - self.log[-1]['f']) / (self.f.grad(self.X).T @ self.d)