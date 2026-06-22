from src.ml.tensor import Tensor
from src.ml.nn.module import Module

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()