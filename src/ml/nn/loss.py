from src.ml.tensor import Tensor

class Loss:
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """
    Implements the Mean Squere Error:
        1/n * sum[(hat(y) - y) ^ 2]
    """
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return ((y_pred - y_true) ** 2).mean()
    
class BCE(Loss):
    """
    Implements the Binary Cross-Entropy:
        - 1/n * sum[y * ln(hat(y)) + (1 - y) * ln(1 - hat(y))]
    """
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:

        y_pred = y_pred + self.eps

        return -(y_true * y_pred.log() + (1 - y_true)*(1 - y_pred).log()).mean()