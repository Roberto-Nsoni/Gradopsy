from visualml.tensor import Tensor
from visualml.ops import CrossEntropy, BCEWithLogits

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


class CrossEntropyFromLogits(Loss):
    """
    Implements the multiclass Cross-Entropy taking the logits z (no Softmax layer before it):
        1/n * sum[-z_y + ln(sum_j exp(z_j))]
    Fused: the node is built by hand so the softmax stays out of the graph.
    """
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        z = y_pred.data
        y = y_true.data.astype(int).reshape(-1)     # labels can come as (n,) or (n, 1)

        ret = Tensor(CrossEntropy.forward(z, y), requires_grad=y_pred.requires_grad)

        if y_pred.requires_grad:
            ctx = {"z": z, "y": y}

            def _backward():
                y_pred.grad += CrossEntropy.backward(ctx, ret.grad)

            ret._prev = {y_pred}
            ret._backward = _backward
            ret._op = "cross_entropy"

        return ret


class BCEWithLogitsLoss(Loss):
    """
    Implements the Binary Cross-Entropy taking the logit z:
        1/n * sum[max(z, 0) - z*y + ln(1 + exp(-|z|))]
    Same trick as the Cross-Entropy: fused so ln(1 - hat(y)) never blows up.
    """
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        z, y = y_pred.data, y_true.data.reshape(y_pred.shape)

        ret = Tensor(BCEWithLogits.forward(z, y), requires_grad=y_pred.requires_grad)

        if y_pred.requires_grad:
            ctx = {"z": z, "y": y}

            def _backward():
                y_pred.grad += BCEWithLogits.backward(ctx, ret.grad)

            ret._prev = {y_pred}
            ret._backward = _backward
            ret._op = "bce_with_logits"

        return ret
