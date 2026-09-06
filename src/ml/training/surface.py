from ml.tensor import Tensor
from ml.data import Surface
from ml.training.trainer import make_snapshot

class SurfaceRunner:
    """
    The same steppable contract as the Trainer, but over a 2D test function instead of a
    model and a dataset. no data, the two coordinates are the parameters.
    """
    def __init__(self, surface: Surface, optimizer_class, start=None, **kwargs):
        x0, y0 = surface.start if start is None else start

        self.x = Tensor(x0, requires_grad=True)
        self.y = Tensor(y0, requires_grad=True)

        self.surface = surface
        self.opt = optimizer_class([("x", self.x), ("y", self.y)], **kwargs)
        self.k = 0

    @property
    def position(self):
        return self.x.data.item(), self.y.data.item()

    def step(self) -> dict:
        def closure():
            self.opt.zero_grad()
            loss = self.surface.f(self.x, self.y)
            loss.backward()
            return loss.data.item()

        loss = self.opt.step(closure)
        self.k += 1

        return make_snapshot(self.k, loss, [self.x, self.y], self.opt)
