import numpy as np
from typing import Iterator

from ml.tensor import Tensor
from ml.ops import Array
from ml.optim.optimizer import Optimizer, Closure, flat_params, set_flat_params, flat_grad, make_fg
from ml.optim.linesearch import blsnw32


def hessian_fd(g, x: Array, h: float = 1e-5) -> Array:
    """
    Hessian by forward finite differences of the analythical gradient:
        H[:, i] ~ (g(x + h*e_i) - g(x)) / h
    Symmetrized at the end, since both halves only agree up to O(h). Costs n+1
    gradient evaluations, so this is only viable on the toy nets of the sandbox.
    """
    n = x.size
    g0 = g(x)
    H = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        H[:, i] = (g(x + e) - g0) / h

    return (H + H.T) / 2


class Newton(Optimizer):
    """
    Damped Newton: the direction solves
        (H + lam*I) d = -g(x)
    with H by finite differences. The damping is not optional here: the loss of a net
    is not convex, so far from a minimum H has negative eigenvalues and plain Newton
    happily walks into a saddle. lam is raised until d is a descent direction.
    Full batch only.
    """

    def __init__(self, params: Iterator[tuple[str, Tensor]], lam: float = 1e-3,
                 c1: float = 1e-4, c2: float = 0.9, h: float = 1e-5):
        super().__init__(params, lr=1.0)
        self.lam = lam
        self.c1, self.c2 = c1, c2
        self.h = h

        self.used_lam = lam     # the damping that ended up working, for the viz
        self.damped_H = None    # H + lam*I, the quadratic the last step actually fitted

    def step(self, closure: Closure) -> float:
        fx = closure()
        gx = flat_grad(self.params)
        x = flat_params(self.params)

        if np.linalg.norm(gx) < 1e-12:
            self.lr = 0.0
            return fx

        f, g = make_fg(self.params, closure)

        H = hessian_fd(g, x, self.h)
        I = np.eye(x.size)

        lam = self.lam
        d = -gx
        for _ in range(20):
            d = np.linalg.solve(H + lam * I, -gx)
            if d @ gx < 0:
                break
            lam *= 10       # too flat/indefinite still: damp harder, d -> -g(x)/lam

        al, _ = blsnw32(f, g, x, d, 1.0, self.c1, self.c2)
        set_flat_params(self.params, x + al * d)

        self.used_lam, self.lr, self.damped_H = lam, al, H + lam * I

        return fx

    def state(self) -> dict:
        """
        `lam` is the damping that ended up working: it grows where the surface is not convex.
        `H` is the quadratic Newton just fitted, damping included, so the viz can draw it.
        """
        return {"lam": self.used_lam, "alpha": self.lr, "H": self.damped_H}

    def quadratic_model(self, closure: Closure) -> tuple[Array, Array, float]:
        """
        The local quadratic Newton is fitting at the current point:
            m(p) = f(x) + g'p + 1/2 p'(H + lam*I) p
        Returns (g, H + lam*I, f) so the viz can draw the bowl and the jump to its minimum.
        """
        fx = closure()
        gx = flat_grad(self.params)
        x = flat_params(self.params)

        _, g = make_fg(self.params, closure)
        H = hessian_fd(g, x, self.h)
        set_flat_params(self.params, x)

        return gx, H + self.used_lam * np.eye(x.size), fx
