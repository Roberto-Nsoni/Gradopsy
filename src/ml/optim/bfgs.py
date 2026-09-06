import numpy as np
from typing import Iterator

from ml.tensor import Tensor
from ml.optim.optimizer import Optimizer, Closure, flat_params, set_flat_params, flat_grad, make_fg
from ml.optim.linesearch import blsnw32


class BFGS(Optimizer):
    """
    Quasi-Newton BFGS. Keeps an approximation H of the INVERSE Hessian, updated with
    the last step s and the change of gradient y:

        s = al * d,
        y = g(x) - g(x - s)
        p = 1 / (y's)
        H = (I - p*s*y') H (I - p*y*s') + p*s*s'

    and moves along d = -H g(x), with the step length from a strong Wolfe line search.
    Full batch only: minibatch noise wrecks the curvature estimate.
    """

    def __init__(self, params: Iterator[tuple[str, Tensor]], c1: float = 1e-4, c2: float = 0.1):
        super().__init__(params, lr=1.0)
        self.c1, self.c2 = c1, c2

        self.H = None
        self.k = 0

        self.d = None
        self.al = 1.0
        self.f_prev = None
        self.descent = True

    def step(self, closure: Closure) -> float:
        fx = closure()
        gx = flat_grad(self.params)
        x = flat_params(self.params)

        f, g = make_fg(self.params, closure)
        n = x.size

        if self.k == 0:
            self.H = np.eye(n)
        elif self.d is not None:
            s = self.al * self.d
            y = gx - g(x - s)               # x - s is exactly the previous iterate

            # Wolfe guarantees y's > 0, but once we sit on the minimum s and y are
            # both ~0 and 1/(y's) blows up. There H is already good, so keep it.
            if y @ s > 1e-12:
                p = 1 / (y @ s)

                I = np.eye(n)
                self.H = (I - p * np.outer(s, y)) @ self.H @ (I - p * np.outer(y, s)) + p * np.outer(s, s)

        d = -self.H @ gx

        self.descent = d @ gx < 0
        if not self.descent:
            set_flat_params(self.params, x)
            return fx

        if self.k == 0:
            almax = 1.0
        else:
            # Quadratic guess from the last two iterates (NW eq. 3.60). It can come out
            # non positive if the loss went up, and then it is not a usable step.
            almax = 2 * (fx - self.f_prev) / (gx @ d)
            if almax <= 0:
                almax = 1.0

        al, _ = blsnw32(f, g, x, d, almax, self.c1, self.c2)
        set_flat_params(self.params, x + al * d)

        self.d, self.al, self.f_prev, self.lr = d, al, fx, al
        self.k += 1

        return fx

    def state(self) -> dict:
        """H is the whole point: on a quadratic it ends up being the inverse Hessian."""
        return {"H": self.H, "alpha": self.al, "descent": self.descent}


class LBFGS(Optimizer):
    """
    Limited memory BFGS: instead of the n x n matrix it stores the last m pairs (s, y)
    and rebuilds d = -H g(x) with the two loop recursion. Same line search as BFGS.
    """

    def __init__(self, params: Iterator[tuple[str, Tensor]], m: int = 10,
                 c1: float = 1e-4, c2: float = 0.1):
        super().__init__(params, lr=1.0)
        self.m = m
        self.c1, self.c2 = c1, c2

        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.k = 0

        self.d = None
        self.al = 1.0
        self.f_prev = None
        self.descent = True

    def direction(self, gx):
        """Two loop recursion: applies H to -g(x) without ever building H."""
        q = gx.copy()
        alphas = []

        for s, y in reversed(self.pairs):
            a = (s @ q) / (y @ s)
            q = q - a * y
            alphas.append(a)

        if self.pairs:                          # H0 = gamma * I, gamma from the last pair
            s, y = self.pairs[-1]
            q = q * (s @ y) / (y @ y)

        for (s, y), a in zip(self.pairs, reversed(alphas)):
            b = (y @ q) / (y @ s)
            q = q + (a - b) * s

        return -q

    def step(self, closure: Closure) -> float:
        fx = closure()
        gx = flat_grad(self.params)
        x = flat_params(self.params)

        f, g = make_fg(self.params, closure)

        if self.k > 0 and self.d is not None:
            s = self.al * self.d
            y = gx - g(x - s)

            if y @ s > 1e-12:               # same as in BFGS, skip the pair
                self.pairs.append((s, y))
                if len(self.pairs) > self.m:
                    self.pairs.pop(0)

        d = self.direction(gx)

        self.descent = d @ gx < 0
        if not self.descent:
            set_flat_params(self.params, x)
            return fx

        if self.k == 0:
            almax = 1.0
        else:
            almax = 2 * (fx - self.f_prev) / (gx @ d)
            if almax <= 0:
                almax = 1.0

        al, _ = blsnw32(f, g, x, d, almax, self.c1, self.c2)
        set_flat_params(self.params, x + al * d)

        self.d, self.al, self.f_prev, self.lr = d, al, fx, al
        self.k += 1

        return fx

    def state(self) -> dict:
        """No H here, just how many (s, y) pairs it is remembering right now."""
        return {"n_pairs": len(self.pairs), "alpha": self.al, "descent": self.descent}
