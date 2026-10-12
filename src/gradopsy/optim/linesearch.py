from typing import Callable

from gradopsy.ops import Array

type Fun = Callable[[Array], float]
type Grad = Callable[[Array], Array]


def blsnw32(f: Fun, g: Grad, x: Array, d: Array, almax: float,
            c1: float = 1e-4, c2: float = 0.9, maxiter: int = 30) -> tuple[float, int]:
    """
    Step length satisfying the strong Wolfe conditions:
        (W1)  f(x + al*d) <= f(x) + c1 * al * g(x)'d          sufficient decrease (Armijo)
        (W2)  |g(x + al*d)'d| <= c2 * |g(x)'d|                curvature

    Nocedal & Wright, algorithm 3.5 (bracketing) + 3.6 (zoom, by bisection here).
    `almax` is the first alpha tried; the bracketing phase doubles it while the
    function keeps going down. Returns (al, iout), iout = number of iterations.
    """
    f0 = f(x)
    df0 = g(x) @ d

    al_prev, f_prev = 0.0, f0
    al = almax

    for i in range(1, maxiter + 1):
        f_al = f(x + al * d)

        # Went too far: the minimum is between the last two alphas
        if f_al > f0 + c1 * al * df0 or (i > 1 and f_al >= f_prev):
            return _zoom(f, g, x, d, al_prev, al, f_prev, f0, df0, c1, c2, maxiter)

        df_al = g(x + al * d) @ d

        if abs(df_al) <= -c2 * df0:     # both Wolfe conditions hold
            return al, i

        # Already climbing again: bracket, but with the interval the other way round
        if df_al >= 0:
            return _zoom(f, g, x, d, al, al_prev, f_al, f0, df0, c1, c2, maxiter)

        al_prev, f_prev = al, f_al
        al = 2 * al

    return al, maxiter


def _zoom(f: Fun, g: Grad, x: Array, d: Array, al_lo: float, al_hi: float, f_lo: float,
          f0: float, df0: float, c1: float, c2: float, maxiter: int) -> tuple[float, int]:
    """
    Shrinks the bracket [al_lo, al_hi] by bisection until a point satisfies both
    Wolfe conditions. al_lo is always the end with the lowest f seen so far.
    """
    al = al_lo

    for i in range(1, maxiter + 1):
        al = (al_lo + al_hi) / 2
        f_al = f(x + al * d)

        if f_al > f0 + c1 * al * df0 or f_al >= f_lo:
            al_hi = al
        else:
            df_al = g(x + al * d) @ d

            if abs(df_al) <= -c2 * df0:
                return al, i

            if df_al * (al_hi - al_lo) >= 0:
                al_hi = al_lo

            al_lo, f_lo = al, f_al

    return al, maxiter


def backtracking(f: Fun, x: Array, d: Array, df0: float, almax: float = 1.0,
                 c1: float = 1e-4, rho: float = 0.5, maxiter: int = 30) -> tuple[float, int]:
    """
    Armijo backtracking. It halves alpha until the sufficient decrease condition holds.
    """
    f0 = f(x)
    al = almax

    for i in range(1, maxiter + 1):
        if f(x + al * d) <= f0 + c1 * al * df0:
            return al, i
        al *= rho

    return al, maxiter
