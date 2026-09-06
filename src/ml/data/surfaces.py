import numpy as np

# 2D test functions for the optimizers. Each one is written with +, * and **, so the SAME
# function works with Tensors (and then autograd gives the gradient the optimizers need) and
# with numpy meshgrids (and then it gives the surface to draw). No duplicated code.

def rosenbrock(x, y):
    """(1 - x)^2 + 100*(y - x^2)^2 . A long curved valley: easy to enter, awful to walk."""
    return (1 - x)**2 + 100*(y - x**2)**2

def double_well(x, y):
    """x^4 - 2x^2 + y^2 . Saddle at the origin and two minima at (+-1, 0). Newton gets stuck."""
    return x**4 - 2*x**2 + y**2

def narrow_valley(x, y):
    """0.5*(x^2 + 50*y^2) . Badly conditioned quadratic: plain gradient descent zigzags."""
    return 0.5*(x**2 + 50*y**2)

def beale(x, y):
    """Beale. Flat plateaus right next to very steep walls."""
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def himmelblau(x, y):
    """(x^2 + y - 11)^2 + (x + y^2 - 7)^2 . Four global minima: the start decides which one."""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def goldstein_price(x, y):
    """Goldstein-Price. Goes from 3 to 10^6, which is why it has to be drawn in log scale."""
    a = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    b = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return a * b

def saddle(x, y):
    """x^2 - y^2 . Unbounded below: it has no minimum, so everyone diverges except Newton."""
    return x**2 - y**2


class Surface:
    """
    One test function plus everything needed to draw it and to start an optimizer on it.
    `minima` is empty when the function has none (see saddle).
    """
    def __init__(self, name, f, x_range, y_range, start, minima, log_scale=False):
        self.name = name
        self.f = f
        self.x_range = x_range
        self.y_range = y_range
        self.start = start
        self.minima = minima
        self.log_scale = log_scale

    def __repr__(self):
        return f"Surface({self.name})"

    @property
    def bounded(self):
        return len(self.minima) > 0

    def grid(self, n=200):
        """Contour of the surface over its own box, for the background of the plot."""
        XX, YY = np.meshgrid(np.linspace(*self.x_range, n), np.linspace(*self.y_range, n))

        return XX, YY, self.f(XX, YY)


SURFACES = {
    s.name: s for s in [
        Surface("rosenbrock",      rosenbrock,      (-2.0, 2.0),  (-1.0, 3.0),  (-1.2, 1.0),   [(1.0, 1.0)], log_scale=True),
        Surface("double_well",     double_well,     (-2.0, 2.0),  (-1.5, 1.5),  (0.001, 0.4),  [(1.0, 0.0), (-1.0, 0.0)]),
        Surface("narrow_valley",   narrow_valley,   (-10.0, 10.0), (-2.0, 2.0), (-9.0, 1.5),   [(0.0, 0.0)]),
        Surface("beale",           beale,           (-4.5, 4.5),  (-4.5, 4.5),  (1.0, 1.0),    [(3.0, 0.5)], log_scale=True),
        Surface("himmelblau",      himmelblau,      (-5.0, 5.0),  (-5.0, 5.0),  (0.0, 0.0),
                [(3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)], log_scale=True),
        Surface("goldstein_price", goldstein_price, (-2.0, 2.0),  (-2.0, 2.0),  (-0.5, 0.5),   [(0.0, -1.0)], log_scale=True),
        Surface("saddle",          saddle,          (-2.0, 2.0),  (-2.0, 2.0),  (0.5, 0.01),   []),
    ]
}
