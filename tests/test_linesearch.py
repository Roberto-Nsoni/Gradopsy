import numpy as np

from ml.optim import blsnw32, backtracking

# Convex quadratic f(x) = 1/2 x'Ax - b'x, and Rosenbrock for something nastier.
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, -1.0])

def f_quad(x):
    return 0.5 * x @ A @ x - b @ x

def g_quad(x):
    return A @ x - b

def f_rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

def g_rosen(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


def assert_strong_wolfe(f, g, x, d, al, c1, c2):
    assert f(x + al * d) <= f(x) + c1 * al * (g(x) @ d)         # (W1) sufficient decrease
    assert abs(g(x + al * d) @ d) <= c2 * abs(g(x) @ d)         # (W2) curvature


def test_wolfe_on_a_quadratic():
    c1, c2 = 1e-4, 0.9
    x = np.array([2.0, 2.0])
    d = -g_quad(x)

    al, _ = blsnw32(f_quad, g_quad, x, d, 1.0, c1, c2)

    assert al > 0
    assert_strong_wolfe(f_quad, g_quad, x, d, al, c1, c2)


def test_wolfe_on_rosenbrock():
    c1, c2 = 1e-4, 0.9
    x = np.array([-1.2, 1.0])
    d = -g_rosen(x)

    al, _ = blsnw32(f_rosen, g_rosen, x, d, 1.0, c1, c2)

    assert_strong_wolfe(f_rosen, g_rosen, x, d, al, c1, c2)


def test_wolfe_from_several_starting_points():
    c1, c2 = 1e-4, 0.45
    rng = np.random.default_rng(0)

    for _ in range(20):
        x = rng.normal(size=2) * 3
        d = -g_quad(x)
        if np.allclose(d, 0):
            continue

        al, _ = blsnw32(f_quad, g_quad, x, d, 1.0, c1, c2)
        assert_strong_wolfe(f_quad, g_quad, x, d, al, c1, c2)


def test_almax_too_small_still_gets_bracketed():
    """With a tiny first alpha the bracketing phase has to grow it, not give up."""
    c1, c2 = 1e-4, 0.9
    x = np.array([5.0, 5.0])
    d = -g_quad(x)

    al, _ = blsnw32(f_quad, g_quad, x, d, 1e-6, c1, c2)

    assert al > 1e-6
    assert_strong_wolfe(f_quad, g_quad, x, d, al, c1, c2)


def test_backtracking_satisfies_armijo():
    c1 = 1e-4
    x = np.array([2.0, 2.0])
    d = -g_quad(x)

    al, _ = backtracking(f_quad, x, d, g_quad(x) @ d, almax=10.0, c1=c1)

    assert f_quad(x + al * d) <= f_quad(x) + c1 * al * (g_quad(x) @ d)
