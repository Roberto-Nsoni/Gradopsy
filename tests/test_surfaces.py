import numpy as np

from ml.tensor import Tensor
from ml.data import SURFACES
from ml.optim import SGD, Momentum, RMSProp, Adam, BFGS, LBFGS, Newton
from ml.training import SurfaceRunner, Recorder


def numeric_gradient(f, x, y, h=1e-6):
    return np.array([(f(x + h, y) - f(x - h, y)) / (2 * h),
                     (f(x, y + h) - f(x, y - h)) / (2 * h)])


def test_every_surface_is_flat_at_its_minima():
    for surface in SURFACES.values():
        for (x, y) in surface.minima:
            assert np.allclose(numeric_gradient(surface.f, x, y), 0, atol=1e-4), surface.name


def test_the_saddle_is_the_only_one_without_minima():
    unbounded = [s.name for s in SURFACES.values() if not s.bounded]

    assert unbounded == ["saddle"]


def test_the_same_formula_works_on_tensors_and_on_numpy():
    """This is what lets one function both drive autograd and draw the contour."""
    for surface in SURFACES.values():
        x, y = 0.7, -0.3
        con_tensor = surface.f(Tensor(x), Tensor(y)).data.item()

        assert np.isclose(con_tensor, surface.f(x, y)), surface.name


def test_grid_shapes():
    for surface in SURFACES.values():
        XX, YY, Z = surface.grid(n=30)

        assert XX.shape == YY.shape == Z.shape == (30, 30)
        assert np.isfinite(Z).all(), surface.name


def test_log_scale_only_where_the_range_is_huge():
    """Log scale needs positive values, so no surface that goes negative can ask for it."""
    for surface in SURFACES.values():
        if surface.log_scale:
            assert surface.grid(n=30)[2].min() > 0, surface.name


def test_autograd_matches_the_numeric_gradient_on_every_surface():
    for surface in SURFACES.values():
        x, y = Tensor(0.6, requires_grad=True), Tensor(-0.4, requires_grad=True)
        surface.f(x, y).backward()

        expected = numeric_gradient(surface.f, 0.6, -0.4)
        assert np.allclose([x.grad, y.grad], expected, atol=1e-5), surface.name


# ------- The point of the whole `descent` demo --------

def run(surface_name, optimizer_class, steps, **kwargs):
    runner = SurfaceRunner(SURFACES[surface_name], optimizer_class, **kwargs)
    for _ in range(steps):
        runner.step()

    return runner


def test_newton_gets_stuck_on_the_saddle_while_everyone_else_escapes():
    """
    Starting next to the saddle of x^4 - 2x^2 + y^2: the first order methods and BFGS roll
    down to a real minimum (f = -1), and Newton stays at (0, 0) with f = 0 because it solves
    g = 0 and a saddle is a stationary point too. This is the payload of the demo.
    """
    # RMSProp gets a looser tolerance on purpose: without a decaying lr it never settles,
    # it hovers around the minimum at a distance of the order of lr.
    for cls, kwargs, steps, atol in [(SGD, {"lr": 0.01}, 500, 1e-3), (Momentum, {"lr": 0.01}, 500, 1e-3),
                                     (RMSProp, {"lr": 0.01}, 500, 1e-2), (Adam, {"lr": 0.05}, 500, 1e-3),
                                     (BFGS, {}, 40, 1e-3), (LBFGS, {}, 40, 1e-3)]:
        runner = run("double_well", cls, steps, **kwargs)
        x, y = runner.position

        assert np.isclose(abs(x), 1.0, atol=atol), f"{cls.__name__} did not escape: x={x}"
        assert np.isclose(y, 0.0, atol=atol), cls.__name__

    stuck = run("double_well", Newton, 20)
    assert np.allclose(stuck.position, (0.0, 0.0), atol=1e-6)


def test_rosenbrock_second_order_beats_first_order():
    """40 BFGS steps against 20000 of SGD."""
    assert np.allclose(run("rosenbrock", BFGS, 40).position, (1.0, 1.0), atol=1e-6)
    assert not np.allclose(run("rosenbrock", SGD, 20000, lr=1e-3).position, (1.0, 1.0), atol=1e-6)


def test_himmelblau_start_decides_the_minimum():
    """Four global minima: where you land depends only on where you started."""
    reached = {tuple(np.round(run("himmelblau", Adam, 2000, start=s, lr=0.05).position, 2))
               for s in [(4.0, 4.0), (-4.0, 4.0), (-4.0, -4.0), (4.0, -4.0)]}

    assert len(reached) == 4


def test_surface_runner_snapshot():
    runner = SurfaceRunner(SURFACES["rosenbrock"], Adam, lr=0.05)
    snapshot = runner.step()

    assert snapshot["k"] == 1
    assert snapshot["params"].shape == (2,)
    assert snapshot["grad_norm"] > 0
    assert set(snapshot["state"]) == {"velocity", "scale"}


def test_start_can_be_overridden():
    runner = SurfaceRunner(SURFACES["beale"], SGD, start=(2.0, -1.0), lr=0.001)

    assert runner.position == (2.0, -1.0)


def test_newton_damping_stays_sane_once_it_arrives():
    """
    Sitting on a stationary point the gradient is 0, so d'g < 0 can never hold. Without the
    early exit the damping loop multiplies lam by 10 twenty times and the panel shows 1e16.
    """
    runner = run("himmelblau", Newton, 30, start=(3.0, 2.0))

    assert runner.opt.state()["lam"] < 1.0
    assert np.allclose(runner.position, (3.0, 2.0), atol=1e-6)


def test_newton_exposes_the_quadratic_it_fitted():
    """The viz draws that quadratic, so it has to be the damped one Newton actually solved."""
    runner = run("rosenbrock", Newton, 5)
    state = runner.opt.state()

    assert state["H"].shape == (2, 2)
    assert np.allclose(state["H"], state["H"].T)                    # simetrizada
    assert np.all(np.linalg.eigvalsh(state["H"]) > 0)               # y definida positiva ahi


def test_newtons_model_is_a_saddle_at_the_saddle():
    """
    Sitting on the double well's saddle the fitted quadratic is itself a saddle: one negative
    eigenvalue. That is exactly why Newton walks into it, and the viz says so instead of
    drawing an ellipse that does not exist.
    """
    runner = run("double_well", Newton, 1, start=(0.02, 0.3))
    eigenvalues = np.linalg.eigvalsh(runner.opt.state()["H"])

    assert eigenvalues.min() < 0
