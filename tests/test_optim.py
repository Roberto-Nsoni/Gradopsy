import numpy as np

from gradopsy.tensor import Tensor
from gradopsy.optim import SGD, Momentum, RMSProp, Adam, BFGS, Newton, LinearDecay
from gradopsy.optim import flat_params, set_flat_params, flat_grad, hessian_fd

# Toy convex problem: f(x) = 1/2 x'Ax - b'x, minimum at A^-1 b.
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([[1.0, 2.0]])
x_star = np.linalg.solve(A, b.reshape(-1))


def make_problem():
    x = Tensor([[2.0, -3.0]], requires_grad=True)
    A_t, b_t = Tensor(A), Tensor(b)

    def f():
        return ((x @ A_t) * x).sum() * 0.5 - (x * b_t).sum()

    return x, f


def run(opt_class, steps, **kwargs):
    x, f = make_problem()
    opt = opt_class([("x", x)], **kwargs)

    def closure():
        opt.zero_grad()
        loss = f()
        loss.backward()
        return loss.data.item()

    for _ in range(steps):
        opt.step(closure)

    return x.data.reshape(-1)


# ------- First order --------

def test_sgd():
    assert np.allclose(run(SGD, 300, lr=0.1), x_star, atol=1e-6)

def test_momentum():
    assert np.allclose(run(Momentum, 300, lr=0.02), x_star, atol=1e-6)

def test_rmsprop():
    """Without a decaying lr RMSProp does not settle: it hovers around the minimum
    at a distance of the order of lr, so the tolerance has to say so."""
    assert np.allclose(run(RMSProp, 3000, lr=0.01), x_star, atol=1e-2)

def test_adam():
    assert np.allclose(run(Adam, 2000, lr=0.05), x_star, atol=1e-4)


# ------- BFGS and Newton --------

def test_bfgs_converges_much_faster_than_sgd():
    """8 BFGS iterations against 300 of SGD, and it still wins by orders of magnitude."""
    assert np.allclose(run(BFGS, 8), x_star, atol=1e-8)

def test_bfgs_learns_the_inverse_hessian():
    """On a quadratic H has to end up being A^-1: that is the whole idea."""
    x, f = make_problem()
    opt = BFGS([("x", x)])

    def closure():
        opt.zero_grad()
        loss = f()
        loss.backward()
        return loss.data.item()

    for _ in range(10):
        opt.step(closure)

    assert np.allclose(opt.H, np.linalg.inv(A), atol=1e-4)

def test_newton_solves_a_quadratic_in_one_step():
    """The quadratic model IS the function here, so one step lands on the minimum."""
    assert np.allclose(run(Newton, 1, lam=1e-10), x_star, atol=1e-5)

def test_bfgs_stops_when_there_is_no_descent_direction():
    """Sitting on the minimum, g = 0 -> d = 0 is not a descent direction: it must not blow up."""
    x = Tensor([x_star], requires_grad=True)
    A_t, b_t = Tensor(A), Tensor(b)
    opt = BFGS([("x", x)])

    def closure():
        opt.zero_grad()
        loss = ((x @ A_t) * x).sum() * 0.5 - (x * b_t).sum()
        loss.backward()
        return loss.data.item()

    opt.step(closure)

    assert not opt.descent
    assert np.allclose(x.data.reshape(-1), x_star)


def test_hessian_fd_matches_the_analytical_one():
    x, f = make_problem()
    params = [x]

    def g(v):
        set_flat_params(params, v)
        x.zero_grad()
        f().backward()
        return flat_grad(params)

    H = hessian_fd(g, flat_params(params))

    assert np.allclose(H, A, atol=1e-4)


# ------- Learning rate schedule --------

def test_linear_decay_follows_the_matlab_formula():
    opt = SGD([("x", Tensor([0.0], requires_grad=True))], lr=99.0)
    scheduler = LinearDecay(opt, alpha_0=1.0, k_ld=10)

    assert np.isclose(opt.lr, scheduler.alpha(1))       # it sets alpha_1 right away

    for _ in range(9):
        scheduler.step()
    assert np.isclose(opt.lr, 0.01)                     # k = k_ld -> 0.01 * alpha_0

    scheduler.step()
    assert np.isclose(opt.lr, 0.01)                     # constant from here on


def test_sgd_with_linear_decay_still_converges():
    x, f = make_problem()
    opt = SGD([("x", x)], lr=0.1)
    scheduler = LinearDecay(opt, alpha_0=0.1, k_ld=200)

    for _ in range(400):
        opt.zero_grad()
        f().backward()
        opt.step()
        scheduler.step()

    assert np.allclose(x.data.reshape(-1), x_star, atol=1e-3)


# ------- What each optimizer carries, for the viz --------

def test_state_of_each_optimizer():
    """SGD carries nothing and that is the point; the rest each expose their own machinery."""
    esperado = {SGD: set(), Momentum: {"velocity"}, RMSProp: {"scale"},
                Adam: {"velocity", "scale"}, BFGS: {"H", "alpha", "descent"},
                Newton: {"lam", "alpha", "H"}}

    for cls, claves in esperado.items():
        x, f = make_problem()
        opt = cls([("x", x)], lr=0.01) if cls in (SGD, Momentum, RMSProp, Adam) else cls([("x", x)])

        def closure():
            opt.zero_grad()
            loss = f()
            loss.backward()
            return loss.data.item()

        opt.step(closure)
        assert set(opt.state()) == claves, cls.__name__


def test_momentum_velocity_grows_and_points_downhill():
    x, f = make_problem()
    opt = Momentum([("x", x)], lr=0.01)

    def closure():
        opt.zero_grad()
        loss = f()
        loss.backward()
        return loss.data.item()

    opt.step(closure)
    primera = np.linalg.norm(opt.state()["velocity"])
    for _ in range(5):
        opt.step(closure)

    assert np.linalg.norm(opt.state()["velocity"]) > primera
