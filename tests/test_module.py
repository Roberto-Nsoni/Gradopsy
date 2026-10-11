import numpy as np

from ml.tensor import Tensor
from ml.nn import Linear, Sequential, MLP, ReLU, Tanh

np.random.seed(0)


def test_sequential_chains_the_modules():
    model = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
    out = model(Tensor(np.random.randn(4, 3)))

    assert out.shape == (4, 2)


def test_sequential_collects_every_parameter():
    model = Sequential(Linear(3, 5), ReLU(), Linear(5, 2))
    names = [name for name, _ in model.parameters()]

    assert names == ["0.W", "0.b", "2.W", "2.b"]


def test_mlp_shapes():
    model = MLP([2, 8, 8, 3], ReLU)
    out = model(Tensor(np.zeros((5, 2))))

    assert out.shape == (5, 3)


def test_mlp_ends_in_linear():
    """No activation on the output: the Cross-Entropy wants the raw logits."""
    model = MLP([2, 4, 3], Tanh)
    layers = list(model._modules.values())

    assert isinstance(layers[-1], Linear)
    assert len(layers) == 3          # Linear, Tanh, Linear


def test_mlp_activations_are_separate_instances():
    model = MLP([2, 4, 4, 3], ReLU)
    activations = [m for m in model._modules.values() if isinstance(m, ReLU)]

    assert len(activations) == 2
    assert activations[0] is not activations[1]


def test_mlp_backward_reaches_every_layer():
    model = MLP([2, 4, 3], ReLU)
    model(Tensor(np.random.randn(6, 2))).sum().backward()

    for _, p in model.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape


def test_linear_without_bias():
    layer = Linear(3, 2, bias=False)

    assert layer.b is None
    assert [name for name, _ in layer.parameters()] == ["W"]


def test_conv_net_shapes():
    import numpy as np
    from ml.tensor import Tensor
    from ml.nn import Conv2D, MaxPool2D, Flatten, Sequential, Linear, ReLU
    net = Sequential(Conv2D(1, 8, 5, padding=2), ReLU(), MaxPool2D(2),
                     Conv2D(8, 16, 5, padding=2), ReLU(), MaxPool2D(2), Flatten(), Linear(16 * 7 * 7, 10))
    assert net(Tensor(np.zeros((3, 1, 28, 28)))).shape == (3, 10)
    assert sum(p.size for _, p in net.parameters()) == 8 * 25 + 8 + 16 * 8 * 25 + 16 + 784 * 10 + 10
