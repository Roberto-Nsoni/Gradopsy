import numpy as np

from ml.tensor import Tensor
from ml.nn import MLP, Linear, Tanh, CrossEntropyFromLogits
from ml.optim import Adam
from ml.data import make_spirals
from ml.training import Trainer


def test_save_and_load_gives_the_same_predictions(tmp_path):
    np.random.seed(0)
    X, y = make_spirals(30, 3, seed=0)

    trained = MLP([2, 8, 3], Tanh)
    trainer = Trainer(trained, CrossEntropyFromLogits(), Adam(trained.parameters(), lr=0.05))
    for _ in range(50):
        trainer.step(X, y)

    path = tmp_path / "model.npz"
    trained.save(path)

    loaded = MLP([2, 8, 3], Tanh)
    assert not np.allclose(loaded(Tensor(X)).data, trained(Tensor(X)).data)   # empieza distinto

    loaded.load(path)
    assert np.array_equal(loaded(Tensor(X)).data, trained(Tensor(X)).data)


def test_saved_keys_are_the_parameter_names(tmp_path):
    path = tmp_path / "layer.npz"
    Linear(3, 2).save(path)

    assert sorted(np.load(path).files) == ["W", "b"]


def test_load_keeps_training(tmp_path):
    """A loaded model is a normal model: the gradients still flow through it."""
    np.random.seed(0)
    X, y = make_spirals(30, 3, seed=0)

    path = tmp_path / "model.npz"
    MLP([2, 8, 3], Tanh).save(path)

    model = MLP([2, 8, 3], Tanh)
    model.load(path)

    trainer = Trainer(model, CrossEntropyFromLogits(), Adam(model.parameters(), lr=0.05))
    first = trainer.step(X, y)["loss"]
    for _ in range(50):
        last = trainer.step(X, y)["loss"]

    assert last < first
