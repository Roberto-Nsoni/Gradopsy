# The exact same race, so native Python and Pyodide can be compared number by number.
import time

from gradopsy.data import SURFACES
from gradopsy.optim import SGD, Momentum, RMSProp, Adam, BFGS, LBFGS, Newton
from gradopsy.training import SurfaceRunner, Recorder

CORREDORES = [("SGD", SGD, {"lr": 0.002}), ("Momentum", Momentum, {"lr": 0.002}),
              ("RMSProp", RMSProp, {"lr": 0.02}), ("Adam", Adam, {"lr": 0.05}),
              ("BFGS", BFGS, {}), ("L-BFGS", LBFGS, {}), ("Newton", Newton, {})]


def carrera(nombre_superficie="double_well", steps=200):
    superficie = SURFACES[nombre_superficie]
    inicio = time.perf_counter()
    resultado = {}

    for nombre, cls, kwargs in CORREDORES:
        runner = SurfaceRunner(superficie, cls, **kwargs)
        rec = Recorder()
        for _ in range(steps):
            rec.add(runner.step())
        resultado[nombre] = [round(v, 10) for v in runner.position]

    resultado["__segundos__"] = round(time.perf_counter() - inicio, 3)
    return resultado


def entrenamiento(steps=300):
    """The `boundary` case: a real MLP training live, which is what D13 has to decide about."""
    import numpy as np
    from gradopsy.nn import MLP, Tanh, CrossEntropyFromLogits
    from gradopsy.data import make_spirals
    from gradopsy.training import Trainer

    np.random.seed(0)
    X, y = make_spirals(100, 3, noise=0.15, seed=0)

    model = MLP([2, 16, 16, 3], Tanh)
    trainer = Trainer(model, CrossEntropyFromLogits(), Adam(model.parameters(), lr=0.05))

    inicio = time.perf_counter()
    for _ in range(steps):
        snap = trainer.step(X, y)

    return {"loss": round(snap["loss"], 8), "__segundos__": round(time.perf_counter() - inicio, 3)}
