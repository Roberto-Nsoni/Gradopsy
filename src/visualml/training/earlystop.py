import numpy as np

from visualml.optim import flat_params, set_flat_params

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience

        self.best_loss = np.inf
        self.best_params = None
        self.best_epoch = 0

        self.epoch = 0
        self.worse = 0

    def update(self, loss: float, params) -> bool:
        """Returns True when it is time to stop."""
        self.epoch += 1

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = flat_params(params)
            self.best_epoch = self.epoch
            self.worse = 0
        else:
            self.worse += 1

        return self.worse >= self.patience

    def restore(self, params) -> None:
        """Puts the model back on the parameters of the best epoch."""
        set_flat_params(params, self.best_params)
