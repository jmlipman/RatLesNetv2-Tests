import torch
from lib.utils import log

class CustomReduceLROnPlateau(object):
    """Reduces the learning rate when a plateau is reached.
    """

    def __init__(self, patience=4, factor=0.1, improvement_thr=0.01,
            limit=3, verbose=True):
        """Description.

           Args:
           `patience`: number of epochs the scheduler will ignore the worsening
            of the validation loss. If patience is 3, the learning rate will
            decrease if the validation loss gets worse 4 consecutive times.
           `factor`: factor in which the learning rate will decrease.
            learning_rate = factor * learning_rate
           `improvement_thr`: % of improvement that the learning must do,
            otherwise we will decrease the lr. The higher, the more strict.
            `limit`: after `limit` times of decreasing the lr, stop training.

        """
        self.losses = []
        self.optimizer = None
        self.patience = patience
        self.factor = factor
        self.improvement_thr = improvement_thr
        self.limit = limit
        self.limit_cnt = limit # When this is -1, stop training
        self.verbose = verbose

    def setOptimizer(self, opt):
        self.optimizer = opt

    def step(self, loss):
        """
           losses is a list of all the validation losses
        """
        if self.optimizer is None:
            raise Exception("LR Scheduler must have set an optimizer!")

        self.losses.append(loss)
        if len(self.losses) > self.patience:
            decreases = [abs(1-self.losses[-i-1]/self.losses[-i]) < self.improvement_thr for i in range(self.patience, 0, -1)]
            if len(decreases) == sum(decreases):
                # Decrease learning rate
                for group in self.optimizer.param_groups:
                    group["lr"] *= self.factor
                self.limit_cnt -= 1
                self.losses = [loss]
                if self.verbose and self.limit_cnt > -1:
                    log("Decrease learning rate to: " + str(group["lr"]))

class CustomReduceLR(object):
    """Reduces the learning rate after certain epochs.
    """

    def __init__(self, epochs=None, factors=None, verbose=True):
        """Description.

           Args:
           `epochs`: List of epochs when the scheduler will be triggered
           `factors`: List of factors in which the learning rate will decrease.
            learning_rate = factor * learning_rate

           Example: epochs=[150, 250], factor=[0.1, 0.1]
           The learning rate will be decreased to lr*0.1 at epoch 150, and lr*0.01 at 250
        """
        if type(epochs) != list or type(factors) != list or len(epochs) != len(factors):
            raise Exception("Scheduler: `epochs` and `factors` must be list of the same size")

        self.limit_cnt = 1 # Not used but left due to compatibility reasons.
        self.optimizer = None
        self.losses = []
        self.curr_factor_i = 0
        self.epochs = epochs
        self.factors = factors
        self.verbose = verbose

    def setOptimizer(self, opt):
        self.optimizer = opt

    def step(self, loss):
        """
           losses is a list of all the validation losses
        """
        if self.optimizer is None:
            raise Exception("LR Scheduler must have set an optimizer!")

        self.losses.append(loss)

        if self.curr_factor_i < len(self.epochs) and len(self.losses) == self.epochs[self.curr_factor_i]:
            print(len(self.losses), self.epochs[self.curr_factor_i])
            for group in self.optimizer.param_groups:
                group["lr"] *= self.factors[self.curr_factor_i]
            if self.verbose and self.limit_cnt > -1:
                log("Decrease learning rate to: " + str(group["lr"]))
            self.curr_factor_i += 1

