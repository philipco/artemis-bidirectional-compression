"""
Created by Philippenko, 6th March 2020.
"""

import torch
from copy import deepcopy

from src.machinery.LocalUpdate import LocalArtemisUpdate
from src.machinery.Parameters import Parameters


class Worker:
    """This class modelize a worker.

    A worker hold an ID, a dataset (X,Y), a local model, a cost model
    and a local update method (to carry out the local step of the federated gradient descent).
    """

    def __init__(self, ID : int, parameters: Parameters) -> None:
        super().__init__()
        self.cost_model = deepcopy(parameters.cost_model)
        self.local_update = LocalArtemisUpdate(parameters, self.cost_model)
        self.model_param = None
        self.ID = ID

    def set_data(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> None:
        """Set data on worker.

        Args:
            X: Data.
            Y: labels.
        """
        self.X, self.Y = X, Y
        self.cost_model.set_data(self.X, self.Y)
