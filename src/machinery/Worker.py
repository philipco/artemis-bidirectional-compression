"""
Created by Philippenko, 6th March 2020.
"""

from src.machinery.Parameters import Parameters


class Worker:
    """This class modelize a worker.

    A worker hold an ID, a dataset (X,Y), a local model, a cost model
    and a local update method (to carry out the local step of the federated gradient descent).
    """

    def __init__(self, ID : int, parameters: Parameters, localUpdate) -> None:
        super().__init__()
        self.local_update = localUpdate(parameters)
        self.idx_last_update = 0
        self.ID = ID

    def set_idx_last_update(self, j: int):
        self.idx_last_update = j
