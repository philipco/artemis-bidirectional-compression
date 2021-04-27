"""
Created by Philippenko, 26th April 2021.
"""
from src.machinery.Parameters import Parameters


class DLParameters(Parameters):

    def __init__(self):
        super().__init__()
        self.optimal_step_size = None


def cast_to_DL(parameters: Parameters) -> DLParameters:
    parameters.__class__ = DLParameters
    return parameters