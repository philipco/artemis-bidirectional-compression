"""
Created by Philippenko, 26th April 2021.

This file gathers all settings used to run a DL experiment.
"""
from src.machinery.Parameters import Parameters


class DLParameters(Parameters):
    """Settings for DL, extend Parameters class."""

    def __init__(self, dataset: str, model, optimal_step_size: int, weight_decay: int, iid: str):
        super().__init__(None)
        self.initialize_DL_params(dataset, model, optimal_step_size, weight_decay, iid)

    def initialize_DL_params(self, dataset: str, model, optimal_step_size: int, weight_decay: int, iid: str):
        self.optimal_step_size = optimal_step_size
        self.dataset = dataset
        self.model = model
        self.weight_decay = weight_decay
        self.iid = iid
        self.criterion = None

    def print(self):
        print("== Settings ==")
        if self.optimal_step_size is not None:
            print("Step size: {0}".format(self.optimal_step_size))
        if self.use_up_memory:
            print("Use UP memory.")
        if self.use_down_memory:
            print("Use DOWN memory.")
        if self.up_error_feedback:
            print("Use UP error-feedback.")
        if self.down_error_feedback:
            print("Use DOWN error-feedback.")
        if self.up_compression_model is not None:
            print("UP compression model: {0}".format(self.up_compression_model.__class__.__name__))
        if self.down_compression_model is not None:
            print("DOWN compression model: {0}".format(self.down_compression_model.__class__.__name__))
        if self.criterion is not None:
            print("Criterion: {0}".format(self.criterion.__class__.__name__))


def cast_to_DL(parameters: Parameters, dataset: str, model, optimal_step_size: int, weight_decay: int, iid: str) \
        -> DLParameters:
    """Casts a Parameters class in a DLParameters class which is the specific format for deep learning."""
    parameters.__class__ = DLParameters
    parameters.initialize_DL_params(dataset, model, optimal_step_size, weight_decay, iid)
    return parameters