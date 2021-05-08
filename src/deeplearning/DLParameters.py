"""
Created by Philippenko, 26th April 2021.
"""
from src.deeplearning.NnModels import SimplestNetwork, MNIST_CNN
from src.machinery.Parameters import Parameters


class DLParameters(Parameters):

    def __init__(self):
        super().__init__()
        self.optimal_step_size = None
        self.dataset = "mnist"
        self.model = MNIST_CNN()

    def print(self):
        print("== Settings ==")
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


def cast_to_DL(parameters: Parameters) -> DLParameters:
    parameters.__class__ = DLParameters
    parameters.optimal_step_size = None
    parameters.dataset = "mnist"
    parameters.model = MNIST_CNN()
    return parameters