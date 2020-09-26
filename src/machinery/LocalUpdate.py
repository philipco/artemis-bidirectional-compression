"""
Created by Philippenko, 6th March 2020.

In this python file is provided tools to implement any local update scheme. This is the part which is perfomed on local
devices in federated learning.
"""
import random
from copy import deepcopy

import scipy.sparse as sp
import torch
import numpy as np

from src.models.CostModel import ACostModel
from src.machinery.Parameters import Parameters
from src.models.QuantizationModel import s_quantization

from abc import ABC, abstractmethod


class AbstractLocalUpdate(ABC):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.parameters = parameters
        # cost_model = cost_model

        self.g_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # Initialization of model's parameter.
        self.model_param = torch.FloatTensor(
            [(-1 ** i) / (2 * self.parameters.n_dimensions) for i in range(self.parameters.n_dimensions)]) \
            .to(dtype=torch.float64)

    def set_initial_v(self, v):
        self.v = v

    @abstractmethod
    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        pass

    def compute_local_gradient(self, cost_model: ACostModel, j: int):
        # TODO : there may be issues with this ?
        if self.parameters.stochastic:
            # If batch size is bigger than number of sample of the device, we only take all its points.
            if self.parameters.batch_size > cost_model.X.shape[0]:
                self.g_i = cost_model.grad(self.model_param)
            else:
                idx = random.sample(list(range(cost_model.X.shape[0])), self.parameters.batch_size)
                if isinstance(cost_model.X, sp.csc.csc_matrix):
                    x = sp.hstack([cost_model.X[i] for i in idx])
                else:
                    x = torch.stack([cost_model.X[i] for i in idx])
                y = torch.stack([cost_model.Y[i] for i in idx])
                assert x.shape[0] == self.parameters.batch_size and y.shape[0] == self.parameters.batch_size, \
                    "The batch doesn't have the correct size, can not compute the local gradient."
                self.g_i = cost_model.grad_i(self.model_param, x, y)
        else:
            self.g_i = cost_model.grad(self.model_param)

    @abstractmethod
    def compute_locally(self, cost_model: ACostModel, j: int):
        pass


class LocalGradientVanillaUpdate(AbstractLocalUpdate):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)

        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute_locally(self, cost_model: ACostModel, j: int):
        self.compute_local_gradient(cost_model, j)

        self.delta_i = deepcopy(self.g_i - self.h_i)
        self.h_i += self.parameters.learning_rate * self.delta_i
        return self.delta_i

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        for tensor in tensor_sent:
            self.v = deepcopy(self.parameters.momentum * self.v + tensor)
            self.model_param = deepcopy(self.model_param - step * self.v)


class LocalDianaUpdate(AbstractLocalUpdate):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        for tensor in tensor_sent:
            self.v = deepcopy(self.parameters.momentum * self.v + tensor)
            self.model_param = deepcopy(self.model_param - step * self.v)

    def compute_locally(self, cost_model: ACostModel, j: int):
        self.compute_local_gradient(cost_model, j)
        if self.g_i is None:
            return None

        self.delta_i = deepcopy(self.g_i - self.h_i)
        quantized_delta_i = s_quantization(self.delta_i, self.parameters.quantization_param)
        self.h_i += self.parameters.learning_rate * quantized_delta_i
        return quantized_delta_i


class LocalArtemisUpdate(AbstractLocalUpdate):
    """This class carry out the local update of the Artemis algorithm."""

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For bidirectional compression :
        self.l_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        learning_rate_down = self.parameters.learning_rate

        for tensor in tensor_sent:

            # l_i must be update with true omega, not with it "unzip" version which corresponds to compress model param.
            # As we override model_param, we need to update l_i in the same operation,
            # to benefit from the true model_param.
            if self.parameters.double_use_memory:
                decompressed_value, self.l_i = tensor + self.l_i, self.l_i + learning_rate_down * tensor
            else:
                decompressed_value = tensor
            if not self.parameters.double_use_memory:
                assert self.l_i.equal(torch.zeros(self.parameters.n_dimensions, dtype=np.float)), \
                    "Downlink memory is not a zero tensor while the double-memory mechanism is switched-off."

            # Updating the model with the new gradients.
            self.v = deepcopy(self.parameters.momentum * self.v + decompressed_value)
            self.model_param = deepcopy(self.model_param - step * self.v)

    def compute_locally(self, cost_model: ACostModel, j: int):
        self.compute_local_gradient(cost_model, j)
        if self.g_i is None:
            return None

        self.delta_i = deepcopy(self.g_i - self.h_i)
        quantized_delta_i = s_quantization(self.delta_i, self.parameters.quantization_param)
        self.h_i += self.parameters.learning_rate * quantized_delta_i
        return quantized_delta_i
