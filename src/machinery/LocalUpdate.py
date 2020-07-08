"""
Created by Philippenko, 6th March 2020.

In this python file is provided tools to implement any local update scheme. This is the part which is perfomed on local
devices in federated learning.
"""

import torch
import numpy as np

from src.models.CostModel import ACostModel
from src.machinery.Parameters import Parameters
from src.models.QuantizationModel import s_quantization

from abc import ABC, abstractmethod


class AbstractLocalUpdate(ABC):

    def __init__(self, parameters: Parameters, cost_model: ACostModel) -> None:
        super().__init__()
        self.parameters = parameters
        self.cost_model = cost_model

        self.g_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # Initialization of model's parameter.
        # self.model_param = torch.zeros(self.parameters.n_dimensions).to(dtype=torch.float64)
        self.model_param = torch.FloatTensor(
            [(-1 ** i) / (2 * self.parameters.n_dimensions) for i in range(self.parameters.n_dimensions)]) \
            .to(dtype=torch.float64)

    def set_initial_v(self, v):
        self.v = v

    @abstractmethod
    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        pass

    @abstractmethod
    def compute(self, j: int):
        pass


class LocalGradientVanillaUpdate(AbstractLocalUpdate):

    def __init__(self, parameters: Parameters, cost_model: ACostModel) -> None:
        super().__init__(parameters, cost_model)

    def compute(self, j: int):
        if self.parameters.stochastic:
            x = torch.stack([self.cost_model.X[j]])
            y = torch.stack([self.cost_model.Y[j]])
            self.g_i = self.cost_model.grad_i(self.model_param, x, y)
        else:
            self.g_i = self.cost_model.grad(self.model_param)
        return self.g_i

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        self.v = self.parameters.momentum * self.v + tensor_sent
        self.model_param = self.model_param - step * self.v


class LocalDianaUpdate(AbstractLocalUpdate):

    def __init__(self, parameters: Parameters, cost_model: ACostModel) -> None:
        super().__init__(parameters, cost_model)
        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        self.v = self.parameters.momentum * self.v + tensor_sent
        self.model_param = self.model_param - step * self.v

    def compute(self, j: int):
        if self.parameters.stochastic:
            x = torch.stack([self.cost_model.X[j]])
            y = torch.stack([self.cost_model.Y[j]])
            self.g_i = self.cost_model.grad_i(self.model_param, x, y)
        else:
            self.g_i = self.cost_model.grad(self.model_param)

        self.delta_i = self.g_i - self.h_i
        quantized_delta_i = s_quantization(self.delta_i, self.parameters.quantization_param)
        self.h_i += self.parameters.learning_rate * quantized_delta_i
        return quantized_delta_i


class LocalArtemisUpdate(AbstractLocalUpdate):
    """This class carry out the local update of the Artemis algorithm."""

    def __init__(self, parameters: Parameters, cost_model: ACostModel) -> None:
        super().__init__(parameters, cost_model)
        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For bidirectional compression :
        self.l_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        learning_rate_down = self.parameters.learning_rate
        if self.parameters.bidirectional:
            # l_i must be update with true omega, not with it "unzip" version which corresponds to compress model param.
            # As we override model_param, we need to update l_i in the same operation,
            # to benefit from the true model_param.
            if self.parameters.double_use_memory:
                decompressed_value, self.l_i = tensor_sent + self.l_i, self.l_i + learning_rate_down * tensor_sent
            else:
                decompressed_value = tensor_sent
        else:
            decompressed_value = tensor_sent

        # Updating the model with the new gradients.
        if self.parameters.compress_gradients and self.parameters.bidirectional:
            self.v = self.parameters.momentum * self.v + decompressed_value
            self.model_param = self.model_param - step * self.v
        else:
            self.model_param = decompressed_value

    def compute(self, j: int):
        if self.parameters.stochastic:
            x = torch.stack([self.cost_model.X[j]])
            y = torch.stack([self.cost_model.Y[j]])
            self.g_i = self.cost_model.grad_i(self.model_param, x, y)
        else:
            self.g_i = self.cost_model.grad(self.model_param)

        self.delta_i = self.g_i - self.h_i
        quantized_delta_i = s_quantization(self.delta_i, self.parameters.quantization_param)
        self.h_i += self.parameters.learning_rate * quantized_delta_i
        return quantized_delta_i
