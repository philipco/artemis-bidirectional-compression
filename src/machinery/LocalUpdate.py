"""
Created by Philippenko, 6th March 2020.

In this python file is provided tools to implement any local update scheme. This is the part which is perfomed on local
devices in federated learning.
"""
import random
from copy import deepcopy, copy

import scipy.sparse as sp
import torch
import numpy as np
from scipy.optimize import least_squares

from src.models.CostModel import ACostModel
from src.machinery.Parameters import Parameters

from abc import ABC, abstractmethod

from src.utils.Constants import BETA


class AbstractLocalUpdate(ABC):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.parameters = parameters
        # cost_model = cost_model

        # Local memory.
        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.previous_h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.averaged_h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.nb_it = 0

        # Local delta (information that is sent to central server).
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.g_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # Initialization of model's parameter.
        self.model_param = torch.FloatTensor([0 for i in range(self.parameters.n_dimensions)]).to(dtype=torch.float64)

        self.error_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def set_initial_v(self, v):
        self.v = v

    @abstractmethod
    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        pass

    def compute_local_gradient(self, cost_model: ACostModel, full_nb_iterations: int):
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

        # Smart initialisation of the memory (it corresponds to the first computed gradient).
        if full_nb_iterations == 1 and self.parameters.use_up_memory:
            self.h_i = self.g_i
            self.averaged_h_i = self.g_i

    @abstractmethod
    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float):
        """

        :param full_nb_iterations: total number of computed iteration
        :return:
        """
        pass

    def optimal_memory(self, worker_mem, coef_mem):
        res = 0
        for i in range(len(coef_mem)):
            res += worker_mem[i] * coef_mem[i]
        return res

    def find_optimal_coef_mem(self):

        def memory(x):
            mem = 0
            for i in range(len(self.h_i)):
                mem += x[i] * self.h_i[i]
            return np.array(self.g_i - mem)

        self.coef_mem = [0 for i in range(len(self.h_i))]
        res_1 = least_squares(memory, self.coef_mem, bounds=(-1, 1))
        self.coef_mem = res_1.x

    def which_mem(self, h_i, averaged_h_i):
        # self.find_optimal_coef_mem()
        # if self.nb_it >= 0:
        #     self.delta_i = self.g_i - self.averaged_h_i
        # else:
        if self.parameters.up_enhanced_up_mem:
            return averaged_h_i
        else:
            return h_i
        # self.delta_i = self.g_i - self.averaged_h_i#self.optimal_memory(self.h_i, self.coef_mem)

    def update_average_mem(self, h_i, average_mem, nb_it):
        rho = 0.95
        # Classic
        # return h_i
        # Weighted average
        # coef1 = rho * (1 - rho ** nb_it)  / (1 - rho ** (nb_it + 1))
        # coef2 = (1 - rho)  / (1 - rho ** (nb_it + 1))
        # return average_mem.mul(coef1) + h_i.mul(coef2)
        # Average
        return (1 - 1 / (nb_it + 1)) * average_mem + 1 / (nb_it + 1) * h_i


class LocalGradientVanillaUpdate(AbstractLocalUpdate):

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None):
        self.compute_local_gradient(cost_model, full_nb_iterations)

        self.delta_i = self.g_i - self.h_i
        if self.parameters.use_up_memory:
            self.h_i += self.parameters.up_learning_rate * self.delta_i
        return self.delta_i

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        for tensor in tensor_sent:
            self.v = deepcopy(self.parameters.momentum * self.v + tensor)
            self.model_param = deepcopy(self.model_param - step * self.v)


class LocalDianaUpdate(AbstractLocalUpdate):

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        for tensor in tensor_sent:
            self.v = deepcopy(self.parameters.momentum * self.v + tensor)
            self.model_param = deepcopy(self.model_param - step * self.v)

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None):
        self.compute_local_gradient(cost_model, full_nb_iterations)
        if self.g_i is None:
            return None

        self.delta_i = self.g_i - self.which_mem(self.h_i, self.averaged_h_i) #deepcopy(self.g_i - self.h_i)
        quantized_delta_i = self.parameters.up_compression_model.compress(self.delta_i)
        if self.parameters.use_up_memory:
            self.previous_h_i = self.h_i
            self.h_i = self.h_i + self.parameters.up_learning_rate * quantized_delta_i + \
                       [0, self.parameters.up_learning_rate * (self.averaged_h_i - self.h_i)][
                           self.parameters.up_enhanced_up_mem]
            self.nb_it += 1
            self.averaged_h_i = self.update_average_mem(self.h_i, self.averaged_h_i, self.nb_it)
        return quantized_delta_i


class LocalArtemisUpdate(AbstractLocalUpdate):
    """This class carry out the local update of the Artemis algorithm."""

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)

        # For bidirectional compression :
        self.H_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):

        for tensor in tensor_sent:

            # H_i must be update with true omega, not with it "unzip" version which corresponds to compress model param.
            if self.parameters.use_down_memory:
                decompressed_value = tensor + self.H_i
                self.H_i = self.H_i + self.parameters.down_learning_rate * tensor
            else:
                decompressed_value = tensor

            self.v = self.parameters.momentum * self.v + decompressed_value
            self.model_param = self.model_param - step * self.v

        if not self.parameters.use_down_memory:
            assert self.H_i.equal(torch.zeros(self.parameters.n_dimensions, dtype=np.float)), \
                "Downlink memory is not a zero tensor while the double-memory mechanism is switched-off."

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None):
        self.compute_local_gradient(cost_model, full_nb_iterations)
        if self.g_i is None:
            return None

        self.delta_i = self.g_i - self.which_mem(self.h_i, self.averaged_h_i) + self.error_i * self.parameters.error_feedback_coef
        quantized_delta_i = self.parameters.up_compression_model.compress(self.delta_i)
        if self.parameters.up_error_feedback:
            self.error_i = self.delta_i - quantized_delta_i
        if self.parameters.use_up_memory:
            # temp = self.h_i
            self.h_i = self.h_i + self.parameters.up_learning_rate * quantized_delta_i + \
                     [0, self.parameters.up_learning_rate * (self.averaged_h_i - self.h_i)][
                         self.parameters.up_enhanced_up_mem]

            # self.h_i = self.h_i + self.parameters.up_learning_rate * (quantized_delta_i + self.h_i - self.averaged_h_i)
            # When using a moment to update the memory
            # if self.parameters.up_enhanced_up_mem:
            #     self.h_i += BETA * (temp - self.previous_h_i)
            # self.previous_h_i = temp
            self.nb_it += 1
            self.averaged_h_i = self.update_average_mem(self.h_i, self.averaged_h_i, self.nb_it)
        return quantized_delta_i

class LocalFedAvgUpdate(AbstractLocalUpdate):

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float):
        self.model_param = copy(tensor_sent)

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float):
        original_model_param = copy(self.model_param)
        for local_iteration in range (self.parameters.nb_local_update):
            self.compute_local_gradient(cost_model, full_nb_iterations)
            self.model_param = copy(self.model_param - step_size * self.g_i)
        return self.parameters.up_compression_model.compress(self.model_param - original_model_param)


class LocalDownCompressModelUpdate(AbstractLocalUpdate):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)

        # Memory for bidirectional compression.
        # There is no need for smart initialiation as we take w_0 = 0.
        self.H_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float,
                                                        H_i: torch.FloatTensor = None):

        if H_i is not None:
            self.H_i = H_i

        if self.parameters.use_down_memory:
            decompressed_value = tensor_sent + self.H_i
            self.H_i = self.H_i + self.parameters.down_learning_rate * tensor_sent
        else:
            decompressed_value = tensor_sent
        self.model_param = decompressed_value

        if not self.parameters.use_down_memory:
            assert self.H_i.equal(torch.zeros(self.parameters.n_dimensions, dtype=np.float)), \
                "Downlink memory is not a zero tensor while the double-memory mechanism is switched-off."

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None):
        self.compute_local_gradient(cost_model, full_nb_iterations)
        if self.g_i is None:
            return None

        self.delta_i = deepcopy((self.g_i - self.h_i) + self.error_i)
        quantized_delta_i = self.parameters.up_compression_model.compress(self.delta_i)
        if self.parameters.use_up_memory:
            self.h_i = deepcopy(self.h_i + self.parameters.up_learning_rate * quantized_delta_i)
        return quantized_delta_i


class LocalSympaUpdate(LocalArtemisUpdate):
    """This class carry out the local update of the Artemis algorithm."""

    def send_global_informations_and_update_local_param(self, tuple_sent: torch.FloatTensor, step: float):

        g, model_param = tuple_sent

        # H_i must be update with true omega, not with it "unzip" version which corresponds to compress model param.
        # As we override model_param, we need to update H_i in the same operation,
        # to benefit from the true model_param.
        if self.parameters.use_down_memory:
            decompressed_value, self.H_i = g + self.H_i, self.H_i + self.parameters.down_learning_rate * g
        else:
            decompressed_value = g

        # Updating the model with the new gradients.
        self.model_param = model_param - step * decompressed_value

        if not self.parameters.use_down_memory:
            assert self.H_i.equal(torch.zeros(self.parameters.n_dimensions, dtype=np.float)), \
                "Downlink memory is not a zero tensor while the double-memory mechanism is switched-off."

        return self.model_param
