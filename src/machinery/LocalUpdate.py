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

from src.models.CostModel import ACostModel
from src.machinery.Parameters import Parameters

from abc import ABC, abstractmethod


class AbstractLocalUpdate(ABC):
    """
    The AbstractLocalUpdate class declares the factory methods while subclasses provide the implementation of this 
    methods.

    This class carries out the local update on each worker.
    """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.parameters = parameters
        # cost_model = cost_model

        # Local memory.
        self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # Local delta (information that is sent to central server).
        self.delta_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.g_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # Initialization of model's parameter.
        self.model_param = torch.FloatTensor([0 for i in range(self.parameters.n_dimensions)]).to(dtype=torch.float64)

        self.error_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def set_initial_v(self, v):
        """Initialize v (gradient-like, updated using previous gradients, required to use a momentum)"""
        self.v = v

    @abstractmethod
    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float) -> None:
        """Send the global model to the local worker and update its model."""
        pass

    def compute_local_gradient(self, cost_model: ACostModel, full_nb_iterations: int) -> None:
        """Compute the local gradient of the worker."""
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
        if self.parameters.fraction_sampled_workers == 1: # TODO : There is issue with PP and multiple memories
            if full_nb_iterations == 1 and self.parameters.use_up_memory:
                self.h_i = self.g_i

    @abstractmethod
    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None)\
            -> torch.FloatTensor:
        """Computes local gradient and then returns the delta vectors (what is sent to central server).
        
        Delta vectors is the vector computed on a worker and sent to the central server. It is either the gradient, 
        either its compressed version.

        :param full_nb_iterations: total number of computed iteration
        :return:
        """
        pass


class LocalGradientVanillaUpdate(AbstractLocalUpdate):
    """Implementation of the vanilla Stochastic Gradient local update.

    Paradigm of SGD: no compression, shares gardients.
    """

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None) \
            -> torch.FloatTensor:
        self.compute_local_gradient(cost_model, full_nb_iterations)

        self.delta_i = self.g_i - self.h_i
        if self.parameters.use_up_memory:
            self.h_i = self.h_i + self.parameters.up_learning_rate * self.delta_i
        return self.delta_i

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float) -> None:
        for tensor in tensor_sent:
            self.v = deepcopy(self.parameters.momentum * self.v + tensor)
            self.model_param = deepcopy(self.model_param - step * self.v)


class LocalDianaUpdate(AbstractLocalUpdate):
    """Implementation of the Diana local update.

    Paradigm of Diana: uplink compression, shares gardients, with or without up memories.
    """

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float) -> None:
        for tensor in tensor_sent:
            self.v = deepcopy(self.parameters.momentum * self.v + tensor)
            self.model_param = deepcopy(self.model_param - step * self.v)

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None) \
            -> torch.FloatTensor:
        self.compute_local_gradient(cost_model, full_nb_iterations)
        if self.g_i is None:
            return None

        self.delta_i = deepcopy(self.g_i - self.h_i)
        quantized_delta_i = self.parameters.up_compression_model.compress(self.delta_i)
        if self.parameters.use_up_memory:
            self.h_i = self.h_i + self.parameters.up_learning_rate * quantized_delta_i
        return quantized_delta_i


class LocalArtemisUpdate(AbstractLocalUpdate):
    """Implementation of the Artemis local update.

    Paradigm of Artemis: uplink/downlink compression, shares gardients, with or without up memories.
    """

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)

        # For bidirectional compression :
        self.H_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float) -> None:

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

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None)\
            -> torch.FloatTensor:
        self.compute_local_gradient(cost_model, full_nb_iterations)
        if self.g_i is None:
            return None

        self.delta_i = (self.g_i - self.h_i) + self.error_i * self.parameters.error_feedback_coef
        quantized_delta_i = self.parameters.up_compression_model.compress(self.delta_i)
        if self.parameters.up_error_feedback:
            self.error_i = self.delta_i - quantized_delta_i
        if self.parameters.use_up_memory:
            self.h_i = self.h_i + self.parameters.up_learning_rate * quantized_delta_i
        return quantized_delta_i

class LocalFedAvgUpdate(AbstractLocalUpdate):
    """Implementation of the FedAvg local update.

    Paradigm of FedAvg: uplink compression (or not), receive a model from the central server, no memory.
    """

    def send_global_informations_and_update_local_param(self, tensor_sent: torch.FloatTensor, step: float) -> None:
        self.model_param = copy(tensor_sent)

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None) \
            -> torch.FloatTensor:
        original_model_param = copy(self.model_param)
        for local_iteration in range (self.parameters.nb_local_update):
            self.compute_local_gradient(cost_model, full_nb_iterations)
            self.model_param = copy(self.model_param - step_size * self.g_i)
        return self.parameters.up_compression_model.compress(self.model_param - original_model_param)


class LocalDownCompressModelUpdate(AbstractLocalUpdate):
    """Implementation of the MCM-like algorithms local update.

    Paradigm of MCM: uplink compression of gradients, downlink compression of model, preserve central model,
    use up/down memories.
    """

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

    def compute_locally(self, cost_model: ACostModel, full_nb_iterations: int, step_size: float = None) \
            -> torch.FloatTensor:
        self.compute_local_gradient(cost_model, full_nb_iterations)
        if self.g_i is None:
            return None

        self.delta_i = deepcopy((self.g_i - self.h_i) + self.error_i)
        quantized_delta_i = self.parameters.up_compression_model.compress(self.delta_i)
        if self.parameters.use_up_memory:
            self.h_i = deepcopy(self.h_i + self.parameters.up_learning_rate * quantized_delta_i)
        return quantized_delta_i


class LocalGhostUpdate(LocalArtemisUpdate):
    """Implementation of Ghost local update.
    This algorithm is impossible to implement in real life.

    Paradigm of Ghost: uplink compression of gradient, downlink compression of model, with or without memories,
    preserve the central model. Ghost updates the local model using the global model (which is impossible in practice).
    """

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
