"""
Created by Philippenko, 6 January 2020.

This class defines the update methods which will be used during the gradient descent. It aim to implement the update
scheme used on the central server.

To add a new update scheme, just extend the abstract class AbstractGradientUpdate which contains methods:
1. to compute the cost at present model's parameters
2. to carry out the update of the model
3. to compute the step size
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod, ABCMeta
import torch
import random
from typing import Tuple
import numpy as np
import time

from src.machinery.Parameters import Parameters
from src.models.QuantizationModel import s_quantization


class AbstractGradientUpdate(ABC):
    """
    The  interface declares the operations that all concrete products of AGradient Descent must implement.
    """

    time_sample = 0

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__()
        self.parameters = parameters
        self.step = 0
        self.all_gradients = []

    def compute_cost(self, model_param):
        """Compute the cost function for the model's parameter."""
        loss, _ = self.parameters.cost_model.cost(model_param)
        return loss.item()

    def initialization(self, nb_it: int, model_param: torch.FloatTensor, L: float):
        self.step = self.__step__(nb_it, L)
        if nb_it == 1:
            if self.parameters.momentum != 0:
                self.v = self.compute_full_gradients(model_param)
            else:
                self.v = torch.zeros_like(model_param)
            # Initialization of v_-1 (for momentum)
            for worker in self.workers:
                worker.local_update.set_initial_v(self.v)

        self.all_gradients = []

    @abstractmethod
    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the model's update.

        This  method must update the model's parameter according to the scheme.

        Args:
            model_param: the parameters of the model before the update
            nb_it: index of epoch (iteration over full local data)
            j: index of inside iterations

        Returns:
            the new model.

        """
        pass

    def __step__(self, it: int, L: float):
        """Compute the step size at iteration *it*."""
        if it == 0:
            return 0
        step = self.parameters.step_formula(it, L, self.parameters.omega_c,
                                            self.parameters.nb_devices)
        return step


class AbstractFLUpdate(AbstractGradientUpdate, metaclass=ABCMeta):

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.workers = workers

    def compute_aggregation(self, local_information_to_aggregate):
        # In Artemis there is no weight associated with the aggregation, all nodes must have the same weight equal
        # to 1 / len(workers), this is why, an average is enough.
        return torch.stack(local_information_to_aggregate).mean(0)


    def compute_full_gradients(self, model_param):
        grad = 0
        for worker in self.workers:
            grad += worker.cost_model.grad(model_param)
        return grad / len(self.workers)

    def compute_cost(self, model_param, cost_models):
        all_loss_i = []
        for worker, cost_model in zip(self.workers, cost_models):
            loss_i, _ = cost_model.cost(model_param)
            all_loss_i.append(loss_i.item())
        return np.mean(all_loss_i)


class ArtemisUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potential memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.value_to_compress = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.omega = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For unidirectional compression:
        self.h = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For bidirectional compression:
        self.l = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L)

        all_delta_i = []

        for worker, cost_model in zip(self.workers, cost_models):
            # We get all the compressed gradient computed on each worker.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, j)

            # If nothing is returned by the device, this device does not participate to the learning at this iterations.
            # This may happened if it is considered that during one epoch each devices should run through all its data
            # exactly once, and if there is different numbers of points on each device.
            if compressed_delta_i is not None:
                all_delta_i.append(compressed_delta_i)

        # Aggregating all delta
        delta = self.compute_aggregation(all_delta_i)
        # Computing new (compressed) gradients
        self.g = self.h + delta

        # We update omega (compression of the sum of compressed gradients).
        self.value_to_compress = self.g - self.l
        omega = s_quantization(self.value_to_compress, self.parameters.quantization_param)

        # Updating the model with the new gradients.
        self.v = self.parameters.momentum * self.v + (omega + self.l)
        model_param = model_param - self.v * self.step

        # Send back omega to all workers and update their local model.
        for worker in self.workers:
            worker.local_update.send_global_informations_and_update_local_param(omega, self.step)

        # Update the second memory if we are using bidirectional compression and that this feature has been turned on.
        if self.parameters.double_use_memory:
            self.l += self.parameters.learning_rate * omega
        self.h += self.parameters.learning_rate * delta
        return model_param


class DianaUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potentiel memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)
        self.value_to_quantized = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For unidirectional compression:
        self.h = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L)

        all_delta_i = []

        for worker, cost_model in zip(self.workers, cost_models):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            quantized_delta_i = worker.local_update.compute_locally(cost_model, j)
            if quantized_delta_i is not None:
                all_delta_i.append(quantized_delta_i)

        # Aggregating all delta
        delta = self.compute_aggregation(all_delta_i)
        # Computing new (compressed) gradients
        self.g = self.h + delta

        self.v = self.parameters.momentum * self.v + self.g
        model_param = model_param - self.v * self.step

        # Send omega to all workers and update their local model.
        for worker in self.workers:
            worker.local_update.send_global_informations_and_update_local_param(self.g, self.step)
        self.h += self.parameters.learning_rate * delta
        return model_param


class GradientVanillaUpdate(AbstractFLUpdate):

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L)

        for worker, cost_model in zip(self.workers, cost_models):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            gradient_i = worker.local_update.compute_locally(cost_model, j)
            if gradient_i is not None:
                self.all_gradients.append(gradient_i)

        # Aggregating all gradients
        self.g = self.compute_aggregation(self.all_gradients)
        self.v = self.parameters.momentum * self.v + self.g

        model_param = model_param - self.v * self.step

        # Send global gradient to all workers and update their local model.
        for worker in self.workers:
            worker.local_update.send_global_informations_and_update_local_param(self.g, self.step)

        return model_param