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

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.step = 0
        self.all_gradients = []

    def compute_cost(self, model_param):
        """Compute the cost function for the model's parameter."""
        loss, _ = self.parameters.cost_model.cost(model_param)
        return loss.item()

    def initialization(self, nb_it: int, model_param: torch.FloatTensor):
        self.step = self.__step__(nb_it)
        if nb_it == 1:
            self.v = self.compute_full_gradients(model_param)

        self.all_gradients = []
        # Initialization of v_-1 (for momentum)
        if nb_it == 1:
            for worker in self.workers:
                worker.local_update.set_initial_v(self.v)

    @abstractmethod
    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
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

    def __step__(self, it: int):
        """Compute the step size at iteration *it*."""
        if it == 0:
            return 0
        step = self.parameters.step_formula(it, self.workers[0].cost_model.L, self.parameters.omega_c,
                                            self.parameters.nb_devices)
        return step


class AbstractFLUpdate(AbstractGradientUpdate, metaclass=ABCMeta):

    def compute_full_gradients(self, model_param):
        grad = 0
        for worker in self.workers:
            grad += worker.cost_model.grad(model_param)
        return grad / len(self.workers)

    def compute_cost(self, model_param):
        all_loss_i = []
        for worker in self.workers:
            loss_i, _ = worker.cost_model.cost(model_param)
            all_loss_i.append(loss_i.item())
        return np.mean(all_loss_i)


class ArtemisUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potential memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters)
        self.workers = workers

        self.value_to_quantized = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.omega = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For unidirectional compression:
        self.h = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For bidirectional compression:
        self.l = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param)

        all_delta_i = []

        for worker in self.workers:
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            quantized_delta_i = worker.local_update.compute_locally(j)
            if quantized_delta_i is not None:
                all_delta_i.append(quantized_delta_i)

        # Aggregating all delta
        delta = torch.stack(all_delta_i).mean(0)
        # Computing new (compressed) gradients
        self.g = self.h + delta

        # If we compress gradients, we update now omega.
        self.value_to_quantized = self.g - self.l
        omega = s_quantization(self.value_to_quantized, self.parameters.quantization_param)

        # Updating the model with the new gradients.
        self.v = self.parameters.momentum * self.v + (omega + self.l)
        model_param = model_param - self.v * self.step

        # Send omega to all workers and update their local model.
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
        super().__init__(parameters)
        self.workers = workers
        self.value_to_quantized = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # For unidirectional compression:
        self.h = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param)

        all_delta_i = []

        for worker in self.workers:
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            quantized_delta_i = worker.local_update.compute_locally(j)
            if quantized_delta_i is not None:
                all_delta_i.append(quantized_delta_i)

        # Aggregating all delta
        delta = torch.stack(all_delta_i).mean(0)
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
        super().__init__(parameters)
        self.workers = workers

        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param)

        for worker in self.workers:
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            gradient_i = worker.local_update.compute_locally(j)
            if gradient_i is not None:
                self.all_gradients.append(gradient_i)

        # Aggregating all gradients
        self.g = torch.stack(self.all_gradients).mean(0)
        self.v = self.parameters.momentum * self.v + self.g

        model_param = model_param - self.v * self.step

        # Send global gradient to all workers and update their local model.
        for worker in self.workers:
            worker.local_update.send_global_informations_and_update_local_param(self.g, self.step)

        return model_param


class BasicGradientUpdate(AbstractGradientUpdate):
    """Implementation of the update procedure for the basic gradient descent."""

    def __step__(self, it):
        return 1 / self.parameters.cost_model.L

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Updating Gradient
        grad = self.parameters.cost_model.grad(model_param)
        # Updating model
        model_param -= (self.__step__(nb_it) * grad)
        return model_param


class StochasticGradientUpdate(AbstractGradientUpdate):
    """Stochastic implementation of the update procedure."""

    def __step__(self, it):
        return 1 / (self.parameters.cost_model.L * math.sqrt(it))

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Randomly selecting a data point
        n_samples, n_dimension = self.parameters.cost_model.X.shape
        start = time.time()
        # random_element = random.sample(list(range(n_samples)), self.parameters.batch_size)
        x = torch.stack([self.parameters.cost_model.X[j]])  # for i in random_element])
        y = torch.stack([self.parameters.cost_model.Y[j]])  # for i in random_element])
        self.time_sample += (time.time() - start)
        # Computing Gradient
        grad = self.parameters.cost_model.grad_i(model_param, x, y)
        # Updating model
        model_param -= (self.__step__(nb_it) * grad)
        return model_param


class MomentumUpdate(StochasticGradientUpdate):
    """Momentum implementation of the update procedure."""

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.model_params = []

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        momentum_coef = 0.4
        # If list of model_params is empty we add the initializer.
        if not self.model_params:
            self.model_params.append(model_param)
        # Randomly selecting a data point
        n_samples, n_dimension = self.parameters.cost_model.X.shape
        random_element = random.sample(list(range(n_samples)), self.parameters.batch_size)
        x = torch.stack([self.parameters.cost_model.X[i] for i in random_element])
        y = torch.stack([self.parameters.cost_model.Y[i] for i in random_element])
        # Updating Gradient
        grad = self.parameters.cost_model.grad_i(model_param, x, y)
        moment = momentum_coef * (model_param - self.model_params[-1]) - self.__step__(nb_it) * grad
        self.model_params.append(model_param)
        # Updating model
        model_param = model_param + moment
        return model_param


class SAGUpdate(StochasticGradientUpdate):
    """SAG implementation of the update procedure."""

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        n_samples, n_dimension = self.parameters.cost_model.X.shape
        self.n_grads = torch.zeros((n_samples, n_dimension))

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Randomly selecting a data point
        n_samples, n_dimension = self.parameters.cost_model.X.shape
        random_element = random.sample(list(range(n_samples)), self.parameters.batch_size)
        x = torch.stack([self.parameters.cost_model.X[i] for i in random_element])
        y = torch.stack([self.parameters.cost_model.Y[i] for i in random_element])
        # Computing new gradients
        for i in random_element:
            # TODO : handle batch size
            self.n_grads[i] = self.parameters.cost_model.grad_i(model_param, x, y)
        # Updating model
        model_param = model_param - self.__step__(nb_it) * torch.mean(self.n_grads, dim=0)
        return model_param


class SVRGUpdate(StochasticGradientUpdate):
    """SVRG implementation of the update procedure."""

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        n_samples, n_dimension = self.parameters.cost_model.X.shape
        nb_inside_it = n_samples
        model_temp = model_param
        list_grad = [model_temp]
        # Compute full fradient
        grad = self.parameters.cost_model.grad(model_param)
        for i in range(nb_inside_it):
            # Randomly selecting a data point
            random_element = random.sample(list(range(n_samples)), self.parameters.batch_size)
            x = torch.stack([self.parameters.cost_model.X[i] for i in random_element])
            y = torch.stack([self.parameters.cost_model.Y[i] for i in random_element])
            # Updating Gradient
            grad_i = self.parameters.cost_model.grad_i(model_param, x, y)
            grad_i_temp = self.parameters.cost_model.grad_i(model_temp, x, y)
            model_temp = model_temp - self.__step__(nb_it * n_samples + i) * (grad_i_temp - grad_i + grad)
            list_grad.append(model_temp)
        # Updating model
        model_param = torch.stack(list_grad).mean(0)
        return model_param


class CoordinateGradientUpdate(AbstractGradientUpdate):

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Randomly selecting a data point
        n_samples, n_dimension = self.parameters.cost_model.X.shape
        j = random.randint(0, n_dimension - 1)
        # Computing Gradient
        grad_j = self.parameters.cost_model.grad_coordinate(model_param, j)
        # Updating model
        model_param[j] = model_param[j] - self.__step__(nb_it) * grad_j
        return model_param

    def __step__(self, it: int):
        return 1 / 3


class AdaGradUpdate(BasicGradientUpdate):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        self.grads = []

    def __step__(self, it: int):
        # n_samples, n_dimension = self.X.shape
        # L = torch.norm(self.X.T.mm(self.X)) / n_samples  # Not completely correct.
        return 1 / math.sqrt(self.parameters.cost_model.L)

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # If list of model_params is empty we add the initializer.
        # Updating Gradient
        grad = self.parameters.cost_model.grad(model_param)
        # Adding new gradient to list of all grad
        self.grads.append(grad)
        # Computing adaptative gradient
        adaptative_grad = grad / torch.sqrt(torch.sum(torch.stack(self.grads) ** 2, dim=0))
        # Updating model
        model_param = model_param - self.__step__(nb_it) * adaptative_grad
        return model_param


class AdaDeltaUpdate(BasicGradientUpdate):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        n_samples, n_dimensions = self.parameters.cost_model.X.shape
        self.accumulated_grads = torch.zeros(n_dimensions).to(dtype=torch.float64)
        self.aggregated_update = torch.zeros(n_dimensions).to(dtype=torch.float64)
        self.decay_rate = 0.5
        self.eps = 1e-3

    def compute(self, model_param: torch.FloatTensor, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Updating Gradient
        grad = self.parameters.cost_model.grad(model_param)
        # Compute the accumulated gradients
        self.accumulated_grads = self.decay_rate * self.accumulated_grads + (1 - self.decay_rate) * grad ** 2
        # Updating model
        step = torch.sqrt(self.aggregated_update + self.eps) / torch.sqrt(self.accumulated_grads + self.eps)
        old_model = model_param
        model_param -= step * grad
        # Compute aggregated model
        self.aggregated_update = self.decay_rate * self.aggregated_update + (1 - self.aggregated_update) * (
                    model_param - old_model) ** 2
        return model_param
