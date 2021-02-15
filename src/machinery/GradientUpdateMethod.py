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

from abc import ABC, abstractmethod, ABCMeta

import torch
from typing import Tuple
import numpy as np

from src.machinery.Parameters import Parameters


class AbstractGradientUpdate(ABC):
    """
    The  interface declares the operations that all concrete products of AGradient Descent must implement.
    """

    time_sample = 0

    def __init__(self, parameters: Parameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.step = 0
        self.all_delta_i = []

    def compute_cost(self, model_param):
        """Compute the cost function for the model's parameter."""
        loss, _ = self.parameters.cost_model.cost(model_param)
        return loss.item()

    @abstractmethod
    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the model's update.

        This  method must update the model's parameter according to the scheme.

        Args:
            model_param: this is used only for initialization at iteration k=1. Deprecated, should be removed !
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
        step = self.parameters.step_formula(it, L, self.parameters.up_compression_model.omega_c,
                                            self.parameters.nb_devices)
        return step


class AbstractFLUpdate(AbstractGradientUpdate, metaclass=ABCMeta):

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters)

        # Delta sent from remote nodes to main server.
        self.all_delta_i = []

        # Local memories hold on the central server.
        self.h = [torch.zeros(parameters.n_dimensions, dtype=np.float) for k in range(self.parameters.nb_devices)]

        # Omega : used to update the model on central server.
        self.omega = torch.zeros(parameters.n_dimensions, dtype=np.float)

        # Sequence of omega : the information that will be send to all active nodes.
        self.omega_k = []

        # For the momentum.
        self.v = torch.zeros(parameters.n_dimensions, dtype=np.float)
        # Sum of local compressed gradients.
        self.g = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.workers = workers
        self.workers_sub_set = None

        # all_error_i is a list of accumulated error. In the case of randomization, there is one accumulated
        # error by remote nodes.
        if self.parameters.randomized:
            self.all_error_i = [[torch.zeros(parameters.n_dimensions, dtype=np.float)
                                for i in range(self.parameters.nb_devices)]]
        else:
            self.all_error_i = [torch.zeros(parameters.n_dimensions, dtype=np.float)]

        # Memory for bidirectional compression:
        if self.parameters.randomized:
            self.H = [torch.zeros(parameters.n_dimensions, dtype=np.float) for _ in range(self.parameters.nb_devices)]
        else:
            self.H = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute_aggregation(self, local_information_to_aggregate):
        # In Artemis there is no weight associated with the aggregation, all nodes must have the same weight equal
        # to 1 / len(workers), this is why, an average is enough.
        true_mean = torch.stack(local_information_to_aggregate).mean(0)
        approximate_mean = torch.stack(local_information_to_aggregate).sum(0) / (self.parameters.fraction_sampled_workers * self.parameters.nb_devices)
        if self.parameters.fraction_sampled_workers == 1.:
            assert true_mean.equal(approximate_mean), "The true and approximate means are not equal !"
        return approximate_mean

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

    def sampling_devices(self, cost_models):
        self.workers_sub_set = []
        # Sampling workers until there is at least one in the subset.
        if self.parameters.fraction_sampled_workers == 1:
            self.workers_sub_set = [(self.workers[i], cost_models[i]) for i in range(self.parameters.nb_devices)]
        else:
            while not self.workers_sub_set:
                s = np.random.binomial(1, self.parameters.fraction_sampled_workers, self.parameters.nb_devices)

                self.workers_sub_set = [(self.workers[i], cost_models[i])
                                        for i in range(self.parameters.nb_devices) if s[i]]

    def initialization(self, nb_it: int, model_param: torch.FloatTensor, L: float, cost_models):

        self.sampling_devices(cost_models)

        if nb_it > 1:
            self.send_back_global_informations_and_update(cost_models)

        # The step size must be updated after updating models (in the case that it is not constant).
        self.step = self.__step__(nb_it, L)

        if nb_it == 1:
            if self.parameters.momentum != 0:
                self.v = self.compute_full_gradients(model_param)
            else:
                self.v = torch.zeros_like(model_param)
            # Initialization of v_-1 (for momentum)
            for worker in self.workers:
                worker.local_update.set_initial_v(self.v)

        self.all_delta_i = []
        self.all_delta_i = []

    def get_set_of_workers(self, cost_models, all=False):
        if all:
            result = list(zip(self.workers, cost_models))
        else:
            result = self.workers_sub_set
        if self.parameters.fraction_sampled_workers == 1:
            assert len(result) == self.parameters.nb_devices
        return result

    def build_randomized_omega(self, cost_models):
        randomized_omega_k = [torch.zeros(self.parameters.n_dimensions, dtype=np.float)
                                for i in range(self.parameters.nb_devices)]
        for (worker, cost_model) in self.get_set_of_workers(cost_models):
            randomized_omega_k[worker.ID] = self.parameters.down_compression_model.compress(self.value_to_compress[worker.ID])
        self.omega = randomized_omega_k
        self.omega_k.append(self.omega)

    def build_omega(self):
        self.omega = self.parameters.down_compression_model.compress(self.value_to_compress)
        self.omega_k.append(self.omega)

    def perform_down_compression(self, value_to_consider, cost_models):

        # We combine with EF and memory to obtain the proper value that will be compressed
        if self.parameters.randomized:
            self.value_to_compress = [(value_to_consider - self.H[i]) + self.all_error_i[-1][i] * self.parameters.down_learning_rate
                                      for i in range(self.parameters.nb_devices)]
        else:
            self.value_to_compress = (value_to_consider - self.H) + self.all_error_i[-1] * self.parameters.down_learning_rate

        # We build omega i.e, what will be sent to remote devices
        if self.parameters.randomized:
            self.build_randomized_omega(cost_models)
        else:
            self.build_omega()

        # We update EF
        if self.parameters.randomized and self.parameters.down_error_feedback:
            self.all_error_i.append([self.value_to_compress[i] - self.omega[i]
                                     for i in range(self.parameters.nb_devices)])
        elif self.parameters.down_error_feedback:
            self.all_error_i.append(self.value_to_compress - self.omega)

    def send_back_global_informations_and_update(self, cost_models):
        """Send to remote servers the global information and update their models."""
        cpt = 0
        for worker, _ in self.get_set_of_workers(cost_models):
            if self.parameters.randomized:
                models_to_send = []
                for k in range(worker.idx_last_update, len(self.omega_k)):
                    models_to_send.append(self.omega_k[k][cpt])
                    cpt += 1
            else:
                models_to_send = self.omega_k[worker.idx_last_update:]
            # models_to_send is a list that send the sequence of missed update
            worker.local_update.send_global_informations_and_update_local_param(models_to_send, self.step)
            if self.parameters.fraction_sampled_workers == 1.:
                assert len(self.omega_k[worker.idx_last_update:]) == 1
            worker.idx_last_update = len(self.omega_k)


class ArtemisUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potential memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.value_to_compress = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def update_model(self, model_param):
        if self.parameters.non_degraded:
            self.v = self.parameters.momentum * self.v + self.g
        elif self.parameters.randomized:
            self.v = self.parameters.momentum * self.v + (torch.mean(torch.stack(self.omega + self.H), dim=0))
        else:
            self.v = self.parameters.momentum * self.v + (self.omega + self.H)
        return model_param - self.step * self.v


    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        # Warning, if one wants to run the case when subset are updating, but then al devices are updated,
        # the following lines must be changed.
        for worker, cost_model in self.get_set_of_workers(cost_models):
            # We get all the compressed gradient computed on each worker.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, nb_inside_it)

            # If nothing is returned by the device, this device does not participate to the learning at this iterations.
            # This may happened if it is considered that during one epoch each devices should run through all its data
            # exactly once, and if there is different numbers of points on each device.
            if compressed_delta_i is not None:
                self.all_delta_i.append(compressed_delta_i + self.h[worker.ID])
                self.h[worker.ID] += self.parameters.up_learning_rate * compressed_delta_i

        # Aggregating all delta
        self.g = self.compute_aggregation(self.all_delta_i)

        self.perform_down_compression(self.g, cost_models)
        model_param = self.update_model(model_param)

        # Update the second memory if we are using bidirectional compression and if this feature has been turned on.
        if self.parameters.use_down_memory:
            self.H += self.parameters.down_learning_rate * self.omega
        return model_param

class SympaUpdate(AbstractFLUpdate):

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.value_to_compress = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def initialization(self, nb_it: int, model_param: torch.FloatTensor, L: float, cost_models):

        # The step size must be updated after updating models (in the case that it is not constant).
        self.step = self.__step__(nb_it, L)

        if nb_it == 1:
            if self.parameters.momentum != 0:
                self.v = self.compute_full_gradients(model_param)
            else:
                self.v = torch.zeros_like(model_param)
            # Initialization of v_-1 (for momentum)
            for worker in self.workers:
                worker.local_update.set_initial_v(self.v)

        self.all_delta_i = []

    def send_back_global_informations_and_update(self, model_param, cost_models):

        all_local_models = []
        for worker, _ in self.get_set_of_workers(cost_models, all=True):
            tuple_to_send = (self.parameters.down_compression_model.compress(self.g), model_param)
            local_model = worker.local_update.send_global_informations_and_update_local_param(tuple_to_send, self.step)
            all_local_models.append(local_model)
        return all_local_models

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L, cost_models)

        for worker, cost_model in self.get_set_of_workers(cost_models, all=True):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, j)
            if compressed_delta_i is not None:
                self.all_delta_i.append(compressed_delta_i + self.h[worker.ID])
                self.h[worker.ID] += self.parameters.up_learning_rate * compressed_delta_i

        # Aggregating all delta
        self.g = self.compute_aggregation(self.all_delta_i)

        all_local_models = self.send_back_global_informations_and_update(model_param, cost_models)
        model_param = torch.mean(torch.stack(all_local_models), dim=0)

        return model_param


class DownCompressModelUpdate(AbstractFLUpdate):

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.value_to_compress = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def build_randomized_omega(self, cost_models):
        randomized_omega_k = [torch.zeros(self.parameters.n_dimensions, dtype=np.float)
                                for i in range(self.parameters.nb_devices)]
        for i in range(self.parameters.nb_devices):
            randomized_omega_k[i] = self.parameters.down_compression_model.compress(self.value_to_compress[i])
        self.omega = randomized_omega_k
        self.omega_k.append(self.omega)

    def send_back_global_informations_and_update(self, cost_models):
        for worker, _ in self.get_set_of_workers(cost_models):
            # We just have to send what has been compressed (i.e omega).
            if self.parameters.randomized:
                models_to_send = self.omega[worker.ID]
            else:
                models_to_send = self.omega
            worker.local_update.send_global_informations_and_update_local_param(models_to_send, self.step)
            # Update the second memory if we are using bidirectional compression and if this feature has been turned on.
            if self.parameters.use_down_memory and self.parameters.randomized:
                self.H[worker.ID] = self.H[worker.ID] + self.parameters.down_learning_rate * self.omega[worker.ID]
        if self.parameters.use_down_memory and not self.parameters.randomized:
            self.H = self.H + self.parameters.down_learning_rate * self.omega

    def update_model(self, model_param):
        self.v = self.parameters.momentum * self.v + self.g
        return model_param - self.step * self.v

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L, cost_models)

        for worker, cost_model in self.get_set_of_workers(cost_models):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, j)
            if compressed_delta_i is not None:
                self.all_delta_i.append(compressed_delta_i + self.h[worker.ID])
                self.h[worker.ID] += self.parameters.up_learning_rate * compressed_delta_i

        # Aggregating all delta
        self.g = self.compute_aggregation(self.all_delta_i)

        model_param = self.update_model(model_param)
        self.perform_down_compression(model_param, cost_models)

        return model_param


class FedAvgUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potentiel memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        local_models = []

        total_nb_of_points = sum([cost_model.X.shape[0] for cost_model in cost_models])

        for worker, cost_model in self.get_set_of_workers(cost_models):
            local_nb_points = cost_model.X.shape[0]
            local_model = worker.local_update.compute_locally(cost_model, j, self.step) + model_param
            local_models.append(local_model * local_nb_points / total_nb_of_points)

        model_param = torch.sum(torch.stack(local_models, dim=0), dim=0)

        self.omega = model_param

        return model_param

    def send_back_global_informations_and_update(self, cost_models):
        """Send to remote servers the global information and update their models."""
        models_to_send = self.omega
        for worker, _ in self.get_set_of_workers(cost_models):
            worker.local_update.send_global_informations_and_update_local_param(models_to_send, self.step)


class DianaUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potentiel memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.value_to_compress = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L, cost_models)

        for worker, cost_model in self.get_set_of_workers(cost_models):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, j)
            if compressed_delta_i is not None:
                self.all_delta_i.append(compressed_delta_i + self.h[worker.ID])
                self.h[worker.ID] += self.parameters.up_learning_rate * compressed_delta_i

        # Aggregating all delta
        self.g = self.compute_aggregation(self.all_delta_i)
        self.v = self.parameters.momentum * self.v + self.g
        model_param = model_param - self.v * self.step

        self.omega = self.g
        self.omega_k.append(self.omega)

        return model_param


class GradientVanillaUpdate(AbstractFLUpdate):

    def compute(self, model_param: torch.FloatTensor, cost_models, nb_it: int, j: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(nb_it, model_param, cost_models[0].L, cost_models)

        for worker, cost_model in self.get_set_of_workers(cost_models):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            delta_i = worker.local_update.compute_locally(cost_model, j)
            if delta_i is not None:
                self.all_delta_i.append(self.h[worker.ID] + delta_i)
                self.h[worker.ID] += self.parameters.up_learning_rate * delta_i

        # Aggregating all delta
        self.g = self.compute_aggregation(self.all_delta_i)
        self.v = self.parameters.momentum * self.v + self.g

        model_param = model_param - self.v * self.step

        # We update omega (compression of the sum of compressed gradients).
        self.omega = self.g
        self.omega_k.append(self.omega)

        return model_param
