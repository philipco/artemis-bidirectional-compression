"""
Created by Philippenko, 6 January 2020.

This class defines the global update methods which will be used during the gradient descent.
It aims to implement the update scheme used on the central server.

To add a new update scheme, just extend the abstract class AbstractGradientUpdate which contains methods:
1. to compute the cost at present model's parameters
2. to carry out the update of the model
3. to compute the step size
"""

from __future__ import annotations

from abc import ABC, abstractmethod, ABCMeta
from math import sqrt

import torch
from typing import Tuple
import numpy as np

from src.machinery.Parameters import Parameters


class AbstractGradientUpdate(ABC):
    """
    The AbstractGradientUpdate class declares the factory methods while subclasses provide the implementation of this methods.

    This class carries out the update of the global model held on the central server.
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
    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the model's update.

        :param full_nb_iterations: total number of computed iteration.
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
    """An abstract class common to all algorithm using a FL paradigm."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters)

        # Delta sent from remote nodes to main server.
        self.all_delta_i = []

        # Local memories hold on the central server.
        if not self.parameters.use_unique_up_memory:
            if self.parameters.use_up_memory: print("Using multiple up memories.")
            self.h = [torch.zeros(parameters.n_dimensions, dtype=np.float) for k in range(self.parameters.nb_devices)]
        else:
            if self.parameters.use_up_memory: print("Using a single up memory.")
            self.h = torch.zeros(parameters.n_dimensions, dtype=np.float)

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
        if self.parameters.randomized and not self.parameters.use_unique_down_memory:
            if self.parameters.use_down_memory: print("Using multiple down memories.")
            self.H = [torch.zeros(parameters.n_dimensions, dtype=np.float) for _ in range(self.parameters.nb_devices)]
        else:
            if self.parameters.use_down_memory: print("Using a single down memory.")
            self.H = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def update_model(self, model_param):
        """Updates the central model."""
        self.v = self.parameters.momentum * self.v + self.g
        return model_param - self.step * self.v

    def compute_aggregation(self, local_information_to_aggregate):
        """Aggregates all gradientds/models received from remote workers."""
        # In Artemis there is no weight associated with the aggregation, all nodes must have the same weight equal
        # to 1 / len(workers), this is why, an average is enough.
        true_mean = torch.stack(local_information_to_aggregate).mean(0)
        approximate_mean = torch.stack(local_information_to_aggregate).sum(0) / (self.parameters.fraction_sampled_workers * self.parameters.nb_devices)
        if self.parameters.fraction_sampled_workers == 1.:
            # If tensors are both NAN, torch return False.
            if not (torch.isnan(approximate_mean).any() and torch.isnan(true_mean).any()):
                assert true_mean.equal(approximate_mean), "The true and approximate means are not equal !"
        return approximate_mean

    def compute_full_gradients(self, model_param):
        """Compute the gradient by using the full dataset held by each worker."""
        grad = 0
        for worker in self.workers:
            grad = grad + worker.cost_model.grad(model_param)
        return grad / len(self.workers)

    def compute_cost(self, model_param, cost_models):
        """Compute the loss by iterating over all worker."""
        all_loss_i = []
        for worker, cost_model in zip(self.workers, cost_models):
            loss_i, _ = cost_model.cost(model_param)
            all_loss_i.append(loss_i.item())
        return np.mean(all_loss_i)

    def sampling_devices(self, cost_models):
        """Samples devices that will be active at this round."""
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
        """Initialize a new round of communication."""

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

    def get_set_of_workers(self, cost_models, all=False):
        """Get the set of active workers."""
        if all:
            result = list(zip(self.workers, cost_models))
        else:
            result = self.workers_sub_set
        if self.parameters.fraction_sampled_workers == 1:
            assert len(result) == self.parameters.nb_devices
        return result

    def build_randomized_omega(self, cost_models):
        """Build omega in a context of a randomized algorithm (like Rand-MCM).
         Omega is the vector that sent to remote workers"""
        randomized_omega_k = [torch.zeros(self.parameters.n_dimensions, dtype=np.float)
                                for i in range(self.parameters.nb_devices)]
        for (worker, cost_model) in self.get_set_of_workers(cost_models):
            randomized_omega_k[worker.ID] = self.parameters.down_compression_model.compress(self.value_to_compress[worker.ID])
        del randomized_omega_k
        self.omega = randomized_omega_k
        self.omega_k.append(self.omega)

    def build_omega(self):
        """Build omega, the vector that is sent to remote workers"""
        self.omega = self.parameters.down_compression_model.compress(self.value_to_compress)
        self.omega_k.append(self.omega)

    def perform_down_compression(self, value_to_consider, cost_models):
        """Perfoms down compression."""

        # We combine with EF and memory to obtain the proper value that will be compressed
        if self.parameters.randomized and not self.parameters.use_unique_down_memory:
            self.value_to_compress = [(value_to_consider - self.H[i]) + self.all_error_i[-1][i] * self.parameters.error_feedback_coef
                                      for i in range(self.parameters.nb_devices)]
        elif self.parameters.randomized and self.parameters.use_unique_down_memory:
            self.value_to_compress = [(value_to_consider - self.H) + self.all_error_i[-1][i] * self.parameters.error_feedback_coef
                                      for i in range(self.parameters.nb_devices)]
        else:
            self.value_to_compress = (value_to_consider - self.H) + self.all_error_i[-1] * self.parameters.error_feedback_coef

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
            
    def receive_all_delta(self, cost_models, full_nb_iterations: int):
        """Retrieve all deltas, which are the vectors sent by all remote workers."""

        # Warning, if one wants to run the case when subset are updating, but then al devices are updated,
        # the following lines must be changed.
        for worker, cost_model in self.get_set_of_workers(cost_models):
            # We get all the compressed gradient computed on each worker.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, full_nb_iterations)

            # Smart initialisation of the memory (it corresponds to the first computed gradient).
            if self.parameters.fraction_sampled_workers==1: # TODO : There is issue with PP and multiple memories
                if full_nb_iterations == 1 and self.parameters.use_up_memory:
                    if self.parameters.use_unique_up_memory:
                        self.h = self.h + worker.local_update.h_i / len(self.get_set_of_workers(cost_models))
                    if not self.parameters.use_unique_up_memory:
                        self.h[worker.ID] = worker.local_update.h_i

            # If nothing is returned by the device, this device does not participate to the learning at this iterations.
            # This may happened if it is considered that during one epoch each devices should run through all its data
            # exactly once, and if there is different numbers of points on each device.
            if compressed_delta_i is not None:
                if self.parameters.use_unique_up_memory:
                    self.all_delta_i.append(compressed_delta_i)
                else:
                    self.all_delta_i.append(compressed_delta_i + self.h[worker.ID])
            if self.parameters.use_up_memory and not self.parameters.use_unique_up_memory:
                self.h[worker.ID] = self.h[worker.ID] + self.parameters.up_learning_rate * compressed_delta_i

        all_delta = self.compute_aggregation(self.all_delta_i)

        # Aggregating all delta
        self.g = all_delta + [0, self.h][self.parameters.use_unique_up_memory]

        if self.parameters.use_up_memory and self.parameters.use_unique_up_memory:
            self.h = self.h + self.parameters.up_learning_rate * all_delta

        if self.parameters.up_compression_model.level != 0:
            if self.parameters.use_up_memory and self.parameters.use_unique_up_memory:
                assert isinstance(self.h, torch.Tensor), "Up memory is not a tensor."
                assert not torch.equal(self.h, torch.zeros(self.parameters.n_dimensions, dtype=np.float)), "Up memory is still null."
            if self.parameters.use_up_memory and not self.parameters.use_unique_up_memory:
                assert not isinstance(self.h, torch.FloatTensor) and len(self.h) == self.parameters.nb_devices, \
                    "Up memory should be a list of length equal to the number of devices."
                # assert all([not torch.equal(e, torch.zeros(self.parameters.n_dimensions, dtype=np.float)) for e in self.h]), \
                #     "Up memories are still null."


class ArtemisUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potential memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

        self.value_to_compress = torch.zeros(parameters.n_dimensions, dtype=np.float)

    def update_model(self, model_param):
        if self.parameters.non_degraded:
            self.v = self.parameters.momentum * self.v + self.g
        elif self.parameters.randomized and not self.parameters.use_unique_down_memory:
            self.v = self.parameters.momentum * self.v + (torch.mean(torch.stack(self.omega + self.H), dim=0))
        else:
            self.v = self.parameters.momentum * self.v + (self.omega + self.H)
        return model_param - self.step * self.v


    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        self.receive_all_delta(cost_models, full_nb_iterations)

        self.perform_down_compression(self.g, cost_models)
        model_param = self.update_model(model_param)

        # Update the second memory if we are using bidirectional compression and if this feature has been turned on.
        if self.parameters.use_down_memory:
            self.H = self.H + self.parameters.down_learning_rate * self.omega
        return model_param


class GhostUpdate(AbstractFLUpdate):

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

    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        for worker, cost_model in self.get_set_of_workers(cost_models, all=True):
            # We send previously computed gradients,
            # and carry out the update step which has just been done on the central server.
            compressed_delta_i = worker.local_update.compute_locally(cost_model, nb_inside_it)
            if compressed_delta_i is not None:
                self.all_delta_i.append(compressed_delta_i + self.h[worker.ID])
                self.h[worker.ID] = self.h[worker.ID] + self.parameters.up_learning_rate * compressed_delta_i

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
        update_H_i = torch.zeros(self.parameters.n_dimensions, dtype=np.float)
        for worker, _ in self.get_set_of_workers(cost_models):
            # We just have to send what has been compressed (i.e omega).
            if self.parameters.randomized:
                models_to_send = self.omega[worker.ID]
            else:
                models_to_send = self.omega
            if self.parameters.use_unique_down_memory and self.parameters.reset_memories and \
                    worker.full_nb_iterations % 4 * sqrt(self.parameters.n_dimensions) == 0:
                worker.local_update.send_global_informations_and_update_local_param(models_to_send, self.step, self.H)
            else:
                worker.local_update.send_global_informations_and_update_local_param(models_to_send, self.step)
            # Update the second memory if we are using bidirectional compression and if this feature has been turned on.
            if self.parameters.use_down_memory and self.parameters.randomized and not self.parameters.use_unique_down_memory:
                self.H[worker.ID] = self.H[worker.ID] + self.parameters.down_learning_rate * self.omega[worker.ID]
            elif self.parameters.use_down_memory and self.parameters.randomized and self.parameters.use_unique_down_memory:
                update_H_i = update_H_i + self.parameters.down_learning_rate * self.omega[worker.ID] / self.parameters.nb_devices
        if self.parameters.use_down_memory and self.parameters.randomized and self.parameters.use_unique_down_memory:
            self.H = self.H + update_H_i
        if self.parameters.use_down_memory and not self.parameters.randomized:
            self.H = self.H + self.parameters.down_learning_rate * self.omega

        if self.parameters.use_down_memory and self.parameters.use_unique_down_memory:
            assert isinstance(self.H, torch.Tensor), "Down memory is not a tensor."
        if self.parameters.use_down_memory and not self.parameters.use_unique_down_memory:
            assert not isinstance(self.H, torch.Tensor) and len(self.H) == self.parameters.nb_devices, \
                "Down memory should be a list of length equal to the number of devices."
            assert any([not torch.equal(e, torch.zeros(self.parameters.n_dimensions, dtype=np.float)) for e in self.H]),\
                "Down memory are still null."


    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        for worker, _ in self.get_set_of_workers(cost_models, all=True):
            worker.full_nb_iterations = full_nb_iterations

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        self.receive_all_delta(cost_models, full_nb_iterations)

        model_param = self.update_model(model_param)
        self.perform_down_compression(model_param, cost_models)

        return model_param


class FedAvgUpdate(AbstractFLUpdate):
    """This class implement the proper update of the Artemis schema.

    It hold two potentiel memories (one for each way), and can either compress gradients, either models."""

    def __init__(self, parameters: Parameters, workers) -> None:
        super().__init__(parameters, workers)

    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        local_models = []

        total_nb_of_points = sum([cost_model.X.shape[0] for cost_model in cost_models])

        for worker, cost_model in self.get_set_of_workers(cost_models):
            local_nb_points = cost_model.X.shape[0]
            local_model = worker.local_update.compute_locally(cost_model, nb_inside_it, self.step) + model_param
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

    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        self.receive_all_delta(cost_models, full_nb_iterations)

        model_param = self.update_model(model_param)

        self.omega = self.g
        self.omega_k.append(self.omega)

        return model_param


class GradientVanillaUpdate(AbstractFLUpdate):

    def compute(self, model_param: torch.FloatTensor, cost_models, full_nb_iterations: int, nb_inside_it: int) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        self.initialization(full_nb_iterations, model_param, cost_models[0].L, cost_models)

        self.receive_all_delta(cost_models, full_nb_iterations)

        model_param = self.update_model(model_param)

        # We update omega (compression of the sum of compressed gradients).
        self.omega = self.g
        self.omega_k.append(self.omega)

        return model_param
