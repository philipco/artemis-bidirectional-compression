"""
Created by Philippenko, 4th January 2020.

It has been decided to use a Factory Pattern to build the gradient descent mechanism.
This pattern is very useful to provide a high level of code flexibility.

This choice is lead by the willingness to define different kind of gradient descent (Adagrad, Adadelta, Momentum ...)
and to not limit oneself to any particular algorithm.

Following the Factory Pattern paradigm, we say that the AGradientDecent class is the client class,
while the AGradientUpdateMethod class is one of its products.

To add a new gradient descent algorithm, just extend the abstract class AGradientDescent which contains methods:
1. to define the method to update the scheme at each iteration
2. to compute the number of iteration required to carry out one epoch
3. to get the name of the algorithm.

The method *run* should not be overridden as this the core gear which launch the gradient descent and is supposed to
work correctly as soon as the update scheme method is correctly defined.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import time
from copy import copy

import numpy as np
import torch
import math
import os
import psutil

from src.machinery.GradientUpdateMethod import *
from src.machinery.LocalUpdate import *
from src.machinery.Parameters import Parameters
from src.machinery.Worker import Worker
from src.models.CompressionModel import SQuantization
from src.utils.Constants import MAX_LOSS


class AGradientDescent(ABC):
    """
    The AGradientDescent class declares the factory methods while subclasses provide
    the implementation of this methods.
    """
    # __slots__ = ('parameters', 'losses', 'model_params', 'model_params', 'averaged_model_params', 'averaged_losses',
    #              'workers', 'memory_info')

    def __init__(self, parameters: Parameters) -> None:
        """Initialization of the gradient descent.

        It initialize all the worker of the network, the sequence of (averaged) losses,
        the sequence of (averaged) models.

        Args:
            parameters: the parameters of the descent.
        """
        super().__init__()
        self.parameters = parameters
        self.losses = []
        self.norm_error_feedback = []
        self.dist_to_model = [torch.tensor(0.)]
        self.var_models = [torch.tensor(0.)]
        self.model_params = []
        self.averaged_model_params = []
        self.averaged_losses = []
        self.memory_info = None

        if self.parameters.use_up_memory and self.parameters.up_compression_model.omega_c != 0 and self.parameters.up_learning_rate is None:
            self.parameters.up_learning_rate = 1 / (2 * (self.parameters.up_compression_model.omega_c + 1))
        elif not self.parameters.use_up_memory:
            self.parameters.up_learning_rate = 0
        if self.parameters.use_down_memory and self.parameters.down_compression_model.omega_c != 0 and self.parameters.down_learning_rate is None:
            self.parameters.down_learning_rate = 1 / (2 * (self.parameters.down_compression_model.omega_c + 1))
        elif not self.parameters.use_down_memory:
            self.parameters.down_learning_rate = 0

        # Creating each worker of the network.
        self.workers = [Worker(i, parameters, self.__local_update__()) for i in range(self.parameters.nb_devices)]

        # Call for the update method of the gradient descent.
        self.update = self.__update_method__()

    @abstractmethod
    def __update_method__(self) -> AbstractGradientUpdate:
        """Factory method for the GD update procedure.

        Returns:
            The update procedure for the gradient descent.
        """
        pass

    def __number_iterations__(self, cost_models) -> int:
        """Return the number of iterations needed to perform one epoch."""
        if self.parameters.stochastic:
            # Devices may have different number of points. Thus to reach an equal weight of participation,
            # we choose that an epoch is constituted of N rounds of communication with the central server,
            # where N is the minimum size of the dataset hold by the different devices.
            n_samples = min([cost_models[i].X.shape[0] for i in range(len(cost_models))])
            return n_samples * self.parameters.nb_epoch / min(n_samples, self.parameters.batch_size)

        return self.parameters.nb_epoch

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the gradient descent algorithm."""
        pass

    def run(self, cost_models) -> float:
        """Run the gradient descent over the data.

        Returns:
            The elapsed time.
        """

        start_time = time.time()

        inside_loop_time = 0
        averaging_time = 0

        # Initialization
        current_model_param = torch.FloatTensor(
            [0 for i in range(self.parameters.n_dimensions)])\
            .to(dtype=torch.float64)

        self.model_params.append(current_model_param)
        self.losses.append(self.update.compute_cost(current_model_param, cost_models))

        if self.parameters.use_averaging:
            self.averaged_model_params.append(self.model_params[-1])
            self.averaged_losses.append(self.losses[-1])

        full_nb_iterations = 0

        if self.parameters.streaming:
            nb_epoch = min([cost_models[i].X.shape[0] for i in range(len(cost_models))])
        else:
            nb_epoch = self.parameters.nb_epoch

        for i in range(1, nb_epoch):

            # If we are in streaming mode, each sample should be used only once !
            if self.parameters.streaming:
                number_of_inside_it = 1
            else:
                number_of_inside_it = self.__number_iterations__(cost_models) / self.parameters.nb_epoch
            # This loops corresponds to the number of loop before considering that one epoch is completed.
            # It is the communication between the central server and all remote devices.
            # This is not the loop carried out on local remote devices.
            # Hence, there is a communication between all devices during this loop.
            # If we use compression, of course all communication are compressed !
            for j in range(0, math.floor(number_of_inside_it)):
                in_loop = time.time()
                full_nb_iterations += 1
                past_model = copy(current_model_param)
                # If in streaming mode, we send the epoch numerous, else the numerous of the inside iteration.
                current_model_param = self.update.compute(current_model_param, cost_models, full_nb_iterations, (j, i)[self.parameters.streaming])
                inside_loop_time += (time.time() - in_loop)

            # We add the past update because at this time, local update has not been yet updated.
            self.model_params.append(past_model)
            self.losses.append(self.update.compute_cost(self.model_params[-1], cost_models))
            if self.parameters.randomized:
                self.norm_error_feedback.append(torch.norm(torch.mean(torch.stack(self.update.all_error_i[-1]), dim=0), p=2))
                self.dist_to_model.append(np.mean(
                    [torch.norm(self.model_params[-1] - w.local_update.model_param)**2 for w in self.workers]
                ))
                self.var_models.append(torch.mean(
                    torch.var(torch.stack([w.local_update.model_param for w in self.workers]))
                ))

            else:
                self.norm_error_feedback.append(torch.norm(self.update.all_error_i[-1], p=2))

            self.dist_to_model.append(np.mean(
                [torch.norm(self.model_params[-1] - w.local_update.model_param) ** 2 for w in self.workers]
            ))
            # if (not self.parameters.randomized):
            #     assert torch.all(torch.norm(self.model_params[-1] - self.workers[0].local_update.model_param) ** 2 == torch.tensor(0.0)), \
            #         "The distance from central server and remote nodes is not null."
            self.var_models.append(torch.mean(
                torch.var(torch.stack([w.local_update.model_param for w in self.workers]))
            ))
            # The norm of error feedback has not been initialized. We initialize it now with the first value.
            if len(self.norm_error_feedback) == 1:
                self.norm_error_feedback.append(self.norm_error_feedback[0])

            start_averaging_time = time.time()
            if self.parameters.use_averaging:
                self.averaged_model_params.append(torch.mean(torch.stack(self.model_params), 0))
                self.averaged_losses.append(self.update.compute_cost(self.averaged_model_params[-1], cost_models))
            averaging_time += time.time() - start_averaging_time

            if self.parameters.verbose:
                if i == 1:
                    print(' | '.join([name.center(8) for name in ["it", "obj"]]))
                if i % (nb_epoch / 5) == 0:
                    print(' | '.join([("%d" % i).rjust(8), ("%.4e" % self.losses[-1]).rjust(8)]))

            # Beyond 1e9, we consider that the algorithm has diverged.
            if self.losses[-1] == math.inf:
                self.losses[-1] = MAX_LOSS
                break
            elif (self.losses[-1] > 1e9):
                self.losses[-1] = MAX_LOSS
                break
        end_time = time.time()
        elapsed_time = end_time - start_time

        # print(self.distance_to_model)

        self.memory_info = psutil.Process(os.getpid()).memory_info().rss / 1e6

        if self.parameters.time_debug:
            for cost_model in cost_models:

                print("Lips time: {0}".format(cost_model.lips_times))
                print("Cost time: {0}".format(cost_model.cost_times))
                print("Grad time: {0}".format(cost_model.grad_i_times))

            print("== Inside time {0}".format(inside_loop_time))
            print("== Averaging time : {0}".format(averaging_time))
            print("== Full time : {0}".format(elapsed_time))
            print("=== Used memory : {0} Mbytes".format(self.memory_info))  # in bytes

        # If we interrupted the run, we need to complete the list of loss to reach the number of iterations.
        # Otherwise it may later cause issues.
        if len(self.losses) != self.parameters.nb_epoch:
            self.losses = self.losses + [self.losses[-1] for i in range(self.parameters.nb_epoch - len(self.losses))]
        if len(self.norm_error_feedback) != self.parameters.nb_epoch:
            self.norm_error_feedback = self.norm_error_feedback + [self.norm_error_feedback[-1] for i in range(self.parameters.nb_epoch - len(self.norm_error_feedback))]
        if len(self.averaged_losses) != self.parameters.nb_epoch and self.parameters.use_averaging == True:
            self.averaged_losses = self.averaged_losses + [self.averaged_losses[-1] for i in range(self.parameters.nb_epoch - len(self.averaged_losses))]
        if len(self.dist_to_model) != self.parameters.nb_epoch:
            self.dist_to_model = self.dist_to_model + [self.dist_to_model[-1] for i in range(
                self.parameters.nb_epoch - len(self.dist_to_model))]
        if len(self.var_models) != self.parameters.nb_epoch:
            self.var_models = self.var_models + [self.var_models[-1] for i in range(
                self.parameters.nb_epoch - len(self.var_models))]

        self.losses = np.array(self.losses)
        if self.parameters.verbose:
            print("Gradient Descent: execution time={t:.3f} seconds".format(t=elapsed_time))
            print("Final loss : {0:.5f}\n".format(self.losses[-1]))

        return elapsed_time


class ArtemisDescent(AGradientDescent):
    """Implementation of Artemis.

    This implementation of Artemis is very flexible and incorporates several possibility. Mainly:
    1. add a moment
    2. use Polyak-Rupper averaging
    3. carry out a stochastic descent or a full batch
    4. use bidirectional or unidirectional compression
    5. use or not memory
    6. use one or two memory (one for each way).
    7. for downlink, compress either gradients either models
    8. set the quantization parameter

    This features can switched on when defining Parameters.
    """

    def __local_update__(self):
        return LocalArtemisUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return ArtemisUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "Artemis"


class SGD_Descent(AGradientDescent):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        # Vanilla SGD doesn't carry out any compression.
        self.parameters.up_compression_model = SQuantization(0, self.parameters.n_dimensions)
        self.parameters.down_compression_model = SQuantization(0, self.parameters.n_dimensions)

    def __local_update__(self):
        return LocalGradientVanillaUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return GradientVanillaUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "VanillaGradient"


class DianaDescent(AGradientDescent):

    def __init__(self, parameters: Parameters) -> None:
        super().__init__(parameters)
        # Diana doesn't carry out a down compression.
        self.parameters.down_compression_model = SQuantization(0, self.parameters.n_dimensions)

    def __local_update__(self):
        return LocalDianaUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return DianaUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "Diana"

class FedAvgDescent(AGradientDescent):

    def __local_update__(self):
        return LocalFedAvgUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return FedAvgUpdate(self.parameters, self.workers)

    def __number_iterations__(self, cost_models) -> int:
        if self.parameters.stochastic:
            # Devices may have different number of points. Thus to reach an equal weight of participation,
            # we choose that an epoch is constituted of N rounds of communication with the central server,
            # where N is the minimum size of the dataset hold by the different devices.
            n_samples = min([cost_models[i].X.shape[0] for i in range(len(cost_models))])
            return n_samples * self.parameters.nb_epoch / (min(n_samples, self.parameters.batch_size) * self.parameters.nb_local_update)

        return self.parameters.nb_epoch

    def __number_iterations__(self, cost_models) -> int:
        """Return the number of iterations needed to perform one epoch."""
        return self.parameters.nb_epoch

    def get_name(self) -> str:
        return "FedAvg"

class DownCompressModelDescent(AGradientDescent):

    def __local_update__(self):
        return LocalDownCompressModelUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return DownCompressModelUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "DwnComprModel"


class SympaDescent(AGradientDescent):

    def __local_update__(self):
        return LocalSympaUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return SympaUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "Sympa"
