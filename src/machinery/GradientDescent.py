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
import numpy as np
import torch
import math

from src.machinery.GradientUpdateMethod import BasicGradientUpdate, StochasticGradientUpdate, MomentumUpdate, SAGUpdate, \
    SVRGUpdate, \
    CoordinateGradientUpdate, AdaGradUpdate, AdaDeltaUpdate, ArtemisUpdate, AbstractGradientUpdate, \
    GradientVanillaUpdate, DianaUpdate
from src.machinery.LocalUpdate import LocalGradientVanillaUpdate, LocalArtemisUpdate, LocalDianaUpdate
from src.models.QuantizationModel import s_quantization_omega_c
from src.machinery.Parameters import Parameters
from src.machinery.Worker import Worker
from src.utils.Constants import MAX_LOSS


class AGradientDescent(ABC):
    """
    The AGradientDescent class declares the factory methods while subclasses provide
    the implementation of this methods.
    """

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
        self.model_params = []
        self.averaged_model_params = []
        self.averaged_losses = []
        self.X, self.Y = None, None

        #if self.parameters.quantization_param != 0:
        self.parameters.omega_c = s_quantization_omega_c(
                self.parameters.n_dimensions,
                self.parameters.quantization_param
            )
        if self.parameters.quantization_param != 0:
            # If learning_rate is None, we set it to optimal value.
            if self.parameters.learning_rate == None:
                self.parameters.learning_rate = 1 / (2 * (self.parameters.omega_c + 1))
            else:
                if not self.parameters.force_learning_rate:
                    self.parameters.learning_rate *= 1 / (1 * (self.parameters.omega_c + 1))

        # If quantization_param == 0, it means there is no compression,
        # which means that we don't want to "predict" values with previous one,
        # and thus, we put learning_rate to zero.
        else:
            self.parameters.learning_rate = 0

        # Creating each worker of the network.
        self.workers = [Worker(i, parameters, self.__local_update__()) for i in range(self.parameters.nb_devices)]

    def set_data(self, X: torch.FloatTensor, Y: torch.FloatTensor) -> None:
        """Set data on each worker and compute coefficient of smoothness."""
        self.X, self.Y = X, Y
        if self.workers is None: # To handle non-Federated Settings.
            self.parameters.cost_model.set_data(X, Y)
            self.parameters.cost_model.L = self.parameters.cost_model.local_L
        else:
            L = 0
            for (worker, x, y) in zip(self.workers, X, Y):
                worker.set_data(x, y)
                L += worker.cost_model.local_L
            for worker in self.workers:
                worker.cost_model.L = L / self.parameters.nb_devices


    @abstractmethod
    def __update_method__(self) -> AbstractGradientUpdate:
        """Factory method for the GD update procedure.

        Returns:
            The update procedure for the gradient descent.
        """
        pass

    def __number_iterations__(self) -> int:
        """Return the number of iterations needed to perform one epoch."""
        if self.parameters.stochastic:
            # Devices may have different number of points. Thus to reach an equal weight of participation,
            # we choose that an epoch is constitutated of N rounds of communication with the central server,
            # where N is the minimum size of the dataset hold by the different devices.
            n_samples = min([self.workers[i].X.shape[0] for i in range(len(self.workers))])
            return n_samples * self.parameters.nb_epoch / min(n_samples, self.parameters.batch_size)

        return self.parameters.nb_epoch

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the gradient descent algorithm."""
        pass

    def run(self) -> float:
        """Run the gradient descent over the data.

        Returns:
            The elapsed time.
        """

        start_time = time.time()

        inside_loop = 0

        # Call for the update method of the gradient descent.
        update = self.__update_method__()

        # Initialization
        current_model_param = torch.FloatTensor(
            [(-1 ** i) / (2 * self.parameters.n_dimensions) for i in range(self.parameters.n_dimensions)])\
            .to(dtype=torch.float64)

        self.model_params.append(current_model_param)
        self.losses.append(update.compute_cost(current_model_param))

        if self.parameters.use_averaging:
            self.averaged_model_params.append(self.model_params[-1])
            self.averaged_losses.append(self.losses[-1])

        full_iterations = 0
        for i in range(1, self.parameters.nb_epoch):

            number_of_inside_it = self.__number_iterations__() / self.parameters.nb_epoch

            # This loops corresponds to the number of loop before considering that an epoch is completed.
            # It is the communication between the central server and all remote devices.
            # This is not the loop carried out on local remote devices.
            # Hence, there is a communication between all devices during this loop.
            # If we use compression, of course all communication are compressed !
            for j in range(0, math.floor(number_of_inside_it)):
                full_iterations += 1
                in_loop = time.time()
                current_model_param = update.compute(current_model_param.clone(), full_iterations, j)
                inside_loop += (time.time() - in_loop)

            self.model_params.append(current_model_param)
            self.losses.append(update.compute_cost(self.model_params[-1]))

            if self.parameters.use_averaging:
                self.averaged_model_params.append(torch.mean(torch.stack(self.model_params), 0))
                self.averaged_losses.append(update.compute_cost(self.averaged_model_params[-1]))

            if self.parameters.verbose:
                if i == 1:
                    print(' | '.join([name.center(8) for name in ["it", "obj"]]))
                if i % (self.parameters.nb_epoch / 5) == 0:
                    print(' | '.join([("%d" % i).rjust(8), ("%.4e" % self.losses[-1]).rjust(8)]))

            # Beyond 1e9, we consider that the algorithm has diverged.
            if self.losses[-1] == math.inf:
                self.losses[-1] = MAX_LOSS
                break
            if (self.losses[-1] > 1e9):
                self.losses[-1] = MAX_LOSS
                break

        end_time = time.time()
        elapsed_time = end_time - start_time

        # If we interrupted the run, we need to complete the list of loss to reach the number of iterations.
        # Otherwise it may later cause issues.
        if len(self.losses) != self.parameters.nb_epoch:
            self.losses = self.losses + [self.losses[-1] for i in range(self.parameters.nb_epoch - len(self.losses))]
        if len(self.averaged_losses) != self.parameters.nb_epoch and self.parameters.use_averaging == True:
            self.averaged_losses = self.averaged_losses + [self.averaged_losses[-1] for i in range(self.parameters.nb_epoch - len(self.averaged_losses))]

        self.losses = np.array(self.losses)
        if self.parameters.verbose:
            print("Gradient Descent: execution time={t:.3f} seconds".format(t=elapsed_time))
            print("Final loss : ", str(self.losses[-1]) + "\n")

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

    def __update_method__(self) -> BasicGradientUpdate:
        return ArtemisUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "Artemis"


class FL_VanillaSGD(AGradientDescent):

    def __local_update__(self):
        return LocalGradientVanillaUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return GradientVanillaUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "VanillaGradient"


class DianaDescent(AGradientDescent):

    def __local_update__(self):
        return LocalDianaUpdate

    def __update_method__(self) -> AbstractGradientUpdate:
        return DianaUpdate(self.parameters, self.workers)

    def get_name(self) -> str:
        return "Diana"


class BasicGradientDescent(AGradientDescent):
    """Basic implementation of the abstract GD class."""

    def __update_method__(self) -> BasicGradientUpdate:
        return BasicGradientUpdate(self.parameters)

    def __number_iterations__(self):
        return self.parameters.nb_epoch

    def get_name(self) -> str:
        return "GD"


class StochasticGradientDescent(AGradientDescent):
    """Stochastic implementation of the abstract GD class."""

    def __update_method__(self) -> StochasticGradientUpdate:
        return StochasticGradientUpdate(self.parameters)

    def __number_iterations__(self):
        n_samples, n_dimensions = self.X.shape
        return self.parameters.nb_epoch * n_samples

    def get_name(self) -> str:
        return "SGD"


class Momentum(StochasticGradientDescent):
    """Basic implementation of the abstract GD class."""

    def __update_method__(self) -> MomentumUpdate:
        return MomentumUpdate(self.parameters)

    def get_name(self) -> str:
        return "Momentum"


class SAG(StochasticGradientDescent):
    """Basic implementation of the abstract GD class."""

    def __update_method__(self) -> SAGUpdate:
        return SAGUpdate(self.parameters)

    def get_name(self) -> str:
        return "SAG"


class SVRGDescent(AGradientDescent):
    """Basic implementation of the abstract GD class."""

    def __update_method__(self) -> SVRGUpdate:
        return SVRGUpdate(self.parameters)

    def __number_iterations__(self):
        return self.parameters.nb_epoch

    def get_name(self) -> str:
        return "SVRG"


class CoordinateGradientDescent(AGradientDescent):
    """Basic implementation of the abstract GD class."""

    def __update_method__(self) -> CoordinateGradientUpdate:
        return CoordinateGradientUpdate(self.parameters)

    def __number_iterations__(self):
        n_samples, n_dimensions = self.X.shape
        return self.parameters.nb_epoch * n_dimensions

    def get_name(self) -> str:
        return "Coordinate GD"


class AdaGradDescent(BasicGradientDescent):
    """Adagrad implementation of the abstract GD class."""

    def __update_method__(self) -> AdaGradUpdate:
        return AdaGradUpdate(self.parameters)

    def get_name(self) -> str:
        return "AdaGrad"

class AdaDeltaDescent(BasicGradientDescent):
    """Adagrad implementation of the abstract GD class."""

    def __update_method__(self) -> AdaDeltaUpdate:
        return AdaDeltaUpdate(self.parameters)

    def get_name(self) -> str:
        return "AdaDelta"

