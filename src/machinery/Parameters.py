"""
Created by Philippenko, 12th March 2020.

This python file provide tools to easily customize a gradient descent based on its hyperparameters.
It also provide predefine parameters to run classical algorithm without introducing an error.
"""
from src.utils.Constants import NB_EPOCH, NB_DEVICES, DIM
from src.models.CostModel import ACostModel, RMSEModel

from math import sqrt


def default_step_formula(stochastic: bool):
    """Default formula to compute the step size at each iteration.

    Two cases are handled, if it is a stochastic run or a full batch descent."""
    if stochastic:
        return lambda it, L, omega, N: 1 / (L * sqrt(it))
    return lambda it, L, omega, N: 1 / L


class Parameters:
    """This class give the ability to tune hyper parameters of each federated gradient descent.

    Exemple of hyperparameters: step size, regularization rate, kind of compression ...
    By default it ran a stochastic non-federated non-regularized and single compressed (if possible) descent
    without parallelization for a RMSE model.
    """

    def __init__(self,
                 cost_model: ACostModel = RMSEModel(),
                 federated: bool = False,
                 n_dimensions: int = DIM,
                 nb_devices: int = NB_DEVICES,
                 batch_size: int = 1,
                 step_formula=None,
                 nb_epoch: int = NB_EPOCH,
                 regularization_rate: int = 0,
                 momentum: int = 0,
                 quantization_param: int = None,
                 learning_rate: int = None,
                 force_learning_rate: bool = False,
                 bidirectional: bool = False,
                 verbose: bool = False,
                 stochastic: bool = True,
                 compress_gradients: bool = True,
                 double_use_memory: bool = False,
                 use_averaging: bool = False) -> None:
        super().__init__()
        self.cost_model = cost_model  # Cost model to use for gradient descent.
        self.federated = federated  # Boolean to say if we do federated learning or not.
        self.n_dimensions = n_dimensions  # Dimension of the problem.
        self.nb_devices = nb_devices  # Number of device on the network.
        self.batch_size = batch_size  # Batch size.
        # To compute the step size at each iteration, we use a lambda function which takes as parameters
        # the number of current epoch, the coefficient of smoothness and the quantization constant omega_c.
        if step_formula == None:
            self.step_formula = default_step_formula(stochastic)
        else:
            self.step_formula = step_formula
        self.nb_epoch = nb_epoch # number of epoch of the run
        self.regularization_rate = regularization_rate # coefficient of regularization
        self.force_learning_rate = force_learning_rate
        self.momentum = momentum # momentum coefficient
        self.quantization_param = quantization_param # quantization parameter
        self.omega_c = 0 # quantization constant involved in the variance inequality of the scheme
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.stochastic = stochastic  # true if runing a stochastic gradient descent
        self.compress_gradients = compress_gradients
        self.double_use_memory = double_use_memory  # Use a
        self.verbose = verbose
        self.use_averaging = use_averaging  # true if using a Polyak-Ruppert averaging.


class PredefinedParameters():
    """Abstract class to predefine (no customizable) parameters required by a given type of algorithms (e.g Artemis, QSGD ...)

    Keep high degree of customization.
    """

    def name(self) -> str:
        """Name of the predefined parameters.
        """
        return "empty"

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int,
               step_formula=None, momentum: float = 0,
               nb_epoch: int = NB_EPOCH,
               use_averaging=False, model: ACostModel = RMSEModel(), stochastic=True):
        """Define parameters to be used during the descent.

        Args:
            n_dimensions: dimensions of the problem.
            nb_devices: number of device in the federated network.
            quantization_param: parameter of quantization.
            step_formula: lambda formul to compute the step size at each iteration.
            momentum: momentum coefficient.
            nb_epoch: number of epoch for the run.
            use_averaging: true if using Polyak-Rupper Averaging.
            model: cost model of the problem (e.g least-square, logistic ...).
            stochastic: true if running stochastic descent.

        Returns:
            Build parameters.
        """
        pass


class SGDWithoutCompression(PredefinedParameters):
    """Predefine parameters to run SGD algorithm in a federated settings.
    """

    def name(self):
        return "SGD"

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int = 0, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False,
               model: ACostModel = RMSEModel(), stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=0,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          bidirectional=False,
                          cost_model=model,
                          use_averaging=use_averaging
                          )


class Qsgd(PredefinedParameters):
    """Predefine parameters to run QSGD algorithm.
    """

    def name(self) -> str:
        return r"QSGD"

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False,
               model: ACostModel = RMSEModel(), stochastic=True) -> None:
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          learning_rate=0,
                          verbose=False,
                          stochastic=stochastic,
                          cost_model=model,
                          use_averaging=use_averaging,
                          bidirectional=False
                          )


class Diana(PredefinedParameters):
    """Predefine parameters to run Diana algorithm.
    """

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False, model: ACostModel = RMSEModel(),
               stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          bidirectional=False,
                          cost_model=model,
                          use_averaging=use_averaging
                          )

    def name(self) -> str:
        return "Diana"


class BiQSGD(PredefinedParameters):
    """Predefine parameters to run Bi-QSGD algorithm.
    """

    def name(self) -> str:
        return "BiQSGD"

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False, model: ACostModel = RMSEModel(),
               stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=1,
                          learning_rate=0,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          cost_model=model,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=False,
                          compress_gradients=True
                          )


class Artemis(PredefinedParameters):
    """Predefine parameters to run Artemis algorithm.
    """

    def name(self) -> str:
        return "Artemis"

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False, model: ACostModel = RMSEModel(),
               stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          cost_model=model,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=False,
                          compress_gradients=True
                          )


class DoreVariant(PredefinedParameters):
    """Predefine parameters to run a variant of algorithm.
    This variant use
    """

    def name(self) -> str:
        return "Dore"

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False, model: ACostModel = RMSEModel(),
               stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          cost_model=model,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=True,
                          compress_gradients=True
                          )


class SGDDoubleModelCompressionWithoutMem(PredefinedParameters):

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False, model: ACostModel = RMSEModel(),
               stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          cost_model=model,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=False,
                          compress_gradients=False
                          )


class SGDDoubleModelCompressionWithMem(PredefinedParameters):

    def define(self, n_dimensions: int, nb_devices: int, quantization_param: int, step_formula=None,
               momentum: float = 0, nb_epoch: int = NB_EPOCH, use_averaging=False, model: ACostModel = RMSEModel(),
               stochastic=True):
        return Parameters(n_dimensions=n_dimensions,
                          nb_devices=nb_devices,
                          nb_epoch=nb_epoch,
                          step_formula=step_formula,
                          quantization_param=quantization_param,
                          momentum=momentum,
                          verbose=False,
                          stochastic=stochastic,
                          cost_model=model,
                          use_averaging=use_averaging,
                          bidirectional=True,
                          double_use_memory=True,
                          compress_gradients=False
                          )


KIND_COMPRESSION = [SGDWithoutCompression(),
                    Qsgd(),
                    Diana(),
                    BiQSGD(),
                    Artemis()
                    ]
