"""
Created by Philippenko, 12th March 2020.

This python file provide tools to easily customize a gradient descent based on its hyperparameters.
It also provide predefine parameters to run classical algorithm without introducing an error.
"""
from src.utils.Constants import NB_EPOCH, NB_WORKERS, DIM
from src.models.CostModel import ACostModel, RMSEModel

from math import sqrt


def constant_step_size_formula(bidirectional: bool, nb_devices: int, n_dimensions: int):
    if sqrt(n_dimensions) >= nb_devices:
        if bidirectional:
            return lambda it, L, omega, N: N / (4 * omega * L * (omega + 1))
            # If omega = 0, it means we don't use compression and hence, the step size can be bigger.
        return lambda it, L, omega, N: 1 / (L * sqrt(it))  # N / (4 * omega * L) if omega != 0 else N / (2 * L)
    else:
        if bidirectional:
            return lambda it, L, omega, N: 1 / (5 * L * (omega + 1))
        # If omega = 0, it means we don't use compression and hence, the step size can be bigger.
        return lambda it, L, omega, N: 1 / (5 * L) if omega != 0 else 1 / (2 * L)

def full_batch_step_size(it, L, omega, N): return 1 / L
def bi_large_dim(it, L, omega, N): return N / (4 * omega * (omega + 1) * L)
def uni_large_dim(it, L, omega, N): return N / (4 * omega * L)
def deacreasing_step_size(it, L, omega, N): return 1 / (L * sqrt(it))
def large_batch_in_large_dim(it, L, omega, N): return N / (8 * L)


def step_formula_when_using_big_batch_size(sto: bool, bi: bool, quantization_param: int): return large_batch_in_large_dim


def step_formula_in_large_dimension(sto: bool, bi: bool, quantization_param: int, batch_size: int):
    """Default formula to compute the step size at each iteration.

    Two cases are handled, if it is a stochastic run or a full batch descent."""

    if batch_size >= 150 or not sto:
        return step_formula_when_using_big_batch_size(sto, bi, quantization_param)
    return bi_large_dim


def default_step_formula(sto: bool):
    """Default formula to compute the step size at each iteration.

    Two cases are handled, if it is a stochastic run or a full batch descent."""

    if sto:
        return deacreasing_step_size
    return full_batch_step_size


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
                 nb_devices: int = NB_WORKERS,
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
        self.step_formula = default_step_formula(stochastic) if sqrt(n_dimensions) < 0.5 * nb_devices \
            else step_formula_in_large_dimension(stochastic, bidirectional, quantization_param, batch_size)
        # To compute the step size at each iteration, we use a lambda function which takes as parameters
        # the number of current epoch, the coefficient of smoothness and the quantization constant omega_c.
        # if step_formula == None:
        #     self.step_formula = constant_step_size_formula(bidirectional, nb_devices, n_dimensions)
        #     self.step_formula = default_step_formula(stochastic)
        # else:
        #     self.step_formula = step_formula
        self.nb_epoch = nb_epoch  # number of epoch of the run
        self.regularization_rate = regularization_rate  # coefficient of regularization
        self.force_learning_rate = force_learning_rate
        self.momentum = momentum  # momentum coefficient
        self.quantization_param = quantization_param  # quantization parameter
        self.omega_c = 0  # quantization constant involved in the variance inequality of the scheme
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.stochastic = stochastic  # true if runing a stochastic gradient descent
        self.compress_gradients = compress_gradients
        self.double_use_memory = double_use_memory  # Use a
        self.verbose = verbose
        self.use_averaging = use_averaging  # true if using a Polyak-Ruppert averaging.

    def print(self):
        print("federated", self.federated)
        print("nb devices:", self.nb_devices)
        print("nb dimension:", self.n_dimensions)
        print("quantization param:", self.quantization_param)
        print("regularization rate:", self.regularization_rate)
        print("cost model", self.cost_model)
        print("omega_c", self.omega_c)
        print("step size", self.step_formula)
        print("momentum:", self.momentum)
        print("stochastic:", self.stochastic)
        print("use avg:", self.use_averaging)
