"""
Created by Philippenko, 12th March 2020.

This file provides tools to easily customize a gradient descent based on its hyperparameters.
"""
from src.models.CompressionModel import *
from src.utils.Constants import NB_EPOCH, NB_DEVICES, DIM

from math import sqrt


def full_batch_step_size(it, L, omega, N): return 1 / L
def deacreasing_step_size(it, L, omega, N): return 1 / (L * sqrt(it))


def default_step_formula(sto: bool):
    """Default formula to compute the step size at each iteration.

    Two cases are handled, if it is a stochastic run or a full batch descent."""

    return full_batch_step_size


class Parameters:
    """This class give the ability to tune hyper parameters of each federated gradient descent.

    Exemple of hyperparameters: step size, regularization rate, kind of compression ...
    By default it ran a stochastic non-federated non-regularized and single compressed (if possible) descent
    without parallelization for a RMSE model.
    """

    def __init__(self,
                 cost_models,
                 n_dimensions: int,
                 nb_devices: int = NB_DEVICES,
                 fraction_sampled_workers: float = 1.,
                 batch_size: int = 1,
                 step_formula=None,
                 nb_epoch: int = NB_EPOCH,
                 regularization_rate: int = 0,
                 momentum: float = 0,
                 up_compression_model: CompressionModel = SQuantization(0),
                 down_compression_model: CompressionModel = SQuantization(0),
                 up_learning_rate: int = None,
                 down_learning_rate: int = None,
                 force_learning_rate: bool = False,
                 verbose: bool = False,
                 stochastic: bool = True,
                 streaming: bool = False,
                 use_up_memory: bool = False,
                 use_down_memory: bool = False,
                 use_unique_up_memory: bool = True,
                 use_averaging: bool = False,
                 time_debug: bool = False,
                 randomized: bool = False,
                 down_error_feedback: bool = False,
                 up_error_feedback: bool = False,
                 nb_local_update: int = 1,
                 non_degraded: bool = False,
                 log_file: str = None) -> None:
        super().__init__()
        self.cost_models = cost_models  # Cost model to use for gradient descent.
        if cost_models:
            self.total_nb_points = np.sum(np.array([cost.X.shape[0] for cost in cost_models]))
        self.n_dimensions = n_dimensions  # Dimension of the problem.
        self.nb_devices = nb_devices  # Number of device on the network.
        self.fraction_sampled_workers = fraction_sampled_workers # Probability of a worker to be active at each round.
        self.batch_size = batch_size  # Batch size.
        self.nb_epoch = nb_epoch  # number of epoch of the run
        self.regularization_rate = regularization_rate  # coefficient of regularization
        self.force_learning_rate = force_learning_rate
        self.momentum = momentum  # momentum coefficient
        self.up_compression_model = up_compression_model
        self.down_compression_model = down_compression_model
        if step_formula is None:
            self.step_formula = default_step_formula(stochastic)
        else:
            self.step_formula = step_formula
        self.up_learning_rate = up_learning_rate  # Learning rate used when up updating memory.
        self.down_learning_rate = down_learning_rate  # Learning rate used when down updating memory.
        self.error_feedback_coef = 1
        self.stochastic = stochastic  # true if running a stochastic gradient descent
        self.streaming = streaming  # True if each sample should be used only once !
        self.use_up_memory = use_up_memory  # use memory when sending to global server
        self.use_down_memory = use_down_memory  # a memory at back communication
        self.reset_memories = False
        self.use_unique_up_memory = False#[use_unique_up_memory, True][fraction_sampled_workers == 1] # either use N memories, either a single one
        self.use_unique_down_memory = False#[use_unique_up_memory, True][fraction_sampled_workers == 1]
        self.verbose = verbose
        self.use_averaging = use_averaging  # true if using a Polyak-Ruppert averaging.
        self.time_debug = time_debug  # True is one want to debug the time spent in each procedure.
        self.randomized = randomized
        self.down_error_feedback = down_error_feedback
        self.up_error_feedback = up_error_feedback
        self.nb_local_update = nb_local_update
        self.non_degraded = non_degraded
        self.log_file = log_file

    def print(self):
        print("nb devices:", self.nb_devices)
        print("nb dimension:", self.n_dimensions)
        print("regularization rate:", self.regularization_rate)
        print("cost model", self.cost_model)
        print("omega_c", self.omega_c)
        print("step size", self.step_formula)
        print("momentum:", self.momentum)
        print("stochastic:", self.stochastic)
        print("use avg:", self.use_averaging)
