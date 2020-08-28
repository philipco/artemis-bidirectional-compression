from src.utils.Constants import NB_EPOCH, NB_WORKERS


class Parameters:

    def __init__(self,
                 n_dimensions: int = 42,
                 nb_devices: int = NB_WORKERS,
                 batch_size: int = 1,
                 nb_epoch: int = NB_EPOCH,
                 quantization_param: int = 1,
                 type_device: str = "cpu",
                 learning_rate: int = None,
                 bidirectional: bool = False,
                 verbose: bool = False,
                 stochastic: bool = True,
                 use_averaging: bool = False) -> None:
        super().__init__()

        assert type_device in ["cpu", "cuda"], "Device must be either 'cpu'; either 'cuda'."

        self.n_dimensions = n_dimensions  # Dimension of the problem.
        self.nb_devices = nb_devices  # Number of device on the network.
        self.batch_size = batch_size  # Batch size.
        self.nb_epoch = nb_epoch  # number of epoch of the run
        self.quantization_param = quantization_param  # quantization parameter
        self.type_device = type_device
        self.omega_c = 0  # quantization constant involved in the variance inequality of the scheme
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.verbose = verbose
        self.stochastic = stochastic  # true if runing a stochastic gradient descent
        self.use_averaging = use_averaging  # true if using a Polyak-Ruppert averaging.