from src.deeplearning.Compression import QuantizationCompressor
from src.deeplearning.NeuralNetworksModel import TwoLayersModel
from src.utils.Constants import NB_EPOCH, NB_WORKERS


class Parameters:

    def __init__(self,
                 model_id: str = "2Layers",
                 nb_devices: int = NB_WORKERS,
                 batch_size: int = 1,
                 nb_epoch: int = NB_EPOCH,
                 quantization_param: int = 1,
                 type_device: str = "cpu",
                 bidirectional: bool = False,
                 verbose: bool = False,
                 stochastic: bool = True,
                 use_averaging: bool = False) -> None:
        super().__init__()

        assert type_device in ["cpu", "cuda"], "Device must be either 'cpu'; either 'cuda'."

        self.model_id = model_id
        self.n_dimensions = self.get_model()().number_of_param()  # Dimension of the problem.
        self.nb_devices = nb_devices  # Number of device on the network.
        self.batch_size = batch_size  # Batch size.
        self.nb_epoch = nb_epoch  # number of epoch of the run
        self.compressor = QuantizationCompressor(self.n_dimensions, quantization_param, nb_devices, nb_epoch)
        self.type_device = type_device
        self.learning_rate = 1 / (2 * (self.compressor.omega_c + 1))
        self.bidirectional = bidirectional
        self.verbose = verbose
        self.stochastic = stochastic  # true if runing a stochastic gradient descent
        self.use_averaging = use_averaging  # true if using a Polyak-Ruppert averaging.

    def get_model(self):
        if self.model_id == "2Layers":
            return TwoLayersModel
        else:
            raise NotImplementedError