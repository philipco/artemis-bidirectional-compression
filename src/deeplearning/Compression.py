from abc import ABC, abstractmethod
from math import sqrt, log

import torch
from torch.distributions.bernoulli import Bernoulli


class CompressionOperator(ABC):

    def __init__(self, dim: int, s: int,  nb_devices: int, nb_epoch: int) -> None:
        super().__init__()
        self.quantization_param = s
        self.n_dimensions = dim
        self.nb_devices = nb_devices
        self.nb_epoch = nb_epoch
        self.omega_c = self.compute_omega_c()

    @abstractmethod
    def compute_omega_c(self):
        pass

    @abstractmethod
    def compress(self):
        pass

    @abstractmethod
    def number_of_bits_needed_to_communicates_compressed(self):
        pass

    @abstractmethod
    def number_of_bits_needed_to_communicates_no_compressed(self):
        pass

    @abstractmethod
    def compute_number_of_bits(self):
        pass


class QuantizationCompressor(CompressionOperator):

    def compute_omega_c(self):
        """Return the value of omega_c (involved in variance) of the s-quantization."""
        # If s==0, it means that there is no compression.
        # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
        if self.quantization_param == 0:
            return sqrt(self.n_dimensions)
        return min(self.n_dimensions / self.quantization_param * self.quantization_param, sqrt(self.n_dimensions) / self.quantization_param)

    def compress(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.quantization_param == 0:
            return x
        norm_x = torch.norm(x, p=2)
        if norm_x == 0:
            return x
        ratio = torch.abs(x) / norm_x
        l = torch.floor(ratio * self.quantization_param)
        p = ratio * self.quantization_param - l
        sampled = Bernoulli(p).sample()
        qtzt = torch.sign(x) * norm_x * (l + sampled) / self.quantization_param
        return qtzt

    def number_of_bits_needed_to_communicates_compressed(self) -> int:
        """Computing the theoretical number of bits used for a single way when using compression (with Elias encoding)."""
        s = self.quantization_param
        frac = 2 * (s ** 2 + self.n_dimensions) / (s * (s + sqrt(self.n_dimensions)))
        return self.nb_devices * (3 + 3 / 2) * log(frac) * s * (s + sqrt(self.n_dimensions)) + 32

    def number_of_bits_needed_to_communicates_no_compressed(self) -> int:
        """Computing the theoretical number of bits used for a single way when using compression (with Elias encoding)."""
        return self.nb_devices * self.n_dimensions * 32

    def compute_number_of_bits(self, bidirectional: bool):
        """Computing the theoretical number of bits used by an algorithm (with Elias encoding)."""
        # Initialization, the first element needs to be removed at the end.
        number_of_bits = [0]
        d = self.n_dimensions
        for i in range(self.nb_epoch):
            if bidirectional:
                s = self.quantization_param
                nb_bits = 2 * self.number_of_bits_needed_to_communicates_compressed()
            elif self.quantization_param != 0:
                s = self.quantization_param
                nb_bits = self.number_of_bits_needed_to_communicates_no_compressed() \
                          + self.number_of_bits_needed_to_communicates_compressed()
            else:
                nb_bits = 2 * self.number_of_bits_needed_to_communicates_no_compressed()

            number_of_bits.append(nb_bits + number_of_bits[-1])
        return number_of_bits[1:]


