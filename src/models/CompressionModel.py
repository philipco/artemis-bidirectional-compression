"""
Created by Constantin Philippenko, 6th March 2020.

This python file provide facilities to quantize tensors.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import bernoulli
from math import sqrt


def prep_grad(vector):
    flat_vector = torch.unsqueeze(vector, 0).flatten()
    dim = vector.shape
    flat_dim = flat_vector.shape[0]
    return flat_vector, dim, flat_dim


class CompressionModel(ABC):
    """
    The CompressionModel class declares the factory methods while subclasses provide the implementation of this methods.

    This class defines the operators of compression.
    """
    
    def __init__(self, level: int, dim: int = None, norm: int = 2, constant: int = 1):
        self.level = level
        self.dim = dim
        self.norm = norm
        self.bucket_size = np.inf
        self.constant = constant
        if dim is not None:
            self.omega_c = self.__compute_omega_c__(flat_dim=dim)
        else:
            self.omega_c = None

    @abstractmethod
    def __compress__(self, vector: torch.FloatTensor, dim_to_use: int):
        """Compresses a vector with the mechanism of the operator of compression."""
        pass

    def compress(self, vector: torch.FloatTensor) -> torch.FloatTensor:
        """Prepare a vector for compression, and compresses it.

        :param vector: The vector to be compressed.
        :return: The compressed vector
        """

        if self.level == 0:
            return vector
        vector, dim, flat_dim = prep_grad(vector)

        if len(vector) > self.bucket_size:
            compressed_vector = torch.zeros_like(vector)
            for i in range(len(vector) // self.bucket_size + 1):
                compressed_vector[self.bucket_size * i: self.bucket_size * (i + 1)] = self.__compress__(
                    vector[self.bucket_size * i: self.bucket_size * (i + 1)])
        else:
            compressed_vector = self.__compress__(vector)

        return compressed_vector.reshape(dim)

    @abstractmethod
    def __omega_c_formula__(self, dim_to_use: int):
        """Proper implementation of the formula to compute omega_c.
        This formula is unique for each operator of compression."""
        pass

    def __compute_omega_c__(self, vector: torch.FloatTensor = None, flat_dim: int = None):
        """Compute the value of omega_c."""
        # If s==0, it means that there is no compression.
        # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
        if flat_dim is None and vector is None:
            raise RuntimeError("The flat dimension and the vector cannot be None together.")
        if flat_dim is None:
            _, _, flat_dim = prep_grad(vector)
        if self.level == 0:
            return 0
        if flat_dim > self.bucket_size:
            return self.__omega_c_formula__(self.bucket_size)
        else:
            return self.__omega_c_formula__(flat_dim)

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the operator of compression."""
        pass

    def get_learning_rate(self, *args, **kwds):
        """Compute the learning rate.

        No argument if the operator already know the dimension.
        Set as unique argument a vector ig the dimension is unknow (in the case of DL)."""
        if self.level == 0:
            return 0
        if len(args) == 1:
            return 1 / (self.constant * (self.__compute_omega_c__(args[0]) + 1))
        return 1 / (self.constant * (self.omega_c + 1))


class TopKSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2):
        super().__init__(level, dim, norm)
        if dim is not None:
            assert 0 <= level < dim, "k must be inferior to the number of dimension and superior to zero."
        self.biased = True

    def __compress__(self, vector: torch.FloatTensor):
        values, indices = torch.topk(abs(vector), self.level)
        compression = torch.zeros_like(vector)
        for i in indices:
            compression[i.item()] = vector[i.item()]
        return compression

    def __omega_c_formula__(self, dim_to_use: int):
        proba = self.level
        return 1 - proba

    def get_name(self) -> str:
        return "Topk"


class RandomSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, biased = False, norm: int = 2, constant: int = 2):
        """

        :param level: number of dimension to select at compression step
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        self.biased = biased
        super().__init__(level, dim, norm)
        assert 0 <= level <= 1, "k must be expressed in percent."

    def __compress__(self, vector: torch.FloatTensor):
        proba = self.level
        indices = bernoulli.rvs(proba, size=len(vector))
        compression = torch.zeros_like(vector)
        for i in range(len(vector)):
            if indices[i]:
                compression[i] = vector[i] * [1/proba, 1][self.biased]
        return compression

    def __omega_c_formula__(self, dim_to_use: int):
        proba = self.level
        if self.biased:
            return 1 - proba
        return (1-proba)/proba

    def get_name(self) -> str:
        if self.biased:
            return "RdkBsd"
        return "Rdk"


class SQuantization(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2, div_omega: int = 1, constant: int = 2):
        self.biased = False
        self.div_omega = div_omega
        super().__init__(level, dim, norm, constant)

    def __compress__(self, vector):
        norm_x = torch.norm(vector, p=self.norm)
        if norm_x == 0:
            return vector
        all_levels = torch.floor(self.level * torch.abs(vector) / norm_x + torch.rand_like(vector)) / self.level
        signed_level = torch.sign(vector) * all_levels
        qtzt = signed_level * norm_x
        return qtzt

    def __omega_c_formula__(self, dim_to_use):
        return min(dim_to_use / (self.level * self.level * self.div_omega),
                   sqrt(dim_to_use) / (self.level * self.div_omega))

    def get_name(self) -> str:
        return "Qtzd"