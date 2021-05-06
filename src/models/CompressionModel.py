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
    """
    
    def __init__(self, level: int, dim: int = None):
        self.level = level
        self.dim = dim
        if dim is not None:
            self.omega_c = self.__compute_omega_c__(dim)
        else:
            self.omega_c = None
    
    @abstractmethod
    def compress(self, vector: torch.FloatTensor):
        pass

    @abstractmethod
    def __compute_omega_c__(self, dim: int):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class TopKSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None):
        super().__init__(level, dim)
        if dim is not None:
            assert 0 <= level < dim, "k must be inferior to the number of dimension and superior to zero."
        self.biased = True

    def compress(self, vector: torch.FloatTensor):
        if self.level == 0:
            return vector
        vector, dim, flat_dim = prep_grad(vector)

        values, indices = torch.topk(abs(vector), self.level)
        compression = torch.zeros_like(vector)
        for i in indices:
            compression[i.item()] = vector[i.item()]
        return compression.reshape(dim)

    def __compute_omega_c__(self, dim: int):
        if self.level == 0:
            return 0
        proba = self.level / self.dim
        return 1 - proba

    def get_name(self) -> str:
        return "Topk"


class RandomSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, biased = True):
        """

        :param level: number of dimension to select at compression step
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        self.biased = biased
        super().__init__(level, dim)
        assert 0 <= level < dim, "k must be expressed in percent."

    def compress(self, vector: torch.FloatTensor):
        if self.level == 0:
            return vector

        vector, dim, flat_dim = prep_grad(vector)

        proba = self.level/self.dim
        indices = bernoulli.rvs(proba, size=len(vector))
        compression = torch.zeros_like(vector)
        for i in range(len(vector)):
            if indices[i]:
                compression[i] = vector[i] * [self.dim/self.level, 1][self.biased]
        return compression.reshape(dim)

    def __compute_omega_c__(self, dim: int):
        proba = self.level / self.dim
        if self.level == 0:
            return 0
        if self.biased:
            return 1 - proba
        return (1-proba)/proba

    def get_name(self) -> str:
        if self.biased:
            return "RdkBsd"
        return "Rdk"


class SQuantization(CompressionModel):

    def __init__(self, level: int, dim: int = None):
        super().__init__(level, dim)
        self.biased = False

    def compress(self, vector: torch.FloatTensor) -> torch.FloatTensor:
        """Implement the s-quantization

        Args:
            x: the tensor to be quantized.
            s: the parameter of quantization.

        Returns:
            The quantizated tensor.
        """

        if self.level == 0:
            return vector
        vector, dim, flat_dim = prep_grad(vector)

        norm_x = torch.norm(vector, p=np.inf)
        if norm_x == 0:
            return vector.reshape(dim)

        all_levels = torch.floor(self.level * torch.abs(vector) / norm_x + torch.rand_like(vector)) / self.level
        signed_level = torch.sign(vector) * all_levels
        qtzt = signed_level * norm_x
        return qtzt.reshape(dim)

    def __compute_omega_c__(self, dim: int):
        """Return the value of omega_c (involved in variance) of the s-quantization."""
        # If s==0, it means that there is no compression.
        # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
        if self.level == 0:
            return 0# TODO This has been changed ! Which impact ? Should be none ... sqrt(dim)
        return min(dim / self.level*self.level, sqrt(dim) / self.level)

    def get_name(self) -> str:
        return "Qtzd"