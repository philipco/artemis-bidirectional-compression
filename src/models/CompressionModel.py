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
    
    def __init__(self, level: int, dim: int = None, norm: int = 2):
        self.level = level
        self.dim = dim
        self.norm = norm
        if dim is not None:
            self.omega_c = self.__compute_omega_c__(flat_dim=dim)
        else:
            self.omega_c = None
    
    @abstractmethod
    def compress(self, vector: torch.FloatTensor, dim: int = None):
        pass

    @abstractmethod
    def __compute_omega_c__(self, vector: torch.FloatTensor = None, flat_dim: int =None):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class TopKSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, norm: int = 2):
        super().__init__(level, dim, norm)
        if dim is not None:
            assert 0 <= level < dim, "k must be inferior to the number of dimension and superior to zero."
        self.biased = True

    def compress(self, vector: torch.FloatTensor, dim: int = None):
        if self.level == 0:
            return vector
        vector, dim, flat_dim = prep_grad(vector)

        values, indices = torch.topk(abs(vector), self.level)
        compression = torch.zeros_like(vector)
        for i in indices:
            compression[i.item()] = vector[i.item()]
        return compression.reshape(dim)

    def __compute_omega_c__(self, vector: torch.FloatTensor = None, flat_dim: int =None):
        if self.level == 0:
            return 0
        proba = self.level / self.dim
        return 1 - proba

    def get_name(self) -> str:
        return "Topk"


class RandomSparsification(CompressionModel):

    def __init__(self, level: int, dim: int = None, biased = True, norm: int = 2):
        """

        :param level: number of dimension to select at compression step
        :param dim: number of dimension in the dataset
        :param biased: set to True to used to biased version of this operators
        """
        self.biased = biased
        super().__init__(level, dim, norm)
        assert 0 <= level < dim, "k must be expressed in percent."

    def compress(self, vector: torch.FloatTensor, dim: int = None):
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

    def __compute_omega_c__(self, vector: torch.FloatTensor = None, flat_dim: int =None):
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

    def __init__(self, level: int, dim: int = None, norm: int = 2, div_omega: int = 1):
        self.biased = False
        self.div_omega = div_omega
        super().__init__(level, dim, norm)


    def compress(self, vector: torch.FloatTensor, dim: str = None) -> torch.FloatTensor:
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

        norm_x = torch.norm(vector, p=self.norm)
        if norm_x == 0:
            return vector.reshape(dim)

        all_levels = torch.floor(self.level * torch.abs(vector) / norm_x + torch.rand_like(vector)) / self.level
        signed_level = torch.sign(vector) * all_levels
        qtzt = signed_level * norm_x
        return qtzt.reshape(dim)

    def __compute_omega_c__(self, vector: torch.FloatTensor = None, flat_dim: int =None):
        """Return the value of omega_c (involved in variance) of the s-quantization."""
        # If s==0, it means that there is no compression.
        # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
        if flat_dim is None and vector is None:
            raise RuntimeError("The flat dimension and the vector cannot be None together.")
        if flat_dim is None:
            _, _, flat_dim = prep_grad(vector)
        if self.level == 0:
            return 0
        return min(flat_dim / (self.level*self.level*self.div_omega), sqrt(flat_dim) / (self.level*self.div_omega))

    def get_name(self) -> str:
        return "Qtzd"