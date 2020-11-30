"""
Created by Constantin Philippenko, 6th March 2020.

This python file provide facilities to quantize tensors.
"""
import random
from abc import ABC, abstractmethod

import torch
from scipy.stats import bernoulli
from torch.distributions.bernoulli import Bernoulli
from math import sqrt


class CompressionModel(ABC):
    """
    """
    
    def __init__(self, level: int, dim: int):
        self.level = level
        self.dim = dim
        self.omega_c = self.__compute_omega_c__(dim)
    
    @abstractmethod
    def compress(self, vector: torch.FloatTensor):
        pass

    @abstractmethod
    def __compute_omega_c__(self, dim: int):
        pass


class TopKSparsification(CompressionModel):

    def compress(self, vector: torch.FloatTensor):
        assert 0 <= self.level < 100, "k must be expressend in percent."
        if self.level == 0:
            return vector
        nb_of_component_to_select = int(len(vector) * self.level / 100)
        values, indices = torch.topk(abs(vector), 3)
        compression = torch.zeros_like(vector)
        for i in indices:
            compression[i.item()] = vector[i.item()]
        return compression

    def __compute_omega_c__(self, dim: int):
        return self.level/dim


class RandomSparsification(CompressionModel):

    def __init__(self, level: int, dim: int, biased = True):
        super().__init__(level, dim)
        self.biased = biased

    def compress(self, vector: torch.FloatTensor):
        assert 0 <= self.level < 100, "k must be expressed in percent."
        if self.level == 0:
            return vector
        proba = self.level/100
        indices = bernoulli.rvs(proba, size=len(vector))
        compression = torch.zeros_like(vector)
        for i in range(len(vector)):
            if indices[i]:
                compression[i] = vector[i] * [1/proba, 1][self.biased]
        return compression

    def __compute_omega_c__(self, dim: int):
        return self.level/dim


class SQuantization(CompressionModel):

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
        norm_x = torch.norm(vector, p=2)
        if norm_x == 0:
            return vector
        ratio = torch.abs(vector) / norm_x
        l = torch.floor(ratio * self.level)
        p = ratio * self.level - l
        sampled = Bernoulli(p).sample()
        qtzt = torch.sign(vector) * norm_x * (l + sampled) / self.level
        return qtzt

    def __compute_omega_c__(self, dim: int):
        """Return the value of omega_c (involved in variance) of the s-quantization."""
        # If s==0, it means that there is no compression.
        # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
        if self.level==0:
            return sqrt(dim)
        return min(dim / self.level*self.level, sqrt(dim) / self.level)


def one_bit_quantization(x, _):
    """Implement the one bit quantization. Not tested, no guarantee to be correct."""
    quantized = torch.zeros_like(x)
    norm_x = torch.norm(x, p=2)
    for j in range(len(x)):
        quantized[j] = norm_x * torch.sign(x)[j]
    return quantized


def bernouilli_quantization(x):
    """Implement the bernouilli quantization. Not tested, no guarantee to be correct."""
    quantized = torch.zeros_like(x)
    norm_x = torch.norm(x, p=2)
    for j in range(len(x)):
        b = bernoulli.rvs(abs(x[j])/norm_x, size=1)[0]
        quantized[j] = norm_x * torch.sign(x)[j] * b
    return quantized