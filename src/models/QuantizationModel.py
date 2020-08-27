"""
Created by Constantin Philippenko, 6th March 2020.

This python file provide facilities to quantize tensors.
"""

import torch
from scipy.stats import bernoulli
from torch.distributions.bernoulli import Bernoulli
from math import sqrt


def s_quantization(x: torch.FloatTensor, s: int) -> torch.FloatTensor:
    """Implement the s-quantization

    Args:
        x: the tensor to be quantized.
        s: the parameter of quantization.

    Returns:
        The quantizated tensor.
    """
    if s == 0:
        return x
    norm_x = torch.norm(x, p=2)
    if norm_x == 0:
        return x
    ratio = torch.abs(x) / norm_x
    l = torch.floor(ratio * s)
    p = ratio * s - l
    sampled = Bernoulli(p).sample()
    qtzt = torch.sign(x) * norm_x * (l + sampled) / s
    return qtzt


def s_quantization_omega_c(dim: int, s: int):
    """Return the value of omega_c (involved in variance) of the s-quantization."""
    # If s==0, it means that there is no compression.
    # But for the need of experiments, we may need to compute the quantization constant associated with s=1.
    if s==0:
        return sqrt(dim)
    return min(dim / s*s, sqrt(dim) / s)


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