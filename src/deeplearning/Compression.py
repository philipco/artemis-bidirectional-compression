import torch
from torch.distributions.bernoulli import Bernoulli


class QuantizationCompressor(object):

    def compress(self, x: torch.FloatTensor, s: int) -> torch.FloatTensor:
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


