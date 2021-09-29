"""
Created by Philippenko, 17th September 2021.
"""
from abc import ABC, abstractmethod

import torch

from src.machinery.Parameters import Parameters
from src.machinery.TailAverager import AnytimeWindowsAverage3Acc, AnytimeExpoAverage


class AbstractMemoryHandler(ABC):

    def __init__(self, parameters: Parameters):
        super().__init__()
        self.parameters = parameters

    @abstractmethod
    def which_mem(self, h_i, averaged_h_i, tail_averaged_h_i):
        pass

    def update_tail_average_mem(self, h_i, nb_it):
        new_val = h_i[-1]  # Used only if we don't have a unique memory.
        if self.parameters.awa_tail_averaging:
            return self.awa_averager.compute_average(nb_it, new_val)
        elif self.parameters.expo_tail_averaging:
            return self.expo_averager.compute_average(nb_it, new_val)
        else:
            assert not self.parameters.use_unique_up_memory, "When using true tail averaging, h_i should be the sequence of all h_i."
            n = (len(h_i) - 1) // 2
            return torch.mean(torch.stack(h_i[n:]), 0)

    def update_average_mem(self, h_i, average_mem, nb_it):
        if not self.parameters.use_unique_up_memory:
            new_val = h_i[-1]
        else:
            new_val = h_i
        return (1 - 1 / (nb_it + 1)) * average_mem + 1 / (nb_it + 1) * new_val

    def update_mem(self, h_i, averaged_h_i, quantized_delta):
        mem = h_i + self.parameters.up_learning_rate * quantized_delta
        if not self.parameters.debiased:
            return mem
        return mem + self.parameters.up_learning_rate * (averaged_h_i - h_i)


class NoMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, h_i, averaged_h_i, tail_averaged_h_i):
        return 0


class ClassicMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, h_i, averaged_h_i, tail_averaged_h_i):
        return h_i


class AverageMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, h_i, averaged_h_i, tail_averaged_h_i):
        return averaged_h_i


class TailAverageMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)
        self.parameters = parameters
        self.expo_averager = AnytimeExpoAverage(parameters)
        self.awa_averager = AnytimeWindowsAverage3Acc(parameters)

    def which_mem(self, h_i, averaged_h_i, tail_averaged_h_i):
        return tail_averaged_h_i
