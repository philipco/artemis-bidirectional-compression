"""
Created by Philippenko, 17th September 2021.
"""
from abc import ABC, abstractmethod

import torch

from src.machinery.Memory import Memory
from src.machinery.Parameters import Parameters
from src.machinery.TailAverager import AnytimeWindowsAverage3Acc, AnytimeExpoAverage


class AbstractMemoryHandler(ABC):

    def __init__(self, parameters: Parameters):
        super().__init__()
        self.parameters = parameters

    @abstractmethod
    def which_mem(self, memory: Memory):
        pass

    def update_memory(self, memory: Memory, quantized_delta_i):
        if self.parameters.use_up_memory:
            memory.set_h_i(self.update_h_i(memory, quantized_delta_i))
            memory.nb_it += 1
            self.update_average_h_i(memory)
            self.update_tail_average_h_i(memory)

    def update_tail_average_h_i(self, memory: Memory):
        new_val = memory.get_current_h_i()
        if self.parameters.awa_tail_averaging:
            memory.tail_averaged_h_i = self.awa_averager.compute_average(memory.nb_it, new_val)
        elif self.parameters.expo_tail_averaging:
            memory.tail_averaged_h_i = self.expo_averager.compute_average(memory.nb_it, new_val)
        else:
            # assert not self.parameters.use_unique_up_memory, "When using true tail averaging, h_i should be the sequence of all h_i."
            n = (len(memory.h_i) - 1) // 2
            memory.tail_averaged_h_i = torch.mean(torch.stack(memory.h_i[n:]), 0)

    def update_average_h_i(self, memory: Memory):
        new_val = memory.get_current_h_i()
        memory.averaged_h_i = (1 - 1 / (memory.nb_it + 1)) * memory.averaged_h_i + 1 / (memory.nb_it + 1) * new_val

    def update_h_i(self, memory: Memory, quantized_delta):
        mem = memory.get_current_h_i() + self.parameters.up_learning_rate * quantized_delta
        if not self.parameters.debiased:
            return mem
        return mem + self.parameters.up_learning_rate * (memory.averaged_h_i - memory.get_current_h_i())


class NoMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, memory: Memory):
        return 0


class ClassicMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, memory: Memory):
        return memory.get_current_h_i()


class AverageMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, memory: Memory):
        return memory.averaged_h_i


class TailAverageMemoryHandler(AbstractMemoryHandler):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)
        self.parameters = parameters
        self.expo_averager = AnytimeExpoAverage(parameters)
        self.awa_averager = AnytimeWindowsAverage3Acc(parameters)

    def which_mem(self, memory: Memory):
        return memory.tail_averaged_h_i
