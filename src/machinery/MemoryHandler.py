"""
Created by Philippenko, 17th September 2021.

This class allows to handle computation related to memory.
"""
from abc import ABC, abstractmethod

import torch

from src.machinery.Memory import Memory
from src.machinery.Parameters import Parameters


class AbstractMemoryHandler(ABC):
    """Abstract class to handle memory."""

    def __init__(self, parameters: Parameters):
        super().__init__()
        self.parameters = parameters

    @abstractmethod
    def which_mem(self, memory: Memory):
        pass

    def update_memory(self, memory: Memory, quantized_delta_i):
        if self.parameters.use_up_memory:
            memory.set_h_i(self.compute_new_h_i(memory, quantized_delta_i))
            memory.nb_it += 1
            self.update_average_h_i(memory)
            if isinstance(self, TailAverageMemoryHandler):
                self.update_tail_average_h_i(memory)

    def update_tail_average_h_i(self, memory: Memory):
        n = memory.nb_it // 2
        new_val = memory.get_current_h_i()
        if self.parameters.awa_tail_averaging:
            memory.tail_averaged_h_i = memory.awa_averager.compute_average(memory.nb_it, new_val)
        elif self.parameters.expo_tail_averaging:
            memory.tail_averaged_h_i = memory.expo_averager.compute_average(memory.nb_it, new_val)
        elif self.parameters.simple_expo_tail_averaging:
            memory.tail_averaged_h_i = memory.simple_expo_averager.compute_average(memory.nb_it, new_val)
        else:
            assert self.parameters.save_all_memories == True, "If we compute the true tail, we need to store all the memories."
            # This is working correctly !
            if n == 0:
                return memory.h_i[-1]
            # Removing the first element of the list that will be no more used.
            if memory.nb_it % 2 == 0:
                memory.h_i = memory.h_i[1:]
            length = n + 1
            if memory.nb_it % 2 == 1:
                length += 1
            memory.tail_averaged_h_i = torch.sum(torch.stack(memory.h_i), 0) / length

    def update_average_h_i(self, memory: Memory):
        new_val = memory.get_current_h_i()
        memory.averaged_h_i = (1 - 1 / (memory.nb_it + 1)) * memory.averaged_h_i + 1 / (memory.nb_it + 1) * new_val

    def compute_new_h_i(self, memory: Memory, quantized_delta):
        # Called only if the use_memory flag is set to True.
        mem = memory.get_current_h_i() + self.parameters.up_learning_rate * quantized_delta
        if not self.parameters.debiased:
            return mem
        return mem + self.parameters.up_learning_rate * (self.which_mem(memory) - memory.get_current_h_i())


class NoMemoryHandler(AbstractMemoryHandler):
    """When there is no memory."""

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, memory: Memory):
        return 0


class ClassicMemoryHandler(AbstractMemoryHandler):
    """When we use the classical (h^i) as memory."""

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, memory: Memory):
        return memory.get_current_h_i()


class AverageMemoryHandler(AbstractMemoryHandler):
    """When we use the average of all (h^i) as memory."""

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

    def which_mem(self, memory: Memory):
        return memory.averaged_h_i


class TailAverageMemoryHandler(AbstractMemoryHandler):
    """When we use the average of all (h^i) as memory."""

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)
        self.parameters = parameters

    def which_mem(self, memory: Memory):
        return memory.tail_averaged_h_i
