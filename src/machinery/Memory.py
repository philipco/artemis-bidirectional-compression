"""
Created by Philippenko, 17th September 2021.

This class allows to store all important information required to handle the memory, and in particular all (h^i).
"""
import numpy as np
import torch

from src.machinery.Parameters import Parameters
from src.machinery.TailAverager import AnytimeWindowsAverage3Acc, AnytimeExpoAverage, ExpoAverage


class Memory:
    """Handle the memory."""

    def __init__(self, parameters: Parameters, is_using_tail: bool):
        super().__init__()
        self.parameters = parameters
        self.is_using_tail = is_using_tail

        self.nb_it = 0

        if self.parameters.save_all_memories:
            self.h_i = [torch.zeros(parameters.n_dimensions, dtype=np.float)]
        else:
            self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.averaged_h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        if is_using_tail:
            self.tail_averaged_h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        if self.parameters.awa_tail_averaging:
            self.awa_averager = AnytimeWindowsAverage3Acc(parameters)
        elif self.parameters.expo_tail_averaging:
            self.expo_averager = AnytimeExpoAverage(parameters)
        elif self.parameters.simple_expo_tail_averaging:
            self.simple_expo_averager = ExpoAverage(parameters)

    def set_h_i(self, new_h):
        if self.parameters.save_all_memories:
            self.h_i.append(new_h)
        else:
            self.h_i = new_h

    def get_current_h_i(self):
        if self.parameters.save_all_memories:
            return self.h_i[-1]
        else:
            return self.h_i

    def smart_initialization(self, first_gradient):
        if self.parameters.save_all_memories:
            self.h_i[-1] = first_gradient
        else:
            self.h_i = first_gradient
        self.averaged_h_i = first_gradient
        if self.is_using_tail:
            self.tail_averaged_h_i = first_gradient