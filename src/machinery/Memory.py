"""
Created by Philippenko, 17th September 2021.

This class allows to store all important information required to handle the memory, and in particular all (h^i).
"""
import numpy as np
import torch

from src.machinery.Parameters import Parameters


class Memory:
    """Handle the memory."""

    def __init__(self, parameters: Parameters):
        super().__init__()
        self.parameters = parameters

        self.nb_it = 0

        if self.parameters.save_all_memories:
            self.h_i = [torch.zeros(parameters.n_dimensions, dtype=np.float)]
        else:
            self.h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

        self.averaged_h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)
        self.tail_averaged_h_i = torch.zeros(parameters.n_dimensions, dtype=np.float)

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

    def smart_initialization_with_unique_memory(self, first_gradient):
        if self.parameters.save_all_memories:
            self.h_i[-1] = self.h_i[-1] + first_gradient
        else:
            self.h_i = self.h_i[-1] + first_gradient
        self.averaged_h_i = self.h_i[-1]
        self.tail_averaged_h_i = self.h_i[-1]

    def smart_initialization(self, first_gradient):
        if self.parameters.save_all_memories:
            self.h_i[-1] = first_gradient
        else:
            self.h_i = first_gradient
        self.averaged_h_i = first_gradient
        self.tail_averaged_h_i = first_gradient