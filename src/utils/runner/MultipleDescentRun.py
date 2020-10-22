"""
Created by Philippenko, 8th June 2020.
"""

from src.machinery import GradientDescent

from src.utils.Utilities import compute_number_of_bits

import numpy as np


class MultipleDescentRun:
    """
    This class gathers the result of multiple gradient descents performed in the same condition with the aim to later
    average the loss, and compute the variance.
    """

    def __init__(self):
        self.multiple_descent = []
        self.losses = []
        self.averaged_losses = []
        self.theoretical_nb_bits = []
        self.artificial = False

    def get_last(self):
        return self.multiple_descent[-1]

    def append(self, new_descent: GradientDescent):
        if not self.theoretical_nb_bits:
            self.theoretical_nb_bits = compute_number_of_bits(new_descent.parameters, len(new_descent.losses))
        self.multiple_descent.append(new_descent)
        self.losses = [d.losses for d in self.multiple_descent]
        self.averaged_losses = [d.averaged_losses for d in self.multiple_descent]

    def append_list(self, my_list, my_list_averaged):
        number_points = len(my_list[0])
        for i in range(number_points):
            loss = []
            loss_avg = []
            for list, list_avg in zip(my_list, my_list_averaged):
                loss.append(list[i])
                loss_avg.append(list_avg[i])
            self.losses.append(np.array(loss))
            self.averaged_losses.append(np.array(loss_avg))
        self.artificial = True