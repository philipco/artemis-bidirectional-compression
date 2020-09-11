"""
Created by Philippenko, 8th June 2020.
"""

from src.machinery import GradientDescent

from src.utils.Utilities import compute_number_of_bits


class MultipleDescentRun:

    def __init__(self):
        self.multiple_descent = []
        self.losses = []
        self.averaged_losses = []
        self.theoretical_nb_bits = []

    def get_last(self):
        return self.multiple_descent[-1]

    def append(self, new_descent: GradientDescent):
        if not self.theoretical_nb_bits:
            self.theoretical_nb_bits = compute_number_of_bits(new_descent.parameters, len(new_descent.losses))
        self.multiple_descent.append(new_descent)
        self.losses = [d.losses for d in self.multiple_descent]
        self.averaged_losses = [d.averaged_losses for d in self.multiple_descent]