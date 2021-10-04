"""
Created by Philippenko, 23rd September 2021.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import math

from src.machinery.Parameters import Parameters


class AnytimeTailAverage(ABC):

    def __init__(self, parameters: Parameters):
        super().__init__()
        self.zeros = torch.zeros(parameters.n_dimensions, dtype=np.float)

    @abstractmethod
    def compute_average(self, nb_it, new_value):
        pass


class AnytimeWindowsAverage2Acc(AnytimeTailAverage):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        self.c = 0.5

        # Number of element in the accumulator 0.
        self.N0 = 0
        # Number of element in the accumulator 1.
        self.N1 = 0

        # Accumulators
        self.accumulator0 = self.zeros
        self.accumulator1 = self.zeros

    def coef_for_update(self, nb_it):
        square = 1 / (self.N0 * self.c * nb_it) + 1 / (self.N1 * self.c * nb_it) - 1 / (self.N0 * self.N1)
        square = math.sqrt(square)
        denom = self.N0 + self.N1
        return (self.N0 - self.N0 * self.N1 * square) / denom

    def compute_average(self, nb_it, new_value):

        # As a sample arrives, we update the second accumulator and leave the first one untouched
        self.N1 += 1
        self.accumulator1 = self.accumulator1 + (new_value - self.accumulator1) / self.N1

        if nb_it in [1, 2]:
            average = new_value
        else:
            # Computing the average.
            average = self.accumulator1 + self.coef_for_update(nb_it) * (self.accumulator0 - self.accumulator1)

        if self.N1 >= self.c * nb_it:
            self.accumulator0 = self.accumulator1
            self.N0 = self.N1
            self.accumulator1 = self.zeros
            self.N1 = self.zeros

        return average


class AnytimeWindowsAverage3Acc(AnytimeTailAverage):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        self.c = 0.5

        # Number of element in the accumulator 0.
        self.N0 = 0
        # Number of element in the accumulator 1.
        self.N1 = 0
        # Number of element in the accumulator 2.
        self.N2 = 0

        # Accumulators
        self.accumulator0 = self.zeros
        self.accumulator1 = self.zeros
        self.accumulator2 = self.zeros

    def coef_for_update(self, nb_it):
        sum_of_N = self.N1 + self.N2

        square = 1 / (self.N0 * self.c * nb_it) + 1 / (sum_of_N * self.c * nb_it) - 1 / (self.N0 * sum_of_N)
        square = math.sqrt(square)
        denom = self.N0 + sum_of_N
        return (self.N0 - self.N0 * sum_of_N * square) / denom

    def compute_average(self, nb_it, new_value):

        # As a sample arrives, we update the second accumulator and leave the first one untouched
        self.N1 += 1
        self.accumulator1 = self.accumulator1 + (new_value - self.accumulator1) / self.N1

        if nb_it in [1, 2]:
            average = new_value
        else:
            sum_of_N = self.N1 + self.N2
            sum_of_acc = (self.N1 * self.accumulator1 + self.N2 * self.accumulator2)
            average_of_acc = sum_of_acc / sum_of_N
            # Computing the average.
            average = average_of_acc + self.coef_for_update(nb_it) * (self.accumulator0 - average_of_acc)

        if self.N2 + self.N1 >= self.c * nb_it:
            self.accumulator0 = self.accumulator1
            self.N0 = self.N1
            self.accumulator1 = self.accumulator2
            self.N1 = self.N2
            self.accumulator2 = self.zeros
            self.N2 = 0

        return average


class AnytimeExpoAverage(AnytimeTailAverage):

    def __init__(self, parameters: Parameters):
        super().__init__(parameters)

        self.c = 0.5

        # Number of element in the accumulator 0.
        self.N0 = 0

        # Accumulator
        self.accumulator = self.zeros

    def compute_average(self, nb_it, new_value):
        if nb_it == 1:
            return new_value
        gamma1 = self.c * (nb_it - 1) / (1 + self.c * (nb_it - 1))
        gamma2 = 1 - math.sqrt((1 - self.c) / (nb_it * (nb_it - 1))) / self.c
        gamma = gamma1 * gamma2
        self.accumulator = gamma * self.accumulator + (1 - gamma) * new_value
        return self.accumulator
