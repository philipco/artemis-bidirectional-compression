"""
Created by Philippenko, 8th June 2020.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from src.deeplearning.Artemis import Artemis
from src.deeplearning.NeuralNetworksModel import TwoLayersModel
from src.deeplearning.Parameters import Parameters
from src.deeplearning.Worker import Worker
from src.utils.Constants import NB_EPOCH, LR
from src.utils.Utilities import compute_number_of_bits
from src.utils.runner.RunnerUtilities import NB_RUN


class ResultsOfSeveralDescents:
    """
    """

    def __init__(self, all_descent, nb_devices_for_the_run):
        self.all_losses = [desc.losses for desc in all_descent.values()]
        self.X_number_of_bits = [desc.theoretical_nb_bits for desc in all_descent.values()]
        self.nb_devices_for_the_run = nb_devices_for_the_run
        self.names = [names for names in all_descent]

    def get_losses_i(self, i: int, averaged: bool = False):
        if averaged:
            return self.all_losses_averaged[i]
        return self.all_losses[i]

    def get_loss(self, obj, averaged: bool = False, in_log = True):
        """Return the sequence of averaged losses for each of the algorithm.

        Args:
            obj: the objective loss which from which we compare.
            averaged: return the loss for the averaged sequence.
            in_log: return the result using log scale.
        """
        if averaged:
            all_losses = self.all_losses_averaged
        else:
            all_losses = self.all_losses
        mean_losses = []
        for losses in all_losses:
            if in_log:
                log_losses = [np.log10(loss - obj) for loss in losses]
                mean_losses.append(np.mean(log_losses, axis=0))
            else:
                mean_losses.append(np.mean(losses - obj, axis=0))
        return mean_losses

    def get_std(self, obj, averaged: bool = False, in_log=True):
        """Return the sequence of standard deviation of the losses for each of the algorithm.

        Args:
            obj: the objective loss which from which we compare.
            averaged: return the loss for the averaged sequence.
            in_log: return the result using log scale.
        """
        if averaged:
            all_losses = self.all_losses_averaged
        else:
            all_losses = self.all_losses
        std_losses = []
        for losses in all_losses:
            if in_log:
                log_losses = [np.log10(loss - obj) for loss in losses]
                std_losses.append(np.std(log_losses, axis=0))
            else:
                std_losses.append(np.std(losses - obj, axis=0))
        return std_losses


class MultipleDescentRun:

    def __init__(self, parameters: Parameters):
        self.multiple_descent = []
        self.losses = []
        self.theoretical_nb_bits = []
        self.parameters = parameters

    def get_last(self):
        return self.multiple_descent[-1]

    def append(self, losses):
        if not self.theoretical_nb_bits:
            self.theoretical_nb_bits = compute_number_of_bits(self.parameters)
        self.losses.append(losses)


def multiple_run_descent(parameters: Parameters,
                         loaders,
                         nb_epoch=NB_EPOCH,
                         use_averaging=False,
                         stochastic=True):
    multiple_descent = MultipleDescentRun(parameters)

    for i in range(NB_RUN):
        artemis = Artemis(loaders, parameters.type_device, variant=0)
        for round_idx in range(nb_epoch):
            artemis.step()
        multiple_descent.append(artemis.losses)
        print("---> final loss:", artemis.losses[-1])
    return multiple_descent