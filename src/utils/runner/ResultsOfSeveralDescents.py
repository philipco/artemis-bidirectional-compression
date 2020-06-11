"""
Created by Philippenko, 8th June 2020.
"""

import numpy as np


class ResultsOfSeveralDescents:
    """This class collect useful results obtained after running multiple gradient descent of different kind.

    The class holds various information: (averaged) losses, the names of the algorithm to obtain this element
    of the sequence, the theoretical number of bits required to exchanged information.
    This terms are organised in two nested sequence of informations.  The first sequence list all algorithm,
    the second nested sequence the list of run for this particular kind of descent.
    For instance: all_losses = [ [algo1_run1, algo1_run2, algo1_run3], [algo2_run1, ...] ... ]
    The class provides method (in log scale or not) to compute the mean and the standard deviation of the nested sequence.
    """

    def __init__(self, all_descent, nb_devices_for_the_run):
        self.all_losses = [desc.losses for desc in all_descent.values()]
        self.all_losses_averaged = [desc.averaged_losses for desc in all_descent.values()]
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
