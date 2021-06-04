"""
Created by Philippenko, 8th June 2020.
"""
from random import random, choice

import numpy as np

from src.deeplearning.DeepLearningRun import DeepLearningRun
from src.machinery.GradientDescent import AGradientDescent
from src.utils.Utilities import compute_number_of_bits


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
        self.all_descent = all_descent
        element = list(self.all_descent.values())[-1].multiple_descent[-1]
        if not self.all_descent[next(iter(self.all_descent))].artificial and isinstance(element, AGradientDescent):
            self.all_final_model = [desc.multiple_descent[-1].model_params[-1] for desc in self.all_descent.values()]
            self.X_number_of_bits = [desc.theoretical_nb_bits for desc in self.all_descent.values()]
            self.omega_c = [desc.omega_c for desc in self.all_descent.values()]
        elif isinstance(element, DeepLearningRun):
            X_number_of_bits = []
            for key, value in self.all_descent.items():
                compress_model = True if 'MCM' in key else False
                params = value.multiple_descent[-1].parameters
                X_number_of_bits.append(compute_number_of_bits(params, params.nb_epoch, compress_model))
            self.X_number_of_bits = X_number_of_bits
        self.nb_devices_for_the_run = nb_devices_for_the_run
        self.update()

    def recompute_nb_bits(self):
        X_number_of_bits = []
        for key, value in self.all_descent.items():
            if 'SGD' == key:
                X_number_of_bits.append(self.X_number_of_bits[0])
            elif 'Diana' == key or 'QSGD' == key:
                X_number_of_bits.append(self.X_number_of_bits[0] / 2)
            else:
                X_number_of_bits.append(self.X_number_of_bits[0] / 16)
        self.X_number_of_bits = X_number_of_bits

    def add_descent(self, descent, name):
        self.all_descent[name] = descent
        self.update()

    def update(self):
        self.all_train_losses = [desc.train_losses for desc in self.all_descent.values()]
        self.all_train_losses_averaged = [desc.averaged_train_losses for desc in self.all_descent.values()]
        self.norm_error_feedback = [desc.norm_error_feedback for desc in self.all_descent.values()]
        self.distance_to_model = [desc.dist_to_model for desc in self.all_descent.values()]
        self.var_models = [desc.var_models for desc in self.all_descent.values()]
        self.names = [names for names in self.all_descent]

        # Required for Deep Learning
        self.all_test_losses = [desc.test_losses for desc in self.all_descent.values()]
        self.all_test_accuracies = [desc.test_accuracies for desc in self.all_descent.values()]

    def get_losses_i(self, i: int, averaged: bool = False):
        if averaged:
            return self.all_train_losses_averaged[i]
        return self.all_train_losses[i]

    def get_loss(self, obj, averaged: bool = False, in_log = True):
        """Return the sequence of averaged losses for each of the algorithm.

        Args:
            obj: the objective loss which from which we compare.
            averaged: return the loss for the averaged sequence.
            in_log: return the result using log scale.
        """
        if averaged:
            all_losses = self.all_train_losses_averaged
        else:
            all_losses = self.all_train_losses
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
            all_losses = self.all_train_losses_averaged
        else:
            all_losses = self.all_train_losses
        std_losses = []
        for losses in all_losses:
            if in_log:
                log_losses = [np.log10(loss - obj) for loss in losses]
                std_losses.append(np.std(log_losses, axis=0))
            else:
                std_losses.append(np.std(losses - obj, axis=0))
        return std_losses

    def getter(self, seq_values, in_log=True):
        res = []
        for values in seq_values:
            if in_log:
                log_e = [np.log10(e) for e in values]
                res.append(np.mean(log_e, axis=0))
            else:
                res.append(np.mean(values, axis=0))
        return res

    def getter_std(self, seq_values, in_log=True):
        res_std = []
        for values in seq_values:
            if in_log:
                log_e = [np.log10(e) for e in values]
                res_std.append(np.std(log_e, axis=0))
            else:
                res_std.append(np.std(values, axis=0))
        return res_std

    def get_error_feedback(self, in_log=True):
        """Return the sequence of error feedback for each of the algorithm."""
        return self.getter(self.norm_error_feedback)

    def get_error_feedback_std(self, in_log=True):
        """Return the sequence of error feedback for each of the algorithm. """
        return self.getter_std(self.norm_error_feedback)

    def get_distance_to_model(self, in_log=True):
        """Return the sequence of average distance between the central model and the remotes one for each of the
        algorithm. """
        return self.getter(self.distance_to_model)

    def get_distance_to_model_std(self, in_log=True):
        return self.getter_std(self.norm_error_feedback)

    def get_var_models(self, in_log=True):
        """Return the sequence of average distance between the central model and the remotes one for each of the
        algorithm. """
        return self.getter(self.var_models)

    def get_var_models_std(self, in_log=True):
        return self.getter_std(self.var_models)

    def get_test_accuracies(self, in_log=False):
        return self.getter(self.all_test_accuracies, False)

    def get_test_accuracies_std(self, in_log=False):
        return self.getter_std(self.all_test_accuracies, False)

    def get_test_losses(self, in_log=False):
        return self.getter(self.all_test_losses, in_log)

    def get_test_losses_std(self, in_log=False):
        return self.getter_std(self.all_test_losses, in_log)


