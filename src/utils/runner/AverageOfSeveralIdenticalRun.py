"""
Created by Philippenko, 8th June 2020.
"""
from src.deeplearning import DeepLearningRun
from src.machinery import GradientDescent

from src.utils.Utilities import compute_number_of_bits

import numpy as np


class AverageOfSeveralIdenticalRun:
    """
    This class gathers the result of multiple gradient descents performed in the same condition with the aim to later
    average the loss, and compute the variance.
    """

    def __init__(self):
        self.multiple_descent = []
        self.train_losses = []
        self.averaged_train_losses = []
        self.norm_error_feedback = []
        self.dist_to_model = []
        self.var_models = []
        self.theoretical_nb_bits = None
        self.omega_c = None
        self.artificial = False

        # Required for Deep Learning
        self.test_losses = []
        self.test_accuracies = []

    def get_last(self):
        return self.multiple_descent[-1]

    def append(self, new_descent: GradientDescent):
        if not self.theoretical_nb_bits:
            compress_model = True if new_descent.get_name() == "DwnComprModel" else False
            self.theoretical_nb_bits = compute_number_of_bits(new_descent.parameters, len(new_descent.train_losses), compress_model)
            self.omega_c = new_descent.parameters.up_compression_model.omega_c
        self.multiple_descent.append(new_descent)
        self.train_losses = [d.train_losses for d in self.multiple_descent]
        self.averaged_train_losses = [d.averaged_train_losses for d in self.multiple_descent]
        self.norm_error_feedback = [d.norm_error_feedback for d in self.multiple_descent]
        self.dist_to_model = [d.dist_to_model for d in self.multiple_descent]
        self.var_models = [d.var_models for d in self.multiple_descent]

    def append_from_DL(self, new_run: DeepLearningRun):
        self.multiple_descent.append(new_run)
        self.train_losses = [d.train_losses for d in self.multiple_descent]
        self.test_losses = [d.test_losses for d in self.multiple_descent]
        self.test_accuracies = [d.test_accuracies for d in self.multiple_descent]

    def append_list(self, my_list, my_list_averaged, my_list_norm_ef, my_list_dist_model, my_list_var_models):
        """Used when running experiments with different step size/compression.

        :param my_list:
        :param my_list_averaged:
        :param my_list_norm_ef:
        :return:
        """
        number_points = len(my_list[0])
        for i in range(number_points):
            loss, loss_avg, norm_ef, dist_model, var_models = [], [], [], [], []
            for list, list_avg, list_ef, list_dist, list_var_models in \
                    zip(my_list, my_list_averaged, my_list_norm_ef, my_list_dist_model, my_list_var_models):
                loss.append(list[i])
                loss_avg.append(list_avg[i])
                norm_ef.append(list_ef[i])
                dist_model.append(list_dist[i])
                var_models.append(list_var_models[i])
            self.train_losses.append(np.array(loss))
            self.averaged_train_losses.append(np.array(loss_avg))
            self.norm_error_feedback.append(np.array(norm_ef))
            self.dist_to_model.append(np.array(dist_model))
            self.var_models.append(np.array(var_models))
        self.artificial = True